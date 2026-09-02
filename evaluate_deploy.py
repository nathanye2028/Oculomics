"""
evaluate_deploy.py
==================
**Deployment evaluation — the answer to RQ3.**

RQ3 asks whether the model can be quantized "without significant degradation in
diagnostic accuracy". Until now only *size* and *latency* were measured, and on
an untrained network. This script closes that gap: it takes a trained mBRSET
checkpoint and reports, on the SAME patient-grouped held-out test split the
model was trained against:

  1. **FP32 vs INT8 accuracy** (AUROC / sensitivity / specificity) -> the RQ3 number
  2. **A calibrated operating point.** AUROC is threshold-free, but a deployed
     screener must pick a threshold. We select it on VAL (never test) and report
     sensitivity/specificity/PPV/NPV on test -- what a clinician actually asks.
  3. **Size + latency** for the FP32 and INT8 artefacts.
  4. Optional **test-time augmentation** (horizontal flip), which is free accuracy.

Split integrity: the split is regenerated from the checkpoint's own seed, so the
test set is exactly the one held out during that run.

    python evaluate_deploy.py --root <mBRSET dir> --ckpt ck_mbrset8/gcg_seed0.pt

Note on ensembling: models from different seeds use DIFFERENT patient splits, so
averaging them would leak test data. Ensembling is therefore not offered here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MBRSETDataset, stratified_split     # noqa: E402
from model import MBRSETClassifier                       # noqa: E402
from fundus_utils import seed_everything, seed_worker    # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def operating_point(scores, labels, target_sens=None):
    """Threshold chosen on VAL. Either the HIGHEST threshold that still meets a
    target sensitivity (screening style: give up as little specificity as the
    sensitivity floor allows) or Youden's J.

    roc_curve returns thresholds in decreasing order, so tpr is non-decreasing
    along the array and the first index with tpr >= target is the largest
    threshold satisfying it. drop_intermediate=False is load-bearing: the
    default prunes collinear ROC points, which can drop exactly the threshold
    that first reaches the target and hand back a lower one (less specificity
    than the data allows). Falls back to Youden's J if the target is
    unreachable.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    if target_sens is not None:
        ok = np.where(tpr >= target_sens)[0]
        i = int(ok[0]) if len(ok) else int(np.argmax(tpr - fpr))
    else:
        i = int(np.argmax(tpr - fpr))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def binary_report(scores, labels, thr):
    """Sensitivity / specificity / PPV / NPV / accuracy at a fixed threshold."""
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    d = lambda a, b: float(a / b) if b else float("nan")
    return {"threshold": float(thr), "sensitivity": d(tp, tp + fn), "specificity": d(tn, tn + fp),
            "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn), "accuracy": d(tp + tn, tp + tn + fp + fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


@torch.no_grad()
def torch_scores(model, loader, device, tta=False):
    """Positive-class probabilities from the PyTorch model."""
    model.eval()
    S, Y = [], []
    for b in loader:
        x, y = b["image"].to(device), b["label"]
        p = torch.softmax(model(x), 1)
        if tta:                                    # horizontal flip averaging
            p = (p + torch.softmax(model(torch.flip(x, dims=[3])), 1)) / 2
        S.append(p[:, 1].cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(S), np.concatenate(Y)


def onnx_scores(path, loader, tta=False):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 1
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    S, Y = [], []
    for b in loader:
        x = b["image"].numpy()
        lg = sess.run(None, {name: x})[0]
        p = torch.softmax(torch.from_numpy(lg), 1)
        if tta:
            lg2 = sess.run(None, {name: np.ascontiguousarray(x[:, :, :, ::-1])})[0]
            p = (p + torch.softmax(torch.from_numpy(lg2), 1)) / 2
        S.append(p[:, 1].numpy()); Y.append(b["label"].numpy())
    return np.concatenate(S), np.concatenate(Y)


LATENCY_NOTE = ("ONNX Runtime CPU proxy on the evaluation machine; NOT device latency. "
                "The deployment number is export_coreml.py on the ANE / an Xcode report.")


def bench_onnx(path, x_np, runs=20, warmup=5):
    """Per-call CPU latency in ms as {p10, median, p90}: the median is the
    number to read (a mean is dragged around by scheduler hiccups), matching
    export_coreml.py's protocol."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 1
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    n = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {n: x_np})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {n: x_np})
        times.append((time.perf_counter() - t) * 1000.0)
    return {"p10": float(np.percentile(times, 10)), "median": float(np.median(times)),
            "p90": float(np.percentile(times, 90))}


def main() -> int:
    p = argparse.ArgumentParser(description="Deployment evaluation: FP32 vs INT8 accuracy, latency, operating point.")
    p.add_argument("--root", required=True,
                   help="Root of the dataset the checkpoint was TRAINED on (mBRSET dir, or "
                        "BRSET dir for a --dataset brset checkpoint; auto-detected from the "
                        "checkpoint's recorded args).")
    p.add_argument("--ckpt", required=True, help="Trained checkpoint (e.g. ck_kd_v4_384/kd_seed1.pt).")
    p.add_argument("--external-root", default=None,
                   help="Optional second dataset (e.g. mBRSET for a BRSET-trained checkpoint): "
                        "reports the deployment operating point on the TARGET domain, with the "
                        "threshold still calibrated on the in-domain VAL split.")
    p.add_argument("--external-dataset", default="mbrset", choices=["mbrset", "brset"])
    p.add_argument("--target-sens", type=float, default=0.85,
                   help="Screening sensitivity to calibrate the threshold for (on VAL).")
    p.add_argument("--tta", action="store_true", help="Horizontal-flip test-time augmentation.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="deploy")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--calib", default="minmax",
                   choices=["minmax", "percentile", "entropy", "sweep"],
                   help="INT8 calibration. MobileNetV3 (Hardswish + SE) is PTQ-hostile: "
                        "MinMax picks scales from activation outliers and can destroy "
                        "accuracy. 'sweep' tries all and reports the best.")
    p.add_argument("--calib-batches", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    ca = ck.get("args", {})
    task = ca.get("task", "dr_referable")
    seed = int(ca.get("seed", 0))
    img_size = int(ca.get("image_size", 224))
    # Newer checkpoints/results record use_gcg directly; older ones only no_gcg.
    if "use_gcg" in ca:
        use_gcg = bool(ca["use_gcg"])
    elif str(ca.get("backbone", "")).startswith("timm:"):
        use_gcg = False   # pre-2026-09-01 timm checkpoints never had a gate, whatever no_gcg says
    else:
        use_gcg = not ca.get("no_gcg", False)
    variant = ca.get("gcg_variant", "baseline")
    backbone = ca.get("backbone", "mobilenetv3_small")
    train_dataset = ca.get("dataset", "mbrset")
    # The checkpoint's recorded --image-ext: BRSET file names may lack an
    # extension and train_mbrset.py appended this one; using a different one
    # here would silently drop every image via drop_missing_files.
    image_ext = ca.get("image_ext", ".jpg")
    seed_everything(seed)
    device = pick_device()

    print(f"[info] ckpt   : {args.ckpt}")
    print(f"[info] config : task={task} seed={seed} size={img_size} backbone={backbone} "
          f"use_gcg={use_gcg}{' variant='+variant if use_gcg else ''} "
          f"trained_on={train_dataset} image_ext={image_ext}")

    # Rebuild the EXACT split this checkpoint was trained with, from the dataset
    # it was trained on — load_any resolves mBRSET vs BRSET layouts and applies
    # brset_dataset's schema/value re-encoding, matching train_mbrset.py.
    from brset_dataset import load_any
    src_data = load_any(args.root, train_dataset, image_ext=image_ext)
    splits = stratified_split(src_data["df"], task=task, val_frac=0.10,
                              test_frac=0.20, group_col="patient", seed=seed)
    mk = lambda df, d=src_data["images_dir"]: MBRSETDataset(
        csv=df, images_dir=d, task=task, split="val",
        image_size=img_size, drop_missing_files=True, fov_crop=True)
    val_ds, test_ds = mk(splits["val"]), mk(splits["test"])
    C = test_ds.num_classes
    if C != 2:
        raise SystemExit(f"This deployment report assumes a binary task; {task} has {C} classes.")
    dl = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, worker_init_fn=seed_worker)
    val_loader, test_loader = dl(val_ds), dl(test_ds)
    print(f"[info] splits : val={len(val_ds)} test={len(test_ds)} (patient-grouped, from seed {seed})")

    ext_loader = None
    if args.external_root:
        ext = load_any(args.external_root, args.external_dataset, image_ext=image_ext)
        ext_ds = MBRSETDataset(csv=ext["df"], images_dir=ext["images_dir"], task=task,
                               split="val", image_size=img_size,
                               drop_missing_files=True, fov_crop=True)
        ext_loader = dl(ext_ds)
        print(f"[info] extern : {args.external_dataset} @ {args.external_root} n={len(ext_ds)}")

    from train_mbrset import backbone_kwargs_for
    model = MBRSETClassifier(num_classes=C, pretrained=False,
                             use_gcg=use_gcg, gcg_variant=variant,
                             backbone=backbone,
                             backbone_kwargs=backbone_kwargs_for(backbone, img_size)).to(device)
    model.load_state_dict(ck["model"])

    # ---------- FP32 ----------
    from sklearn.metrics import roc_auc_score
    vs, vy = torch_scores(model, val_loader, device, tta=args.tta)
    ts, ty = torch_scores(model, test_loader, device, tta=args.tta)
    fp32_auroc = float(roc_auc_score(ty, ts))
    thr, vsens, vspec = operating_point(vs, vy, target_sens=args.target_sens)
    fp32_op = binary_report(ts, ty, thr)
    print(f"\n[info] threshold calibrated on VAL for sensitivity>={args.target_sens}: "
          f"{thr:.4f}  (val sens={vsens:.3f} spec={vspec:.3f})")

    ext_report = None
    if ext_loader is not None:
        es, ey = torch_scores(model, ext_loader, device, tta=args.tta)
        ext_auroc = float(roc_auc_score(ey, es))
        ext_report = {"auroc": ext_auroc, **binary_report(es, ey, thr), "n": int(len(ey))}
        print(f"\n=== TARGET-DOMAIN operating point ({args.external_dataset}, threshold from "
              "in-domain VAL) ===")
        print(f"  AUROC={ext_auroc:.4f}  sens={ext_report['sensitivity']:.3f}  "
              f"spec={ext_report['specificity']:.3f}  ppv={ext_report['ppv']:.3f}  "
              f"npv={ext_report['npv']:.3f}  (n={len(ey)})")
        print("  NB: the threshold was chosen on source-domain val; a threshold shift is part")
        print("      of the domain gap and belongs in the report, not hidden by re-tuning.")

    # ---------- export + INT8 ----------
    fp32_pt = os.path.join(args.out_dir, "model_fp32.pt")
    onnx_fp32 = os.path.join(args.out_dir, "model_fp32.onnx")
    onnx_int8 = os.path.join(args.out_dir, "model_int8.onnx")
    torch.save(model.state_dict(), fp32_pt)
    dummy = torch.randn(1, 3, img_size, img_size)
    # dynamo=False pins the TorchScript exporter: torch >= 2.9 flips the default
    # to the dynamo exporter, whose graph differs enough to change what
    # quant_pre_process / quantize_static see.
    torch.onnx.export(model.cpu().eval(), dummy, onnx_fp32, input_names=["input"],
                      output_names=["logits"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    model.to(device)

    res = {"ckpt": args.ckpt, "task": task, "seed": seed, "image_size": img_size,
           "use_gcg": use_gcg, "backbone": backbone, "train_dataset": train_dataset,
           "image_ext": image_ext,
           "tta": args.tta, "n_test": int(len(ty)), "external": ext_report,
           "external_dataset": args.external_dataset if ext_report else None,
           "latency_note": LATENCY_NOTE}

    int8_ok = True
    try:
        from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                              QuantType, QuantFormat)
        from onnxruntime.quantization.shape_inference import quant_pre_process

        class _Calib(CalibrationDataReader):
            """Calibrate on REAL validation images (not noise) -> better INT8 ranges."""
            def __init__(self, loader, n=8):
                xs = []
                for i, b in enumerate(loader):
                    xs.append({"input": b["image"].numpy()})
                    if i + 1 >= n:
                        break
                self._it = iter(xs)

            def get_next(self):
                return next(self._it, None)

        from onnxruntime.quantization import CalibrationMethod
        _CM = {"minmax": CalibrationMethod.MinMax,
               "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}
        prepped = os.path.join(args.out_dir, "model_fp32.prep.onnx")
        quant_pre_process(onnx_fp32, prepped)

        if args.calib == "sweep":
            # Try every calibration method; select the winner on VAL, never test.
            # Selecting on test and then reporting that same test AUROC as the
            # RQ3 number is model selection on the evaluation set — the reported
            # quantization cost would be optimistically biased.
            fp32_val_auroc = float(roc_auc_score(vy, vs))
            best_name, best_auc = None, -1.0
            for nm, cm in _CM.items():
                cand = os.path.join(args.out_dir, f"model_int8_{nm}.onnx")
                try:
                    quantize_static(prepped, cand, _Calib(val_loader, n=args.calib_batches),
                                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                                    activation_type=QuantType.QUInt8, per_channel=True,
                                    calibrate_method=cm)
                    s_, y_ = onnx_scores(cand, val_loader, tta=args.tta)
                    a = float(roc_auc_score(y_, s_))
                    print(f"[calib] {nm:<11} INT8 val AUROC={a:.4f}  (FP32 val {fp32_val_auroc:.4f}, "
                          f"delta {a-fp32_val_auroc:+.4f})")
                    if a > best_auc:
                        best_auc, best_name = a, nm
                        os.replace(cand, onnx_int8)
                except Exception as e:  # noqa
                    print(f"[calib] {nm:<11} FAILED: {str(e)[:60]}")
            int8_ok = best_name is not None
            # The winner was os.replace'd into model_int8.onnx as it was found;
            # whatever is still under a per-method name lost. Delete those so
            # nobody later scores a loser by picking the wrong file.
            for nm in _CM:
                loser = os.path.join(args.out_dir, f"model_int8_{nm}.onnx")
                if os.path.isfile(loser):
                    os.remove(loser)
            if int8_ok:
                print(f"[calib] best = {best_name} (val AUROC {best_auc:.4f}; "
                      "test evaluated once below; losing candidates deleted)")
            else:
                print("[calib] every calibration method failed")
            res["calib_method"] = best_name
        else:
            res["calib_method"] = args.calib
            quantize_static(prepped, onnx_int8, _Calib(val_loader, n=args.calib_batches),
                            calibrate_method=_CM[args.calib],
                            quant_format=QuantFormat.QDQ,
                            weight_type=QuantType.QInt8,
                            activation_type=QuantType.QUInt8,
                            # per-channel is necessary (depthwise convs) but on a
                            # real trained MobileNetV3 it is NOT sufficient --
                            # calibration method matters just as much.
                            per_channel=True)
    except Exception as e:  # noqa
        int8_ok = False
        print(f"[warn] INT8 quantization failed: {e}")

    mb = lambda p_: os.path.getsize(p_) / 1e6
    x_np = dummy.numpy()
    lat_fp32 = bench_onnx(onnx_fp32, x_np, args.runs)
    res["fp32"] = {"auroc": fp32_auroc, **fp32_op,
                   "size_mb": mb(onnx_fp32), "latency_ms_cpu_onnx": lat_fp32}
    fmt_lat = lambda l: f"{l['median']:.1f} [{l['p10']:.1f}-{l['p90']:.1f}]"

    print("\n" + "=" * 74)
    print(f"DEPLOYMENT REPORT — {task}   (test n={len(ty)}, patient-grouped hold-out)")
    print(f"  model: backbone={backbone} use_gcg={use_gcg} size={img_size} seed={seed}")
    print("=" * 74)
    print(f"{'variant':<12}{'AUROC':>8}{'Sens':>8}{'Spec':>8}{'PPV':>8}{'size MB':>10}"
          f"{'CPU ms median [p10-p90]':>26}")
    print(f"{'FP32':<12}{fp32_auroc:>8.4f}{fp32_op['sensitivity']:>8.3f}"
          f"{fp32_op['specificity']:>8.3f}{fp32_op['ppv']:>8.3f}{mb(onnx_fp32):>10.2f}"
          f"{fmt_lat(lat_fp32):>26}")

    if int8_ok:
        # Quantization preserves RANKING (AUROC) but shifts the probability
        # calibration, so the FP32 threshold does not transfer -- reusing it made
        # measured sensitivity collapse 1.000 -> 0.263 at identical AUROC.
        # Re-calibrate on the INT8 model's own VAL scores.
        ivs, ivy = onnx_scores(onnx_int8, val_loader, tta=args.tta)
        thr_int8, _, _ = operating_point(ivs, ivy, target_sens=args.target_sens)
        is_, iy = onnx_scores(onnx_int8, test_loader, tta=args.tta)
        int8_auroc = float(roc_auc_score(iy, is_))
        int8_op = binary_report(is_, iy, thr_int8)
        print(f"[info] INT8 threshold re-calibrated on VAL: {thr_int8:.4f} "
              f"(FP32 threshold {thr:.4f} does not transfer after quantization)")
        lat_int8 = bench_onnx(onnx_int8, x_np, args.runs)
        res["int8"] = {"auroc": int8_auroc, **int8_op,
                       "size_mb": mb(onnx_int8), "latency_ms_cpu_onnx": lat_int8}
        print(f"{'INT8':<12}{int8_auroc:>8.4f}{int8_op['sensitivity']:>8.3f}"
              f"{int8_op['specificity']:>8.3f}{int8_op['ppv']:>8.3f}"
              f"{mb(onnx_int8):>10.2f}{fmt_lat(lat_int8):>26}")
        print("-" * 74)
        d = int8_auroc - fp32_auroc
        res["quantization_delta_auroc"] = d
        print(f"RQ3 -> quantization cost: {d:+.4f} AUROC   "
              f"({mb(onnx_fp32)/mb(onnx_int8):.2f}x smaller, "
              f"{lat_fp32['median']/lat_int8['median']:.2f}x faster on CPU-ONNX)")
        verdict = ("negligible (<0.01 AUROC)" if abs(d) < 0.01 else
                   "modest (<0.02)" if abs(d) < 0.02 else "SIGNIFICANT — report it")
        print(f"     degradation is {verdict}")

    print(f"\nconfusion @ threshold {thr:.4f}: "
          f"TP={fp32_op['tp']} FP={fp32_op['fp']} TN={fp32_op['tn']} FN={fp32_op['fn']}")
    print(f"NOTE: {LATENCY_NOTE}")

    out = os.path.join(args.out_dir, f"deploy_{task}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
