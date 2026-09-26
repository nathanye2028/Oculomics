#!/usr/bin/env python3
"""
export_retinareach.py
=====================
Core ML export of **RetinaReach** in the form that lets the phone re-calibrate
itself: two graphs over the same weights plus a small JSON of device profiles.

    RetinaReach.mlpackage            image (0-255 RGB) + bn_stats  ->  probabilities [1, K]
    RetinaReachCalibrator.mlpackage  images [B,3,S,S] (0-255)      ->  bn_stats of that batch
    RetinaReachFused.mlpackage       (--fused) BN folded, one fixed profile: the latency baseline
    retinareach_profiles.json        statistics order, trained + per-profile statistics vectors,
                                     envelopes, thresholds, capability rows, preprocessing

Why two graphs
--------------
Core ML folds BatchNorm into the convolutions at conversion, so the AdaBN
statistics of a camera the model has never seen cannot be written into a
shipped ``.mlpackage``. Here BN reads its mean/variance from an input vector
instead (``retinareach.StatInputNet``), and a second graph
(``retinareach.BatchStatCollector``) measures that vector from a batch of the
clinic's own captures. The app's whole calibration procedure is then:

    1. run the calibrator on the first N captures in batches of B;
    2. average its outputs (equal weight per batch -- AdaBN's cumulative mean);
    3. compare the MEASURED vector to each profile (``retinareach.camera_distance``)
       -> which targets to show (``retinareach.device_capability``); the
       envelopes were built from unblended statistics, so never compare a
       blended vector;
    4. for inference only, optionally shrink toward ``trained_stats`` with
       pseudo-count N0 (``retinareach.blend_vector``:
       vec = (N0 * trained + N * measured) / (N0 + N));
    5. store that vector; every later screen passes it with the image.
No labels, no gradient, no re-export.

Precision: the inference graph runs fp16 (the ANE); the calibrator defaults to
fp32 because it squares activations and runs once per camera, not per patient.

    # a trained model
    python export_retinareach.py --checkpoint ck_retinareach/seed0.pt \\
        --verify-images <dir of captures> --verify-n 64 --benchmark --fused
    # before one exists: the same graphs on any trained V4 trunk (numerics/latency only)
    python export_retinareach.py --backbone timm:mobilenetv4_conv_medium.e500_r256_in1k \\
        --init-trunk-from ck_retinal_age_v5/medium_mix_seed0.pt --image-size 512 \\
        --verify-images <dir> --benchmark --fused

``--verify-images`` is the fidelity gate for on-device calibration: the same
uint8 captures go through (a) PyTorch AdaBN and (b) the Core ML calibrator +
inference graph, and the statistics and probabilities are compared. Latency
on a Mac is an optimistic bound (see export_coreml.py); an iPhone number needs
an Xcode performance report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_coreml import FOV_TOL, list_images, preprocess_for_verify  # noqa: E402
from retinareach import (PROBE_TARGETS, SOURCE_TARGETS, BatchStatCollector,  # noqa: E402
                         RetinaReachNet, StatInputNet, _Normalise, bn_layers, camera_distance,
                         get_bn_stats, self_calibrate, stats_from_lists, stats_to_vector,
                         vector_to_stats)


class FusedWrapper(nn.Module):
    """Plain eval-mode network (BN folded by the converter) -- the baseline the
    calibratable graph's latency is compared against."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net, self.pre = net.eval(), _Normalise()

    def forward(self, x):
        return torch.sigmoid(self.net(self.pre(x)))


def build(args) -> tuple:
    """(net, checkpoint-or-None). A probe build copies a trained trunk from any
    train_mbrset / train_retinal_age checkpoint (``backbone.*`` -> ``trunk.*``)
    and leaves the heads random: numerics and latency only, no predictions."""
    if args.checkpoint:
        from retinareach import build_from_checkpoint
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        return build_from_checkpoint(ck), ck
    torch.manual_seed(0)
    net = RetinaReachNet(SOURCE_TARGETS, PROBE_TARGETS, backbone=args.backbone,
                         pretrained=False, probe_size=args.image_size)
    if args.init_trunk_from:
        src = torch.load(args.init_trunk_from, map_location="cpu", weights_only=False)["model"]
        dst = net.state_dict()
        n = 0
        for k, v in src.items():
            kk = "trunk." + k[len("backbone."):] if k.startswith("backbone.") else None
            if kk in dst and dst[kk].shape == v.shape:
                dst[kk] = v
                n += 1
        missing = [k for k in dst if k.startswith("trunk.") and k not in
                   {"trunk." + s[len("backbone."):] for s in src if s.startswith("backbone.")}]
        if n == 0 or missing:
            raise SystemExit(f"[fatal] --init-trunk-from: {n} tensors matched, {len(missing)} trunk "
                             f"tensors missing (e.g. {missing[:3]}); wrong --backbone?")
        net.load_state_dict(dst)
        print(f"[info] trunk: {n} tensors from {args.init_trunk_from} (heads random: probe build)")
    return net.eval(), None


def convert(module: nn.Module, example: tuple, inputs, out_name: str, precision: str,
            min_target: str):
    import warnings
    import coremltools as ct
    with warnings.catch_warnings():
        # The collector turns tensor shapes into Python ints (batch x H x W for the
        # Bessel factor); the graph is fixed-shape by design, so baking them is right.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        traced = torch.jit.trace(module.eval(), example, strict=False)
    prec = ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16
    return ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name=out_name)],
                      convert_to="mlprogram", compute_precision=prec,
                      minimum_deployment_target=getattr(ct.target, min_target),
                      compute_units=ct.ComputeUnit.ALL)


def profiles_payload(net: RetinaReachNet, ck: Optional[dict], args) -> dict:
    order = [n for n, _ in bn_layers(net)]
    trained = get_bn_stats(net)
    out = {"targets": net.targets, "source_targets": net.source_targets,
           "probe_targets": net.probe_targets, "image_size": args.image_size,
           "calib_batch": args.calib_batch,
           "stats_layout": {"order": order, "channels": [m.num_features for _, m in bn_layers(net)],
                            "format": "per layer: mean[C] then running variance[C]"},
           "trained_stats": stats_to_vector(trained, order).tolist(),
           "preprocessing": {"input": "RGB 0-255, no normalisation (ImageNet mean/std in graph)",
                             "fov_crop": f"crop to the bounding box of max(R,G,B) > {FOV_TOL}",
                             "resize": f"full-frame resize to {args.image_size}x{args.image_size} "
                                       f"(no centre crop)"},
           "probe_build": ck is None, "profiles": {}, "thresholds": {}}
    if ck is not None:
        out["thresholds"] = ck.get("thresholds", {})
        for name, p in ck.get("profiles", {}).items():
            out["profiles"][name] = {
                "dataset": p["dataset"], "n_pool": p["n_pool"], "envelope": p["envelope"],
                "capability": [{k: r.get(k) for k in ("target", "protocol", "status", "reason",
                                                      "auroc", "delta", "sensitivity")}
                               for r in p["capability"]],
                "bn_stats": stats_to_vector(stats_from_lists(p["bn_stats"]), order).tolist()}
    return out


# --------------------------------------------------------------------------- #
# Verification: Core ML calibration == PyTorch AdaBN on the same bytes
# --------------------------------------------------------------------------- #
def verify(paths: Dict[str, str], net: RetinaReachNet, args) -> bool:
    import coremltools as ct
    from PIL import Image
    files = list_images(args.verify_images, args.verify_n)
    B = args.calib_batch
    n_cal = (len(files) // B) * B
    if n_cal < B:
        print(f"[skip] verification needs >= {B} images, found {len(files)}")
        return True
    u8 = [preprocess_for_verify(f, args.image_size, "cls")[1] for f in files]
    raw = torch.from_numpy(np.stack(u8)).permute(0, 3, 1, 2).float()      # [N,3,S,S] 0-255
    batches = [raw[i:i + B] for i in range(0, n_cal, B)]
    norm = _Normalise()
    order = [n for n, _ in bn_layers(net)]

    # (a) PyTorch: AdaBN on those bytes, then the eval-mode network
    cal, info = self_calibrate(net, [{"image": norm(b)} for b in batches], torch.device("cpu"))
    ref_vec = stats_to_vector(get_bn_stats(cal), order)
    with torch.no_grad():
        ref_p = torch.sigmoid(cal(norm(raw))).numpy()
        src_p = torch.sigmoid(net(norm(raw))).numpy()

    # (b) Core ML: calibrator over the same batches, averaged; then inference
    cu = getattr(ct.ComputeUnit, args.verify_units)
    calib = ct.models.MLModel(paths["calibrator"], compute_units=cu)
    infer = ct.models.MLModel(paths["model"], compute_units=cu)
    ml_vec = np.mean([calib.predict({"images": b.numpy()})["bn_stats"] for b in batches], axis=0)
    ml_p = np.concatenate([
        infer.predict({"image": Image.fromarray(u), "bn_stats": ml_vec.astype(np.float32)})
        ["probabilities"] for u in u8])
    ml_p_refvec = np.concatenate([
        infer.predict({"image": Image.fromarray(u), "bn_stats": ref_vec.numpy()})["probabilities"]
        for u in u8])

    # statistics error, relative to each layer's own scale
    ml_stats = vector_to_stats(torch.from_numpy(ml_vec.astype(np.float32)), net)
    ref_stats = get_bn_stats(cal)
    rel = []
    for k in ref_stats:
        for i in (0, 1):
            scale = float(ref_stats[k][1].abs().mean().sqrt()) if i == 0 else \
                float(ref_stats[k][1].abs().mean())
            rel.append(float((ml_stats[k][i] - ref_stats[k][i]).abs().max()) / max(scale, 1e-6))
    d_ml_ref = camera_distance(ml_stats, ref_stats)
    d_src_ref = camera_distance(get_bn_stats(net), ref_stats)
    err_total = float(np.abs(ml_p - ref_p).max())
    err_infer = float(np.abs(ml_p_refvec - ref_p).max())
    shift = float(np.abs(ref_p - src_p).max())
    ok = err_total <= args.tol
    print(f"\n=== on-device calibration fidelity ({len(files)} captures from {args.verify_images}, "
          f"{len(batches)} batches of {B}, Core ML on {args.verify_units}) ===")
    print(f"  calibrator vs AdaBN statistics: max error {max(rel):.2e} of the layer scale; "
          f"camera distance {d_ml_ref:.2e} (vs {d_src_ref:.3g} between trained and calibrated)")
    print(f"  probabilities, Core ML calibrator+model vs PyTorch AdaBN model: max |diff| "
          f"{err_total:.4f}  (inference graph alone, PyTorch statistics: {err_infer:.4f})")
    print(f"  for scale: calibration itself moved probabilities by up to {shift:.4f} on these images")
    print(f"  [{'PASS' if ok else 'FAIL'}] tolerance {args.tol}")
    paths["_verify"] = {"n_images": len(files), "batches": len(batches), "stats_max_rel_err": max(rel),
                        "camera_distance_ml_vs_ref": d_ml_ref, "camera_distance_src_vs_ref": d_src_ref,
                        "prob_max_err_total": err_total, "prob_max_err_inference": err_infer,
                        "calibration_effect_max": shift, "tol": args.tol, "pass": ok}
    return ok


def benchmark(path: str, inputs: dict, label: str, runs: int, warmup: int,
              units=("CPU_AND_NE", "ALL", "CPU_ONLY")) -> Dict[str, float]:
    import coremltools as ct
    out = {}
    for u in units:
        try:
            m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, u))
            for _ in range(warmup):
                m.predict(inputs)
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter()
                m.predict(inputs)
                ts.append((time.perf_counter() - t0) * 1000)
            out[u] = float(np.median(ts))
            print(f"  {label:<34}{u:<12} median {out[u]:7.2f} ms  "
                  f"(p10 {np.percentile(ts, 10):.2f} / p90 {np.percentile(ts, 90):.2f})")
        except Exception as e:  # noqa
            print(f"  {label:<34}{u:<12} unavailable ({str(e)[:60]})")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Core ML export of RetinaReach (self-calibrating).")
    p.add_argument("--checkpoint", default=None, help="train_retinareach.py checkpoint")
    p.add_argument("--backbone", default="timm:mobilenetv4_conv_small.e2400_r224_in1k",
                   help="probe build only (no --checkpoint)")
    p.add_argument("--init-trunk-from", default=None,
                   help="probe build: copy a trained trunk from a train_mbrset/retinal-age ckpt")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--calib-batch", type=int, default=None,
                   help="calibrator batch size (default: the checkpoint's, else 16)")
    p.add_argument("--collector-precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                   help="inference graph precision")
    p.add_argument("--fused", action="store_true",
                   help="also export a BN-folded graph (no statistics input): the latency baseline, "
                        "and the fast path for a camera whose profile is known at export time")
    p.add_argument("--fused-profile", default="trained",
                   help="statistics baked into the fused graph: 'trained' or a profile name from "
                        "the checkpoint (e.g. handheld)")
    p.add_argument("--min-target", default="iOS17")
    p.add_argument("--out-dir", default="edge_export/retinareach")
    p.add_argument("--verify-images", default=None)
    p.add_argument("--verify-n", type=int, default=64)
    p.add_argument("--verify-units", default="CPU_AND_NE",
                   choices=["CPU_AND_NE", "ALL", "CPU_ONLY", "CPU_AND_GPU"])
    p.add_argument("--tol", type=float, default=0.02,
                   help="max |probability difference| Core ML vs PyTorch for PASS")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args(argv)

    import coremltools as ct
    if not args.checkpoint:
        args.image_size = args.image_size or 384     # the probe build needs it before build()
    net, ck = build(args)
    if ck is not None:
        args.image_size = args.image_size or int(ck["config"].get("image_size", 384))
        args.calib_batch = args.calib_batch or int(ck["args"].get("calib_batch", 16))
    args.image_size = args.image_size or 384
    args.calib_batch = args.calib_batch or 16
    S, B = args.image_size, args.calib_batch
    os.makedirs(args.out_dir, exist_ok=True)

    sin, col = StatInputNet(net), BatchStatCollector(net)
    L = sin.stats_len
    vec0 = stats_to_vector(get_bn_stats(net), [n for n, _ in bn_layers(net)])
    print(f"[info] {net.backbone_name} @ {S}px: {len(net.targets)} heads, "
          f"{len(bn_layers(net))} BN layers -> statistics vector of {L} floats; calibrator batch {B}")

    paths = {}
    ex_img = torch.randint(0, 256, (1, 3, S, S)).float()
    m = convert(sin, (ex_img, vec0), [
        ct.ImageType(name="image", shape=(1, 3, S, S), color_layout=ct.colorlayout.RGB,
                     scale=1.0, bias=[0.0, 0.0, 0.0]),
        ct.TensorType(name="bn_stats", shape=(L,))], "probabilities", args.precision, args.min_target)
    m.short_description = ("RetinaReach screening model. Inputs: FOV-cropped fundus image "
                           "(0-255 RGB) and the device's BatchNorm statistics vector. Output: "
                           "per-target probabilities.")
    meta = {"targets": ",".join(net.targets), "stats_len": str(L), "calib_batch": str(B),
            "image_size": str(S), "profiles_file": "retinareach_profiles.json"}
    for k, v in meta.items():
        m.user_defined_metadata[k] = v
    paths["model"] = os.path.join(args.out_dir, "RetinaReach.mlpackage")
    m.save(paths["model"])

    c = convert(col, (torch.randint(0, 256, (B, 3, S, S)).float(),),
                [ct.TensorType(name="images", shape=(B, 3, S, S))], "bn_stats",
                args.collector_precision, args.min_target)
    c.short_description = (f"RetinaReach calibrator: a batch of {B} captures (0-255 RGB) -> the "
                           f"BatchNorm statistics of that batch. Average over batches.")
    for k, v in meta.items():
        c.user_defined_metadata[k] = v
    paths["calibrator"] = os.path.join(args.out_dir, "RetinaReachCalibrator.mlpackage")
    c.save(paths["calibrator"])
    if args.fused:
        fused_net = net
        if args.fused_profile != "trained":
            from retinareach import build_from_checkpoint
            if ck is None:
                raise SystemExit("[fatal] --fused-profile needs --checkpoint (profiles live there)")
            fused_net = build_from_checkpoint(ck, stats=args.fused_profile)
        f = convert(FusedWrapper(fused_net), (ex_img,), [
            ct.ImageType(name="image", shape=(1, 3, S, S), color_layout=ct.colorlayout.RGB,
                         scale=1.0, bias=[0.0, 0.0, 0.0])], "probabilities", args.precision,
            args.min_target)
        f.user_defined_metadata["bn_stats"] = args.fused_profile
        f.user_defined_metadata["targets"] = ",".join(net.targets)
        paths["fused"] = os.path.join(args.out_dir, f"RetinaReachFused_{args.fused_profile}.mlpackage")
        f.save(paths["fused"])
    with open(os.path.join(args.out_dir, "retinareach_profiles.json"), "w") as fh:
        json.dump(profiles_payload(net, ck, args), fh)
    print("[ok] saved " + ", ".join(paths.values()) + " and retinareach_profiles.json")

    ok = True
    if args.verify_images:
        ok = verify(paths, net, args)
    bench = {}
    if args.benchmark:
        from PIL import Image
        rng = np.random.default_rng(1)
        img = Image.fromarray(rng.integers(0, 256, (S, S, 3), dtype=np.uint8))
        print(f"\n=== latency on this Mac ({args.warmup} warm-up + {args.runs} runs; optimistic "
              f"bound, not an iPhone number) ===")
        bench["calibratable"] = benchmark(paths["model"], {"image": img, "bn_stats": vec0.numpy()},
                                          "inference, BN from input vector", args.runs, args.warmup)
        if "fused" in paths:
            bench["fused"] = benchmark(paths["fused"], {"image": img}, "inference, BN folded",
                                       args.runs, args.warmup)
        bench["calibrator"] = benchmark(
            paths["calibrator"], {"images": rng.integers(0, 256, (B, 3, S, S)).astype(np.float32)},
            f"calibrator, one batch of {B}", max(5, args.runs // 5), 3)
    with open(os.path.join(args.out_dir, "export_report.json"), "w") as fh:
        json.dump({"args": vars(args), "stats_len": L, "latency_ms": bench,
                   "verify": paths.get("_verify")}, fh, indent=2, default=float)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
