#!/usr/bin/env python3
"""
train_phone_ceiling.py
======================
The camera ladder's **in-domain ceiling**: train AND test on the handheld camera
(mBRSET) with RetinaReach's stage-1 model, recipe and patient split, so the phone
gap splits into two parts measured on the same phone test patients:

* **transfer gap** = phone-trained minus tabletop-trained (``--compare``): what any
  better tabletop->phone transfer (artifact augmentation, style transfer, mixing
  phone photos in) could at most recover;
* **image limit** = the phone-trained model's own distance from the IDx-DR
  targets: what no transfer method recovers, because a model that has seen phone
  photos and their labels still does no better.

    python train_phone_ceiling.py --root <mBRSET> --seed 0 \\
        --split-file exp_rr_canon/seed0/split_handheld.csv --compare ck_rr_canon/seed0.pt \\
        --out exp_phone_ceiling/seed0 --ckpt ck_phone_ceiling/seed0.pt
    python train_phone_ceiling.py --summarize exp_phone_ceiling        # mean over seed*/

Identical to ``train_retinareach.py`` stage 1: ``RetinaReachNet`` (same backbone,
image size, heads ``dr_referable`` + ``edema``), masked BCE, sampler balanced on
``dr_referable``, AdamW + warmup-cosine, EMA (default 0.999, the sweep's EXTRA),
early stopping on mean val AUROC, and ``MultiTargetDataset`` (same decode / FOV
crop / augmentation). The split is the trainer's ``multitask_split(df,
'dr_referable', seed=42)``; ``--split-file`` takes a run's own
``split_handheld.csv`` instead and reports whether the two agree.

Different by design: the training photos and labels are the phone's (train
split) and thresholds are fixed on the PHONE validation split at
``--target-sens``. Phone labels break RetinaReach's label-free premise, so this
model is a reference line on the ladder, never a candidate to ship.

``--mix-root`` trains the same model on the tabletop camera's train split AND the
phone's (the model named ``mixed``), the retinal-age track's fix for transfer.
``--phone-share`` (default 0.5) is the fraction of each epoch's draws that are
phone photos, ``dr_referable`` balanced within each camera; selection,
thresholds and the test stay on the phone. ``--mix-camera`` keeps one tabletop
camera (the Canon sweep: 'Canon CR') and ``--mix-split-file`` takes that sweep's
``split_tabletop.csv``, so the tabletop photos are exactly the compared
checkpoint's training photos. Read against the phone-only run: mixed above
phone-only means the small phone train set, not the photos, held phone-only
down. The mixed model and each ``--compare`` checkpoint (as trained) are also
scored on the tabletop test split (``home_camera.csv``): does mixing phone
photos in cost the home camera anything?

    python train_phone_ceiling.py --root <mBRSET> --mix-root <BRSET> --mix-camera 'Canon CR' \\
        --split-file exp_rr_canon/seed0/split_handheld.csv \\
        --mix-split-file exp_rr_canon/seed0/split_tabletop.csv --compare ck_rr_canon/seed0.pt \\
        --seed 0 --out exp_phone_mixed/seed0 --ckpt ck_phone_mixed/seed0.pt

``--compare`` scores RetinaReach checkpoints on the same phone test rows, as
trained (``<tag>_trained``: the ladder's "as trained" row, shipped tabletop
threshold) and with full AdaBN on the phone pool (``<tag>_calibrated``), and
gives the paired AUROC difference phone-trained minus each, patient bootstrap.

Per model x target: image-level AUROC with a patient-cluster CI, patient-level
AUROC, sensitivity / specificity / flagged fraction at the shipped threshold, and
the **test-oracle specificity at --target-sens** -- where the ROC curve itself
crosses the target sensitivity. No threshold rule (shipped, EM, anything
label-free) can beat it, so it says whether the ranking can reach the IDx-DR box
at all. Per model x rule: per-patient referral (``referral_outcomes``: refer if
any gradable eye flags, retake if none is gradable) against the IDx-DR targets.

Outputs (``--out``): ``results.json``, ``ceiling.csv``, ``referral.csv``,
``transfer_gap.csv`` (``test`` = handheld, or tabletop for the home camera),
``predictions_<model>.csv`` (file, patient, p_<t>, threshold_<t>),
``split_handheld.csv``; mixed runs add ``home_camera.csv``,
``predictions_home_<model>.csv`` and ``split_tabletop.csv``. ``--ckpt`` loads
with ``retinareach.build_from_checkpoint``.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capability_gate import multitask_split, paired_delta_bootstrap, patient_auroc  # noqa: E402
from dataset import DeviceAug  # noqa: E402
from fundus_utils import seed_everything, seed_worker  # noqa: E402
from metadata_model import (_auc, cluster_bootstrap, load_table, operating_point,  # noqa: E402
                            threshold_at_sensitivity)
from referral_outcomes import RULES, eye_table, patient_outcomes  # noqa: E402
from retinareach import DEFAULT_BACKBONE, SOURCE_TARGETS, RetinaReachNet, build_from_checkpoint  # noqa: E402
from train_mbrset import ModelEMA, pick_device, splits_from_file  # noqa: E402
from train_retinareach import MultiTargetDataset, balanced_weights, predict  # noqa: E402

PHONE = "phone"
MIXED = "mixed"


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def device_split(df: pd.DataFrame, strat: str, split_seed: int,
                 split_file: Optional[str] = None) -> tuple:
    """``(splits, agrees)``: the trainer's patient-grouped partition, or the one
    in ``split_file`` when given. ``agrees`` says whether the file matches the
    recomputed partition (None without a file): a mismatch means the label table
    or sklearn changed since that run, and the file is what fixed its test patients."""
    mine = multitask_split(df, strat, seed=split_seed)
    if not split_file:
        return mine, None
    theirs = splits_from_file(df, split_file)
    agrees = all(set(mine[k]["file"].astype(str)) == set(theirs[k]["file"].astype(str))
                 for k in ("train", "val", "test"))
    return theirs, agrees


def mixed_weights(y_tab: np.ndarray, y_phone: np.ndarray, phone_share: float,
                  balance: bool = True) -> torch.Tensor:
    """Sampler weights over ``[tabletop train; phone train]``: phone rows take
    ``phone_share`` of the draws whatever the two sizes, and ``dr_referable`` is
    balanced within each camera (``balance``) or drawn as found."""
    parts = []
    for y, share in ((y_tab, 1.0 - phone_share), (y_phone, phone_share)):
        w = balanced_weights(y) if balance else torch.ones(len(y), dtype=torch.double)
        parts.append(w / w.sum() * share)
    return torch.cat(parts)


# --------------------------------------------------------------------------- #
# Stage 1 on the phone (train_retinareach.py's loop, phone images in)
# --------------------------------------------------------------------------- #
def fit(a, model, train_loader, val_loader, y_val: np.ndarray, dev, name: str = PHONE) -> tuple:
    """Train ``model`` in place; returns ``(best val mean AUROC, best epoch)``."""
    nS = len(model.source_targets)
    gpu_aug = a.gpu_aug if a.gpu_aug is not None else dev.type == "cuda"
    ema = ModelEMA(model, a.ema_decay) if a.ema_decay > 0 else None
    use_amp = a.amp if a.amp is not None else dev.type == "cuda"
    amp_dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda")
    dev_aug = DeviceAug() if gpu_aug else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    warm = max(0, min(a.warmup_epochs, a.epochs - 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs - warm))
    sched = (torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm), cos],
        milestones=[warm]) if warm > 0 else cos)

    def val_score(m) -> float:
        pv = predict(m, val_loader, dev)[:, :nS]
        aucs = [_auc(y[~np.isnan(y)].astype(int), q[~np.isnan(y)])
                for y, q in zip(y_val[:, :nS].T, pv.T) if len(np.unique(y[~np.isnan(y)])) == 2]
        aucs = [x for x in aucs if x == x]
        return float(np.mean(aucs)) if aucs else float("nan")

    cuda = dev.type == "cuda"
    best, best_ep, since, best_state = -1.0, -1, 0, None
    print(f"\n=== {name} trunk + heads {model.source_targets}, up to {a.epochs} epochs ===")
    for ep in range(1, a.epochs + 1):
        model.train()
        run, nb, te = 0.0, 0, time.time()
        for b in train_loader:
            x = b["image"].to(dev, non_blocking=cuda)
            # the label slice is strided: never a non-blocking copy off CUDA (see predict)
            y = b["label"][:, :nS].contiguous().to(dev, non_blocking=cuda)
            if dev_aug is not None:
                x = dev_aug(x)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)[:, :nS]
            mask = ~torch.isnan(y)
            bce = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y),
                                                     reduction="none")
            loss = (bce * mask).sum() / mask.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            run = run + loss.detach(); nb += 1
            if a.log_every and nb % a.log_every == 0:
                print(f"    step {nb}/{len(train_loader)} loss={float(run) / nb:.4f}", flush=True)
        em = ema.ema if ema is not None else model
        v = val_score(em)
        tag = ""
        if v == v and v > best:
            best, best_ep, since = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in em.state_dict().items()}
            tag = "  <- best"
        else:
            since += 1
        print(f"  epoch {ep:3d}/{a.epochs} loss={float(run) / max(nb, 1):.4f} "
              f"val mean AUROC={v:.4f} [{time.time() - te:.0f}s]{tag}", flush=True)
        sched.step()
        if a.patience and since >= a.patience:
            print(f"  early stop: no val gain for {a.patience} epochs")
            break
    if best_state is None:          # val AUROC undefined every epoch: keep the last weights
        best_state = {k: t.detach().cpu().clone()
                      for k, t in (ema.ema if ema is not None else model).state_dict().items()}
        print("[warn] val AUROC was never defined; keeping the final weights")
    model.load_state_dict(best_state)
    model.eval()
    return best, best_ep


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_rows(model_name: str, probs: np.ndarray, test, targets: Sequence[str],
               thresholds: Dict[str, float], a) -> List[dict]:
    """One row per target: AUROC (+ patient CI, patient-level), the shipped
    operating point, and the test-oracle operating point at ``a.target_sens``."""
    y = test.label_array()
    rows = []
    for k, t in enumerate(targets):
        ok = ~np.isnan(y[:, k]) & ~np.isnan(probs[:, k])
        yy, qq, gg = y[ok, k].astype(int), probs[ok, k], test.patients[ok]
        if len(np.unique(yy)) < 2:
            continue
        thr = thresholds.get(t, float("nan"))
        op = operating_point(yy, qq, thr) if thr == thr else {}
        orc = operating_point(yy, qq, threshold_at_sensitivity(yy, qq, a.target_sens))
        lo, hi = cluster_bootstrap(yy, qq, gg, a.n_boot, a.seed)["auroc_ci"]
        rows.append({"model": model_name, "target": t, "n": int(len(yy)), "positives": int(yy.sum()),
                     "pos_patients": int(len(np.unique(gg[yy == 1]))),
                     "auroc": _auc(yy, qq), "auroc_lo": lo, "auroc_hi": hi,
                     "patient_auroc": patient_auroc(yy, qq, gg), "threshold": thr,
                     "sensitivity": op.get("sensitivity", float("nan")),
                     "specificity": op.get("specificity", float("nan")),
                     "flagged": op.get("flagged_fraction", float("nan")),
                     "oracle_sens": orc["sensitivity"], "oracle_spec": orc["specificity"],
                     "oracle_meets_spec": bool(orc["specificity"] >= a.target_spec)})
    return rows


def predictions_frame(probs: np.ndarray, test, targets: Sequence[str],
                      thresholds: Dict[str, float]) -> pd.DataFrame:
    """``train_retinareach.py``'s predictions_<device>_<protocol>.csv layout, so
    ``referral_outcomes.eye_table`` reads it unchanged."""
    out = pd.DataFrame({"file": test.df["file"], "patient": test.patients})
    for k, t in enumerate(targets):
        out[f"p_{t}"] = probs[:, k]
        out[f"threshold_{t}"] = thresholds.get(t, np.nan)
    return out


def referral_rows(model_name: str, pred: pd.DataFrame, table: pd.DataFrame, a) -> List[dict]:
    rows = []
    for rule in RULES:
        eyes = eye_table(pred, table, rule)
        if eyes is None:
            continue
        o = patient_outcomes(eyes, n_boot=a.n_boot, seed=a.seed)
        o.update(model=model_name, rule=rule,
                 meets_targets=bool(o.get("sensitivity", 0) >= a.target_sens
                                    and o.get("specificity", 0) >= a.target_spec))
        rows.append(o)
    return rows


def gap_rows(model_name: str, probs: Dict[str, np.ndarray], test, targets: Sequence[str],
             test_name: str, a) -> List[dict]:
    """Paired AUROC ``model_name`` minus every other model in ``probs``, same
    rows, patient bootstrap."""
    y = test.label_array()
    rows = []
    for name in probs:
        if name == model_name:
            continue
        for k, t in enumerate(targets):
            ok = ~np.isnan(y[:, k]) & ~np.isnan(probs[name][:, k])
            yy = y[ok, k].astype(int)
            if len(np.unique(yy)) < 2:
                continue
            d = paired_delta_bootstrap(yy, probs[model_name][ok, k], probs[name][ok, k],
                                       test.patients[ok], a.n_boot, a.seed)
            rows.append({"model": model_name, "test": test_name, "target": t, "versus": name, **d})
    return rows


def checkpoint_tag(path: str) -> str:
    """'ck_rr_canon/seed0.pt' -> 'ck_rr_canon_seed0': a model name safe in file names."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{parent}_{stem}").strip("_")


# --------------------------------------------------------------------------- #
# Multi-seed summary
# --------------------------------------------------------------------------- #
def family(model: str) -> str:
    """'ck_rr_canon_seed3_trained' -> 'compare_trained': rows that pool over seeds."""
    if model in (PHONE, MIXED):
        return model
    return "compare_" + model.rsplit("_", 1)[-1]


def summarize(d: str) -> int:
    dirs = sorted(glob.glob(os.path.join(d, "seed*", "ceiling.csv")))
    if not dirs:
        raise SystemExit(f"[fatal] no seed*/ceiling.csv under {d}")

    def load(name: str) -> pd.DataFrame:
        parts = []
        for f in dirs:
            run = os.path.dirname(f)
            try:                    # a run without --compare writes an empty transfer_gap.csv
                df = pd.read_csv(os.path.join(run, name))
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
            parts.append(df.assign(seed=int(re.sub(r"\D", "", os.path.basename(run)) or -1)))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    lines: List[str] = []
    say = lambda text: (print(text), lines.append(text))
    ceil, ref = load("ceiling.csv"), load("referral.csv")
    ceil["family"] = ceil["model"].map(family)
    cols = ["auroc", "patient_auroc", "sensitivity", "specificity", "flagged", "oracle_spec"]
    g = ceil.groupby(["target", "family"], sort=False)
    tab = g[cols].mean()
    tab.insert(0, "seeds", g["seed"].nunique())
    tab["auroc_sd"] = g["auroc"].std()
    tab["oracle_meets_spec"] = g["oracle_meets_spec"].sum().astype(int)
    say(f"=== image level, mean over seeds ({len(dirs)} run(s)); oracle_spec = specificity where "
        f"the test ROC curve crosses the target sensitivity ===")
    say(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    if len(ref):
        ref["family"] = ref["model"].map(family)
        g = ref.groupby(["rule", "family"], sort=False)
        rt = g[["prevalence", "sensitivity", "specificity", "per1000_missed", "per1000_unnecessary"]].mean()
        rt["seeds_meeting_targets"] = g["meets_targets"].sum().astype(int)
        say("\n=== per patient at the shipped threshold (IDx-DR targets), mean over seeds ===")
        say(rt.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) <= 1 else f"{v:.1f}"))
    home = load("home_camera.csv")
    if len(home):
        home["family"] = home["model"].map(family)
        g = home.groupby(["target", "family"], sort=False)
        ht = g[["auroc", "patient_auroc", "sensitivity", "specificity", "oracle_spec"]].mean()
        ht.insert(0, "seeds", g["seed"].nunique())
        ht["auroc_sd"] = g["auroc"].std()
        say("\n=== home camera: the tabletop test split, each model at its own shipped threshold, "
            "mean over seeds ===")
        say(ht.to_string(float_format=lambda v: f"{v:.3f}"))
    gap = load("transfer_gap.csv")
    if len(gap):
        if "test" not in gap:       # written before mixed runs existed: phone test, phone model
            gap = gap.assign(test="handheld", model=PHONE)
        gap["family"] = gap["versus"].map(family)
        g = gap.groupby(["test", "target", "family"], sort=False)
        gt = g[["delta", "delta_lo", "delta_hi"]].mean()
        gt["seeds_ci_above_0"] = g["delta_lo"].apply(lambda s: int((s > 0).sum()))
        gt["seeds_ci_below_0"] = g["delta_hi"].apply(lambda s: int((s < 0).sum()))
        trained = "/".join(sorted(gap["model"].unique()))
        say(f"\n=== transfer gap: AUROC {trained} minus the tabletop-trained model, same test rows "
            f"(handheld = phone test, tabletop = home-camera test; patient bootstrap per seed) ===")
        say(gt.to_string(float_format=lambda v: f"{v:+.3f}"))
    with open(os.path.join(d, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {d}/summary.txt")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phone-trained, phone-tested reference model for "
                                            "the camera ladder (in-domain ceiling).")
    p.add_argument("--summarize", default=None, metavar="DIR",
                   help="print the mean over DIR/seed*/ and exit (no training)")
    p.add_argument("--root", help="handheld dataset root (labels + images)")
    p.add_argument("--dataset", default="mbrset", choices=["mbrset", "brset"])
    p.add_argument("--split-file", default=None,
                   help="a RetinaReach run's split_handheld.csv: train / val / test exactly as "
                        "that run (and its --compare checkpoint) used them")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--compare", nargs="*", default=[],
                   help="RetinaReach checkpoints scored on the same phone test rows")
    p.add_argument("--mix-root", default=None,
                   help="tabletop dataset root: train on its train split AND the phone's (model "
                        "'mixed'); selection, thresholds and the test stay on the phone")
    p.add_argument("--mix-dataset", default="brset", choices=["brset", "mbrset"])
    p.add_argument("--mix-camera", default=None,
                   help="keep this tabletop camera only (the Canon sweep: 'Canon CR')")
    p.add_argument("--mix-split-file", default=None,
                   help="a RetinaReach run's split_tabletop.csv: its tabletop train photos are the "
                        "mixed model's, its tabletop test rows the home-camera check's")
    p.add_argument("--phone-share", type=float, default=0.5,
                   help="fraction of each epoch's draws that are phone photos (mixed runs)")
    p.add_argument("--targets", nargs="+", default=list(SOURCE_TARGETS))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--imbalance", default="sampler", choices=["sampler", "none"])
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="default = the Canon sweep's (run_retinareach.sh EXTRA)")
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--gpu-aug", dest="gpu_aug", action="store_true", default=None)
    p.add_argument("--no-gpu-aug", dest="gpu_aug", action="store_false")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--target-spec", type=float, default=0.825)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default="exp_phone_ceiling/seed0")
    p.add_argument("--ckpt", default="ck_phone_ceiling/seed0.pt")
    p.add_argument("--log-every", type=int, default=100)
    a = p.parse_args(argv)
    if a.summarize:
        return summarize(a.summarize)
    if not a.root:
        p.error("--root is required unless --summarize")
    if a.mix_root and os.path.realpath(a.mix_root) == os.path.realpath(a.root):
        p.error("--mix-root is the phone --root")
    if a.mix_root and not 0 < a.phone_share < 1:
        p.error("--phone-share must be strictly between 0 and 1")
    if (a.mix_camera or a.mix_split_file) and not a.mix_root:
        p.error("--mix-camera / --mix-split-file need --mix-root")
    name = MIXED if a.mix_root else PHONE

    seed_everything(a.seed, deterministic=not a.nondeterministic)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.ckpt)), exist_ok=True)
    t0 = time.time()

    # ---- data ------------------------------------------------------------ #
    from brset_dataset import filter_camera, load_any
    table, csv = load_table(a.root, a.dataset)
    img_dir = load_any(a.root, a.dataset)["images_dir"]
    split, agrees = device_split(table, a.targets[0], a.split_seed, a.split_file)
    if agrees is False:
        print(f"[warn] --split-file differs from multitask_split(seed={a.split_seed}) on this label "
              f"table: using the file (it fixed that run's test patients)")
    pd.concat([v.assign(split=k)[["file", "patient", "split"]] for k, v in split.items()]
              ).to_csv(os.path.join(a.out, "split_handheld.csv"), index=False)
    gpu_aug = a.gpu_aug if a.gpu_aug is not None else dev.type == "cuda"
    mk = lambda sp, d, part, mode, size=a.image_size: MultiTargetDataset(
        sp[part], d, a.targets, mode, size, device_aug=gpu_aug and mode == "train")
    train_ds, val_ds, test_ds = (mk(split, img_dir, "train", "train"), mk(split, img_dir, "val", "val"),
                                 mk(split, img_dir, "test", "val"))
    for nm, d in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if len(d) == 0:
            raise SystemExit(f"[fatal] {PHONE} {nm} has no images on disk under {img_dir}")
    print(f"[data] {a.dataset} @ {csv}: {len(table)} rows / {table['patient'].nunique()} patients; "
          f"images on disk train/val/test {len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
          + ("" if agrees is None else f"; split file matches recomputed split: {agrees}"))

    # ---- mixed: the tabletop camera's train split joins the phone's -------- #
    tab_split, tab_dir, tab_agrees, tab_train, tab_test = None, None, None, None, None
    if a.mix_root:
        tab, tab_csv = load_table(a.mix_root, a.mix_dataset)
        n_all = len(tab)
        if a.mix_camera:
            try:
                tab = filter_camera(tab, a.mix_camera.strip())
            except ValueError as e:
                raise SystemExit(f"[fatal] --mix-camera: {e}")
        tab_dir = load_any(a.mix_root, a.mix_dataset)["images_dir"]
        tab_split, tab_agrees = device_split(tab, a.targets[0], a.split_seed, a.mix_split_file)
        if tab_agrees is False:
            print(f"[warn] --mix-split-file differs from multitask_split(seed={a.split_seed}): "
                  f"using the file (it fixed that run's tabletop photos)")
        pd.concat([v.assign(split=k)[["file", "patient", "split"]] for k, v in tab_split.items()]
                  ).to_csv(os.path.join(a.out, "split_tabletop.csv"), index=False)
        tab_train, tab_test = mk(tab_split, tab_dir, "train", "train"), mk(tab_split, tab_dir, "test", "val")
        for nm, d in (("train", tab_train), ("test", tab_test)):
            if len(d) == 0:
                raise SystemExit(f"[fatal] tabletop {nm} has no images on disk under {tab_dir}")
        print(f"[data] tabletop {a.mix_dataset} @ {tab_csv}"
              + (f", camera {a.mix_camera!r} ({len(tab)} of {n_all} rows)" if a.mix_camera else "")
              + f": images on disk train/test {len(tab_train)}/{len(tab_test)}"
              + ("" if tab_agrees is None else f"; split file matches recomputed split: {tab_agrees}"))
        print(f"[data] {MIXED}: {len(tab_train)} tabletop + {len(train_ds)} phone training photos, "
              f"phone share of draws {a.phone_share:.2f}")

    g = torch.Generator(); g.manual_seed(a.seed)
    dl = lambda d, **kw: DataLoader(d, batch_size=a.batch_size, num_workers=a.num_workers,
                                    worker_init_fn=seed_worker, pin_memory=dev.type == "cuda", **kw)
    y_train = train_ds.label_array()
    if tab_train is not None:
        fit_ds = torch.utils.data.ConcatDataset([tab_train, train_ds])
        w = mixed_weights(tab_train.label_array()[:, 0], y_train[:, 0], a.phone_share,
                          balance=a.imbalance == "sampler")
        sampler = WeightedRandomSampler(w, len(fit_ds), replacement=True, generator=g)
    else:
        fit_ds = train_ds
        sampler = (WeightedRandomSampler(balanced_weights(y_train[:, 0]), len(train_ds),
                                         replacement=True, generator=g)
                   if a.imbalance == "sampler" else None)
    train_loader = dl(fit_ds, sampler=sampler, shuffle=sampler is None, drop_last=True,
                      generator=g, persistent_workers=a.num_workers > 0)
    val_loader, test_loader = dl(val_ds), dl(test_ds)
    tab_loader = dl(tab_test) if tab_test is not None else None

    # ---- train (phone photos, plus the tabletop's when mixed) --------------- #
    model = RetinaReachNet(a.targets, (), backbone=a.backbone, pretrained=not a.no_pretrained,
                           dropout=a.dropout).to(dev)
    best, best_ep = fit(a, model, train_loader, val_loader, val_ds.label_array(), dev, name)
    print(f"[train] selected epoch {best_ep} (val mean AUROC {best:.4f})")

    # thresholds on the phone's own validation split: the phone is this model's home
    pv = predict(model, val_loader, dev)
    yv = val_ds.label_array()
    thr = {}
    for k, t in enumerate(a.targets):
        ok = ~np.isnan(yv[:, k])
        thr[t] = threshold_at_sensitivity(yv[ok, k].astype(int), pv[ok, k], a.target_sens)
    probs = {name: predict(model, test_loader, dev)}
    home_probs = {name: predict(model, tab_loader, dev)} if tab_loader is not None else {}
    thresholds = {name: thr}
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {**model.config(), "image_size": a.image_size},
                "thresholds": {"trained": thr}, "profiles": {}, "args": vars(a),
                "best_epoch": best_ep, "best_val": best}, a.ckpt)

    # ---- the tabletop-trained models on the same phone (and tabletop) test rows #
    compare_info = {}
    as_targets = lambda q, net, n: np.column_stack([q[:, net.index(t)] if t in net.targets
                                                    else np.full(n, np.nan) for t in a.targets])
    for path in a.compare:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck_args, tag = ck["args"], checkpoint_tag(path)
        size = int(ck["config"].get("image_size", 384))
        if int(ck_args.get("split_seed", 42)) != a.split_seed and not a.split_file:
            print(f"[warn] {path}: split_seed {ck_args.get('split_seed')} != {a.split_seed}; its "
                  f"handheld test patients are not these (pass its run's --split-file)")
        if a.mix_root and (ck_args.get("source_camera") or None) != (a.mix_camera or None):
            print(f"[warn] {path}: trained on camera {ck_args.get('source_camera') or 'all'}, the "
                  f"mixed model on {a.mix_camera or 'all'}: the home-camera rows are not its own")
        ds = test_ds if size == a.image_size else mk(split, img_dir, "test", "val", size)
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers,
                            worker_init_fn=seed_worker)
        target_name = ck_args.get("target_name", "handheld")
        for proto, stats in (("trained", "trained"), ("calibrated", target_name)):
            if stats != "trained" and stats not in ck.get("profiles", {}):
                print(f"[warn] {path}: no {stats!r} profile; skipping {proto}")
                continue
            net = build_from_checkpoint(ck, stats=stats).to(dev)
            m = f"{tag}_{proto}"
            probs[m] = as_targets(predict(net, loader, dev), net, len(ds))
            thresholds[m] = {t: float(ck["thresholds"][proto].get(t, np.nan)) for t in a.targets}
            if tab_test is not None and proto == "trained":     # its home camera, its home stats
                hds = tab_test if size == a.image_size else mk(tab_split, tab_dir, "test", "val", size)
                hl = DataLoader(hds, batch_size=a.batch_size, num_workers=a.num_workers,
                                worker_init_fn=seed_worker)
                home_probs[m] = as_targets(predict(net, hl, dev), net, len(hds))
            del net
        compare_info[tag] = {"path": os.path.abspath(path), "seed": ck_args.get("seed"),
                             "source_camera": ck_args.get("source_camera"), "image_size": size}
        print(f"[compare] {path}: seed {ck_args.get('seed')}, source camera "
              f"{ck_args.get('source_camera') or 'all'}, {size} px")

    # ---- report ------------------------------------------------------------- #
    ceil_rows, ref_rows, home_rows = [], [], []
    for m, q in probs.items():
        ceil_rows += score_rows(m, q, test_ds, a.targets, thresholds[m], a)
        pred = predictions_frame(q, test_ds, a.targets, thresholds[m])
        pred.to_csv(os.path.join(a.out, f"predictions_{m}.csv"), index=False)
        ref_rows += referral_rows(m, pred, table, a)
    for m, q in home_probs.items():
        home_rows += score_rows(m, q, tab_test, a.targets, thresholds[m], a)
        predictions_frame(q, tab_test, a.targets, thresholds[m]).to_csv(
            os.path.join(a.out, f"predictions_home_{m}.csv"), index=False)
    gaps = gap_rows(name, probs, test_ds, a.targets, "handheld", a)
    if home_probs:
        gaps += gap_rows(name, home_probs, tab_test, a.targets, "tabletop", a)
    ceil, ref, gap = pd.DataFrame(ceil_rows), pd.DataFrame(ref_rows), pd.DataFrame(gaps)
    ceil.to_csv(os.path.join(a.out, "ceiling.csv"), index=False)
    ref.to_csv(os.path.join(a.out, "referral.csv"), index=False)
    gap.to_csv(os.path.join(a.out, "transfer_gap.csv"), index=False)
    if home_rows:
        pd.DataFrame(home_rows).to_csv(os.path.join(a.out, "home_camera.csv"), index=False)

    fmt = lambda v: f"{v:.3f}"
    print(f"\n=== phone test, image level (target sens {a.target_sens}; oracle = where the test "
          f"ROC curve crosses it) ===")
    print(ceil[["model", "target", "n", "pos_patients", "auroc", "auroc_lo", "auroc_hi", "sensitivity",
                "specificity", "oracle_spec"]].to_string(index=False, float_format=fmt))
    if len(ref):
        print(f"\n=== per patient at the shipped threshold (IDx-DR: sens >= {a.target_sens}, "
              f"spec >= {a.target_spec}) ===")
        print(ref[["model", "rule", "patients", "retakes", "sensitivity", "specificity",
                   "meets_targets"]].to_string(index=False, float_format=fmt))
    if home_rows:
        print(f"\n=== home camera: tabletop test split ({len(tab_test)} photos), each model at its "
              f"own shipped threshold ===")
        print(pd.DataFrame(home_rows)[["model", "target", "n", "auroc", "auroc_lo", "auroc_hi",
                                       "sensitivity", "specificity"]]
              .to_string(index=False, float_format=fmt))
    if len(gap):
        print(f"\n=== transfer gap: AUROC {name} minus each, same rows, patient bootstrap ===")
        print(gap.drop(columns="model").to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    n = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    if tab_train is not None:
        n.update(tabletop_train=len(tab_train), tabletop_test=len(tab_test))
    results = {"args": vars(a), "model": name, "best_epoch": best_ep, "best_val_mean_auroc": best,
               "split_file_agrees": agrees, "mix_split_file_agrees": tab_agrees, "n": n,
               "thresholds": thresholds, "compare": compare_info,
               "ceiling": ceil_rows, "referral": ref_rows, "home_camera": home_rows,
               "transfer_gap": gaps,
               "minutes": (time.time() - t0) / 60,
               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] {a.out}/ and {a.ckpt} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
