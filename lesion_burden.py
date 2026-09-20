#!/usr/bin/env python3
"""
lesion_burden.py
================
Does sub-threshold retinopathy explain the retinal age gap of diabetics whose
eyes are graded DR 0?

``analyze_age_gap.py`` finds that BRSET diabetics with no visible retinopathy
read about two years older than non-diabetics (three with no other eye disease
in either group). One explanation it cannot rule out is lesions below the
graders' threshold: the clock is very sensitive to lesions (+4.7 y for referable
DR), so a handful of microaneurysms a grader waved through could be what it is
reading. This script tests that with the repository's own lesion segmenter.

It takes exactly the cohort of the "diabetes before retinopathy" block (worst
eye DR grade 0, no macular edema, known diabetes status, gradable images), runs
a ``train_idrid.py`` segmentation checkpoint over every image, reduces each
probability map to a per-lesion **burden** (area above threshold inside the
field of view, in parts per million, plus a blob count), averages the two eyes,
and reports:

1. whether grade-0 diabetics carry more predicted lesion than non-diabetics
   (share of patients with any predicted lesion; adjusted difference in
   log10(1 + ppm));
2. whether burden tracks the age gap at all;
3. the mediation-style check that matters: the diabetes effect on the gap
   before and after adjusting for the burden of every lesion type. If the
   effect barely moves, lesions the segmenter can see do not explain it; if it
   collapses, they do. The same is repeated among patients with no other
   ophthalmic flag.

The segmenter was trained on IDRiD / FGADR, not BRSET, so absolute burden is
uncalibrated and it will fire on some non-lesions (reflexes, drusen). That
affects both groups alike; every comparison here is relative, and the
covariates (age, age², sex, camera) are the same as in ``analyze_age_gap.py``.
The per-image table is written incrementally, so an interrupted run resumes.

    python lesion_burden.py --seg-checkpoint checkpoints/gcg_seed0.pt \\
        --predictions exp_retinal_age_v5/predictions_pooled.csv --condition medium_mix \\
        --root <BRSET> --brset-csv <BRSET>/labels_brset.csv --out exp_retinal_age_v5/lesion_burden
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_age_gap import (MIN_CASES, _f, attach_brset_flags, covariates, load_predictions,   # noqa: E402
                             ols, to_patients)
from fundus_utils import crop_to_fov, pick_device, tiled_predict                 # noqa: E402
from idrid_dataset import IMAGENET_MEAN, IMAGENET_STD                           # noqa: E402
from train_retinal_age import PATHOLOGY_COLS, _flag                              # noqa: E402

FOV_TOL = 12


# --------------------------------------------------------------------------- #
# Cohort: the same patients as analyze_age_gap.prelesion
# --------------------------------------------------------------------------- #
def grade0_cohort(df: pd.DataFrame, gap_col: str = "gap_corrected") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(patient table, image rows) for gradable images of patients whose worst eye is DR
    grade 0, who have no macular edema and a known diabetes status."""
    img = df[df["gradable"].map(_flag) == 1.0] if "gradable" in df.columns else df
    pat = to_patients(img, gap_col)
    keep = (pat["dr_grade"] == 0) & pat["diabetes"].notna() & pat["gap"].notna()
    if "final_edema" in pat.columns:
        keep &= pat["final_edema"] != 1.0
    pat = pat[keep].reset_index(drop=True)
    rows = img[img["patient"].isin(set(pat["patient"]))].reset_index(drop=True)
    return pat, rows


# --------------------------------------------------------------------------- #
# Scoring one image
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def score_image(net, path: str, lesions: Sequence[str], device, size: int = 512, tiled: bool = False,
                tile: int = 512, overlap: int = 64, thr: float = 0.5) -> Dict[str, float]:
    """FOV crop -> (resize to ``size`` | native-resolution tiles) -> sigmoid maps -> per-lesion
    area in ppm of the field of view, blob count and max probability."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    rgb, _ = crop_to_fov(rgb, tol=FOV_TOL)
    if not tiled:
        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    fov = rgb.max(axis=2) > FOV_TOL
    x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    if tiled:
        probs = tiled_predict(net, x, tile, overlap, device, fg_map=torch.from_numpy(fov),
                              n_classes=len(lesions)).float().cpu().numpy()
    else:
        probs = torch.sigmoid(net(x[None].to(device)))[0].float().cpu().numpy()
    n_fov = max(int(fov.sum()), 1)
    out: Dict[str, float] = {"fov_px": float(n_fov)}
    union = np.zeros_like(fov)
    for c, code in enumerate(lesions):
        m = (probs[c] > thr) & fov
        union |= m
        out[f"ppm_{code}"] = 1e6 * float(m.sum()) / n_fov
        out[f"blobs_{code}"] = float(ndimage.label(m)[1])
        out[f"pmax_{code}"] = float(probs[c][fov].max()) if n_fov > 1 else 0.0
    out["ppm_any"] = 1e6 * float(union.sum()) / n_fov
    return out


def score_all(net, lesions, rows: pd.DataFrame, images_dir: str, out_csv: str, device, **kw) -> pd.DataFrame:
    """Score every row's image once; the CSV is appended to, so a rerun resumes."""
    done = pd.read_csv(out_csv) if os.path.isfile(out_csv) else pd.DataFrame(columns=["file"])
    seen = set(done["file"].astype(str))
    todo = [f for f in rows["file"].astype(str).unique() if f not in seen]
    print(f"[info] {len(seen)} images already scored, {len(todo)} to go")
    buf: List[Dict[str, object]] = []

    def flush(i: int) -> None:
        nonlocal buf
        if buf:
            pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.isfile(out_csv), index=False)
            buf = []
        print(f"  {i}/{len(todo)} scored", flush=True)

    for i, f in enumerate(todo, 1):
        p = os.path.join(images_dir, f)
        if os.path.isfile(p):
            try:
                buf.append({"file": f, **score_image(net, p, lesions, device, **kw)})
            except Exception as e:                                # one unreadable image must not end the run
                print(f"[warn] {f}: {e}")
        if len(buf) >= 200:
            flush(i)
    flush(len(todo))                                              # whatever is left, missing files included
    return pd.read_csv(out_csv) if os.path.isfile(out_csv) else done


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def patient_burden(scores: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Mean of every burden column over a patient's images. ``patient`` keeps the dtype it has in
    ``rows`` (BRSET ids are integers) so the result merges back onto the patient table."""
    m = rows[["file", "patient"]].astype({"file": str}).merge(scores.astype({"file": str}), on="file", how="inner")
    cols = [c for c in scores.columns if c.startswith(("ppm_", "blobs_", "pmax_"))]
    return m.groupby("patient", sort=False)[cols].mean().reset_index()


def _delta(y: np.ndarray, x: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    o = ols(y, np.column_stack([x, X]))
    return {"d": float(o["coef"][0]), "lo": float(o["lo"][0]), "hi": float(o["hi"][0]), "p": float(o["p"][0])}


def mediation(pat: pd.DataFrame, burden_cols: Sequence[str]) -> Optional[Dict[str, object]]:
    """Diabetes effect on the gap before / after adjusting for log burden of every lesion type."""
    X, _ = covariates(pat)
    keep = X.notna().all(axis=1) & pat[list(burden_cols)].notna().all(axis=1) & pat["gap"].notna()
    d, Xd = pat[keep], X[keep]
    x = d["diabetes"].to_numpy(float)
    if (x == 1).sum() < MIN_CASES or (x == 0).sum() < MIN_CASES:
        return None
    g = d["gap"].to_numpy(float)
    B = np.log10(1.0 + d[list(burden_cols)].to_numpy(float))
    before = _delta(g, x, Xd.to_numpy(float))
    after = _delta(g, x, np.column_stack([B, Xd.to_numpy(float)]))
    att = 1.0 - after["d"] / before["d"] if before["d"] else float("nan")
    return {"n1": int((x == 1).sum()), "n0": int((x == 0).sum()), "before": before, "after": after,
            "attenuation": float(att)}


def burden_report(pat: pd.DataFrame, lesions: Sequence[str]) -> List[str]:
    codes = list(lesions) + ["any"]
    L = [f"\n## Lesion burden in DR-grade-0 patients (n={len(pat)}: "
         f"{int((pat['diabetes'] == 1).sum())} diabetic, {int((pat['diabetes'] == 0).sum())} non-diabetic)"]
    X, used = covariates(pat)
    ok = X.notna().all(axis=1)
    L.append(f"\n### 1. Do grade-0 diabetics carry more predicted lesion? (adjusted for {', '.join(used)})")
    L.append("| lesion | patients with any predicted pixels: diabetic / non-diabetic | median ppm: diabetic / non-diabetic | "
             "adjusted Δ log10(1+ppm) [95% CI] | p |")
    L.append("|---|---|---|---|---|")
    dm, nd = pat[pat["diabetes"] == 1.0], pat[pat["diabetes"] == 0.0]
    for c in codes:
        col = f"ppm_{c}"
        r = _delta(np.log10(1.0 + pat.loc[ok, col].to_numpy(float)), pat.loc[ok, "diabetes"].to_numpy(float),
                   X[ok].to_numpy(float))
        L.append(f"| {c} | {100 * (dm[col] > 0).mean():.1f}% / {100 * (nd[col] > 0).mean():.1f}% | "
                 f"{dm[col].median():.0f} / {nd[col].median():.0f} | {_f(r['d'], 3, sign=True)} "
                 f"[{_f(r['lo'], 3, sign=True)}, {_f(r['hi'], 3, sign=True)}] | {_f(r['p'], 4)} |")
    L.append("\n### 2. Does burden track the age gap? (Spearman of log burden vs corrected gap)")
    L.append("| lesion | diabetics | non-diabetics |")
    L.append("|---|---|---|")
    for c in codes:
        cells = []
        for grp in (dm, nd):
            rho, p = stats.spearmanr(grp[f"ppm_{c}"], grp["gap"]) if len(grp) > 3 else (np.nan, np.nan)
            cells.append(f"ρ={_f(rho, 3, sign=True)} (p={_f(p, 4)})")
        L.append(f"| {c} | {cells[0]} | {cells[1]} |")
    L.append("\n### 3. Does burden explain the diabetes effect on the gap?")
    L.append("| cohort | n diabetic / ref | Δ gap, covariates only | Δ gap, + log burden of every lesion | attenuation |")
    L.append("|---|---|---|---|---|")
    cohorts = [("all grade-0, no edema", pat)]
    pc = [c for c in PATHOLOGY_COLS if c in pat.columns and pat[c].notna().any()]
    if pc:
        cohorts.append(("no other ophthalmic flag", pat[(pat[pc] == 0.0).all(axis=1)]))
    for lab, sub in cohorts:
        m = mediation(sub, [f"ppm_{c}" for c in lesions])
        if m is None:
            L.append(f"| {lab} | too few patients | | | |")
            continue
        b, a = m["before"], m["after"]
        L.append(f"| {lab} | {m['n1']} / {m['n0']} | {_f(b['d'], sign=True)} [{_f(b['lo'], sign=True)}, {_f(b['hi'], sign=True)}] | "
                 f"{_f(a['d'], sign=True)} [{_f(a['lo'], sign=True)}, {_f(a['hi'], sign=True)}] | {100 * m['attenuation']:.0f}% |")
    L.append("\nReading: if grade-0 diabetics carry no more predicted lesion than non-diabetics (table 1) and the "
             "diabetes effect barely attenuates (table 3), lesions this segmenter can see do not explain the gap. A "
             "large attenuation means they do. The segmenter was trained on other cameras, so burden is relative, "
             "not calibrated; lesions it cannot see remain possible.")
    return L


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sub-threshold lesion burden vs the retinal age gap in DR-grade-0 patients.")
    p.add_argument("--seg-checkpoint", required=True, help="train_idrid.py checkpoint (gcg_unet).")
    p.add_argument("--predictions", required=True, help="predictions_pooled.csv (or one run's predictions CSV).")
    p.add_argument("--condition", default=None)
    p.add_argument("--root", required=True, help="BRSET root (holds fundus_photos/).")
    p.add_argument("--brset-csv", default=None, help="Raw BRSET labels CSV: joins the other ophthalmic flags.")
    p.add_argument("--dataset", default="brset")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--size", type=int, default=512, help="Whole-image input size (ignored with --tiled).")
    p.add_argument("--tiled", action="store_true", help="Native-resolution tiles: slower, keeps microaneurysms.")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="Score at most this many patients per group (smoke test).")
    p.add_argument("--out", default="lesion_burden")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    df, cond = load_predictions(args.predictions, args.condition)
    df = df[df["dataset"].astype(str).str.lower() == args.dataset].copy()
    if args.brset_csv:
        df, _ = attach_brset_flags(df, args.brset_csv, image_ext=args.image_ext)
    pat, rows = grade0_cohort(df)
    if args.limit:
        keep = pd.concat([g.head(args.limit) for _, g in pat.groupby("diabetes")])["patient"]
        pat = pat[pat["patient"].isin(set(keep))].reset_index(drop=True)
        rows = rows[rows["patient"].isin(set(keep))].reset_index(drop=True)
    print(f"[info] cohort: {len(pat)} patients ({int((pat['diabetes'] == 1).sum())} diabetic), {len(rows)} images"
          + (f"  (condition {cond})" if cond else ""))

    from brset_dataset import load_any
    images_dir = load_any(args.root, args.dataset, image_ext=args.image_ext)["images_dir"]
    from eval_fgadr import load_model
    device = pick_device(args.device)
    net, lesions = load_model(args.seg_checkpoint, device, "gcg_unet", None)
    os.makedirs(args.out, exist_ok=True)
    scores = score_all(net, lesions, rows, images_dir, os.path.join(args.out, "lesion_burden_images.csv"), device,
                       size=args.size, tiled=args.tiled, tile=args.tile, overlap=args.overlap, thr=args.thr)
    pat = pat.merge(patient_burden(scores, rows), on="patient", how="inner")
    pat.to_csv(os.path.join(args.out, "lesion_burden_patients.csv"), index=False)
    L = [f"# Lesion burden vs retinal age gap", f"segmenter `{args.seg_checkpoint}` ({', '.join(lesions)}; "
         f"{'tiled native resolution' if args.tiled else f'whole image at {args.size} px'}, threshold {args.thr}); "
         f"predictions `{args.predictions}`" + (f" (condition `{cond}`)" if cond else "")]
    L += burden_report(pat, lesions)
    text = "\n".join(L)
    print(text)
    with open(os.path.join(args.out, "lesion_burden.md"), "w") as f:
        f.write(text + "\n")
    print(f"\n[info] wrote {args.out}/lesion_burden.md, lesion_burden_patients.csv, lesion_burden_images.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
