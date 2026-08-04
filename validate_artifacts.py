"""
validate_artifacts.py
=====================
**Validates the artifact-reduction pipeline (RQ1) against expert labels.**

``artifacts.py`` computes image-quality signals (focus, glare, illumination
uniformity) using thresholds that were *chosen by hand*. mBRSET is the first
dataset available here with expert judgements of exactly those properties:

    final_quality   : is the image gradable?          (expert)
    final_artifacts : are acquisition artifacts present? (expert)

So we can finally ask the research question quantitatively:

    Do the computed quality signals agree with ophthalmologist judgement
    on real smartphone-captured fundus images?

For each metric we report **AUROC** against both expert labels (threshold-free,
so it does not depend on the hand-picked cut-offs), plus the best achievable
operating point (Youden's J) — which also *calibrates* the thresholds that
``artifacts.py`` should use instead of the current guesses.

Run (on the server, where mBRSET lives):
    python validate_artifacts.py \
        --root /data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0 \
        --workers 8 --out artifact_validation.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifacts import assess_quality  # noqa: E402

METRICS = ["focus", "glare_frac", "illumination_spread", "quality_score"]
# Higher metric value should predict the POSITIVE class; flip where it doesn't.
HIGHER_IS_GOOD = {"focus": True, "glare_frac": False,
                  "illumination_spread": False, "quality_score": True}


def _binarize(s: pd.Series) -> pd.Series:
    """Map yes/no, 1/0, adequate/inadequate -> 1.0/0.0 (NaN if unknown)."""
    def one(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip().lower()
        if t in {"yes", "y", "1", "1.0", "true", "present", "adequate", "gradable"}:
            return 1.0
        if t in {"no", "n", "0", "0.0", "false", "absent", "inadequate", "ungradable"}:
            return 0.0
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(one)


def _score_one(args):
    path, fname = args
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        q = assess_quality(rgb)
        return {"file": fname, **{k: q[k] for k in METRICS}, "gradable_pred": q["gradable"]}
    except Exception as e:  # noqa
        return {"file": fname, **{k: np.nan for k in METRICS},
                "gradable_pred": np.nan, "error": str(e)[:80]}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = ~(np.isnan(scores) | np.isnan(labels))
    if m.sum() < 10 or len(np.unique(labels[m])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[m], scores[m]))


def best_threshold(scores, labels):
    """Youden's J operating point -> (threshold, sensitivity, specificity)."""
    from sklearn.metrics import roc_curve
    m = ~(np.isnan(scores) | np.isnan(labels))
    if m.sum() < 10 or len(np.unique(labels[m])) < 2:
        return (float("nan"),) * 3
    fpr, tpr, thr = roc_curve(labels[m], scores[m])
    j = int(np.argmax(tpr - fpr))
    return float(thr[j]), float(tpr[j]), float(1 - fpr[j])


def main() -> int:
    p = argparse.ArgumentParser(description="Validate artifact metrics against mBRSET expert labels.")
    p.add_argument("--root", required=True, help="mBRSET 1.0 directory (contains images/ + labels_mbrset.csv).")
    p.add_argument("--limit", type=int, default=0, help="Only score the first N images (0 = all).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="artifact_validation.csv")
    args = p.parse_args()

    csv_path = os.path.join(args.root, "labels_mbrset.csv")
    img_dir = os.path.join(args.root, "images")
    df = pd.read_csv(csv_path)
    print(f"[info] labels : {len(df)} rows  |  images dir: {img_dir}")

    # Resolve filenames on disk (the CSV 'file' column may omit the extension).
    on_disk = {f: f for f in os.listdir(img_dir)}
    stems = {os.path.splitext(f)[0]: f for f in on_disk}
    df["_file"] = df["file"].astype(str).map(lambda v: on_disk.get(v) or stems.get(os.path.splitext(v)[0]))
    missing = int(df["_file"].isna().sum())
    df = df.dropna(subset=["_file"])
    if args.limit:
        df = df.iloc[:args.limit]
    print(f"[info] scoring: {len(df)} images ({missing} CSV rows had no image file)")

    for col in ("final_quality", "final_artifacts"):
        print(f"[info] {col} raw values: {df[col].value_counts(dropna=False).to_dict()}")

    tasks = [(os.path.join(img_dir, f), f) for f in df["_file"]]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_score_one, tasks, chunksize=16))
    got = pd.DataFrame(rows)
    n_err = int(got.get("error", pd.Series(dtype=object)).notna().sum()) if "error" in got else 0
    if n_err:
        print(f"[warn] {n_err} images failed to score")

    merged = df.merge(got, left_on="_file", right_on="file", suffixes=("", "_m"))
    # Expert labels -> binary. 'gradable' is the positive class for quality;
    # 'clean' (no artifacts) is the positive class for artifacts, so invert.
    y_quality = _binarize(merged["final_quality"]).to_numpy()
    y_clean = 1.0 - _binarize(merged["final_artifacts"]).to_numpy()

    print("\n" + "=" * 76)
    print("RQ1 — do computed quality signals agree with expert judgement? (AUROC)")
    print("=" * 76)
    print(f"{'metric':<22}{'vs gradable':>14}{'vs artifact-free':>20}")
    results = {}
    for k in METRICS:
        s = merged[k].to_numpy(dtype=float)
        if not HIGHER_IS_GOOD[k]:
            s = -s                                  # orient so higher = better quality
        a_q, a_a = auroc(s, y_quality), auroc(s, y_clean)
        results[k] = (a_q, a_a)
        print(f"{k:<22}{a_q:>14.3f}{a_a:>20.3f}")
    print("(0.5 = chance; >0.7 = useful; <0.5 means the signal is inverted)")

    print("\n" + "-" * 76)
    print("Calibrated operating points (Youden's J) — use these instead of hand-picked cut-offs")
    print("-" * 76)
    for k in METRICS:
        s = merged[k].to_numpy(dtype=float)
        sgn = 1.0 if HIGHER_IS_GOOD[k] else -1.0
        thr, sens, spec = best_threshold(sgn * s, y_quality)
        if not np.isnan(thr):
            print(f"{k:<22} threshold={sgn*thr:>9.4f}  sensitivity={sens:.3f}  specificity={spec:.3f}")

    # How well does the current hand-tuned rule do?
    if "gradable_pred" in merged:
        pred = merged["gradable_pred"].astype(float).to_numpy()
        m = ~(np.isnan(pred) | np.isnan(y_quality))
        if m.sum():
            acc = float((pred[m] == y_quality[m]).mean())
            tp = float(((pred[m] == 1) & (y_quality[m] == 1)).sum())
            fn = float(((pred[m] == 0) & (y_quality[m] == 1)).sum())
            fp = float(((pred[m] == 1) & (y_quality[m] == 0)).sum())
            print(f"\ncurrent hand-tuned 'gradable' rule: accuracy={acc:.3f}  "
                  f"sensitivity={tp/max(tp+fn,1):.3f}  precision={tp/max(tp+fp,1):.3f}")

    merged.to_csv(args.out, index=False)
    print(f"\n[info] per-image metrics + labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
