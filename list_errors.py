#!/usr/bin/env python3
"""
list_errors.py
==============
Which test images did one RetinaReach run get wrong? Joins a seed's
``predictions_<device>_<protocol>.csv`` (score and shipped threshold per image)
to the graders' labels and lists every **missed** image (labelled positive,
score below the threshold) and every **false alarm** (labelled negative,
flagged), per camera, with the raw ICDR grade, edema and image-quality columns.

    python list_errors.py --dir exp_rr_canon --seed 0 --protocol trained \\
        --target dr_referable --brset <BRSET root> --mbrset <mBRSET root> --copy

Writes ``<dir>/seed<s>/errors_<target>_<protocol>.csv``. ``--copy`` also copies
the images into ``<dir>/seed<s>/errors_<target>_<protocol>/<device>/<missed|false_alarm>/``,
each file name prefixed with its score, to look through by eye.

Image level, one seed: the referral tables count patients (a patient is referred
when any of their images is flagged), so these counts are larger than theirs.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brset_dataset import load_any                       # noqa: E402
from metadata_model import load_table, target_vector     # noqa: E402

RAW_COLS = ["final_icdr", "final_edema", "final_quality", "camera"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", default="exp_rr_canon")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--protocol", default="trained", choices=["trained", "calibrated"])
    p.add_argument("--target", default="dr_referable", choices=["dr_referable", "edema"])
    p.add_argument("--brset", required=True, help="BRSET root (tabletop and the held-out camera)")
    p.add_argument("--mbrset", required=True, help="mBRSET root (handheld)")
    p.add_argument("--copy", action="store_true", help="also copy the wrong images into folders")
    a = p.parse_args()

    run = os.path.join(a.dir, f"seed{a.seed}")
    files = sorted(glob.glob(os.path.join(run, f"predictions_*_{a.protocol}.csv")))
    if not files:
        sys.exit(f"[fatal] no predictions_*_{a.protocol}.csv under {run}")

    tables = {}
    out = []
    for path in files:
        device = re.match(rf"predictions_(.+)_{a.protocol}\.csv$", os.path.basename(path)).group(1)
        ds = "mbrset" if device == "handheld" else "brset"
        root = a.mbrset if ds == "mbrset" else a.brset
        if ds not in tables:
            df, _ = load_table(root, ds)
            df = df.assign(label=target_vector(df, a.target))
            tables[ds] = (df[["file", "label"] + [c for c in RAW_COLS if c in df.columns]]
                          .drop_duplicates("file"), load_any(root, ds)["images_dir"])
        lab, img_dir = tables[ds]
        pr = pd.read_csv(path)
        score, thr = f"p_{a.target}", f"threshold_{a.target}"
        if score not in pr.columns or pr[thr].isna().all():
            print(f"[skip] {device}: no {a.target} score or threshold")
            continue
        m = pr[["file", "patient", score, thr]].merge(lab, on="file", how="left")
        if m["label"].isna().all():
            print(f"[skip] {device}: no {a.target} labels joined (file names do not match)")
            continue
        m = m.dropna(subset=["label", score])
        flagged = m[score] >= m[thr]
        m["outcome"] = np.select([(m["label"] == 1) & ~flagged, (m["label"] == 0) & flagged],
                                 ["missed", "false_alarm"], "correct")
        m["device"], m["image_path"] = device, [os.path.join(img_dir, f) for f in m["file"]]
        pos, neg = int((m["label"] == 1).sum()), int((m["label"] == 0).sum())
        n_miss, n_fa = int((m["outcome"] == "missed").sum()), int((m["outcome"] == "false_alarm").sum())
        print(f"[{device:<13}] {len(m):5d} test images: missed {n_miss:4d} of {pos:4d} positive, "
              f"false alarms {n_fa:4d} of {neg:5d} negative  (threshold {m[thr].iloc[0]:.3f})")
        out.append(m[m["outcome"] != "correct"].rename(columns={score: "score", thr: "threshold"}))

    if not out:
        sys.exit("[fatal] nothing scored")
    err = pd.concat(out, ignore_index=True)
    # worst first: missed images with the lowest scores, false alarms with the highest
    err["_k"] = np.where(err["outcome"] == "missed", err["score"], -err["score"])
    err = err.sort_values(["device", "outcome", "_k"]).drop(columns="_k")
    csv = os.path.join(run, f"errors_{a.target}_{a.protocol}.csv")
    err.to_csv(csv, index=False)
    print(f"\nwrote {csv} ({len(err)} images)")
    if "final_icdr" in err.columns:
        print("\nICDR grade of the wrong images:")
        print(err.groupby(["device", "outcome"])["final_icdr"].value_counts(dropna=False)
              .unstack(fill_value=0).to_string())

    if a.copy:
        base = os.path.join(run, f"errors_{a.target}_{a.protocol}")
        missing = 0
        for r in err.itertuples():
            dst = os.path.join(base, r.device, r.outcome)
            os.makedirs(dst, exist_ok=True)
            if os.path.isfile(r.image_path):
                shutil.copy2(r.image_path, os.path.join(dst, f"{r.score:.3f}_{os.path.basename(r.file)}"))
            else:
                missing += 1
        print(f"copied the images into {base}/" + (f"  ({missing} not found on disk)" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
