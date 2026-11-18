#!/usr/bin/env python3
"""
plot_score_distributions.py
===========================
The score-distribution figure: for one training run, where the model's scores
fall for healthy and for referable photos on each camera, with the shipped
cutoff marked. It shows why a cutoff fixed on the home camera stops working on
the phone. Reads a finished RetinaReach run's saved predictions -- no GPU, no
retraining, seconds::

    # on the machine that holds the run and the datasets
    python plot_score_distributions.py --dir exp_rr_canon --seed 0

    # anywhere with pandas + matplotlib, from the table the first call wrote
    python3 plot_score_distributions.py --from-csv score_distributions_seed0.csv

Three panels, top to bottom: the home camera as trained, the phone as trained,
the phone after recalibration (AdaBN). Each panel draws the healthy and the
referable photos as two histograms, EACH scaled to the same area, so the rarer
referable class stays visible; the x axis is the score on a logit (stretched)
scale, because most scores sit very close to 0 or 1.

Outputs (``--out``, default ``<dir>/figures``):
``score_distributions_seed<s>.csv`` (panel, truth, score, threshold: no file or
patient names) and, when matplotlib is installed, ``score_distributions_seed<s>.png``.
The printed table gives each panel's sensitivity and specificity at the cutoff,
to check against ``ladder.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAPER, INK, MUTED, AXIS = "#F7F5F0", "#1F2D3A", "#5B6770", "#B9B6AA"
HEALTHY, REFERABLE = "#1F6F78", "#A8502A"
EPS = 1e-5                                     # scores are clipped to [EPS, 1 - EPS] before the logit
TICKS = (0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def collect(run_dir: str, seed: int, target: str, labels: Sequence[str]) -> pd.DataFrame:
    """One row per test photo and panel: ``panel``, ``truth`` (1 / 0), ``score``,
    ``threshold``. Truth comes from the dataset's own label table, joined on the
    file name exactly as ``referral_outcomes.eye_table`` does."""
    import json
    from metadata_model import load_table, target_vector
    from referral_outcomes import devices_of, run_dirs

    base = dict(run_dirs(run_dir)).get(seed)
    if base is None:
        raise SystemExit(f"[fatal] no seed {seed} (results.json) under {run_dir}")
    with open(os.path.join(base, "results.json")) as f:
        results = json.load(f)
    a = results["args"]
    where = {name: (root, ds) for name, root, ds in devices_of(results)}
    wanted = [(a["source_name"], "trained"), (a["target_name"], "trained"),
              (a["target_name"], "calibrated")]
    rows = []
    for (device, proto), label in zip(wanted, labels):
        p = os.path.join(base, f"predictions_{device}_{proto}.csv")
        if not os.path.exists(p):
            print(f"[skip] {p} not found")
            continue
        pred = pd.read_csv(p)
        if f"p_{target}" not in pred or pred[f"threshold_{target}"].isna().all():
            print(f"[skip] {p} has no score or threshold for {target}")
            continue
        table = load_table(*where[device])[0]
        df = pred.merge(table.drop(columns=[c for c in ("patient",) if c in table.columns]),
                        on="file", how="left", validate="many_to_one")
        y = target_vector(df, target)
        ok = ~np.isnan(y)
        rows.append(pd.DataFrame({"panel": label, "truth": y[ok].astype(int),
                                  "score": df.loc[ok, f"p_{target}"].to_numpy(),
                                  "threshold": df.loc[ok, f"threshold_{target}"].to_numpy()}))
    if not rows:
        raise SystemExit(f"[fatal] no predictions_<device>_<protocol>.csv usable under {base}")
    return pd.concat(rows, ignore_index=True)


def panel_stats(scores: pd.DataFrame) -> pd.DataFrame:
    """Per panel: photos, referable share, cutoff, sensitivity and specificity at it."""
    out = []
    for label, g in scores.groupby("panel", sort=False):
        flag, y = g["score"] >= g["threshold"], g["truth"] == 1
        out.append({"panel": label, "photos": len(g), "referable_share": float(y.mean()),
                    "threshold": float(g["threshold"].iloc[0]),
                    "sensitivity": float(flag[y].mean()) if y.any() else float("nan"),
                    "specificity": float((~flag[~y]).mean()) if (~y).any() else float("nan")})
    return pd.DataFrame(out)


def draw(scores: pd.DataFrame, out_png: str, bins: int = 44, width: float = 6.0,
         height: float = 6.6, font: float = 15.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": font,
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    stats = panel_stats(scores)
    edges = np.linspace(logit(EPS), logit(1 - EPS), bins + 1)
    fig, axes = plt.subplots(len(stats), 1, figsize=(width, height), sharex=True, facecolor=PAPER)
    for ax, (_, s) in zip(np.atleast_1d(axes), stats.iterrows()):
        g = scores[scores["panel"] == s["panel"]]
        for cls, colour in ((0, HEALTHY), (1, REFERABLE)):
            x = logit(g.loc[g["truth"] == cls, "score"])
            if len(x):
                ax.hist(x, bins=edges, density=True, color=colour, alpha=0.55, linewidth=0)
                ax.hist(x, bins=edges, density=True, histtype="step", color=colour, linewidth=1.4)
        ax.axvline(float(logit(s["threshold"])), color=INK, linewidth=1.8, linestyle=(0, (4, 3)))
        ax.set_title(f"{s['panel']}:  caught {s['sensitivity']:.0%},  cleared {s['specificity']:.0%}",
                     loc="left", fontsize=font, fontweight="bold", color=INK, pad=6)
        ax.set_facecolor(PAPER)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="x", colors=MUTED, length=3, labelsize=font - 3)
    last = np.atleast_1d(axes)[-1]
    last.set_xticks(logit(TICKS))
    last.set_xticklabels([f"{t:g}" for t in TICKS], rotation=45, ha="right")
    last.set_xlim(edges[0], edges[-1])
    last.set_xlabel("Model score (stretched scale)", color=MUTED, fontsize=font - 1)
    fig.text(0.02, 0.012, "healthy", color=HEALTHY, fontsize=font - 1, fontweight="bold")
    fig.text(0.21, 0.012, "referable", color=REFERABLE, fontsize=font - 1, fontweight="bold")
    fig.text(0.44, 0.012, "- - cutoff", color=INK, fontsize=font - 1, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.9)
    fig.savefig(out_png, dpi=300, facecolor=PAPER)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Score distributions per camera, with the shipped cutoff.")
    p.add_argument("--dir", default="exp_rr_canon", help="a sweep (seed*/) or a single run dir")
    p.add_argument("--seed", type=int, default=0, help="which training run to draw")
    p.add_argument("--target", default="dr_referable")
    p.add_argument("--from-csv", default=None, metavar="CSV",
                   help="draw from a table an earlier call wrote (no datasets needed)")
    p.add_argument("--labels", nargs=3, metavar="LABEL",
                   default=["Canon tabletop", "Phone, as trained", "Phone, recalibrated"],
                   help="panel names: home camera, phone as trained, phone recalibrated")
    p.add_argument("--panels", type=int, default=3, choices=(2, 3),
                   help="2 leaves the recalibrated phone panel out")
    p.add_argument("--out", default=None, help="default: <dir>/figures, or the CSV's folder")
    a = p.parse_args(argv)

    if a.from_csv:
        scores = pd.read_csv(a.from_csv)
        out = a.out or os.path.dirname(os.path.abspath(a.from_csv))
        stem = os.path.splitext(os.path.basename(a.from_csv))[0]
    else:
        scores = collect(a.dir, a.seed, a.target, a.labels)
        out = a.out or os.path.join(a.dir, "figures")
        stem = f"score_distributions_seed{a.seed}"
        os.makedirs(out, exist_ok=True)
        scores.to_csv(os.path.join(out, stem + ".csv"), index=False)
        print(f"wrote {os.path.join(out, stem + '.csv')}")
    keep = list(dict.fromkeys(scores["panel"]))[:a.panels]
    scores = scores[scores["panel"].isin(keep)]
    print(panel_stats(scores).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    try:
        draw(scores, os.path.join(out, stem + ".png"))
        print(f"wrote {os.path.join(out, stem + '.png')}")
    except ImportError:
        print("matplotlib is not installed here: copy the CSV to a machine that has it and run\n"
              f"  python3 plot_score_distributions.py --from-csv {stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
