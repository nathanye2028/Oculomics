#!/usr/bin/env python3
"""
plot_retinareach.py
===================
Poster figures from a finished RetinaReach sweep (``run_retinareach.sh`` ->
``summarize_retinareach.py``). Reads only the sweep's CSV/JSON outputs -- no torch
-- so any Python with pandas + matplotlib runs it (on the Mac: the system
``python3``; the repo's .venv has no matplotlib)::

    python3 plot_retinareach.py --dir exp_retinareach            # -> exp_retinareach/figures/

Five figures, each with a CSV of exactly what it plots (the table view):

1. ``fig1_threshold_transfer``  sensitivity at the shipped threshold and AUROC per
   source target and camera, as trained vs self-calibrated (mean over seeds,
   whiskers = seed SD); the target sensitivity and the gate's floor marked.
2. ``fig2_capability``          the final capability matrix (targets x cameras,
   self-calibrated protocol): status label + AUROC and image-minus-metadata delta.
3. ``fig3_calibration_size``    sensitivity and AUROC on the handheld test split vs
   the number of unlabelled captures, with and without the source prior; the
   uncalibrated and full-pool values as reference lines.
4. ``fig4_mechanism``           camera decodability as trained vs calibrated (one
   line per seed), and AUROC with the vessels removed vs the same area elsewhere.
5. ``fig5_unfamiliar``          how often the phone shows each target on the
   unfamiliar camera vs N, against what that camera's own labels support.

Styling: the reference data-viz palette (light surface, hairline recessive
axes, 2 px lines, >= 8 px markers with a surface ring, at most three series
colours per chart, statuses in the status palette and always labelled in text).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]            # slots 1-3 (validated all-pairs)
STATUS_FILL = {"SUPPORTED": "#0ca30c", "THRESHOLD_DRIFT": "#fab219",
               "NO_IMAGE_EVIDENCE": "#ec835a", "UNDERPOWERED": "#e1e0d9",
               "NOT_VALIDATED": "#f0efec"}
STATUS_LABEL = {"SUPPORTED": "Supported", "THRESHOLD_DRIFT": "Threshold drift",
                "NO_IMAGE_EVIDENCE": "No image evidence", "UNDERPOWERED": "Underpowered",
                "NOT_VALIDATED": "Not validated"}


# --------------------------------------------------------------------------- #
# Loading (per-seed CSVs, as summarize_retinareach.load does, without torch)
# --------------------------------------------------------------------------- #
def load_runs(d: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, List[pd.DataFrame]] = {"cap": [], "unf": [], "sweep": [], "ablation": [],
                                          "lookup": []}
    results = {}
    files = {"cap": "capability.csv", "unf": "capability_unfamiliar.csv",
             "sweep": "calibration_sweep.csv", "ablation": "anatomy_ablation.csv",
             "lookup": "unfamiliar_lookup.csv"}
    for base in sorted(glob.glob(os.path.join(d, "seed*"))):
        m = re.fullmatch(r"seed(\d+)", os.path.basename(base))
        if not m or not os.path.exists(os.path.join(base, "results.json")):
            continue
        s = int(m.group(1))
        with open(os.path.join(base, "results.json")) as fh:
            res = json.load(fh)
        if res.get("shuffle_source_labels"):
            continue
        results[s] = res
        for k, f in files.items():
            p = os.path.join(base, f)
            if os.path.exists(p):
                t = pd.read_csv(p).assign(seed=s)
                if k == "sweep" and res.get("prior_mode") != "in_forward" and "prior_strength" in t:
                    t = t[t["prior_strength"] <= 0]        # pre-fix post-hoc prior rows are invalid
                out[k].append(t)
    frames = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}
    frames["results"] = results
    return frames


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
    return plt


def hairgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def ref_lines(ax, items, min_gap: float = 0.07) -> None:
    """Horizontal reference lines labelled OUTSIDE the plot on the right, nudged
    apart in axes space when two values are close (labels never sit on a line)."""
    import matplotlib.transforms as mt
    lo, hi = ax.get_ylim()
    pos = []
    for v, lab in sorted(items, key=lambda t: t[0]):
        if v != v:
            continue
        ax.axhline(v, color=MUTED, lw=0.8, zorder=0)
        f = (v - lo) / (hi - lo)
        if pos and f - pos[-1] < min_gap:
            f = pos[-1] + min_gap
        pos.append(f)
        ax.text(1.02, f, lab, transform=mt.blended_transform_factory(ax.transAxes, ax.transAxes),
                color=MUTED, fontsize=8, va="center", ha="left", clip_on=False)


def text_on(fill: str) -> str:
    """White or ink, whichever contrasts more with ``fill`` (WCAG ratio)."""
    L = _luma(fill)
    white, ink = 1.05 / (L + 0.05), (L + 0.05) / (_luma(INK) + 0.05)
    return "#ffffff" if white >= ink else INK


def dot(ax, x, y, color, label=None, z=3):
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=1.5, label=label, zorder=z)


def save(fig, out_dir: str, name: str, table: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=220, bbox_inches="tight")
    table.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    print(f"  {name}.png  ({len(table)} rows in {name}.csv)")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_threshold_transfer(plt, cap: pd.DataFrame, source: List[str], devices: List[str],
                           target_sens: float, tol: float, out: str) -> None:
    rows = []
    for t in source:
        for dv in devices:
            g = cap[(cap["target"] == t) & (cap["device"] == dv)]
            for pr in ("trained", "calibrated"):
                v = g[g["protocol"] == pr]
                for m in ("sensitivity", "auroc"):
                    s = v[m].dropna() if m in v else pd.Series(dtype=float)
                    if len(s):
                        rows.append({"target": t, "device": dv, "protocol": pr, "metric": m,
                                     "mean": s.mean(), "sd": s.std(ddof=1) if len(s) > 1 else 0.0,
                                     "seeds": len(s)})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return
    keys = [(t, d) for t in source for d in devices
            if ((tab["target"] == t) & (tab["device"] == d)).any()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(keys) + 1.6), sharey=True)
    colors = {"trained": SERIES[1], "calibrated": SERIES[0]}
    names = {"trained": "as trained (ships today)", "calibrated": "self-calibrated (RetinaReach)"}
    for ax, m, title in ((axes[0], "sensitivity", "Sensitivity at the shipped threshold"),
                         (axes[1], "auroc", "AUROC")):
        hairgrid(ax)
        for i, (t, d) in enumerate(keys):
            y = len(keys) - 1 - i
            pts = {}
            for pr in ("trained", "calibrated"):
                r = tab[(tab["target"] == t) & (tab["device"] == d) & (tab["protocol"] == pr)
                        & (tab["metric"] == m)]
                if len(r):
                    pts[pr] = (float(r["mean"].iloc[0]), float(r["sd"].iloc[0]))
            if len(pts) == 2:
                ax.plot([pts["trained"][0], pts["calibrated"][0]], [y, y], color=AXIS, lw=1.5, zorder=1)
            for pr, (mu, sd) in pts.items():
                off = 0.2 if pr == "trained" else -0.2            # seed-SD whiskers never overlap
                if sd > 0:                                        # clamped to the metric's 0-1 range
                    ax.plot([max(mu - sd, 0.0), min(mu + sd, 1.0)], [y + off, y + off],
                            color=colors[pr], lw=1, zorder=2)
                dot(ax, mu, y, colors[pr], names[pr] if (i == 0 and ax is axes[0]) else None)
        if m == "sensitivity":
            for xv, lab, ha, dx in ((target_sens, f"target {target_sens:.2f}", "left", 0.01),
                                    (target_sens - tol, f"gate floor {target_sens - tol:.2f}",
                                     "right", -0.01)):
                ax.axvline(xv, color=MUTED, lw=0.8, zorder=0)
                ax.text(xv + dx, len(keys) - 0.45, lab, color=MUTED, fontsize=8, ha=ha, va="bottom")
        ax.set_xlim(0, 1.02)
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([f"{t}  ·  {d}" for t, d in keys][::-1])
    axes[0].set_ylim(-0.6, len(keys) - 0.1)
    fig.suptitle("Does self-calibration restore the tabletop threshold?", x=0.0, y=1.0,
                 ha="left", va="bottom", fontsize=14, fontweight="bold")
    fig.legend(loc="lower left", bbox_to_anchor=(0.0, 0.93), ncol=2)
    fig.text(0.0, -0.04, "Dots = mean over seeds; whiskers = seed SD (orange above, blue below). "
                         "The threshold is fixed on the tabletop validation split in both cases.",
             fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.84)
    save(fig, out, "fig1_threshold_transfer", tab)
    plt.close(fig)


def _luma(hexcol: str) -> float:
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def fig_capability(plt, final: pd.DataFrame, targets: List[str], devices: List[str], out: str,
                   unfamiliar: Optional[List[str]] = None) -> None:
    c = final[final["protocol"] == "calibrated"]
    if c.empty:
        return
    devices = [d for d in devices if (c["device"] == d).any()]
    targets = [t for t in targets if (c["target"] == t).any()]
    fig, ax = plt.subplots(figsize=(2.3 * len(devices) + 2.2, 0.62 * len(targets) + 1.2))
    ax.set_xlim(0, len(devices))
    ax.set_ylim(0, len(targets))
    ax.axis("off")
    gap = 0.03
    rows = []
    for i, t in enumerate(targets):
        y = len(targets) - 1 - i
        ax.text(-0.05, y + 0.5, t, ha="right", va="center", fontsize=9.5, color=INK2)
        for j, d in enumerate(devices):
            r = c[(c["target"] == t) & (c["device"] == d)]
            st = r["final"].iloc[0] if len(r) else "NOT_VALIDATED"
            fill = STATUS_FILL.get(st, STATUS_FILL["NOT_VALIDATED"])
            ax.add_patch(plt.Rectangle((j + gap, y + gap), 1 - 2 * gap, 1 - 2 * gap, color=fill, lw=0))
            ink = text_on(fill)
            auc = r["auroc"].iloc[0] if len(r) and "auroc" in r else np.nan
            dl = r["delta"].iloc[0] if len(r) and "delta" in r else np.nan
            ax.text(j + 0.5, y + (0.62 if auc == auc else 0.5), STATUS_LABEL.get(st, st),
                    ha="center", va="center", fontsize=9, fontweight="bold", color=ink)
            if auc == auc:
                sub = f"AUROC {auc:.2f}" + (f" · Δ {dl:+.2f}".replace("-", "−") if dl == dl else "")
                ax.text(j + 0.5, y + 0.3, sub, ha="center", va="center", fontsize=8, color=ink)
            rows.append({"target": t, "device": d, "status": st, "auroc": auc, "delta_vs_metadata": dl})
    for j, d in enumerate(devices):
        ax.text(j + 0.5, len(targets) + 0.34, d, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
        if d in (unfamiliar or []):
            ax.text(j + 0.5, len(targets) + 0.08, "unfamiliar camera (labels only score)",
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_title("What RetinaReach reports it can screen for, by camera", pad=40)
    ax.text(0, -0.35, "Δ = image AUROC − intake-form metadata AUROC on the same patients. Only "
                      "'Supported' targets are shown on the device.", fontsize=8, color=MUTED)
    save(fig, out, "fig2_capability", pd.DataFrame(rows))
    plt.close(fig)


def fig_calibration_size(plt, sweep: pd.DataFrame, source: List[str], device: str, out: str) -> None:
    if sweep.empty:
        return
    fig, axes = plt.subplots(len(source), 2, figsize=(10, 3.1 * len(source)), squeeze=False)
    rows = []
    priors = sorted(sweep.loc[sweep["kind"] == "subset", "prior_strength"].unique())[:2]
    for i, t in enumerate(source):
        for j, (m, title) in enumerate(((f"sens_{t}", "sensitivity at the shipped threshold"),
                                        (f"auroc_{t}", "AUROC"))):
            ax = axes[i][j]
            hairgrid(ax, "y")
            if m not in sweep:
                continue
            sub = sweep[sweep["kind"] == "subset"]
            for k, pr in enumerate(priors):
                g = sub[sub["prior_strength"] == pr].groupby("n")[m].agg(["mean", "std"]).reset_index()
                lab = "AdaBN only" if pr == 0 else f"with source prior (N0 = {pr:g})"
                ax.fill_between(g["n"], g["mean"] - g["std"], g["mean"] + g["std"],
                                color=SERIES[k], alpha=0.10, lw=0)
                ax.plot(g["n"], g["mean"], color=SERIES[k], lw=1.5,
                        label=lab if (i == 0 and j == 0) else None)
                for _, r in g.iterrows():
                    dot(ax, r["n"], r["mean"], SERIES[k])
                    rows.append({"target": t, "metric": m.split("_")[0], "prior": pr, "n": r["n"],
                                 "mean": r["mean"], "sd": r["std"]})
            xmax = sub["n"].max()
            ax.set_ylim(-0.03, 1.05)
            refs = []
            for kind, lab in (("trained", "no calibration"), ("full_pool", "full pool")):
                v = sweep.loc[sweep["kind"] == kind, m].mean()
                if v == v:
                    refs.append((v, lab))
                    rows.append({"target": t, "metric": m.split("_")[0], "prior": None, "n": kind,
                                 "mean": v, "sd": None})
            ref_lines(ax, refs)
            ax.set_xscale("log", base=2)
            ns = sorted(sub["n"].unique())
            ax.set_xticks(ns)
            ax.set_xticklabels([str(int(n)) for n in ns])
            ax.minorticks_off()
            ax.set_xlim(min(ns) / 1.3, xmax * 1.3)
            ax.set_title(f"{t} · {title}", fontsize=10.5)
            if i == len(source) - 1:
                ax.set_xlabel(f"unlabelled {device} captures used to calibrate")
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    fig.suptitle("How many unlabelled captures does a clinic need?", x=0.0, y=1.0, ha="left",
                 va="bottom", fontsize=14, fontweight="bold")
    fig.legend(loc="lower left", bbox_to_anchor=(0.0, 0.94), ncol=2)
    fig.text(0.0, -0.02, "Mean over seeds and random draws; band = SD. Sensitivity at the "
                         "threshold shipped with self-calibration.", fontsize=8, color=MUTED)
    save(fig, out, "fig3_calibration_size", pd.DataFrame(rows))
    plt.close(fig)


def fig_mechanism(plt, results: dict, abl: pd.DataFrame, out: str) -> None:
    dec = {s: r.get("camera_decodability") for s, r in results.items() if r.get("camera_decodability")}
    has_abl = not abl.empty
    ncols = 1 + int(has_abl)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 + 5.6 * has_abl, 3.8), squeeze=False)
    rows = []
    ax = axes[0][0]
    hairgrid(ax, "y")
    if dec:
        for s, d in sorted(dec.items()):
            ax.plot([0, 1], [d["trained"]["auroc"], d["calibrated"]["auroc"]], color=AXIS, lw=1, zorder=1)
            rows.append({"panel": "decodability", "seed": s, "trained": d["trained"]["auroc"],
                         "calibrated": d["calibrated"]["auroc"]})
        mt = np.mean([d["trained"]["auroc"] for d in dec.values()])
        mc = np.mean([d["calibrated"]["auroc"] for d in dec.values()])
        ax.plot([0, 1], [mt, mc], color=SERIES[0], lw=1.5, zorder=2)
        dot(ax, 0, mt, SERIES[0])
        dot(ax, 1, mc, SERIES[0])
        ax.text(1.06, mc, f"{mc:.2f}", va="center", fontsize=9, color=INK)
        ax.text(-0.06, mt, f"{mt:.2f}", va="center", ha="right", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, zorder=0)
    ax.text(1.0, 0.505, "chance", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["as trained", "self-calibrated"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("camera identity AUROC from the embedding")
    ax.set_title("Is the camera still in the representation?", fontsize=11)
    if has_abl:
        ax = axes[0][1]
        hairgrid(ax, "x")
        g = abl.groupby(["target", "device"])[["auroc_intact", "auroc_vessels", "auroc_control"]].mean()
        g = g.reset_index()
        labels = {"auroc_intact": "intact", "auroc_vessels": "vessels removed",
                  "auroc_control": "same area removed elsewhere"}
        for i, r in g.iterrows():
            y = len(g) - 1 - i
            for k, col in enumerate(("auroc_intact", "auroc_vessels", "auroc_control")):
                dot(ax, r[col], y + (0.18, 0.0, -0.18)[k], SERIES[k], labels[col] if i == 0 else None)
            rows.append({"panel": "vessel_ablation", "target": r["target"], "device": r["device"],
                         "intact": r["auroc_intact"], "vessels_removed": r["auroc_vessels"],
                         "control_removed": r["auroc_control"]})
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"{t}  ·  {d}" for t, d in zip(g["target"], g["device"])][::-1])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("AUROC (self-calibrated)")
        ax.set_title("Does the model read the vessels?", fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3)
    fig.tight_layout()
    save(fig, out, "fig4_mechanism", pd.DataFrame(rows))
    plt.close(fig)


def fig_unfamiliar(plt, lookup: pd.DataFrame, out: str) -> None:
    """Small multiples: one panel per (unfamiliar camera, target) the camera's own
    labels could judge or the phone ever showed -- coinciding lines never hide
    each other. The panel title carries what the labels support."""
    if lookup.empty:
        return
    panels = []
    for dv in dict.fromkeys(lookup["device"]):
        lk = lookup[lookup["device"] == dv]
        obs = lk.groupby("target")["observed"].agg(lambda v: v.mode().iloc[0])
        shown_ever = lk.groupby("target")["shown_rate"].max()
        panels += [(dv, t, obs[t]) for t in obs.index if obs[t] != "NOT_VALIDATED" or shown_ever[t] > 0]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels) + 0.8, 3.2), squeeze=False,
                             sharey=True)
    rows = []
    for j, (dv, t, ob) in enumerate(panels):
        ax = axes[0][j]
        hairgrid(ax, "y")
        g = lookup[(lookup["device"] == dv) & (lookup["target"] == t)].groupby("n")["shown_rate"].mean()
        unsafe = ob != "SUPPORTED"
        col = SERIES[1] if unsafe else SERIES[0]
        ax.plot(g.index, g.values, color=col, lw=1.5)
        for n, v in g.items():
            dot(ax, n, v, col)
            rows.append({"device": dv, "target": t, "n": n, "shown_rate": v, "observed": ob,
                         "unsafe_rate": v if unsafe else 0.0})
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(g.index))
        ax.set_xticklabels([str(int(n)) for n in g.index])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{t}\nits labels: {STATUS_LABEL.get(ob, ob)}", fontsize=10)
        ax.set_xlabel(f"unlabelled {dv} captures")
        if j == 0:
            ax.set_ylabel("share of draws shown on the phone")
    fig.suptitle("Unfamiliar camera: is what the phone shows backed by that camera's labels?",
                 x=0.0, y=1.0, ha="left", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.0, -0.06, "Orange = the labels do NOT support the target, so every draw that shows it "
                         "is an unsafe report; blue = supported, so a hidden draw is a missed one.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, out, "fig5_unfamiliar", pd.DataFrame(rows))
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RetinaReach poster figures.")
    p.add_argument("--dir", default="exp_retinareach")
    p.add_argument("--out", default=None, help="default: <dir>/figures")
    a = p.parse_args(argv)
    runs = load_runs(a.dir)
    if not runs["results"]:
        raise SystemExit(f"[fatal] no seed*/results.json under {a.dir}")
    first = next(iter(runs["results"].values()))["args"]
    source, probes = first["source_targets"], first["probe_targets"]
    S, T = first["source_name"], first["target_name"]
    unf = [spec[2] if len(spec) == 3 else spec[1] for spec in first.get("unfamiliar") or []]
    out = a.out or os.path.join(a.dir, "figures")
    plt = setup()
    print(f"figures -> {out}")
    cap_all = pd.concat([runs["cap"], runs["unf"]], ignore_index=True) if len(runs["unf"]) else runs["cap"]
    fig_threshold_transfer(plt, cap_all, source, [S, T] + unf, first["target_sens"],
                           first["sens_tolerance"], out)
    final_p = os.path.join(a.dir, "capability_final.csv")
    if os.path.exists(final_p):
        final = pd.read_csv(final_p)
        summ_p = os.path.join(a.dir, "summary.json")
        if os.path.exists(summ_p):
            with open(summ_p) as fh:
                uc = json.load(fh).get("unfamiliar_capability") or []
            if uc:
                final = pd.concat([final, pd.DataFrame(uc)], ignore_index=True)
        fig_capability(plt, final, source + probes, [S, T] + unf, out, unfamiliar=unf)
    else:
        print("  (no capability_final.csv: run summarize_retinareach.py first for fig2)")
    fig_calibration_size(plt, runs["sweep"], source, T, out)
    fig_mechanism(plt, runs["results"], runs["ablation"], out)
    fig_unfamiliar(plt, runs["lookup"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
