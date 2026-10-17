#!/usr/bin/env python3
"""
summarize_systemic.py
=====================
Per-task summary of the systemic (oculomics) runs written by ``run_systemic.sh``
(``<dir>/<task>/<condition>_seed<n>.json`` from ``train_mbrset.py``). In-domain
runs: there is no external block, so this is NOT summarize_xfer.py.

Two paired questions per task, both paired by seed (same patient split):

1. **Is there retinal signal beyond age?** image AUROC minus the age+sex
   logistic baseline recorded in the same run (``covariate_baseline``). A CI
   that excludes 0 on the positive side is the claim; a CI that includes 0 means
   the model may have learned the patient's age from the retina.
   With ``incremental`` in the runs (``--covariate-baseline`` on a binary task),
   the sharper version: covariates + image score vs covariates alone, both fit
   on the val patients and scored on the test patients, overall and per subset
   (e.g. non-diabetic DR-grade-0), paired by seed.
2. **Does DR pre-training transfer?** ``--treatment`` minus ``--control`` on test
   AUROC (default drinit vs ctrl). Optional -- with ``--control`` alone the
   script only answers question 1.

Fold runs (``train_mbrset.py --fold K``, ``<cond>_fold<K>_seed<n>.json`` plus their
``_test.csv`` rows) are pooled OUT OF FOLD instead: once every fold of a seed is
present, each labelled patient has exactly one score from a model that never saw
them, and question 1's sharp version is answered on all of them at once, overall
and per before-lesions subset, with a patient-cluster bootstrap CI. Several
complete seeds are then paired like any other runs.

    python summarize_systemic.py --dir exp_systemic --treatment drinit --control ctrl
    python summarize_systemic.py --dir exp_systemic --tasks hypertension --control ctrl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import paired_stats          # noqa: E402  (same statistics as every other experiment)
from covariate_baseline import incremental_block  # noqa: E402

_FOLD_RUN = re.compile(r"^(?P<cond>.+)_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.json$")


def load_task(d: str) -> Dict[str, Dict[int, dict]]:
    """{condition: {seed: record}} from <d>/<cond>_seed<n>.json (fold runs excluded:
    see :func:`load_folds`)."""
    out: Dict[str, Dict[int, dict]] = {}
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
        if _FOLD_RUN.match(os.path.basename(p)):
            continue
        base = os.path.basename(p)[: -len(".json")]
        cond, _, seed = base.rpartition("_seed")
        try:
            seed = int(seed)
        except ValueError:
            continue
        with open(p) as f:
            out.setdefault(cond, {})[seed] = json.load(f)
    return out


def load_folds(d: str) -> Dict[str, Dict[int, Dict[int, dict]]]:
    """{condition: {seed: {fold: record}}} from <d>/<cond>_fold<k>_seed<n>.json; each
    record gains ``_path`` so its test rows can be found next to it."""
    out: Dict[str, Dict[int, Dict[int, dict]]] = {}
    for p in sorted(glob.glob(os.path.join(d, "*_fold*_seed*.json"))):
        m = _FOLD_RUN.match(os.path.basename(p))
        if not m:
            continue
        with open(p) as f:
            rec = json.load(f)
        rec["_path"] = p
        out.setdefault(m["cond"], {}).setdefault(int(m["seed"]), {})[int(m["fold"])] = rec
    return out


def _fold_rows(rec: dict) -> Optional[pd.DataFrame]:
    """A fold run's test rows: the recorded predictions_csv, else the ``_test.csv``
    beside its JSON (results copied off the machine that wrote them)."""
    for p in (rec.get("predictions_csv"), os.path.splitext(rec["_path"])[0] + "_test.csv"):
        if p and os.path.isfile(p):
            return pd.read_csv(p)
    return None


def pooled_out_of_fold(by_seed: Dict[int, Dict[int, dict]], n_boot: int = 2000) -> dict:
    """Per seed whose every fold is present: the test rows of all folds pooled,
    then covariates vs covariates + image (each fold's two models were fit on that
    fold's val patients) on every patient at once, overall and per ``in_<subset>``
    column. With two or more complete seeds, the pooled numbers are also paired
    across seeds."""
    seeds, missing = {}, {}
    for s, folds in sorted(by_seed.items()):
        n = next(iter(folds.values())).get("n_folds") or len(folds)
        if sorted(folds) != list(range(n)):
            missing[s] = sorted(folds)
            continue
        frames = [_fold_rows(folds[k]) for k in range(n)]
        if any(f is None for f in frames):
            missing[s] = [k for k in range(n) if frames[k] is None]
            continue
        rows = pd.concat(frames, ignore_index=True)
        if not {"p_covariates", "p_combined"} <= set(rows.columns):
            missing[s] = "no covariate scores in the test rows (run with --covariate-baseline)"
            continue
        if rows.groupby(rows["patient"].astype(str))["fold"].nunique().max() > 1:
            raise SystemExit(f"[fatal] seed {s}: a patient is tested in two folds; "
                             f"the folds are not one partition")
        ok = rows[["label", "p_image", "p_covariates", "p_combined"]].notna().all(1).to_numpy()
        blocks = {"all": ok}
        for c in sorted(c for c in rows.columns if c.startswith("in_")):
            blocks[c[3:]] = ok & rows[c].astype(bool).to_numpy()
        y, g = rows["label"].to_numpy(), rows["patient"].astype(str).to_numpy()
        seeds[s] = {"n_folds": n, "fold_seed": folds[0].get("fold_seed"),
                    "patients": int(rows.loc[ok, "patient"].nunique()),
                    "blocks": {name: incremental_block(np.nan_to_num(y), g,
                                                       rows["p_covariates"].to_numpy(),
                                                       rows["p_combined"].to_numpy(),
                                                       rows["p_image"].to_numpy(), m,
                                                       n_boot=n_boot, seed=s)
                               for name, m in blocks.items()}}
    paired = {}
    if len(seeds) >= 2:
        names = set.intersection(*(set(v["blocks"]) for v in seeds.values()))
        for name in sorted(names):
            comb = {s: v["blocks"][name].get("auroc_combined") for s, v in seeds.items()}
            cov = {s: v["blocks"][name].get("auroc_covariates") for s, v in seeds.items()}
            if all(x is not None for x in list(comb.values()) + list(cov.values())):
                paired[name] = _ci(paired_stats(comb, cov))
    return {"seeds": seeds, "incomplete": missing, "paired_over_seeds": paired}


def _pooled_lines(cond: str, pooled: dict, feats: str) -> List[str]:
    L = []
    for s, v in pooled["seeds"].items():
        L.append(f"  POOLED OUT OF FOLD  {cond} seed {s}: {v['n_folds']} folds (fold seed "
                 f"{v['fold_seed']}), {v['patients']} patients, each tested once")
        for name, b in v["blocks"].items():
            if "delta" not in b:
                L.append(f"    [{name}] {b.get('reason', 'not computed')} (n={b['n']})")
                continue
            lo, hi = b.get("delta_ci", [float("nan")] * 2)
            L.append(f"    [{name:<20}] {feats}={b['auroc_covariates']:.4f}  +image={b['auroc_combined']:.4f}  "
                     f"delta={b['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]  image alone={b['auroc_image']:.4f}  "
                     f"({b['pos_patients']} positive patients)")
    for s, why in pooled["incomplete"].items():
        L.append(f"  [{cond} seed {s}] not pooled: {why if isinstance(why, str) else f'folds present {why}'}")
    for name, st in pooled["paired_over_seeds"].items():
        L.append(_fmt_stats(st, f"{cond}: pooled +image vs {feats} [{name}]") + "   <- over seeds")
    return L


def discover_tasks(root: str) -> List[str]:
    return sorted(t for t in os.listdir(root)
                  if os.path.isdir(os.path.join(root, t))
                  and glob.glob(os.path.join(root, t, "*_seed*.json")))


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _ci(st):
    lo, hi = st["ci95"]
    return {**st, "ci95": [lo, hi]}


def _fmt_stats(st, label):
    lo, hi = st["ci95"]
    if st["n"] < 2:
        return f"  {label:<34} delta={st['mean_delta']:+.4f}  no CI (n={st['n']})"
    verdict = "SIGNIFICANT" if st["significant"] else "inconclusive (CI includes 0)"
    return (f"  {label:<34} delta={st['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
            f"{st['n_positive']}/{st['n']} seeds+  {verdict}")


def incremental_by_seed(by_seed: Dict[int, dict]) -> Dict[str, dict]:
    """{block: paired_stats(covariates+image, covariates)} over seeds, for the
    ``test`` block and every subset recorded in the runs' ``incremental``."""
    blocks: Dict[str, Dict[str, Dict[int, float]]] = {}
    for s, r in by_seed.items():
        inc = r.get("incremental") or {}
        named = [("all", inc.get("test"))] + list((inc.get("subsets") or {}).items())
        for name, b in named:
            if b and "delta" in b:
                d = blocks.setdefault(name, {"comb": {}, "cov": {}})
                d["comb"][s], d["cov"][s] = b["auroc_combined"], b["auroc_covariates"]
    return {name: _ci(paired_stats(d["comb"], d["cov"])) for name, d in blocks.items()}


def summarize(root: str, tasks: Optional[List[str]], control: str,
              treatment: Optional[str] = None, n_boot: int = 2000) -> dict:
    tasks = tasks or discover_tasks(root)
    L = [f"\n{'='*78}", f"SYSTEMIC (OCULOMICS) TARGETS  --  {root}",
         f"  control={control}" + (f"  treatment={treatment}" if treatment else "") +
         f"  tasks={tasks}", f"{'='*78}"]
    report: Dict[str, dict] = {}
    overview = []
    for task in tasks:
        runs = load_task(os.path.join(root, task))
        folds = load_folds(os.path.join(root, task))
        if control not in runs and control not in folds:
            L.append(f"\n[{task}] no {control!r} runs found; skipped")
            continue
        pooled = None
        if control in folds:
            pooled = pooled_out_of_fold(folds[control], n_boot=n_boot)
            any_rec = next(iter(next(iter(folds[control].values())).values()))
            feats = "+".join((any_rec.get("incremental") or {}).get("features") or ["covariates"])
            L += ["", f"--- {task} ---  fold runs"] + _pooled_lines(control, pooled, feats)
        if control not in runs:
            report[task] = {"pooled_out_of_fold": pooled}
            continue
        conds = [control] + ([treatment] if treatment and treatment in runs else [])
        if treatment and treatment not in runs:
            L.append(f"\n[{task}] no {treatment!r} runs; reporting {control!r} only")
        seeds = sorted(set.intersection(*(set(runs[c]) for c in conds)))
        rec = {"seeds": seeds, "conditions": {}, "paired_image_minus_covariate": {},
               "paired_incremental": {}, "paired_treatment_minus_control": None,
               "pooled_out_of_fold": pooled}
        L += ["", f"--- {task} ---  paired seeds={seeds}",
              f"  {'condition':<10}{'image AUROC':<20}{'covariate AUROC':<20}{'image-covariate':<18}"
              f"{'prev':>7}{'n_test':>8}"]
        for c in conds:
            img = {s: runs[c][s]["test"]["auroc"] for s in seeds}
            cbs = {s: (runs[c][s].get("covariate_baseline") or {}).get("auroc") for s in seeds}
            cbs = {s: v for s, v in cbs.items() if v is not None and v == v}
            r0 = runs[c][seeds[0]]
            prev = (r0["test_pos"] / r0["n_test"]) if r0.get("test_pos") is not None and r0.get("n_test") else None
            rec["conditions"][c] = {"test_auroc": img, "covariate_auroc": cbs,
                                    "prevalence": prev, "n_test": r0.get("n_test"),
                                    "covariate_features": (r0.get("covariate_baseline") or {}).get("features"),
                                    "init_from": (r0.get("init_from") or {}).get("path")}
            im, isd = agg(img.values()); cm, csd = agg(cbs.values())
            st = paired_stats(img, cbs) if cbs else None
            rec["paired_image_minus_covariate"][c] = _ci(st) if st else None
            delta = f"{st['mean_delta']:+.4f}" if st else "-"
            L.append(f"  {c:<10}{f'{im:.4f}+/-{isd:.4f}':<20}"
                     f"{(f'{cm:.4f}+/-{csd:.4f}' if cbs else 'not recorded'):<20}{delta:<18}"
                     f"{(f'{prev:.1%}' if prev is not None else '-'):>7}{(r0.get('n_test') or 0):>8}")
        feats = {c: "+".join(rec["conditions"][c]["covariate_features"] or ["?"]) for c in conds}
        L += ["", "  PAIRED (same seed, same split):"]
        for c in conds:
            st = rec["paired_image_minus_covariate"][c]
            if st:
                L.append(_fmt_stats(st, f"{c}: image minus {feats[c]} baseline")
                         + "   <- retinal signal beyond age?")
            else:
                L.append(f"  {c}: no covariate baseline recorded (run with --covariate-baseline)")
            inc = incremental_by_seed({s: runs[c][s] for s in seeds})
            rec["paired_incremental"][c] = inc
            for name, st in inc.items():
                L.append(_fmt_stats(st, f"{c}: +image vs {feats[c]} [{name}]")
                         + "   <- does the photo add?")
        if treatment in conds:
            st = paired_stats(rec["conditions"][treatment]["test_auroc"],
                              rec["conditions"][control]["test_auroc"])
            rec["paired_treatment_minus_control"] = _ci(st)
            L.append(_fmt_stats(st, f"{treatment} minus {control} (image AUROC)")
                     + "   <- does DR pre-training transfer?")
        report[task] = rec
        best = max(conds, key=lambda c: agg(rec["conditions"][c]["test_auroc"].values())[0])
        overview.append((task, best, agg(rec["conditions"][best]["test_auroc"].values())[0],
                         agg(rec["conditions"][best]["covariate_auroc"].values())[0],
                         rec["conditions"][best]["prevalence"]))

    if not report:
        raise SystemExit(f"[fatal] no <condition>_seed<n>.json or <condition>_fold<k>_seed<n>.json "
                         f"under {root}/<task>/ for {tasks}")

    L += ["", "-" * 78, "OVERVIEW (best condition per task, mean over seeds):",
          f"  {'task':<24}{'cond':<10}{'image AUROC':>12}{'covariate':>10}{'prev':>8}"]
    for task, c, im, cm, prev in overview:
        L.append(f"  {task:<24}{c:<10}{im:>12.4f}{(f'{cm:.4f}' if cm == cm else '-'):>10}"
                 f"{(f'{prev:.1%}' if prev is not None else '-'):>8}")
    L += ["", "NB: in-domain, patient-grouped. Self-reported / chart comorbidities; AUROC is",
          "    rank-based, so the prevalence column is context, not a correction. Quote image",
          "    AND covariate AUROC together; '+image vs covariates' is the oculomics claim."]
    out = "\n".join(L)
    print(out)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("```\n" + out + "\n```\n")
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump({"control": control, "treatment": treatment, "tasks": report}, f, indent=2)
    print(f"\n[info] wrote {root}/summary.md and summary.json")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise run_systemic.sh results per task.")
    p.add_argument("--dir", default="exp_systemic", help="Root holding <task>/<cond>_seed<n>.json.")
    p.add_argument("--tasks", nargs="+", default=None, help="Tasks (default: every subdir with runs).")
    p.add_argument("--control", default="ctrl", help="Condition prefix of the control arm.")
    p.add_argument("--treatment", default=None,
                   help="Condition prefix of the treatment arm (e.g. drinit); omit for control-only.")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="Patient-cluster bootstrap resamples for the pooled out-of-fold CIs.")
    args = p.parse_args()
    if not os.path.isdir(args.dir):
        print(f"[fatal] {args.dir} is not a directory", file=sys.stderr)
        return 2
    summarize(args.dir, args.tasks, args.control, args.treatment, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
