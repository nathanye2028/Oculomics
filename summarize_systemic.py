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
2. **Does DR pre-training transfer?** ``--treatment`` minus ``--control`` on test
   AUROC (default drinit vs ctrl). Optional -- with ``--control`` alone the
   script only answers question 1.

    python summarize_systemic.py --dir exp_systemic --treatment drinit --control ctrl
    python summarize_systemic.py --dir exp_systemic --tasks hypertension --control ctrl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import paired_stats          # noqa: E402  (same statistics as every other experiment)


def load_task(d: str) -> Dict[str, Dict[int, dict]]:
    """{condition: {seed: record}} from <d>/<cond>_seed<n>.json."""
    out: Dict[str, Dict[int, dict]] = {}
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
        base = os.path.basename(p)[: -len(".json")]
        cond, _, seed = base.rpartition("_seed")
        try:
            seed = int(seed)
        except ValueError:
            continue
        with open(p) as f:
            out.setdefault(cond, {})[seed] = json.load(f)
    return out


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


def summarize(root: str, tasks: Optional[List[str]], control: str,
              treatment: Optional[str] = None) -> dict:
    tasks = tasks or discover_tasks(root)
    L = [f"\n{'='*78}", f"SYSTEMIC (OCULOMICS) TARGETS ON mBRSET  --  {root}",
         f"  control={control}" + (f"  treatment={treatment}" if treatment else "") +
         f"  tasks={tasks}", f"{'='*78}"]
    report: Dict[str, dict] = {}
    overview = []
    for task in tasks:
        runs = load_task(os.path.join(root, task))
        if control not in runs:
            L.append(f"\n[{task}] no {control!r} runs found; skipped")
            continue
        conds = [control] + ([treatment] if treatment and treatment in runs else [])
        if treatment and treatment not in runs:
            L.append(f"\n[{task}] no {treatment!r} runs; reporting {control!r} only")
        seeds = sorted(set.intersection(*(set(runs[c]) for c in conds)))
        rec = {"seeds": seeds, "conditions": {}, "paired_image_minus_covariate": {},
               "paired_treatment_minus_control": None}
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
        raise SystemExit(f"[fatal] no <condition>_seed<n>.json under {root}/<task>/ for {tasks}")

    L += ["", "-" * 78, "OVERVIEW (best condition per task, mean over seeds):",
          f"  {'task':<24}{'cond':<10}{'image AUROC':>12}{'covariate':>10}{'prev':>8}"]
    for task, c, im, cm, prev in overview:
        L.append(f"  {task:<24}{c:<10}{im:>12.4f}{(f'{cm:.4f}' if cm == cm else '-'):>10}"
                 f"{(f'{prev:.1%}' if prev is not None else '-'):>8}")
    L += ["", "NB: in-domain, patient-grouped, smartphone-only. Self-reported / chart",
          "    comorbidities in a diabetic cohort; AUROC is rank-based, so the prevalence",
          "    column is context, not a correction. Quote image AND covariate AUROC together."]
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
    args = p.parse_args()
    if not os.path.isdir(args.dir):
        print(f"[fatal] {args.dir} is not a directory", file=sys.stderr)
        return 2
    summarize(args.dir, args.tasks, args.control, args.treatment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
