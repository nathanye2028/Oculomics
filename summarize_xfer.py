#!/usr/bin/env python3
"""
summarize_xfer.py
=================
Paired treatment-vs-control comparison for the **BRSET -> mBRSET transfer**
runs written by ``train_mbrset.py --external-test-root``. The two conditions are
named with ``--treatment`` / ``--control`` (default gcg vs nogcg; run_kd_xfer.sh
uses kd vs ctrl) and files must be named ``<condition>_seed<n>.json`` — the seed
is what pairs them.

Which number is the result
--------------------------
Read **external AUROC** first. That is the model's performance on unconstrained
smartphone captures, which is the thing the project claims to improve and the
only number that corresponds to deploying anything.

The **domain gap** (external minus in-domain) is descriptive, not the endpoint,
and it can move for the wrong reason: a variant that improves in-domain more
than out-of-domain *widens* the gap while being strictly better on both. Never
report a narrowed gap as a win without showing external AUROC went up too.

AUROC is rank-based and largely prevalence-invariant, so a referable-rate
difference between BRSET and mBRSET (roughly 3x for dr_referable) does not by
itself move it. Accuracy, macro-F1 and kappa DO shift with prevalence and are
reported here only for completeness -- do not quote their deltas as evidence
about image quality. The prevalences printed at the bottom come from the runs'
own ``test_pos`` / ``external_pos`` counts, not from a constant.

Usage
-----
    python summarize_xfer.py --dir exp_xfer                        # gcg vs nogcg
    python summarize_xfer.py --dir exp_kd --treatment kd --control ctrl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import paired_stats          # noqa: E402  (same statistics as every other experiment)

METRICS = ["auroc", "acc", "f1", "kappa"]


def load(d):
    """{condition: {seed: record}} from <cond>_seed<n>.json."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
        base = os.path.basename(p)[: -len(".json")]
        cond, _, seed = base.rpartition("_seed")
        try:
            seed = int(seed)
        except ValueError:
            continue
        with open(p) as f:
            r = json.load(f)
        if not r.get("external"):
            print(f"[skip] {base}: no external test block", file=sys.stderr)
            continue
        out.setdefault(cond, {})[seed] = r
    return out


def agg(vals):
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _prevalences(runs, conds, seeds):
    """(in-domain test prevalence, external prevalence) from the first run that
    recorded ``test_pos`` / ``external_pos`` with ``n_test`` / ``n_external``;
    None if no run has them (binary tasks only -- the trainer writes null
    otherwise)."""
    for c in conds:
        for s in seeds:
            r = runs[c][s]
            tp, ep = r.get("test_pos"), r.get("external_pos")
            nt, ne = r.get("n_test") or 0, r.get("n_external") or 0
            if tp is not None and ep is not None and nt > 0 and ne > 0:
                return tp / nt, ep / ne
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paired <treatment>-vs-<control> comparison of BRSET -> mBRSET transfer "
                    "runs. Result files in --dir must be named <condition>_seed<n>.json "
                    "(as written by run_xfer.sh / run_kd_xfer.sh); runs are paired by seed.")
    p.add_argument("--dir", default="exp_xfer",
                   help="Directory of <condition>_seed<n>.json files from train_mbrset.py.")
    p.add_argument("--treatment", default="gcg",
                   help="Condition name (the <condition> file prefix) of the treatment arm.")
    p.add_argument("--control", default="nogcg",
                   help="Condition name of the control arm; paired per seed with --treatment.")
    args = p.parse_args()

    runs = load(args.dir)
    if args.treatment not in runs or args.control not in runs:
        print(f"[fatal] need both {args.treatment!r} and {args.control!r} in {args.dir}; "
              f"found {sorted(runs)}", file=sys.stderr)
        return 2

    seeds = sorted(set(runs[args.treatment]) & set(runs[args.control]))
    if not seeds:
        print("[fatal] no seed appears in both conditions; nothing is paired.", file=sys.stderr)
        return 2
    unpaired = (set(runs[args.treatment]) ^ set(runs[args.control]))
    r_first = runs[args.control][seeds[0]]
    ext_name = r_first.get("external_dataset") or "external"
    src = r_first.get("train_dataset") or "source"
    L = [f"\n{'='*76}",
         f"{src} -> {ext_name} TRANSFER ({r_first.get('task', '?')}): {args.treatment} vs {args.control}",
         f"  paired seeds={seeds}" + (f"   (dropped unpaired: {sorted(unpaired)})" if unpaired else ""),
         f"{'='*76}"]

    # ---- per-condition table ------------------------------------------------ #
    L.append(f"\n{'condition':<10}{'in-domain AUROC':<20}{f'{ext_name} AUROC':<20}{'gap':<16}{'n(ext)':>7}")
    per = {}
    for cond in (args.control, args.treatment):
        ind = [runs[cond][s]["test"]["auroc"] for s in seeds]
        ext = [runs[cond][s]["external"]["auroc"] for s in seeds]
        gap = [e - i for e, i in zip(ext, ind)]
        per[cond] = {"in": dict(zip(seeds, ind)), "ext": dict(zip(seeds, ext)),
                     "gap": dict(zip(seeds, gap))}
        n_ext = runs[cond][seeds[0]].get("n_external", 0)
        im, isd = agg(ind); em, esd = agg(ext); gm, gsd = agg(gap)
        L.append(f"{cond:<10}{f'{im:.4f}+/-{isd:.4f}':<20}{f'{em:.4f}+/-{esd:.4f}':<20}"
                 f"{f'{gm:+.4f}':<16}{n_ext:>7}")

    # ---- the comparison ----------------------------------------------------- #
    L += ["", "-" * 76,
          f"PAIRED {args.treatment} - {args.control} (same seed, same split):"]

    ext_stats = paired_stats(per[args.treatment]["ext"], per[args.control]["ext"])
    gap_stats = paired_stats(per[args.treatment]["gap"], per[args.control]["gap"])

    for label, st, note in (
        (f"{ext_name} AUROC", ext_stats, "<- THE RESULT: performance on the external set"),
        ("domain gap", gap_stats, "   (descriptive; only meaningful if the row above improved)"),
    ):
        lo, hi = st["ci95"]
        if st["n"] < 2:
            L.append(f"  {label:<16} delta={st['mean_delta']:+.4f}  no CI (n={st['n']})  {note}")
            continue
        verdict = "SIGNIFICANT" if st["significant"] else "inconclusive (CI includes 0)"
        L.append(f"  {label:<16} delta={st['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
                 f"{st['n_positive']}/{st['n']} seeds+  {verdict}")
        L.append(f"  {'':<16} {note}")

    # ---- test-time BN adaptation (present when runs used --bn-adapt) ------- #
    # Say WHICH runs lack it rather than dropping the table silently: a missing
    # block usually means one arm was run without --bn-adapt, or it hit a
    # BN-free backbone (train_mbrset.py records external_bnadapt=null then).
    no_bn = [f"{c}_seed{s}" for c in (args.control, args.treatment) for s in seeds
             if not runs[c][s].get("external_bnadapt")]
    have_bn = not no_bn
    bn_stats = None
    if no_bn:
        print(f"[skip] BN-adaptation table: no external_bnadapt in {no_bn}", file=sys.stderr)
    if have_bn:
        L += ["", "-" * 76,
              f"TEST-TIME BN ADAPTATION (label-free AdaBN on {ext_name} images; report separately):",
              f"  {'condition':<10}{'zero-shot':<18}{'BN-adapted':<18}{'adapt effect (paired)':<24}"]
        per_bn = {}
        for cond in (args.control, args.treatment):
            zs = [runs[cond][s]["external"]["auroc"] for s in seeds]
            ad = [runs[cond][s]["external_bnadapt"]["auroc"] for s in seeds]
            per_bn[cond] = dict(zip(seeds, ad))
            eff = paired_stats(dict(zip(seeds, ad)), dict(zip(seeds, zs)))
            zm, zsd = agg(zs); am, asd = agg(ad)
            lo, hi = eff["ci95"]
            L.append(f"  {cond:<10}{f'{zm:.4f}+/-{zsd:.4f}':<18}{f'{am:.4f}+/-{asd:.4f}':<18}"
                     f"{eff['mean_delta']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                     f"{'SIG' if eff['significant'] else 'n.s.'}")
        bn_stats = paired_stats(per_bn[args.treatment], per_bn[args.control])
        lo, hi = bn_stats["ci95"]
        L.append(f"  {args.treatment}-{args.control} on BN-adapted AUROC: "
                 f"delta={bn_stats['mean_delta']:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
                 f"{bn_stats['n_positive']}/{bn_stats['n']} seeds+  "
                 f"{'SIGNIFICANT' if bn_stats['significant'] else 'inconclusive'}")

    # ---- secondary metrics, flagged as prevalence-sensitive ----------------- #
    L += ["", "-" * 76,
          f"Secondary metrics on {ext_name} (prevalence-sensitive -- context only, not evidence):",
          f"  {'metric':<10}{args.control:<22}{args.treatment:<22}{'delta':<12}"]
    for m in METRICS[1:]:
        cv = [runs[args.control][s]["external"][m] for s in seeds]
        tv = [runs[args.treatment][s]["external"][m] for s in seeds]
        cm, csd = agg(cv); tm, tsd = agg(tv)
        L.append(f"  {m:<10}{f'{cm:.4f}+/-{csd:.4f}':<22}{f'{tm:.4f}+/-{tsd:.4f}':<22}{tm-cm:+.4f}")

    # Prevalence from the runs themselves (train_mbrset.py records positive
    # counts for binary tasks); older JSONs lack the counts, so the sentence is
    # omitted rather than filled with a number from some other dataset/task.
    prev = _prevalences(runs, (args.control, args.treatment), seeds)
    if prev is not None:
        L += ["", f"NB: the in-domain test split is {prev[0]:.1%} positive and {ext_name} {prev[1]:.1%}.",
              "    AUROC is rank-based and does not move on prevalence alone;",
              "    accuracy/F1/kappa do. Quote the AUROC delta."]
    else:
        L += ["", "NB: AUROC is rank-based and does not move on prevalence alone;",
              "    accuracy/F1/kappa do. Quote the AUROC delta."]

    out = "\n".join(L)
    print(out)
    with open(os.path.join(args.dir, "summary.md"), "w") as f:
        f.write("```\n" + out + "\n```\n")
    with open(os.path.join(args.dir, "summary.json"), "w") as f:
        json.dump({"seeds": seeds, "per_condition": per,
                   "paired_external_auroc": {**ext_stats, "ci95": list(ext_stats["ci95"])},
                   "paired_domain_gap": {**gap_stats, "ci95": list(gap_stats["ci95"])},
                   "paired_external_bnadapt": ({**bn_stats, "ci95": list(bn_stats["ci95"])}
                                               if bn_stats else None)},
                  f, indent=2)
    print(f"\n[info] wrote {args.dir}/summary.md and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
