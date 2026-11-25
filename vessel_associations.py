"""
vessel_associations.py
======================
Recommendation 2: measure the vessels, test ONE pre-registered feature per disease,
and put the retinal age gap on the same footing.

Nothing is trained on the rare label. For each disease in the committed
pre-registration (``vessel_prereg.json``):

    disease ~ feature(per SD) + covariates      (brset_comorbidities.logit_or)

on patients whose AutoMorph-gradable images were averaged over both eyes; the
odds ratio per standard deviation of the feature, its Wald 95 % CI and p, and a
Holm adjustment across the primary tests. Every primary feature is also tested
against the negative control (BRSET hypothyroidism), where nothing is expected.
With ``--age-gap`` the same model is fit with the cross-fitted, bias-corrected
retinal age gap (``gap_corrected`` of the 5-fold clock: every patient's predicted
age from a model that never saw them) for every pre-registered condition plus the
control: the forest plot. ``--exploratory`` adds every feature x disease pair,
BH-corrected and labelled exploratory.

Inputs:
  * ``--automorph`` the automorph_stage.py folder after AutoMorph ran:
    ``name_map.csv``, ``Results/M1/results_ensemble.csv`` (quality),
    ``Results/M3/Disc_Features.csv`` / ``Macular_Features.csv`` (measurements);
  * ``--dataset/--root`` the label table (train_mbrset.load_table, with the parsed
    comorbidities for BRSET);
  * ``--prereg`` the committed pre-registration; a draft or an uncommitted copy is
    refused unless ``--allow-draft``, and then every output says DRAFT.

    python vessel_associations.py --automorph automorph/brset_canon --dataset brset --root <BRSET> \\
        --prereg vessel_prereg.json --age-gap <pooled 5-fold clock predictions> --out exp_vessels/brset

Outputs in ``--out``: ``results.csv`` (one row per test), ``summary.md``,
``forest.csv`` + ``forest.png`` (matplotlib; without it, redraw on the Mac with
``python3 vessel_associations.py --plot-only <out>/forest.csv``) and
``patient_measurements.csv`` (patient-level, stays on the box: it holds patient ids).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

M1_CSV = os.path.join("Results", "M1", "results_ensemble.csv")
M3_CSVS = ((os.path.join("Results", "M3", "Disc_Features.csv"), "disc"),
           (os.path.join("Results", "M3", "Macular_Features.csv"), "macula"))
NON_FEATURES = {"Name", "stem", "centred"}


# --------------------------------------------------------------------------- #
# AutoMorph outputs
# --------------------------------------------------------------------------- #
def _stem(name) -> str:
    return os.path.basename(str(name)).split(".")[0]


def read_quality(stage: str):
    """Per staged image: rejected or not, by AutoMorph's own rule (M1
    merge_quality_assessment.py): Prediction 2, or Prediction 1 with softmax_bad >= 0.25."""
    import pandas as pd
    path = os.path.join(stage, M1_CSV)
    if not os.path.isfile(path):
        raise SystemExit(f"[fatal] no AutoMorph quality file at {path}")
    q = pd.read_csv(path)
    need = {"Name", "Prediction", "softmax_bad"}
    if not need <= set(q.columns):
        raise SystemExit(f"[fatal] {path} lacks {sorted(need - set(q.columns))}; columns: {list(q.columns)}")
    pred = pd.to_numeric(q["Prediction"], errors="coerce")
    bad = pd.to_numeric(q["softmax_bad"], errors="coerce")
    q["stem"] = q["Name"].map(_stem)
    q["rejected"] = (pred == 2) | ((pred == 1) & (bad >= 0.25)) | pred.isna()
    return q[["stem", "rejected"]].drop_duplicates("stem")


def read_measurements(stage: str):
    """Disc- and macula-centred feature tables stacked (an image is in one of them);
    AutoMorph's -1 / empty 'invalid' values become NaN."""
    import pandas as pd
    frames = []
    for rel, centred in M3_CSVS:
        p = os.path.join(stage, rel)
        if os.path.isfile(p):
            f = pd.read_csv(p)
            f["stem"], f["centred"] = f["Name"].map(_stem), centred
            frames.append(f)
    if not frames:
        raise SystemExit(f"[fatal] no {' or '.join(r for r, _ in M3_CSVS)} under {stage}")
    m = pd.concat(frames, ignore_index=True, sort=False)
    dup = m["stem"].duplicated(keep="first")
    if dup.any():
        print(f"[warn] {int(dup.sum())} images measured as both disc- and macula-centred; keeping the first")
        m = m.loc[~dup]
    for c in m.columns:
        if c not in NON_FEATURES:
            v = pd.to_numeric(m[c], errors="coerce")
            m[c] = v.where(v != -1)
    return m.reset_index(drop=True)


def feature_columns(m) -> List[str]:
    return [c for c in m.columns if c not in NON_FEATURES and m[c].notna().any()
            and m[c].dtype.kind in "fi"]


def _numeric(s):
    """A covariate / label column as numbers; yes/no text through dataset._binary_flag."""
    import pandas as pd
    from dataset import _binary_flag
    v = pd.to_numeric(s, errors="coerce")
    nn = s.notna()
    # text only when most RECORDED values fail to parse: a sparse numeric column
    # (many NaN) must not go through the yes/no parser, which keeps only 0 and 1
    if nn.any() and v[nn].isna().mean() > 0.5:
        v = s.map(_binary_flag).astype(float)
    return v


def patient_measurements(stage: str, table, outcomes: Sequence[str], covariates: Sequence[str]):
    """Per patient: every measurement averaged over the gradable images of each eye,
    then over the eyes; the outcomes (patient-level) and covariates beside them.
    Returns (frame, counts)."""
    import pandas as pd
    from dataset import LABEL_REGISTRY
    nm = pd.read_csv(os.path.join(stage, "name_map.csv"))
    nm["stem"] = nm["automorph_name"].map(_stem)
    q, m = read_quality(stage), read_measurements(stage)
    feats = feature_columns(m)
    img = nm[["stem", "file"]].merge(q, on="stem", how="left").merge(m, on="stem", how="left")
    counts = {"staged": int(len(nm)), "graded": int(img["rejected"].notna().sum()),
              "rejected": int(img["rejected"].fillna(False).astype(bool).sum())}
    img = img[img["rejected"].eq(False)]
    counts["measured_gradable"] = int(img[feats].notna().any(axis=1).sum())
    t = table.assign(file=table["file"].astype(str)).drop_duplicates("file")
    keep = ["file", "patient"] + [c for c in ("laterality", "camera") if c in t.columns]
    img = img.merge(t[keep], on="file", how="inner")
    eye = img["laterality"].astype(str) if "laterality" in img.columns else img["file"]
    per_eye = img.groupby([img["patient"].astype(str), eye])[feats].mean()
    per_pat = per_eye.groupby(level=0).mean()
    per_pat["n_eyes"] = per_eye.groupby(level=0).size()
    out = per_pat.join(patient_frame(t, outcomes, covariates), how="left")
    counts["patients"] = int(len(out))
    return out, feats, counts


def _label_cols(outcome: str) -> Tuple[str, ...]:
    from dataset import LABEL_REGISTRY
    return tuple(LABEL_REGISTRY[outcome].source_cols)


def patient_frame(rows, outcomes: Sequence[str], covariates: Sequence[str]):
    """Patient-indexed outcomes (patient-level labels: max over the patient's rows)
    and covariates (first row) from image-level table rows."""
    import pandas as pd
    from dataset import LABEL_REGISTRY
    rows = rows.assign(patient=rows["patient"].astype(str))
    lab = {}
    for o in outcomes:
        spec = LABEL_REGISTRY[o]
        if any(c not in rows.columns for c in spec.source_cols):
            raise SystemExit(f"[fatal] outcome {o}: the table lacks {spec.source_cols}")
        lab[o] = np.array([spec.fn(dict(zip(spec.source_cols, r)))
                           for r in rows[list(spec.source_cols)].to_numpy()], dtype=float)
    pl = pd.DataFrame(lab, index=rows.index).assign(patient=rows["patient"]).groupby("patient").max(min_count=1)
    cov = pd.DataFrame({c: _numeric(rows[c]) for c in covariates if c in rows.columns},
                       index=rows.index).assign(patient=rows["patient"]).groupby("patient").first()
    return pl.join(cov, how="left")


# --------------------------------------------------------------------------- #
# Pre-registration
# --------------------------------------------------------------------------- #
def load_prereg(path: str, allow_draft: bool) -> Tuple[dict, dict]:
    with open(path) as f:
        pr = json.load(f)
    prov = {"path": os.path.abspath(path), "status": pr.get("status"), "dated": pr.get("dated")}
    problems = []
    if pr.get("status") != "final":
        problems.append(f"status is {pr.get('status')!r}, not 'final'")
    try:
        datetime.date.fromisoformat(str(pr.get("dated")))
    except ValueError:
        problems.append(f"'dated' is {pr.get('dated')!r}, not YYYY-MM-DD")
    d = os.path.dirname(os.path.abspath(path))
    git = lambda *a: subprocess.run(["git", *a], cwd=d, capture_output=True, text=True)
    try:
        inside = git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"
    except FileNotFoundError:
        inside = False
    if not inside:
        problems.append("not inside a git checkout, so the registration date cannot be shown")
    else:
        name = os.path.basename(path)
        dirty = git("status", "--porcelain", "--", name).stdout.strip()
        log = git("log", "-1", "--format=%H %cI", "--", name).stdout.strip()   # empty before a first commit
        prov["commit"], prov["committed"] = (log.split() + ["", ""])[:2]
        if dirty or not log:
            problems.append("the file has uncommitted changes or was never committed")
    prov["problems"] = problems
    if problems and not allow_draft:
        raise SystemExit("[fatal] the pre-registration is not final and committed: " + "; ".join(problems)
                         + ". Commit it before opening any result (or --allow-draft to test the code; "
                           "every output is then stamped DRAFT).")
    return pr, prov


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def odds_ratio_per_sd(df, outcome: str, exposure: str, covariates: Sequence[str]) -> dict:
    """``outcome ~ exposure (z-scored) + covariates`` on complete cases."""
    from brset_comorbidities import logit_or
    cols = [outcome, exposure] + [c for c in covariates if c in df.columns]
    d = df[cols].apply(lambda s: s.astype(float)).dropna()
    d = d[d[outcome].isin([0.0, 1.0])]
    res = {"n": int(len(d)), "cases": int(d[outcome].sum()) if len(d) else 0}
    used = [c for c in covariates if c in d.columns and d[c].nunique() > 1]
    res["covariates"] = used
    if res["cases"] < 2 or res["n"] - res["cases"] < 2 or d[exposure].std() == 0:
        res["reason"] = "too few cases or controls, or a constant exposure"
        return res
    sd = float(d[exposure].std())
    z = (d[exposure] - d[exposure].mean()) / sd
    X = np.column_stack([z.to_numpy()] + [d[c].to_numpy() for c in used])
    o = logit_or(d[outcome].to_numpy(), X, [exposure] + used)[exposure]
    se = (math.log(o["hi"]) - math.log(o["lo"])) / (2 * 1.96) if o["lo"] > 0 and o["hi"] < float("inf") else float("nan")
    zstat = math.log(o["or"]) / se if se and se == se and se > 0 else float("nan")
    p = math.erfc(abs(zstat) / math.sqrt(2)) if zstat == zstat else float("nan")
    res.update({"or": o["or"], "lo": o["lo"], "hi": o["hi"], "p": p, "stable": bool(o["stable"]),
                "sd": sd})
    if res["cases"] < 10:
        res["note"] = "fewer than 10 cases: read as a case series"
    return res


def beta_per_sd(df, outcome: str, exposure: str, covariates: Sequence[str]) -> dict:
    """``outcome ~ exposure (z-scored) + covariates`` by least squares, for a measured
    continuous outcome (AI-READI blood pressure, log UACR, MoCA): the change in the
    outcome per SD of the exposure, normal-approximation 95 % CI and p."""
    cols = [outcome, exposure] + [c for c in covariates if c in df.columns]
    d = df[cols].apply(lambda s: s.astype(float)).dropna()
    used = [c for c in covariates if c in d.columns and d[c].nunique() > 1]
    res = {"n": int(len(d)), "cases": int(len(d)), "covariates": used, "effect": "beta_per_sd"}
    if len(d) < len(used) + 10 or d[exposure].std() == 0:
        res["reason"] = "too few complete rows, or a constant exposure"
        return res
    sd = float(d[exposure].std())
    X = np.column_stack([np.ones(len(d)), ((d[exposure] - d[exposure].mean()) / sd).to_numpy()]
                        + [d[c].to_numpy() for c in used])
    yv = d[outcome].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    s2 = float(resid @ resid) / max(len(yv) - X.shape[1], 1)
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    b = float(beta[1])
    res.update(beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se, sd=sd,
               p=math.erfc(abs(b / se) / math.sqrt(2)) if se > 0 else float("nan"), stable=True)
    return res


def effect_per_sd(df, outcome: str, exposure: str, covariates: Sequence[str]) -> dict:
    """Odds ratio per SD for a yes/no outcome, beta per SD for a continuous one."""
    from dataset import LABEL_REGISTRY
    spec = LABEL_REGISTRY[outcome]
    if spec.num_classes is None and not spec.multilabel:
        return beta_per_sd(df, outcome, exposure, covariates)
    r = odds_ratio_per_sd(df, outcome, exposure, covariates)
    r["effect"] = "or_per_sd"
    return r


def holm(ps: Sequence[float]) -> List[float]:
    ps = list(ps)
    order = sorted(range(len(ps)), key=lambda i: (ps[i] != ps[i], ps[i]))
    out, run, m = [float("nan")] * len(ps), 0.0, sum(p == p for p in ps)
    for rank, i in enumerate(order):
        if ps[i] != ps[i]:
            continue
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def bh(ps: Sequence[float]) -> List[float]:
    ps = list(ps)
    idx = [i for i in range(len(ps)) if ps[i] == ps[i]]
    order = sorted(idx, key=lambda i: ps[i])
    m, out, prev = len(idx), [float("nan")] * len(ps), 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, ps[i] * m / (rank + 1))
        out[i] = prev
    return out


def _direction_ok(r: dict, expected: Optional[str]) -> Optional[bool]:
    if not expected or ("or" not in r and "beta" not in r):
        return None
    if "beta" in r:
        return (r["beta"] < 0) if expected == "lower" else (r["beta"] > 0)
    return (r["or"] < 1) if expected == "lower" else (r["or"] > 1)


# --------------------------------------------------------------------------- #
# Forest plot (stdlib csv + matplotlib only, so system python3 can redraw it)
# --------------------------------------------------------------------------- #
FOREST_COLS = ("analysis", "outcome", "exposure", "role", "n", "cases", "or", "lo", "hi", "p")


def plot_forest(forest_csv: str, png: str, title: str = "") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    with open(forest_csv) as f:
        rows = [r for r in csv.DictReader(f) if r.get("or") not in ("", "nan", None)]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        o, lo, hi = float(r["or"]), float(r["lo"]), float(r["hi"])
        ctrl = r["role"] == "negative_control"
        colour = "#8a8a8a" if ctrl else ("#1F6F78" if r["analysis"] == "vessel" else "#A8502A")
        ax.plot([max(lo, 1e-3), min(hi, 1e3)], [yi, yi], color=colour, lw=1.6)
        ax.plot([o], [yi], marker="D" if ctrl else "s", color=colour, ms=6)
    ax.axvline(1.0, color="#333333", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['outcome'].replace('cm_', '')}  ~  {r['exposure']}"
                        f"{'  (control)' if r['role'] == 'negative_control' else ''}  "
                        f"[{r['cases']}/{r['n']}]" for r in rows], fontsize=8)
    ax.set_xlabel("odds ratio per SD (95% CI), adjusted")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
def run(args) -> int:
    import pandas as pd
    from train_mbrset import load_table
    pr, prov = load_prereg(args.prereg, args.allow_draft)
    spec = (pr.get("datasets") or {}).get(args.dataset)
    if not spec:
        raise SystemExit(f"[fatal] {args.prereg} has no entry for dataset {args.dataset!r}")
    camera = args.camera if args.camera is not None else spec.get("camera")
    covariates = list(spec.get("covariates") or ["age", "sex"])
    diseases = {k: v for k, v in (spec.get("diseases") or {}).items() if v}
    control = spec.get("negative_control")
    gap_spec = spec.get("age_gap") or {}
    gap_conditions = list(gap_spec.get("conditions") or [])
    outcomes = list(dict.fromkeys(list(diseases) + ([control] if control else []) +
                                  (gap_conditions if args.age_gap else [])))
    trigger = next((o for o in outcomes if o.startswith("cm_")), outcomes[0] if outcomes else "dr_referable")
    table = load_table(args.root, args.dataset, trigger, camera=camera, image_ext=args.image_ext)["df"]
    # a pre-registered covariate the table lacks (or never records) would be dropped
    # from every model while the header still named it: a deviation, not a detail
    gone = [c for c in covariates if c not in table.columns or _numeric(table[c]).isna().all()]
    if gone:
        msg = f"pre-registered covariate(s) {gone} missing from the table (columns: {sorted(table.columns)[:40]})"
        if not args.allow_draft:
            raise SystemExit(f"[fatal] {msg}")
        prov["problems"].append(msg)
    absent = [o for o in outcomes if not set(_label_cols(o)) <= set(table.columns)]
    if absent:
        raise SystemExit(f"[fatal] outcome(s) {absent} have no column in the {args.dataset} table; "
                         f"is the pre-registration entry for the right dataset?")
    draft = bool(prov["problems"])
    stamp = ("DRAFT, NOT PRE-REGISTERED: " + "; ".join(prov["problems"])) if draft else \
        f"pre-registered {prov['dated']} (commit {prov.get('commit', '')[:10]} on {prov.get('committed', '')})"
    os.makedirs(args.out, exist_ok=True)
    results: List[dict] = []
    L = [f"VESSELS AND RETINAL AGE GAP  --  {args.dataset}{f' ({camera})' if camera else ''}", stamp,
         f"covariates: {' + '.join(covariates)}; effect = odds ratio per SD of the exposure"]

    pat = None
    if args.automorph:
        pat, feats, counts = patient_measurements(args.automorph, table, outcomes, covariates)
        pat.to_csv(os.path.join(args.out, "patient_measurements.csv"))
        L.append(f"AutoMorph: {counts['staged']} staged, {counts['graded']} graded, {counts['rejected']} "
                 f"rejected; {counts['patients']} patients with a gradable measured eye")
        missing = {d: v["feature"] for d, v in diseases.items() if v["feature"] not in feats}
        if missing:
            raise SystemExit(f"[fatal] pre-registered feature(s) not among AutoMorph's columns: {missing}. "
                             f"Columns: {feats}. Fix only the spelling, as a dated amendment.")
        L += ["", "PRIMARY TESTS (one per disease, pre-registered)"]
        prim = []
        for d, v in diseases.items():
            r = effect_per_sd(pat, d, v["feature"], covariates)
            r.update(analysis="vessel", outcome=d, exposure=v["feature"], role="primary",
                     expected=v.get("direction"), direction_ok=_direction_ok(r, v.get("direction")))
            prim.append(r)
        for r, ph in zip(prim, holm([r.get("p", float("nan")) for r in prim])):
            r["p_holm"] = ph
        results += prim
        if control:
            for f_ in dict.fromkeys(v["feature"] for v in diseases.values()):
                r = effect_per_sd(pat, control, f_, covariates)
                r.update(analysis="vessel", outcome=control, exposure=f_, role="negative_control")
                results.append(r)
        if args.exploratory:
            expl = []
            for d in diseases:
                for f_ in feats:
                    if f_ == diseases[d]["feature"] or pat[f_].notna().mean() < 0.5:
                        continue
                    r = effect_per_sd(pat, d, f_, covariates)
                    r.update(analysis="vessel", outcome=d, exposure=f_, role="exploratory")
                    expl.append(r)
            for r, q in zip(expl, bh([r.get("p", float("nan")) for r in expl])):
                r["q_bh"] = q
            results += expl

    if args.age_gap:
        g = pd.read_csv(args.age_gap)
        if args.gap_col not in g.columns or "file" not in g.columns:
            raise SystemExit(f"[fatal] {args.age_gap} needs 'file' and {args.gap_col!r}; has {list(g.columns)}")
        t = table.assign(file=table["file"].astype(str)).drop_duplicates("file")
        gg = g.assign(file=g["file"].astype(str))[["file", args.gap_col]].merge(t, on="file", how="inner")
        if "final_quality" in gg.columns and not args.keep_ungradable:
            from dataset import _yes_no
            ok = gg["final_quality"].map(_yes_no) == 1
            L.append(f"age gap: {int((~ok).sum())} ungradable images dropped (they read older)")
            gg = gg[ok]
        rows = gg.assign(patient=gg["patient"].astype(str))
        per = rows.groupby("patient")[args.gap_col].mean().rename("age_gap")
        # every patient with a gap, whether or not AutoMorph measured them
        dfg = patient_frame(rows, outcomes, covariates).join(per, how="inner")
        L.append(f"age gap: {len(dfg)} patients with a cross-fitted, bias-corrected gap ({args.gap_col})")
        gres = []
        for c in list(dict.fromkeys(gap_conditions + ([control] if control else []))):
            if c not in dfg.columns:
                continue
            r = effect_per_sd(dfg, c, "age_gap", covariates)
            r.update(analysis="age_gap", outcome=c, exposure="age_gap", role="negative_control" if c == control
                     else "primary", expected=gap_spec.get("direction"),
                     direction_ok=_direction_ok(r, gap_spec.get("direction")))
            if "sd" in r and "or" in r:
                r["or_per_year"] = float(r["or"] ** (1.0 / r["sd"]))
            gres.append(r)
        prim_g = [r for r in gres if r["role"] == "primary"]
        for r, ph in zip(prim_g, holm([r.get("p", float("nan")) for r in prim_g])):
            r["p_holm"] = ph
        results += gres

    if not results:
        raise SystemExit("[fatal] nothing to test: give --automorph and/or --age-gap")
    res_df = pd.DataFrame(results)
    res_df.insert(0, "dataset", args.dataset)
    res_df["preregistration"] = stamp
    res_df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    forest = res_df[(res_df["role"] != "exploratory") &
                    (res_df.get("effect", pd.Series("or_per_sd", index=res_df.index)) != "beta_per_sd")]
    forest.reindex(columns=list(FOREST_COLS)).to_csv(os.path.join(args.out, "forest.csv"), index=False)
    for analysis in ("vessel", "age_gap"):
        sub = res_df[res_df["analysis"] == analysis]
        if sub.empty:
            continue
        L += ["", f"--- {analysis} ---"]
        for r in sub.to_dict("records"):
            is_beta = r.get("effect") == "beta_per_sd"
            val = r.get("beta") if is_beta else r.get("or")
            if val is None or val != val:
                L.append(f"  [{r['role']}] {r['outcome']} ~ {r['exposure']}: {r.get('reason', 'not computed')} "
                         f"(n={r['n']}, cases={r['cases']})")
                continue
            extra = []
            if r.get("p_holm") == r.get("p_holm") and r.get("p_holm") is not None:
                extra.append(f"Holm p {r['p_holm']:.3g}")
            if r.get("q_bh") == r.get("q_bh") and r.get("q_bh") is not None:
                extra.append(f"BH q {r['q_bh']:.3g}")
            if r.get("direction_ok") is not None and r.get("direction_ok") == r.get("direction_ok"):
                extra.append("expected direction" if r["direction_ok"] else "OPPOSITE to the expected direction")
            if r.get("or_per_year") == r.get("or_per_year") and r.get("or_per_year") is not None:
                extra.append(f"OR per year {r['or_per_year']:.3f}")
            if isinstance(r.get("note"), str):
                extra.append(r["note"])
            if r.get("stable") is False:
                extra.append("UNSTABLE fit (separation)")
            used = r.get("covariates")
            if isinstance(used, list) and len(used) != len(covariates):
                extra.append(f"adjusted for {'+'.join(used) or 'nothing'} only (a covariate was constant here)")
            label = f"beta/SD {val:+.3f}" if is_beta else f"OR/SD {val:.3f}"
            count = f"n={int(r['n'])}" if is_beta else f"{int(r['cases'])}/{int(r['n'])}"
            L.append(f"  [{r['role']:<16}] {r['outcome']:<24} ~ {r['exposure']:<26} {label} "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}]  p {r['p']:.3g}  ({count})"
                     + (f"  {'; '.join(extra)}" if extra else ""))
    if args.exploratory:
        L += ["", "Exploratory rows are hypothesis-generating only; the primaries above are the result."]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("```\n" + txt + "\n```\n")
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump({"preregistration": prov, "args": vars(args)}, f, indent=2)
    png = os.path.join(args.out, "forest.png")
    if plot_forest(os.path.join(args.out, "forest.csv"), png, title=stamp[:90]):
        print(f"[info] wrote {png}")
    else:
        print(f"[info] no matplotlib here: on the Mac, python3 vessel_associations.py --plot-only "
              f"{os.path.join(args.out, 'forest.csv')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pre-registered vessel and age-gap associations.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--plot-only", default=None, metavar="FOREST_CSV",
                   help="only redraw forest.png beside this forest.csv (stdlib + matplotlib)")
    p.add_argument("--automorph", default=None, help="automorph_stage.py folder after AutoMorph ran")
    p.add_argument("--dataset", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--camera", default=None, help="default: the pre-registration's camera")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--prereg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vessel_prereg.json"))
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--age-gap", default=None, help="per-image cross-fitted clock predictions (file, gap column)")
    p.add_argument("--gap-col", default="gap_corrected")
    p.add_argument("--keep-ungradable", action="store_true")
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.plot_only:
        png = os.path.join(os.path.dirname(os.path.abspath(a.plot_only)), "forest.png")
        if not plot_forest(a.plot_only, png):
            raise SystemExit("[fatal] matplotlib is not available, or the CSV has no estimable rows")
        print(f"[info] wrote {png}")
        return 0
    if not (a.dataset and a.root and a.out):
        p.error("--dataset, --root and --out are required (or --plot-only)")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
