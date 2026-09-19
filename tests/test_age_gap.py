"""Tests for analyze_age_gap.py: the numpy OLS / logistic / BH helpers against known
answers, patient aggregation, gap-column choice, and an end-to-end run on a synthetic
predictions table with planted diabetes / DR / hypertension effects."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

from analyze_age_gap import (assoc_binary, attach_brset_flags, bh_fdr, covariates, logit, ols,
                             pick_gap_col, prelesion, prelesion_lines, to_patients)


def test_ols_matches_scipy_and_logit_recovers_planted_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 1.5 + 2.0 * x + rng.normal(size=500)
    o = ols(y, x[:, None])
    lr = stats.linregress(x, y)
    assert abs(o["coef"][0] - lr.slope) < 1e-9 and abs(o["p"][0] - lr.pvalue) < 1e-9
    assert o["lo"][0] < 2.0 < o["hi"][0]
    z = rng.normal(size=4000)
    p = 1 / (1 + np.exp(-(-0.5 + 1.2 * z)))
    yb = (rng.random(4000) < p).astype(float)
    lg = logit(yb, z[:, None])
    assert lg is not None and abs(lg["coef"][0] - 1.2) < 0.15 and lg["p"][0] < 1e-6
    assert logit(np.array([0, 0, 1, 1.0]), np.array([0, 1, 2, 3.0])[:, None]) is None   # separated


def test_bh_fdr():
    q = bh_fdr([0.01, 0.04, 0.03, np.nan, 0.5])
    # m=4 finite p's: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.0533, 0.5*4/4=0.5, then step-up
    assert np.isnan(q[3]) and abs(q[0] - 0.04) < 1e-12 and abs(q[1] - 0.05333333) < 1e-6
    assert abs(q[2] - 0.05333333) < 1e-6 and q[4] == 0.5
    assert all(a <= b + 1e-12 for a, b in zip(sorted(x for x in q if x == x), sorted(x for x in q if x == x)))


def _table(n_pat=600, seed=1):
    """Two eyes per patient; gap = 2*diabetes + 1.5*referable + 1.0*htn - 0.02*(age-55) + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        age = float(rng.integers(30, 85)); dm = float(rng.random() < 0.4)
        grade = int(rng.choice([0, 0, 1, 2, 3])) if dm else 0
        htn = float(rng.random() < 0.5); sex = float(rng.integers(0, 2))
        cam = "A" if rng.random() < 0.6 else "B"
        base = 2.0 * dm + 1.5 * (grade >= 2) + 1.0 * htn - 0.02 * (age - 55) + rng.normal(0, 2.0)
        for eye in (1, 2):
            g = base + rng.normal(0, 0.7)
            rows.append(dict(dataset="brset", file=f"b{p}_{eye}.jpg", patient=f"p{p}", split="test",
                             cohort="healthy" if (dm == 0 and grade == 0) else "nonhealthy", age=age, sex=sex,
                             dr_grade=grade if eye == 2 else 0, gradable=1.0 if rng.random() > 0.05 else 0.0,
                             diabetes="yes" if dm else "no", dm_time=(rng.integers(1, 25) if dm else np.nan),
                             final_edema="no", insulin="yes" if (dm and rng.random() < 0.4) else "no",
                             camera=cam, systemic_hypertension="yes" if htn else "no",
                             pred_age=age + g + 0.02 * (age - 55), gap=g + 0.02 * (age - 55), gap_corrected=g))
    df = pd.DataFrame(rows)
    ext = df.sample(min(300, len(df)), random_state=0).copy()
    ext["dataset"] = "mbrset"; ext["cohort"] = "external"; ext["split"] = "external"
    ext["file"] = "m" + ext["file"]; ext["patient"] = "m" + ext["patient"]
    ext["diabetes"] = np.nan; ext["camera"] = np.nan
    ext["gap_bnadapt"] = ext["gap_corrected"] - 4.0
    ext["gap_bnadapt_recal_corrected"] = ext["gap_corrected"]
    return pd.concat([df, ext], ignore_index=True)


def test_to_patients_averages_eyes_and_takes_worst_eye():
    df = _table(20)
    pat = to_patients(df[df["dataset"] == "brset"], "gap_corrected")
    assert len(pat) == 20 and (pat["n_images"] == 2).all()
    p0 = df[(df["patient"] == "p0") & (df["dataset"] == "brset")]
    assert abs(pat.loc[pat["patient"] == "p0", "gap"].iloc[0] - p0["gap_corrected"].mean()) < 1e-9
    assert pat.loc[pat["patient"] == "p0", "dr_grade"].iloc[0] == p0["dr_grade"].max()
    assert set(pat["diabetes"].dropna()) <= {0.0, 1.0} and "ungradable" in pat.columns


def test_pick_gap_col_prefers_calibrated_for_external():
    df = _table(30)
    assert pick_gap_col(df[df["dataset"] == "brset"])[0] == "gap_corrected"
    ext = df[df["dataset"] == "mbrset"]
    assert pick_gap_col(ext)[0] == "gap_bnadapt_recal_corrected"
    assert pick_gap_col(ext.drop(columns=["gap_bnadapt_recal_corrected"]))[0] == "gap_bnadapt"
    assert pick_gap_col(ext, override="gap")[0] == "gap"


def test_assoc_recovers_planted_effects_with_covariates():
    df = _table(800)
    pat = to_patients(df[(df["dataset"] == "brset") & (df["gradable"] == 1.0)], "gap_corrected")
    X, used = covariates(pat)
    assert "sex" in used and any(u.startswith("camera") for u in used)
    r = assoc_binary(pat, X, "diabetes", "diabetes")
    # planted 2.0, plus 1.5 * P(referable | diabetic) = 0.6 carried by the diabetics -> ~2.6
    assert r["n_exposed"] > 100 and 2.1 < r["adj_delta"] < 3.1 and r["p_adj"] < 1e-6
    assert r["adj_lo"] < 2.6 < r["adj_hi"] and r["or_per5y"] > 1.0 and r["or_lo"] > 1.0
    assert r["quintiles"] and r["quintiles"][-1][1] > r["quintiles"][0][1]      # prevalence rises with gap
    # referable DR within diabetics: the planted +1.5 on top of diabetes
    r2 = assoc_binary(pat, X, "dr_referable", "referable DR", within="diabetes")
    assert 0.8 < r2["adj_delta"] < 2.2 and r2["p_adj"] < 0.01
    # a null exposure stays null
    pat["coin"] = (np.random.default_rng(3).random(len(pat)) < 0.5).astype(float)
    r3 = assoc_binary(pat, X, "coin", "coin")
    assert abs(r3["adj_delta"]) < 0.6 and r3["p_adj"] > 0.01
    assert assoc_binary(pat.head(15), X.head(15), "diabetes", "x")["note"].startswith("too few")


def test_prelesion_recovers_the_grade0_diabetes_effect_strata_and_duration():
    df = _table(1500, seed=4)
    b = df[(df["dataset"] == "brset") & (df["gradable"] == 1.0)].copy()
    # plant a duration dose-response on top of the flat +2.0: +1 y per decade of diabetes
    dm = b["diabetes"] == "yes"
    b.loc[dm, "gap_corrected"] += 0.1 * b.loc[dm, "dm_time"].astype(float)
    pat = to_patients(b, "gap_corrected")
    pl = prelesion(pat)
    assert pl is not None and pl["n"] < len(pat)                 # grade >= 1 patients are excluded
    o = pl["overall"]
    # planted: +2.0 flat + 0.1 * mean duration (~12.5 y) = ~3.2; nothing from DR (grade 0 only)
    assert o["n_exposed"] > 100 and 2.6 < o["adj_delta"] < 3.9 and o["p_adj"] < 1e-6
    labs = [l for l, _ in pl["strata"]]
    assert {"camera = A", "camera = B"} <= set(labs) and sum(l.startswith("age ") for l in labs) == 4
    for lab, r in pl["strata"]:                                   # the effect lives in every stratum
        if r["n_exposed"] >= 20 and r["n_ref"] >= 20:
            assert r["adj_delta"] > 1.5, (lab, r["adj_delta"])
    du = pl["duration"]
    assert du is not None and du["lo"] < 1.0 < du["hi"] and du["p"] < 0.01
    assert len(pl["tertiles"]) == 3 and pl["tertiles"][-1]["mean"] > pl["tertiles"][0]["mean"]
    assert pl["insulin"] is not None and abs(pl["insulin"]["adj_delta"]) < 1.0      # no planted insulin effect
    text = "\n".join(prelesion_lines("brset", pl))
    assert "diabetes before retinopathy" in text and "| camera = A |" in text and "per 10 years" in text
    # a set where every patient is diabetic (mBRSET) has no contrast to make
    allmd = pat.copy(); allmd["diabetes"] = 1.0
    assert prelesion(allmd) is None and prelesion(pat.drop(columns=["diabetes"])) is None


def test_attach_brset_flags_joins_on_image_id(tmp_path):
    df = _table(10)
    raw = pd.DataFrame({"image_id": ["b0_1", "b0_2", "b1_1"], "amd": [1, 0, 0], "drusens": [0, 0, 1]})
    raw.to_csv(tmp_path / "labels_brset.csv", index=False)
    out, cols = attach_brset_flags(df, str(tmp_path / "labels_brset.csv"))
    assert cols == ["amd", "drusens"]
    b = out[out["dataset"] == "brset"].set_index("file")
    assert b.loc["b0_1.jpg", "amd"] == 1.0 and b.loc["b1_1.jpg", "drusens"] == 1.0 and np.isnan(b.loc["b2_1.jpg", "amd"])
    assert out.loc[out["dataset"] == "mbrset", "amd"].isna().all()


def test_end_to_end_cli(tmp_path):
    df = _table(500)
    df.insert(0, "condition", "student"); df["n_seeds"] = 3
    pred = tmp_path / "predictions_pooled.csv"; df.to_csv(pred, index=False)
    raw = pd.DataFrame({"image_id": df.loc[df["dataset"] == "brset", "file"].str.replace(".jpg", "", regex=False),
                        "amd": (np.random.default_rng(0).random((df["dataset"] == "brset").sum()) < 0.1).astype(int)})
    raw.to_csv(tmp_path / "labels_brset.csv", index=False)
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analyze_age_gap.py")
    res = subprocess.run([sys.executable, script, "--predictions", str(pred), "--brset-csv",
                          str(tmp_path / "labels_brset.csv"), "--out", str(tmp_path / "assoc")],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "## brset" in out and "## mbrset" in out and "prevalence by quintile" in out
    assert "gap_bnadapt_recal_corrected" in out and "quality artefact check" in out
    assert "| amd |" in out and "| systemic hypertension |" in out and "by DR grade" in out
    assert "brset: diabetes before retinopathy" in out and "mbrset: diabetes before retinopathy" not in out
    tab = pd.read_csv(tmp_path / "assoc" / "age_gap_associations.csv")
    d = tab[(tab["dataset"] == "brset") & (tab["column"] == "diabetes")].iloc[0]
    assert 2.1 < d["adj_delta"] < 3.1 and d["q_adj"] < 0.001
    h = tab[(tab["dataset"] == "mbrset") & (tab["column"] == "systemic_hypertension")].iloc[0]
    assert 0.3 < h["adj_delta"] < 1.7
    assert (tmp_path / "assoc" / "age_gap_associations.md").exists()
