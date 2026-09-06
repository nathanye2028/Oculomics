"""CPU-only, synthetic, network-free tests for the retinal-age track: the
healthy-cohort rule (patient-level, NaN-safe), the age-stratified patient split,
the regression metrics and the age-gap bias correction, and a 1-epoch end-to-end
run of train_retinal_age.py on tiny synthetic BRSET/mBRSET trees."""
import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from train_retinal_age import (AGE_BIN_LABELS, _flag, age_bin, bias_apply, bias_fit,
                               build_cohort, main, regression_metrics, split_by_patient)


# ---- flags ------------------------------------------------------------------ #
def test_flag_is_strict_and_nan_safe():
    assert _flag("yes") == 1.0 and _flag("No") == 0.0 and _flag(" sim ") == 1.0
    assert _flag(1) == 1.0 and _flag(0.0) == 0.0 and _flag(True) == 1.0
    for bad in (2, "unknown", "", None, np.nan, "1/2"):
        assert np.isnan(_flag(bad)), bad


# ---- cohort ----------------------------------------------------------------- #
def _brset_like():
    # mBRSET-schema frame as load_brset would return it (quality already 1/0).
    rows = [
        # p1: healthy non-diabetic, both eyes gradable
        dict(file="a.jpg", patient="p1", age=50, final_icdr=0, diabetes="no", final_quality=1.0),
        dict(file="b.jpg", patient="p1", age=50, final_icdr=0, diabetes="no", final_quality=1.0),
        # p2: non-diabetic but ONE eye has DR -> both eyes excluded (patient-level)
        dict(file="c.jpg", patient="p2", age=61, final_icdr=0, diabetes="no", final_quality=1.0),
        dict(file="d.jpg", patient="p2", age=61, final_icdr=2, diabetes="no", final_quality=1.0),
        # p3: diabetic, no DR -> excluded under nodm, kept under dr0
        dict(file="e.jpg", patient="p3", age=45, final_icdr=0, diabetes="yes", final_quality=1.0),
        # p4: healthy but one image ungradable -> that IMAGE excluded, the other kept
        dict(file="f.jpg", patient="p4", age=33, final_icdr=0, diabetes="no", final_quality=0.0),
        dict(file="g.jpg", patient="p4", age=33, final_icdr=0, diabetes="no", final_quality=1.0),
        # p5: diabetes unknown -> NOT healthy under nodm (never a silent negative)
        dict(file="h.jpg", patient="p5", age=70, final_icdr=0, diabetes=np.nan, final_quality=1.0),
        # p6: no age -> dropped entirely
        dict(file="i.jpg", patient="p6", age=np.nan, final_icdr=0, diabetes="no", final_quality=1.0),
    ]
    return pd.DataFrame(rows)


def test_cohort_nodm_is_patient_level_and_nan_safe():
    c = build_cohort(_brset_like(), healthy="nodm")
    assert "i.jpg" not in set(c["file"])                       # no age -> dropped
    h = set(c.loc[c["healthy"], "file"])
    assert h == {"a.jpg", "b.jpg", "g.jpg"}
    ex = dict(zip(c["file"], c["exclusion"]))
    assert ex["c.jpg"] == "dr" and ex["d.jpg"] == "dr"          # fellow eye disqualified
    assert ex["e.jpg"] == "diabetes"
    assert ex["f.jpg"] == "ungradable"
    assert ex["h.jpg"] == "diabetes"                            # unknown diabetes = not healthy


def test_cohort_dr0_keeps_diabetics_without_dr_and_all_keeps_everyone():
    c = build_cohort(_brset_like(), healthy="dr0")
    # p3 (diabetic, no DR) and p5 (diabetes unknown, no DR) are both healthy under dr0
    assert set(c.loc[c["healthy"], "file"]) == {"a.jpg", "b.jpg", "e.jpg", "g.jpg", "h.jpg"}
    c = build_cohort(_brset_like(), healthy="gradable")
    assert set(c.loc[~c["healthy"], "file"]) == {"f.jpg"}
    c = build_cohort(_brset_like(), healthy="all")
    assert c["healthy"].all()


def test_cohort_nodm_refuses_missing_diabetes_column():
    with pytest.raises(ValueError, match="diabetes"):
        build_cohort(_brset_like().drop(columns=["diabetes"]), healthy="nodm")
    # ... but dr0 works without it
    assert build_cohort(_brset_like().drop(columns=["diabetes"]), healthy="dr0")["healthy"].any()


def test_cohort_exclude_pathology_reads_raw_csv():
    raw = pd.DataFrame({"image_id": ["a", "b", "g"], "amd": [0, 0, 1], "drusens": [0, 0, 0]})
    c = build_cohort(_brset_like(), healthy="nodm", exclude_pathology=True, raw_csv=raw)
    assert set(c.loc[c["healthy"], "file"]) == {"a.jpg", "b.jpg"}     # p4 has AMD on g.jpg
    with pytest.raises(ValueError):
        build_cohort(_brset_like(), healthy="nodm", exclude_pathology=True)


# ---- split ------------------------------------------------------------------ #
def test_split_by_patient_is_disjoint_and_complete():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"file": [f"{i}.jpg" for i in range(n)], "patient": [f"p{i//2}" for i in range(n)],
                       "age": np.repeat(rng.integers(20, 90, n // 2), 2).astype(float),
                       "healthy": np.repeat(rng.random(n // 2) < 0.6, 2)})
    sp = split_by_patient(df, seed=1)
    assert set(sp.unique()) == {"train", "val", "test"} and len(sp) == n
    psets = {k: set(df.loc[sp == k, "patient"]) for k in ("train", "val", "test")}
    for a, b in itertools.combinations(psets, 2):
        assert not (psets[a] & psets[b])
    frac = sp.value_counts(normalize=True)
    assert 0.12 < frac["test"] < 0.28 and 0.05 < frac["val"] < 0.16
    assert split_by_patient(df, seed=1).equals(sp)                      # deterministic
    with pytest.raises(ValueError):
        split_by_patient(df.drop(columns=["patient"]), seed=0)


# ---- metrics ----------------------------------------------------------------- #
def test_regression_metrics_and_bins():
    age = np.array([25, 35, 45, 55, 65, 75, 85], float)
    pred = age + np.array([2, -2, 1, -1, 3, -3, 0], float)
    m = regression_metrics(age, pred, patient=["p1", "p1", "p2", "p3", "p4", "p5", "p6"])
    assert m["n"] == 7 and abs(m["mae"] - 12 / 7) < 1e-9 and abs(m["mean_gap"]) < 1e-9
    assert sum(b["n"] for b in m["by_bin"].values()) == 7
    assert list(m["by_bin"]) == list(AGE_BIN_LABELS) and m["by_bin"]["<30"]["mae"] == 2.0
    assert m["n_patients"] == 6                       # p1's two eyes averaged: (25+35)/2 vs (27+33)/2
    assert abs(m["patient_mae"] - (0 + 1 + 1 + 3 + 3 + 0) / 6) < 1e-9
    assert regression_metrics([], [])["n"] == 0
    assert list(age_bin([10, 30, 39, 40, 79, 80, 99])) == [0, 1, 1, 2, 5, 6, 6]


def test_bias_correction_removes_age_dependence():
    rng = np.random.default_rng(3)
    age = rng.uniform(20, 90, 2000)
    gap = 8.0 - 0.15 * age + rng.normal(0, 1.0, 2000)     # regression-to-the-mean pattern
    fit = bias_fit(age, gap)
    assert abs(fit["a"] - 8.0) < 0.3 and abs(fit["b"] + 0.15) < 0.01
    corr = bias_apply(age, gap, fit)
    assert abs(np.corrcoef(age, corr)[0, 1]) < 0.05 and abs(corr.mean()) < 0.1
    assert bias_fit([50.0], [3.0]) == {"a": 3.0, "b": 0.0, "n": 1}


# ---- end to end -------------------------------------------------------------- #
def _fundus(rng, path, age):
    a = np.zeros((72, 80, 3), np.uint8)
    a[8:64, 10:70] = rng.integers(40, 200, (56, 60, 3), dtype=np.uint8)
    a[8:64, 10:70, 0] = np.clip(60 + age * 2 + rng.integers(-10, 10), 0, 255)   # a learnable signal
    Image.fromarray(a).save(path)


def _make_trees(tmp_path, n_pat=40, n_ext=24):
    rng = np.random.default_rng(0)
    B = tmp_path / "BRSET"; (B / "fundus_photos").mkdir(parents=True)
    rows = []
    for p in range(n_pat):
        age = int(rng.integers(20, 85))
        dm = "yes" if p % 5 == 0 else "no"
        for eye in (1, 2):
            i = p * 2 + eye
            icdr = 3 if (p % 7 == 0 and eye == 2) else 0
            _fundus(rng, B / "fundus_photos" / f"img{i:05d}.jpg", age)
            rows.append(dict(image_id=f"img{i:05d}", patient_id=p, DR_ICDR=icdr, patient_age=age,
                             patient_sex=int(rng.integers(1, 3)), exam_eye=eye, diabetes=dm,
                             quality="Inadequate" if i % 11 == 0 else "Adequate",
                             artifacts=1, amd=int(p % 13 == 0), drusens=0))
    pd.DataFrame(rows).to_csv(B / "labels_brset.csv", index=False)
    M = tmp_path / "mBRSET"; (M / "images").mkdir(parents=True)
    rows = []
    for i in range(n_ext):
        age = int(rng.integers(30, 80))
        _fundus(rng, M / "images" / f"{i//2}.{i%2+1}.jpg", age)
        rows.append(dict(file=f"{i//2}.{i%2+1}.jpg", patient=i // 2, age=age, sex=int(rng.integers(0, 2)),
                         final_icdr=int(rng.integers(0, 4)), final_quality="yes" if i % 9 else "no",
                         final_edema="no", systemic_hypertension="yes" if i % 3 == 0 else "no"))
    pd.DataFrame(rows).to_csv(M / "labels_mbrset.csv", index=False)
    return str(B), str(M)


def test_end_to_end_smoke(tmp_path):
    B, M = _make_trees(tmp_path)
    ck = tmp_path / "ck"; out = tmp_path / "exp"
    common = ["--root", B, "--external-test-root", M, "--image-size", "64", "--backbone", "mobilenetv3_small",
              "--no-pretrained", "--num-workers", "0", "--batch-size", "8", "--no-amp",
              "--ckpt-dir", str(ck), "--epochs", "1", "--warmup-epochs", "0"]
    assert main(common + ["--inspect"]) == 0
    assert not ck.exists()                                     # --inspect touches nothing
    rc = main(common + ["--seed", "0", "--run-name", "student_seed0", "--bn-adapt",
                        "--results-json", str(out / "student_seed0.json")])
    assert rc == 0
    r = json.load(open(out / "student_seed0.json"))
    assert r["task"] == "retinal_age" and r["healthy"] == "nodm"
    assert r["cohort"]["n_healthy_images"] < r["cohort"]["n_images"]
    for k in ("val", "test_healthy", "test_all", "external", "external_bnadapt", "bias_correction"):
        assert r[k] is not None, k
    assert r["test_healthy"]["n"] > 0 and r["external"]["n"] > 0
    assert set(r["test_healthy"]["by_bin"]) == set(AGE_BIN_LABELS)
    assert r["external_by_dr"] and set(r["external_by_dr"]) == {"dr0", "dr1", "referable"}
    # every image with an age is exactly one of: trained on, val, or scored (never trained on)
    assert r["n_train"] + r["n_val"] + r["n_scored"] == r["cohort"]["n_images"]
    assert (ck / "student_seed0.pt").exists() and (ck / "student_seed0.done").exists()

    pr = pd.read_csv(r["predictions_csv"])
    assert set(pr["dataset"]) == {"brset", "mbrset"}
    br = pr[pr["dataset"] == "brset"]
    # nothing the model trained on is in the table: healthy rows only from the test split
    assert (br.loc[br["cohort"] == "healthy", "split"] == "test").all()
    assert (br["cohort"] == "nonhealthy").sum() > 0 and "systemic_hypertension" in pr.columns
    ext = pr[pr["dataset"] == "mbrset"]
    assert ext["pred_age_bnadapt"].notna().all() and ext["gap_corrected"].notna().all()
    assert len(ext) == r["external"]["n"]
    # gap_corrected = gap - (a + b*age) row by row
    fit = r["bias_correction"]
    np.testing.assert_allclose(pr["gap_corrected"], pr["gap"] - (fit["a"] + fit["b"] * pr["age"]), atol=1e-6)

    # a second seed + the summariser (pooled predictions, per-bin table)
    assert main(common + ["--seed", "1", "--run-name", "student_seed1",
                          "--results-json", str(out / "student_seed1.json")]) == 0
    import subprocess, sys
    res = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                       "summarize_retinal_age.py"), "--dir", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "MAE by age bin" in res.stdout and (out / "summary.md").exists()
    pooled = pd.read_csv(out / "predictions_pooled.csv")
    assert pooled["n_seeds"].max() == 2 and {"condition", "dataset", "file", "pred_age"} <= set(pooled.columns)
