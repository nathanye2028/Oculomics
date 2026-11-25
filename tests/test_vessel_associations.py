"""Recommendation 2: automorph_stage.py lays out symlinks + name map + resolution
file; vessel_associations.py drops AutoMorph-rejected images by AutoMorph's own
rule, averages the eyes per patient, refuses a draft or uncommitted
pre-registration, fits the pre-registered feature (OR per SD) plus the negative
control, and puts the cross-fitted age gap on the same forest."""
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from test_folds import _brset_with_flags


def _automorph_outputs(stage, root, seed=0):
    """Fake AutoMorph Results/: hypertensive patients have narrower arterioles."""
    rng = np.random.default_rng(seed)
    nm = pd.read_csv(stage / "name_map.csv")
    lab = pd.read_csv(root / "labels_brset.csv").set_index("image_id")
    q, disc, mac = [], [], []
    for i, r in enumerate(nm.itertuples()):
        stem = r.automorph_name.split(".")[0]
        htn = lab.loc[stem, "comorbidities"] == "hypertension"
        pred, bad = (2, 0.9) if i % 17 == 0 else ((1, 0.3) if i % 19 == 0 else (0, 0.01))
        q.append({"Name": f"{stem}.png", "softmax_good": 0.9, "softmax_usable": 0.05,
                  "softmax_bad": bad, "good_sd": 0, "usable_sd": 0, "bad_sd": 0, "Prediction": pred})
        row = {"Name": f"{stem}.png", "Fractal_dimension": 1.4 + rng.normal(0, 0.02),
               "Vessel_density": 0.08 + rng.normal(0, 0.01)}
        if i % 3:
            row.update(CRAE_Knudtson_zone_b=150 - 12 * htn + rng.normal(0, 6),
                       CRVE_Knudtson_zone_b=220 + rng.normal(0, 10))
            disc.append(row)
        else:
            mac.append(row)
    (stage / "Results" / "M1").mkdir(parents=True, exist_ok=True)
    (stage / "Results" / "M3").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(q).to_csv(stage / "Results" / "M1" / "results_ensemble.csv", index=False)
    pd.DataFrame(disc).to_csv(stage / "Results" / "M3" / "Disc_Features.csv", index=False)
    pd.DataFrame(mac).to_csv(stage / "Results" / "M3" / "Macular_Features.csv", index=False)


def _prereg(path, status="draft"):
    pr = {"status": status, "dated": "2026-11-19" if status == "final" else "YYYY-MM-DD",
          "datasets": {"brset": {"camera": "Canon CR", "covariates": ["age", "sex", "diabetes"],
                                 "negative_control": "cm_hypothyroidism",
                                 "diseases": {"cm_hypertension": {"feature": "CRAE_Knudtson_zone_b",
                                                                  "direction": "lower", "source": "x"}},
                                 "age_gap": {"exposure": "gap_corrected", "direction": "higher",
                                             "conditions": ["cm_hypertension"]}}}}
    path.write_text(json.dumps(pr))
    return path


def _brset_probabilistic(tmp_path, n_patients):
    """test_folds' BRSET tree, with hypertension drawn from age (not decided by it:
    a deterministic label separates and every fit is unstable)."""
    root = _brset_with_flags(tmp_path, n_patients=n_patients)
    lab = pd.read_csv(root / "labels_brset.csv")
    rng = np.random.default_rng(3)
    per = lab.groupby("patient_id")["patient_age"].first()
    draw = {p: rng.random() < 1 / (1 + np.exp(-(a - 58) / 10)) for p, a in per.items()}
    rec = lab["comorbidities"] != "NA"
    lab.loc[rec, "comorbidities"] = ["hypertension" if draw[p] else "0" for p in lab.loc[rec, "patient_id"]]
    lab.to_csv(root / "labels_brset.csv", index=False)
    return root


def test_stage_and_pre_registered_associations(tmp_path):
    import automorph_stage
    import vessel_associations as va
    root = _brset_probabilistic(tmp_path, n_patients=200)
    stage = tmp_path / "am"
    assert automorph_stage.main(["--dataset", "brset", "--root", str(root), "--camera", "Canon CR",
                                 "--out", str(stage)]) == 0
    nm = pd.read_csv(stage / "name_map.csv")
    assert len(nm) == len(os.listdir(stage / "images")) and nm["res_mm_per_px"].gt(0).all()
    assert os.path.islink(stage / "images" / nm["automorph_name"][0])
    assert set(pd.read_csv(stage / "resolution_information.csv").columns) == {"fundus", "res"}
    _automorph_outputs(stage, root)

    q = va.read_quality(str(stage))
    assert q["rejected"].sum() > 0                                   # Prediction 2, or 1 with bad >= 0.25
    pre = _prereg(tmp_path / "prereg.json")
    lab = pd.read_csv(root / "labels_brset.csv")
    age = lab.drop_duplicates("image_id").set_index("image_id")
    gap = pd.DataFrame({"file": [f"{i}.jpg" for i in age.index],
                        "gap_corrected": [3.0 * (age.loc[i, "comorbidities"] == "hypertension")
                                          + np.random.default_rng(len(i)).normal(0, 2) for i in age.index]})
    gap.to_csv(tmp_path / "oof.csv", index=False)
    argv = ["--automorph", str(stage), "--dataset", "brset", "--root", str(root), "--prereg", str(pre),
            "--age-gap", str(tmp_path / "oof.csv"), "--out", str(tmp_path / "res")]
    with pytest.raises(SystemExit, match="not final"):
        va.main(argv)
    assert va.main(argv + ["--allow-draft", "--exploratory"]) == 0
    res = pd.read_csv(tmp_path / "res" / "results.csv")
    prim = res[(res["role"] == "primary") & (res["analysis"] == "vessel")].iloc[0]
    assert prim["outcome"] == "cm_hypertension" and prim["or"] < 1 and prim["hi"] < 1
    assert bool(prim["direction_ok"]) and res["preregistration"].str.startswith("DRAFT").all()
    assert ((res["role"] == "negative_control") & (res["outcome"] == "cm_hypothyroidism")).any()
    gap_row = res[(res["analysis"] == "age_gap") & (res["outcome"] == "cm_hypertension")].iloc[0]
    assert gap_row["or"] > 1
    assert (res["role"] == "exploratory").any() and "q_bh" in res.columns
    pm = pd.read_csv(tmp_path / "res" / "patient_measurements.csv")
    assert pm["n_eyes"].max() == 2                                     # eyes averaged per patient
    assert set(pd.read_csv(tmp_path / "res" / "forest.csv")["role"]) <= {"primary", "negative_control"}
    # a pre-registered covariate the table lacks stops the run (or stamps it, with --allow-draft)
    miss = json.loads(pre.read_text())
    miss["datasets"]["brset"]["covariates"] = ["age", "sex", "bmi"]
    (tmp_path / "miss.json").write_text(json.dumps(miss))
    argv_m = [a if a != str(pre) else str(tmp_path / "miss.json") for a in argv]
    with pytest.raises(SystemExit):
        va.main(argv_m)
    assert va.main(argv_m + ["--allow-draft", "--out", str(tmp_path / "res_m")]) == 0
    assert pd.read_csv(tmp_path / "res_m" / "results.csv")["preregistration"].str.contains("bmi").all()
    # a misspelt pre-registered column fails before any test is run
    bad = json.loads(pre.read_text())
    bad["datasets"]["brset"]["diseases"]["cm_hypertension"]["feature"] = "CRAE_zone_b"
    pre.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="not among"):
        va.main(argv + ["--allow-draft"])


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_prereg_must_be_final_and_committed(tmp_path):
    from vessel_associations import load_prereg
    g = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q")
    g("config", "user.email", "t@t"); g("config", "user.name", "t")
    p = _prereg(tmp_path / "prereg.json", status="final")
    with pytest.raises(SystemExit, match="uncommitted"):
        load_prereg(str(p), allow_draft=False)
    g("add", "prereg.json"); g("commit", "-q", "-m", "prereg")
    pr, prov = load_prereg(str(p), allow_draft=False)
    assert prov["commit"] and not prov["problems"]
    p.write_text(p.read_text().replace("2026-11-19", "2026-11-20"))    # edited after committing
    with pytest.raises(SystemExit, match="uncommitted"):
        load_prereg(str(p), allow_draft=False)


def test_multiplicity_helpers():
    from vessel_associations import bh, holm
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(holm([float("nan"), 0.02])[0]) and holm([float("nan"), 0.02])[1] == pytest.approx(0.02)


def test_numeric_keeps_sparse_numbers():
    from vessel_associations import _numeric
    s = pd.Series([np.nan] * 8 + [2.0, 3.0])                         # mostly missing, but numbers
    assert _numeric(s).dropna().tolist() == [2.0, 3.0]
    assert _numeric(pd.Series(["yes", "no", None])).tolist()[:2] == [1.0, 0.0]
