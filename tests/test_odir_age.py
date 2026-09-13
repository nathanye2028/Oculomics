"""ODIR-5K as a labelled-age training source: keyword parsing, the adapter's age
fields, the loader dispatch, and the patient-level ``--healthy normal`` rule."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brset_dataset import DATASETS, load_any                     # noqa: E402
from public_fundus import _odir_tokens, load_odir, odir_age_fields   # noqa: E402
from train_retinal_age import HEALTHY_DEFS, build_cohort         # noqa: E402


def test_odir_tokens_split_fullwidth_and_ascii_commas():
    assert _odir_tokens("laser spot，moderate non proliferative retinopathy") == \
        ["laser spot", "moderate non proliferative retinopathy"]
    assert _odir_tokens("Normal Fundus, lens dust") == ["normal fundus", "lens dust"]
    assert _odir_tokens(np.nan) == [] and _odir_tokens(None) == [] and _odir_tokens("") == []


def test_odir_age_fields_from_keywords():
    f = odir_age_fields
    assert f("normal fundus") == {"final_quality": 1.0, "normal_fundus": 1.0, "final_icdr": 0.0}
    # a quality remark alone: ungradable, and NOTHING can be read about the fundus
    r = f("lens dust")
    assert r["final_quality"] == 0.0 and np.isnan(r["normal_fundus"]) and np.isnan(r["final_icdr"])
    # normal fundus with a quality remark: still a normal reading, but not gradable
    r = f("normal fundus，lens dust")
    assert r["final_quality"] == 0.0 and r["normal_fundus"] == 1.0 and r["final_icdr"] == 0.0
    # a non-DR finding: abnormal, DR grade 0
    assert f("cataract") == {"final_quality": 1.0, "normal_fundus": 0.0, "final_icdr": 0.0}
    assert f("hypertensive retinopathy")["final_icdr"] == 0.0          # not diabetic retinopathy
    # graded DR, both spellings, "suspected" and "severe proliferative" included
    assert f("mild nonproliferative retinopathy")["final_icdr"] == 1.0
    assert f("laser spot，moderate non proliferative retinopathy")["final_icdr"] == 2.0
    assert f("suspected moderate non proliferative retinopathy")["final_icdr"] == 2.0
    assert f("severe nonproliferative retinopathy")["final_icdr"] == 3.0
    assert f("severe proliferative diabetic retinopathy")["final_icdr"] == 4.0
    assert f("proliferative diabetic retinopathy，drusen")["final_icdr"] == 4.0
    # DR named without a grade: abnormal, grade unknown (never a guessed number)
    r = f("suspected diabetic retinopathy")
    assert r["normal_fundus"] == 0.0 and np.isnan(r["final_icdr"])
    # nothing at all
    r = f(np.nan)
    assert r["final_quality"] == 1.0 and np.isnan(r["normal_fundus"]) and np.isnan(r["final_icdr"])


def _odir_tree(tmp_path, n_pat=12):
    """Kaggle full_df.csv layout (one row per eye) + preprocessed_images/."""
    rng = np.random.default_rng(3)
    root = tmp_path / "odir"; (root / "preprocessed_images").mkdir(parents=True)
    kw = {0: ("normal fundus", "normal fundus"),
          1: ("normal fundus", "cataract"),                  # one bad eye disqualifies the patient
          2: ("normal fundus，lens dust", "normal fundus"),  # ungradable eye, fellow eye healthy
          3: ("moderate non proliferative retinopathy", "mild nonproliferative retinopathy"),
          4: ("lens dust", "normal fundus")}                 # unreadable eye, fellow eye healthy
    rows = []
    for p in range(n_pat):
        lk, rk = kw.get(p, ("normal fundus", "normal fundus"))
        age = int(rng.integers(20, 85))
        for side, k in (("left", lk), ("right", rk)):
            fn = f"{p}_{side}.jpg"
            Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(root / "preprocessed_images" / fn)
            rows.append({"ID": p, "Patient Age": age, "Patient Sex": "Male" if p % 2 else "Female",
                         "Left-Fundus": f"{p}_left.jpg", "Right-Fundus": f"{p}_right.jpg",
                         "Left-Diagnostic Keywords": lk, "Right-Diagnostic Keywords": rk,
                         "N": int(lk == rk == "normal fundus"), "D": 0, "G": 0, "C": 0, "A": 0, "H": 0, "M": 0, "O": 0,
                         "filepath": f"../input/x/{fn}", "labels": "['N']", "target": "[1,0,0,0,0,0,0,0]",
                         "filename": fn})
    pd.DataFrame(rows).to_csv(root / "full_df.csv", index=False)
    return str(root)


def test_load_odir_carries_age_fields_and_load_any_dispatches(tmp_path):
    root = _odir_tree(tmp_path)
    assert "odir" in DATASETS
    src = load_any(root, "odir")
    df = src["df"]
    assert src["source"] == "odir" and src["images_dir"].endswith("preprocessed_images")
    for c in ("file", "patient", "age", "sex", "final_quality", "final_icdr", "normal_fundus", "keywords", "camera"):
        assert c in df.columns, c
    assert (df["camera"] == "odir").all() and len(df) == 24 and df["patient"].nunique() == 12
    eye = df.set_index("file")
    assert eye.loc["0_left.jpg", "normal_fundus"] == 1.0 and eye.loc["1_right.jpg", "normal_fundus"] == 0.0
    assert eye.loc["2_left.jpg", "final_quality"] == 0.0 and eye.loc["2_left.jpg", "normal_fundus"] == 1.0
    assert eye.loc["3_left.jpg", "final_icdr"] == 2.0 and eye.loc["3_right.jpg", "final_icdr"] == 1.0
    assert np.isnan(eye.loc["4_left.jpg", "normal_fundus"])
    assert df["glaucoma"].notna().sum() > 0                    # the glaucoma branch's labels still ride along
    # the same adapter through the direct entry point
    assert len(load_odir(root)["df"]) == 24


def test_cohort_normal_rule_is_patient_level_and_refuses_without_column(tmp_path):
    assert "normal" in HEALTHY_DEFS
    df = load_any(_odir_tree(tmp_path), "odir")["df"]
    c = build_cohort(df, healthy="normal").set_index("file")
    # both eyes normal -> healthy
    assert c.loc["0_left.jpg", "healthy"] and c.loc["0_right.jpg", "healthy"]
    # one cataract eye disqualifies the fellow (normal) eye: patient-level, reason 'abnormal'
    assert not c.loc["1_left.jpg", "healthy"] and c.loc["1_left.jpg", "exclusion"] == "abnormal"
    assert not c.loc["1_right.jpg", "healthy"]
    # a normal reading with lens dust: that eye is 'ungradable', the fellow eye stays healthy
    assert not c.loc["2_left.jpg", "healthy"] and c.loc["2_left.jpg", "exclusion"] == "ungradable"
    assert c.loc["2_right.jpg", "healthy"]
    # DR patient: excluded, and the DR grade travels as dr_grade
    assert not c.loc["3_left.jpg", "healthy"] and c.loc["3_left.jpg", "dr_grade"] == 2.0
    # an eye whose keywords say nothing (lens dust only) is NOT normal -> the patient is excluded
    assert not c.loc["4_right.jpg", "healthy"] and c.loc["4_right.jpg", "exclusion"] == "abnormal"
    assert c["diabetes"].isna().all()                          # ODIR has no systemic diabetes column
    # a BRSET-schema frame has no normal_fundus column: refuse rather than train on everyone
    with pytest.raises(ValueError, match="normal_fundus"):
        build_cohort(df.drop(columns=["normal_fundus"]), healthy="normal")
    # nodm cannot be evaluated on ODIR either
    with pytest.raises(ValueError, match="diabetes"):
        build_cohort(df, healthy="nodm")
