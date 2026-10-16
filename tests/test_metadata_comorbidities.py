"""CPU-only synthetic tests for the BRSET comorbidity targets in metadata_model.py:
task registration, the patient join, feature rules (no overlapping group, no
diabetes restatement), the one-camera filter, and an end-to-end Canon-only run
scored on an mBRSET-like external table."""
import json

import numpy as np
import pandas as pd
import pytest

from brset_comorbidities import attach_flags, mbrset_equivalents
from dataset import LABEL_REGISTRY
from metadata_model import filter_camera, features_for, main
from test_metadata_model import _mbrset_like


def _brset_rows(n_patients=400, seed=0):
    """BRSET-schema CSV rows (raw column names, 1/2 sex, BOM-free). Hypertension
    and heart disease rise with age; Nikon patients are mostly unrecorded."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        cam = "NIKON NF5050" if rng.random() < 0.35 else "Canon CR"
        age = float(rng.uniform(25, 90))
        dm = rng.random() < 0.3
        if cam.startswith("NIKON") and rng.random() < 0.95:
            text = "NA"
        else:
            items = []
            if rng.random() < 1 / (1 + np.exp(-(age - 60) / 7)):
                items.append("hypertension")
            if rng.random() < 1 / (1 + np.exp(-(age - 80) / 5)):
                items.append("cardiopathy")
            if dm:
                items.append("diabetes")
            text = " , ".join(items) if items else "0"
        for eye in (1, 2):
            rows.append({"image_id": f"img{pid:05d}_{eye}", "patient_id": pid, "camera": cam,
                         "patient_age": age, "patient_sex": int(rng.integers(1, 3)),
                         "exam_eye": eye, "comorbidities": text,
                         "diabetes": "yes" if dm else "No",
                         "insuline": rng.choice(["yes", "no"]) if dm else "",
                         "diabetes_time_y": float(rng.uniform(0, 20)) if dm else np.nan,
                         "nationality": "Brazil", "DR_ICDR": 0, "macular_edema": 0})
    return pd.DataFrame(rows)


def _write_brset(tmp_path, **kw):
    d = tmp_path / "BRSET" / "physionet.org" / "files" / "brazilian-ophthalmological" / "1.0.1"
    d.mkdir(parents=True)
    _brset_rows(**kw).to_csv(d / "labels_brset.csv", index=False, encoding="utf-8-sig")
    return tmp_path / "BRSET", d / "labels_brset.csv"


def test_sex_is_one_for_male_in_every_adapter(tmp_path):
    """mBRSET's raw 1 = male is the convention; BRSET (1 = male, 2 = female per the
    real CSV's prostate / breast-cancer patients) and ODIR ("Male"/"Female") must
    be recoded to match, or a sex coefficient fit on one dataset reverses on another."""
    from brset_comorbidities import _sex01
    from brset_dataset import _normalise_sex, load_any
    assert _normalise_sex(pd.Series([1, 2], dtype=float)).tolist() == [1.0, 0.0]
    assert _sex01(pd.Series([1.0])) == 1.0 and _sex01(pd.Series([2.0])) == 0.0
    (tmp_path / "preprocessed_images").mkdir()
    pd.DataFrame({"ID": [1, 1, 2, 2], "Patient Age": [60, 60, 50, 50],
                  "Patient Sex": ["Male", "Male", "Female", "Female"],
                  "Left-Diagnostic Keywords": ["normal fundus"] * 4,
                  "Right-Diagnostic Keywords": ["normal fundus"] * 4,
                  "filename": ["1_left.jpg", "1_right.jpg", "2_left.jpg", "2_right.jpg"]}
                 ).to_csv(tmp_path / "full_df.csv", index=False)
    odir = load_any(str(tmp_path), "odir")["df"].set_index("file")
    assert odir.loc["1_left.jpg", "sex"] == 1.0 and odir.loc["2_left.jpg", "sex"] == 0.0


def test_comorbidity_tasks_registered_nan_safe():
    spec = LABEL_REGISTRY["cm_hypertension"]
    assert spec.source_cols == ("cm_hypertension",)
    assert spec.fn({"cm_hypertension": 1.0}) == 1.0
    assert spec.fn({"cm_hypertension": 0.0}) == 0.0
    assert np.isnan(spec.fn({"cm_hypertension": np.nan}))
    assert "cm_brain_any" in LABEL_REGISTRY and "cm_alzheimer" in LABEL_REGISTRY


def test_attach_flags_aligns_rows_and_keeps_unrecorded_nan(tmp_path):
    from brset_dataset import load_brset
    _, csv = _write_brset(tmp_path, n_patients=60)
    df = load_brset(str(csv))                        # patient read as int here
    out = attach_flags(df, str(csv))
    assert len(out) == len(df) and list(out["file"]) == list(df["file"])
    raw = pd.read_csv(csv)
    na = raw["comorbidities"].isna().to_numpy()      # pandas reads "NA" as NaN
    assert out.loc[na, "cm_hypertension"].isna().all()
    assert out.loc[~na, "cm_hypertension"].notna().all()
    # both eyes of a patient carry the same patient-level flag
    assert (out.groupby("patient")["cm_hypertension"].nunique(dropna=False) == 1).all()


def test_attach_flags_refuses_a_bad_join(tmp_path):
    from brset_dataset import load_brset
    _, csv = _write_brset(tmp_path, n_patients=30)
    df = load_brset(str(csv))
    df["patient"] = df["patient"] + 10_000
    with pytest.raises(ValueError, match="matched only"):
        attach_flags(df, str(csv))


def test_feature_rules_for_comorbidity_targets():
    cols = ["age", "sex", "dm_time", "insulin", "diabetes", "nationality",
            "cm_hypertension", "cm_heart_disease", "cm_cardiovascular_event",
            "cm_myocardial_infarction", "cm_stroke", "cm_diabetes", "cm_thyroid"]
    full = features_for("cm_heart_disease", "full", cols)
    assert "cm_hypertension" in full and "cm_thyroid" in full and "cm_stroke" in full
    for banned in ("cm_heart_disease", "cm_cardiovascular_event", "cm_myocardial_infarction",
                   "cm_diabetes"):
        assert banned not in full
    assert not [c for c in features_for("cm_heart_disease", "clinical", cols) if c.startswith("cm_")]
    dm = features_for("cm_diabetes", "clinical", cols)
    assert not {"diabetes", "dm_time", "insulin"} & set(dm)
    # cardiovascular_event contains stroke, so stroke is not a feature for it
    assert "cm_stroke" not in features_for("cm_cardiovascular_event", "full", cols)


def test_filter_camera():
    df = pd.DataFrame({"camera": ["Canon CR", "NIKON NF5050", " Canon CR "], "x": [1, 2, 3]})
    assert filter_camera(df, "Canon CR")["x"].tolist() == [1, 3]
    assert len(filter_camera(df, None)) == 3
    with pytest.raises(ValueError, match="cameras present"):
        filter_camera(df, "Topcon")


def test_mbrset_equivalents_copy_structured_columns():
    ext = mbrset_equivalents(_mbrset_like(n_patients=20))
    assert (ext["cm_hypertension"] == ext["systemic_hypertension"]).all()
    assert "cm_kidney" not in ext.columns            # not equivalent; dropped after the 2026-10-16 run


def test_cli_canon_only_comorbidity_baseline_with_mbrset_external(tmp_path):
    root, _ = _write_brset(tmp_path, n_patients=500, seed=3)
    mroot = tmp_path / "mbrset"
    mroot.mkdir()
    _mbrset_like(n_patients=200).to_csv(mroot / "labels_mbrset.csv", index=False)
    out = tmp_path / "out"
    assert main(["--dataset", "brset", "--root", str(root), "--out", str(out),
                 "--camera", "Canon CR", "--tasks", "cm_hypertension", "cm_heart_disease",
                 "--feature-sets", "age_sex", "full", "--models", "logreg",
                 "--n-boot", "50", "--cv-repeats", "1", "--n-shuffle", "5",
                 "--external-root", str(mroot), "--external-dataset", "mbrset"]) == 0
    table = pd.read_csv(out / "metadata_results.csv").set_index(["task", "feature_set"])
    htn = table.loc[("cm_hypertension", "age_sex")]
    assert htn["test_auroc"] > 0.75                  # planted: hypertension rises with age
    assert not np.isnan(htn["ext_auroc"])            # scored on mBRSET's own labels
    assert np.isnan(table.loc[("cm_heart_disease", "age_sex"), "ext_auroc"])   # no mBRSET equivalent
    blob = json.loads((out / "metadata_results.json").read_text())
    assert blob["args"]["camera"] == "Canon CR"
    full = next(r for r in blob["results"]
                if r["task"] == "cm_heart_disease" and r["feature_set"] == "full")
    assert "cm_hypertension" in full["features"] and "cm_cardiovascular_event" not in full["features"]
    # only Canon rows of recorded patients were used
    n_used = full["n_train"] + full["n_test"]
    raw = _brset_rows(n_patients=500, seed=3)
    canon_recorded = ((raw["camera"] == "Canon CR") & (raw["comorbidities"] != "NA")).sum()
    assert n_used == canon_recorded
