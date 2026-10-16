"""CPU-only synthetic tests for brset_comorbidities.py: the vocabulary on the
strings seen in the real CSV, missing-vs-none, patient consolidation, the
diabetes cross-check orientation, and an audit that must flag a planted camera
confound (and must not flag an age-only signal)."""
import numpy as np
import pandas as pd
import pytest

from brset_comorbidities import (GROUPS, audit, logit_or, main, parse_comorbidities,
                                 patient_flags, read_brset_csv, render)


@pytest.mark.parametrize("text, expect", [
    ("hypertension", {"hypertension"}),
    ("hypertension , hypothyroidism", {"hypertension", "hypothyroidism"}),
    ("diabetes1, hypertension", {"diabetes", "diabetes_type1", "hypertension"}),
    ("diabetes, hypertension , cardiopathy", {"diabetes", "hypertension", "cardiopathy"}),
    ("acute myocardium infarct", {"myocardial_infarction"}),
    ("encephalic vascular accident", {"stroke"}),
    ("cardiac insufficiency", {"heart_failure"}),
    ("arrythmia", {"arrhythmia"}),
    ("chronic kidney disease", {"kidney_disease"}),
    ("kidney transplant", {"kidney_transplant"}),
    ("alzheimer", {"alzheimer"}),
    ("multiple sclerosis", {"multiple_sclerosis"}),
    ("hydroxichloroquine", {"antimalarial_use"}),
    ("rheumatoid arthritis", {"rheumatoid_arthritis"}),
    ("sickle cell anemia", {"sickle_cell"}),
    ("hypophysis adenoma", {"pituitary_adenoma"}),
    ("chron disease", {"crohn"}),
    ("tabagism", {"smoking"}),
    ("ocular hypertension", {"ocular_hypertension"}),
    ("glaucoma", {"ocular_condition"}),
    # first real audit's unmapped list (2026-10-15)
    ("parkison", {"parkinson"}),
    ("hipoitireoidismo", {"hypothyroidism"}),
    ("intracranial hypertension", {"intracranial_hypertension"}),
    ("meningioma", {"brain_tumor"}),
    ("prolactinoma", {"pituitary_adenoma"}),
    ("leucemia", {"cancer"}),
    ("devic", {"neuromyelitis_optica"}),
    ("herpetic encephalitis", {"encephalitis"}),
    ("hepatic transplant", {"liver_disease"}),
    ("hepatitis c", {"chronic_infection"}),
    ("deep vascular thrombosis", {"thrombosis"}),
    ("thrombocytopenia", {"blood_disorder"}),
    ("catheterism", {"cardiopathy"}),
    ("dpoc", {"asthma_copd"}),
    ("chronic obstructive pulmonary disease", {"asthma_copd"}),
    ("hipocolesterolemia", {"dyslipidemia"}),
    ("muscular dystrophy", {"genetic_syndrome"}),
    ("cone dystrophy", {"ocular_condition"}),
    ("neurofibromatosis", {"genetic_syndrome"}),
])
def test_vocabulary_on_real_strings(text, expect):
    atoms, unmapped = parse_comorbidities(text)
    assert atoms == expect
    assert unmapped == []


@pytest.mark.parametrize("text", ["na", "NA", "", None, float("nan"), " N/A "])
def test_missing_is_not_none(text):
    assert parse_comorbidities(text) == (None, [])


def test_zero_means_recorded_none():
    assert parse_comorbidities("0") == (set(), [])


def test_unmapped_items_are_returned():
    atoms, unmapped = parse_comorbidities("hypertension, zebra fever")
    assert atoms == {"hypertension"}
    assert unmapped == ["zebra fever"]


def test_patient_has_with_diabetes_is_not_hypertension():
    # The Portuguese acronym HAS is NOT matched: "has" is an English word too.
    atoms, _ = parse_comorbidities("patient has diabetes")
    assert "hypertension" not in atoms


def _rows(pid, cam, age, sex, dm, texts, dr=0):
    return [{"image_id": f"img{pid}_{i}", "patient_id": str(pid), "camera": cam,
             "patient_age": str(age), "patient_sex": str(sex), "diabetes": dm,
             "insuline": "", "DR_ICDR": str(dr), "comorbidities": t}
            for i, t in enumerate(texts)]


def test_patient_consolidation_keeps_unrecorded_nan():
    raw = pd.DataFrame(
        _rows(1, "Canon CR", 60, 1, "No", ["na", "hypertension , hypothyroidism"])
        + _rows(2, "Canon CR", 50, 2, "No", ["na", "na"])
        + _rows(3, "NIKON NF5050", 40, 2, "No", ["0", "0"])
        + _rows(4, "Canon CR", 70, 1, "yes", ["diabetes", "diabetes, hypertension"]))
    pf, unmapped, stats = patient_flags(raw)
    pf = pf.set_index("patient_id")
    assert pf.loc["1", "recorded"] == 1 and pf.loc["1", "cm_hypertension"] == 1
    assert pf.loc["1", "cm_heart_disease"] == 0
    assert np.isnan(pf.loc["2", "cm_hypertension"]) and pf.loc["2", "recorded"] == 0
    assert pf.loc["3", "cm_hypertension"] == 0 and pf.loc["3", "recorded"] == 1
    assert pf.loc["4", "cm_hypertension"] == 1 and pf.loc["4", "cm_diabetes"] == 1
    assert stats["inconsistent_history"] == 1          # patient 4 only; "na" is not a history
    assert pf.loc["1", "sex"] == 0 and pf.loc["2", "sex"] == 1
    assert pf.loc["4", "diabetes_col"] == 1 and pf.loc["1", "diabetes_col"] == 0
    assert not unmapped


def test_bom_header_is_stripped(tmp_path):
    p = tmp_path / "labels_brset.csv"
    pd.DataFrame(_rows(1, "Canon CR", 60, 1, "No", ["NA"])).to_csv(p, index=False,
                                                                   encoding="utf-8-sig")
    raw = read_brset_csv(str(p))
    assert raw.columns[0] == "image_id"
    assert raw.loc[0, "comorbidities"] == "NA"          # literal, not pandas NaN
    assert patient_flags(raw)[0].loc[0, "recorded"] == 0


def test_logit_or_recovers_a_known_effect():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 6000).astype(float)
    z = rng.normal(size=6000)
    y = (rng.random(6000) < 1 / (1 + np.exp(-(-1 + 0.7 * x + 0.3 * z)))).astype(float)
    fit = logit_or(y, np.column_stack([x, z]), ["x", "z"])
    assert fit["x"]["lo"] < np.exp(0.7) < fit["x"]["hi"]
    assert fit["x"]["stable"]


def _synthetic_brset(n=1500, seed=1):
    """Canon is the majority camera. Nikon patients are mostly unrecorded;
    hypertension depends on age only; dyslipidemia depends on camera."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        cam = "NIKON NF5050" if rng.random() < 0.35 else "Canon CR"
        age = float(rng.uniform(20, 90))
        sex = int(rng.integers(1, 3))
        dm = "yes" if rng.random() < 0.2 else "No"
        p_rec = 0.2 if cam.startswith("NIKON") else 0.8
        if rng.random() > p_rec:
            text = "na"
        else:
            items = []
            if rng.random() < 1 / (1 + np.exp(-(age - 60) / 8)):
                items.append("hypertension")
            if rng.random() < (0.30 if cam.startswith("NIKON") else 0.05):
                items.append("dyslipidemia")
            if dm == "yes" and rng.random() < 0.7:
                items.append("diabetes")
            text = " , ".join(items) if items else "0"
        rows += _rows(pid, cam, age, sex, dm, [text, text])
    return pd.DataFrame(rows)


def test_audit_flags_planted_camera_effects_only():
    pf, unmapped, stats = patient_flags(_synthetic_brset())
    res = audit(pf, unmapped, stats)
    miss = res["missingness_adjusted"]["or"]["cam_other"]
    assert res["missingness_adjusted"]["contrast"] == "NIKON NF5050 vs Canon CR"
    assert miss["hi"] < 1                                # Nikon far less often recorded
    assert res["targets"]["dyslipidemia"]["camera_flag"] is True
    assert res["targets"]["hypertension"]["camera_flag"] is False
    assert res["targets"]["hypertension"]["age_pos"] > res["targets"]["hypertension"]["age_neg"]
    assert set(res["targets"]) == set(GROUPS)


def test_diabetes_crosscheck_orientation():
    raw = pd.DataFrame(
        sum([_rows(f"a{i}", "Canon CR", 60, 1, "yes", ["diabetes"]) for i in range(3)], [])
        + sum([_rows(f"b{i}", "Canon CR", 60, 1, "yes", ["hypertension"]) for i in range(2)], [])
        + sum([_rows(f"c{i}", "Canon CR", 60, 1, "No", ["diabetes"]) for i in range(1)], [])
        + sum([_rows(f"d{i}", "Canon CR", 60, 1, "No", ["0"]) for i in range(4)], []))
    pf, unmapped, stats = patient_flags(raw)
    dc = audit(pf, unmapped, stats)["diabetes_crosscheck"]
    assert dc == {"n": 10, "text_yes_col_yes": 3, "text_yes_col_no": 1,
                  "text_no_col_yes": 2, "text_no_col_no": 4}
    report = render(audit(pf, unmapped, stats), "x.csv")
    assert "| text lists diabetes | 3 | 1 |" in report
    assert "lists diabetes for 60%" in report           # 3 of the 5 column-diabetics


def test_brain_any_groups_brain_conditions_but_not_migraine():
    raw = pd.DataFrame(
        _rows(1, "Canon CR", 80, 1, "No", ["parkison"])
        + _rows(2, "Canon CR", 70, 2, "No", ["encephalic vascular accident"])
        + _rows(3, "Canon CR", 40, 2, "No", ["migraine"]))
    pf = patient_flags(raw)[0].set_index("patient_id")
    assert list(pf["cm_brain_any"]) == [1, 1, 0]
    assert list(pf["cm_neurodegenerative"]) == [1, 0, 0]


def test_perfect_diabetes_agreement_is_not_called_a_completeness_check():
    raw = pd.DataFrame(
        sum([_rows(f"a{i}", "Canon CR", 60, 1, "yes", ["diabetes"]) for i in range(3)], [])
        + sum([_rows(f"d{i}", "Canon CR", 60, 1, "No", ["0"]) for i in range(4)], []))
    pf, unmapped, stats = patient_flags(raw)
    report = render(audit(pf, unmapped, stats), "x.csv")
    assert "NOT an independent check" in report
    assert "false negatives" not in report


def test_one_camera_target_is_unstable_without_overflow_warnings():
    """A condition seen on only one camera separates perfectly: the fit must be
    marked unstable, never flagged, and must not emit overflow warnings."""
    raw = _synthetic_brset(n=1500)
    stroke_rows = raw.index[(raw["camera"] == "Canon CR") & (raw["comorbidities"] == "0")][:40]
    raw.loc[stroke_rows, "comorbidities"] = "stroke"
    pf, unmapped, stats = patient_flags(raw)
    with np.errstate(over="raise", invalid="raise"):
        res = audit(pf, unmapped, stats)
    stroke = res["targets"]["stroke"]
    assert stroke["camera_or"]["stable"] is False and stroke["camera_flag"] is False
    assert "n/a (no cases on one camera)" in render(res, "x.csv")


def test_main_writes_outputs(tmp_path):
    raw = _synthetic_brset(n=300)
    raw.loc[0, "comorbidities"] = "zebra fever"
    d = tmp_path / "BRSET" / "physionet.org" / "files" / "brazilian-ophthalmological" / "1.0.1"
    d.mkdir(parents=True)
    raw.to_csv(d / "labels_brset.csv", index=False, encoding="utf-8-sig")
    out = tmp_path / "out"
    assert main(["--root", str(tmp_path / "BRSET"), "--out", str(out)]) == 0
    flags = pd.read_csv(out / "brset_comorbidity_flags.csv")
    assert len(flags) == 300 and "cm_hypertension" in flags.columns
    assert "zebra fever" in (out / "comorbidity_audit.md").read_text()
    assert (out / "comorbidity_audit.json").exists()
