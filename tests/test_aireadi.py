"""Recommendation 3: aireadi_dataset.py converts colour fundus DICOMs (YBR -> RGB,
IR skipped, Aurora = handheld), builds the measured targets from OMOP tables
(refusing an ambiguous lab name), and the converted folder works as
--dataset aireadi for the regression probe and for scoring BRSET models."""
import json

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")


def _dicom(path, pid, manufacturer, model, lat, rgb=(180, 60, 30), photometric="RGB", samples=3):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    h, w = 40, 48
    if samples == 3:
        arr = np.zeros((h, w, 3), np.uint8)
        arr[6:34, 8:40] = rgb
        if photometric == "YBR_FULL":
            from pydicom.pixel_data_handlers.util import convert_color_space
            arr = convert_color_space(arr, "RGB", "YBR_FULL")
    else:
        arr = np.full((h, w), 90, np.uint8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.Manufacturer, ds.ManufacturerModelName, ds.ImageLaterality = pid, manufacturer, model, lat
    ds.Rows, ds.Columns, ds.SamplesPerPixel = h, w, samples
    ds.PhotometricInterpretation = photometric if samples == 3 else "MONOCHROME2"
    if samples == 3:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = arr.tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


def _download(tmp_path, n=80, seed=0, ambiguous=False):
    rng = np.random.default_rng(seed)
    root = tmp_path / "fairhub"
    parts, meas, obs = [], [], []
    for i in range(n):
        pid = str(1000 + i)
        age = float(rng.uniform(40, 80))
        group = ["healthy", "pre_diabetes_lifestyle_controlled", "oral_medication_and_or_non_insulin_injectable_medication_controlled",
                 "insulin_dependent"][i % 4]
        parts.append({"participant_id": pid, "age": age, "study_group": group, "clinical_site": "UW",
                      "recommended_split": "train"})
        sbp = 100 + 0.5 * age + rng.normal(0, 10)
        for v in (sbp - 2, sbp + 2):
            meas.append({"person_id": pid, "measurement_source_value": "Systolic (mmHg)", "value_as_number": v,
                         "unit_source_value": "mmHg"})
        meas.append({"person_id": pid, "measurement_source_value": "Diastolic (mmHg)",
                     "value_as_number": 75 + rng.normal(0, 5), "unit_source_value": "mmHg"})
        meas.append({"person_id": pid, "measurement_source_value": "Urine Albumin (mg/dL)",
                     "value_as_number": 3.0 if i % 5 == 0 else 1.0, "unit_source_value": "mg/dL"})
        meas.append({"person_id": pid, "measurement_source_value": "Urine creatinine (mg/dL)",
                     "value_as_number": 100.0, "unit_source_value": "mg/dL"})
        obs.append({"person_id": pid, "observation_source_value": "Total Score", "value_as_number": 24 + i % 6,
                    "observation_concept_id": 37174522})
        for k, (man, model) in enumerate((("Optomed", "Aurora"), ("iCare", "Eidon"))):
            for lat in ("L", "R"):
                _dicom(root / "retinal_photography" / "cfp" / model.lower() / pid / f"{k}_{lat}.dcm", pid, man,
                       model, lat, photometric="YBR_FULL" if k == 1 else "RGB")
        _dicom(root / "retinal_photography" / "ir" / "eidon" / pid / "ir.dcm", pid, "iCare", "Eidon", "L", samples=1)
    if ambiguous:
        meas.append({"person_id": "1000", "measurement_source_value": "Albumin urine dipstick",
                     "value_as_number": 1.0, "unit_source_value": "mg/dL"})
    (root / "clinical_data").mkdir(parents=True)
    pd.DataFrame(parts).to_csv(root / "participants.tsv", sep="\t", index=False)
    pd.DataFrame(meas).to_csv(root / "clinical_data" / "measurement.csv", index=False)
    pd.DataFrame(obs).to_csv(root / "clinical_data" / "observation.csv", index=False)
    return root


def test_convert_and_targets(tmp_path, capsys):
    import aireadi_dataset as ar
    from brset_dataset import load_any
    root, out = _download(tmp_path, n=12), tmp_path / "AIREADI"
    assert ar.main(["inspect", "--root", str(root)]) == 0
    assert "Systolic (mmHg)" in capsys.readouterr().out
    assert ar.main(["convert", "--root", str(root), "--out", str(out), "--workers", "1"]) == 0
    lab = pd.read_csv(out / "labels_aireadi.csv")
    assert len(lab) == 12 * 4                                          # IR skipped
    assert set(lab["device_class"]) == {"handheld", "tabletop"}
    assert set(lab.loc[lab["device_class"] == "handheld", "camera"]) == {"Optomed Aurora"}
    from PIL import Image
    eidon = lab[lab["camera"] == "iCare Eidon"]["file"].iloc[0]          # stored as YBR_FULL
    px = np.asarray(Image.open(out / "images" / eidon).convert("RGB"))[20, 24].astype(int)
    assert np.abs(px - np.array([180, 60, 30])).max() < 12             # came back as RGB
    assert ar.main(["targets", "--root", str(root), "--out", str(out)]) == 0
    lab = pd.read_csv(out / "labels_aireadi.csv")
    per = lab.drop_duplicates("patient").set_index("patient")
    assert per.loc[1000, "uacr"] == pytest.approx(30.0) and per.loc[1000, "albuminuria"] == 1
    assert per.loc[1001, "albuminuria"] == 0 and per["moca"].notna().all()
    assert set(per["diabetes_stage"]) == {0, 1, 2, 3} and set(per["diabetes"]) == {0, 1}
    assert per["htn_measured"].isin([0, 1]).all() and np.allclose(per["log_uacr"], np.log(per["uacr"]))
    assert load_any(str(out), "aireadi")["images_dir"].endswith("images")
    # convert again: targets survive, nothing re-encoded
    assert ar.main(["convert", "--root", str(root), "--out", str(out), "--workers", "1"]) == 0
    assert "sbp" in pd.read_csv(out / "labels_aireadi.csv").columns
    # a generic "Total Score" without the MoCA concept could be another questionnaire's
    o = pd.read_csv(root / "clinical_data" / "observation.csv")
    o.drop(columns=["observation_concept_id"]).to_csv(root / "clinical_data" / "observation.csv", index=False)
    with pytest.raises(SystemExit, match="another questionnaire"):
        ar.main(["targets", "--root", str(root), "--out", str(out)])
    amb = _download(tmp_path / "b", n=4, ambiguous=True)
    ar.main(["convert", "--root", str(amb), "--out", str(tmp_path / "B"), "--workers", "1"])
    with pytest.raises(SystemExit, match="Pin one"):
        ar.main(["targets", "--root", str(amb), "--out", str(tmp_path / "B")])


def _features(out, signal_col, seed=0, d=16):
    lab = pd.read_csv(out / "labels_aireadi.csv")
    rng = np.random.default_rng(seed)
    per = lab.drop_duplicates("patient").set_index("patient")[signal_col]
    z = (per - per.mean()) / per.std()
    feats = np.stack([np.r_[2.0 * z[p], rng.normal(size=d - 1)] for p in lab["patient"]]).astype(np.float32)
    path = out / "f.npz"
    np.savez(path, files=np.asarray(lab["file"].astype(str).tolist(), dtype=str), feats=feats,
             meta=json.dumps({"weights": "fake", "pool": "cls", "root": str(out), "weights_sha256": "x"}))
    return path


def test_regression_probe_and_external_score(tmp_path):
    import aireadi_dataset as ar
    import probe_systemic
    import score_external
    from retfound import save_probe_head
    root, out = _download(tmp_path, n=90), tmp_path / "AIREADI"
    ar.main(["convert", "--root", str(root), "--out", str(out), "--workers", "1"])
    ar.main(["targets", "--root", str(root), "--out", str(out)])
    feats = _features(out, "sbp")
    res = tmp_path / "exp"
    assert probe_systemic.main(["--dataset", "aireadi", "--root", str(out), "--task", "sbp", "--features",
                                str(feats), "--out", str(res), "--n-folds", "3", "--covariate-features", "age",
                                "diabetes_stage", "--n-boot", "50"]) == 0
    rec = json.loads((res / "sbp" / "probe_fold0_seed0.json").read_text())
    assert rec["probe"]["kind"] == "regressor" and rec["test"]["r"] > 0.5
    assert (res / "sbp" / "summary_regression_probe.md").exists()
    pooled = probe_systemic.summarize_regression(str(res / "sbp"), "probe", n_boot=50)
    assert pooled[0]["all"]["r2_combined"] > pooled[0]["all"]["r2_covariates"]
    # a BRSET hypertension probe scored against AI-READI's measured flag, by device
    w = np.zeros(16); w[0] = 1.5
    save_probe_head(str(tmp_path / "p_fold0_seed0_probe.npz"), w, 0.0,
                    {"kind": "classifier", "task": "cm_hypertension", "features_meta": {"pool": "cls",
                                                                                         "weights_sha256": "x"}})
    assert score_external.main(["--task", "cm_hypertension", "--dataset", "aireadi", "--root", str(out),
                                "--probes", str(tmp_path / "p_fold*_probe.npz"), "--features", str(feats),
                                "--out", str(tmp_path / "ext"), "--n-boot", "50"]) == 0
    s = json.loads((tmp_path / "ext" / "summary.json").read_text())["results"]
    assert {"all", "device_class=handheld", "device_class=tabletop"} <= set(s)
    assert s["all"]["probe:p_fold0_seed0"]["auroc"] > 0.7 and "context:age_alone" in s["all"]


def test_target_coding_edge_cases(tmp_path):
    """Below-detection albumin counts as low, albumin 0 is albuminuria 0, a half-missing
    BP pair is a flag only when the known reading is high, MoCA pinned in observation.csv,
    the diabetes-stage indicators, and a score of BRSET models refused on their own data."""
    import aireadi_dataset as ar
    import score_external
    from retfound import save_probe_head
    root, out = _download(tmp_path, n=8), tmp_path / "AIREADI"
    M = pd.read_csv(root / "clinical_data" / "measurement.csv", dtype={"person_id": str})
    alb = M["measurement_source_value"] == "Urine Albumin (mg/dL)"
    M.loc[alb & (M["person_id"] == "1000"), "value_as_number"] = np.nan
    M["value_source_value"] = ""
    M.loc[alb & (M["person_id"] == "1000"), "value_source_value"] = "<0.4"
    M.loc[alb & (M["person_id"] == "1001"), "value_as_number"] = 0.0
    dia = M["measurement_source_value"] == "Diastolic (mmHg)"
    M = M[~(dia & M["person_id"].isin(["1002", "1003"]))]
    sys_ = M["measurement_source_value"] == "Systolic (mmHg)"
    M.loc[sys_ & (M["person_id"] == "1002"), "value_as_number"] = 120.0
    M.loc[sys_ & (M["person_id"] == "1003"), "value_as_number"] = 140.0
    M.to_csv(root / "clinical_data" / "measurement.csv", index=False)
    ar.main(["convert", "--root", str(root), "--out", str(out), "--workers", "1"])
    assert ar.main(["targets", "--root", str(root), "--out", str(out), "--moca-source", "Total Score"]) == 0
    per = pd.read_csv(out / "labels_aireadi.csv").drop_duplicates("patient").set_index("patient")
    assert per.loc[1000, "uacr"] == pytest.approx(0.2 / 100 * 1000) and per.loc[1000, "albuminuria"] == 0
    assert per.loc[1001, "uacr"] == 0 and per.loc[1001, "albuminuria"] == 0 and np.isfinite(per.loc[1001, "log_uacr"])
    assert np.isnan(per.loc[1002, "htn_measured"]) and per.loc[1003, "htn_measured"] == 1
    assert per["moca"].notna().all()
    assert {"dm_pre", "dm_oral", "dm_insulin"} <= set(per.columns)
    assert (per[["dm_pre", "dm_oral", "dm_insulin"]].sum(axis=1) == (per["diabetes_stage"] > 0)).all()
    save_probe_head(str(tmp_path / "own_fold0_seed0_probe.npz"), np.zeros(16), 0.0,
                    {"kind": "classifier", "task": "htn_measured", "dataset": "aireadi"})
    with pytest.raises(SystemExit, match="trained on --dataset aireadi"):
        score_external.main(["--task", "htn_measured", "--dataset", "aireadi", "--root", str(out),
                             "--probes", str(tmp_path / "own_fold*_probe.npz"), "--features", str(tmp_path / "x.npz"),
                             "--out", str(tmp_path / "e")])
