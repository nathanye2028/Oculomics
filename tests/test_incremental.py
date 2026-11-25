"""CPU-only tests for the "does the photo add to age+sex?" comparison
(covariate_baseline.incremental_value), its row alignment and no-DM/DR-0 subset
in train_mbrset.py, and a one-epoch Canon-only cm_hypertension run end to end."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from covariate_baseline import incremental_value


def _frame(n_patients, seed, image_signal):
    """Two eyes per patient. Hypertension depends on age AND a latent the image
    sees (``image_signal`` scales how much); the image probability is that latent
    plus noise, so with image_signal=0 the photo carries nothing beyond age."""
    rng = np.random.default_rng(seed)
    rows, probs = [], []
    for pid in range(n_patients):
        age = rng.uniform(30, 85)
        latent = rng.normal()
        htn = rng.random() < 1 / (1 + np.exp(-((age - 60) / 8 + image_signal * latent)))
        dm = rng.random() < 0.3
        for eye in (0, 1):
            rows.append({"file": f"{pid}_{eye}.jpg", "patient": pid, "age": age,
                         "sex": float(rng.integers(0, 2)), "cm_hypertension": float(htn),
                         "diabetes": "yes" if dm else "No", "final_icdr": 0})
            probs.append(1 / (1 + np.exp(-(latent + rng.normal(0, 0.5)))))
    return pd.DataFrame(rows), np.array(probs)


def test_real_image_signal_adds_beyond_age_sex():
    val, pv = _frame(500, 0, image_signal=1.5)
    test, pt = _frame(500, 1, image_signal=1.5)
    r = incremental_value(val, pv, test, pt, "cm_hypertension", n_boot=300)
    t = r["test"]
    assert r["reason"] is None and t["delta"] > 0.03
    assert t["delta_ci"][0] > 0 and t["p_no_gain"] < 0.05
    assert t["auroc_combined"] > t["auroc_covariates"]


def test_uninformative_image_adds_nothing():
    val, pv = _frame(500, 2, image_signal=0.0)
    test, pt = _frame(500, 3, image_signal=0.0)
    t = incremental_value(val, pv, test, pt, "cm_hypertension", n_boot=300)["test"]
    assert abs(t["delta"]) < 0.02
    assert t["delta_ci"][0] < 0 < t["delta_ci"][1]


def test_subsets_and_refusals():
    val, pv = _frame(300, 4, image_signal=1.0)
    test, pt = _frame(300, 5, image_signal=1.0)
    mask = (test["diabetes"] == "No").to_numpy()
    r = incremental_value(val, pv, test, pt, "cm_hypertension", subsets={"no_dm": mask}, n_boot=100)
    assert r["subsets"]["no_dm"]["n"] == int(mask.sum())
    assert "delta" in r["subsets"]["no_dm"]
    assert "missing" in incremental_value(val.drop(columns=["sex"]), pv, test, pt,
                                          "cm_hypertension")["reason"]
    assert "not binary" in incremental_value(val, pv, test, pt, "dr_grade")["reason"]


def test_rows_for_and_nodm_dr0_mask():
    from train_mbrset import nodm_dr0_mask, rows_for

    class DS:
        files = np.array(["b.jpg", "a.jpg"])
    df = pd.DataFrame({"file": ["a.jpg", "b.jpg", "c.jpg"], "patient": [1, 1, 2],
                       "diabetes": ["No", "No", "yes"], "final_icdr": [0, 2, 0]})
    rows = rows_for(DS, df)
    assert rows["file"].tolist() == ["b.jpg", "a.jpg"]
    # patient 1 is non-diabetic but one eye is grade 2 -> excluded as a patient
    assert nodm_dr0_mask(rows).tolist() == [False, False]
    assert nodm_dr0_mask(df.drop(columns=["diabetes"])) is None
    ok = pd.DataFrame({"patient": [3, 3], "diabetes": ["No", "No"], "final_icdr": [0, 0]})
    assert nodm_dr0_mask(ok).tolist() == [True, True]


def _brset_dir(tmp_path, n_patients=60, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    root = tmp_path / "BRSET"
    (root / "fundus_photos").mkdir(parents=True)
    rows = []
    for pid in range(1, n_patients + 1):
        cam = "NIKON NF5050" if pid % 4 == 0 else "Canon CR"
        age = float(rng.uniform(30, 85))
        htn = age > 58
        text = "NA" if (cam.startswith("NIKON") or pid % 7 == 0) else ("hypertension" if htn else "0")
        for eye in (1, 2):
            iid = f"img{pid:05d}_{eye}"
            shade = 120 + (40 if htn else 0)
            Image.fromarray(np.full((64, 64, 3), shade, np.uint8)).save(root / "fundus_photos" / f"{iid}.jpg")
            rows.append({"image_id": iid, "patient_id": pid, "camera": cam, "patient_age": age,
                         "patient_sex": int(rng.integers(1, 3)), "exam_eye": eye,
                         "comorbidities": text, "diabetes": "No", "DR_ICDR": 0,
                         "macular_edema": 0, "quality": "Adequate", "artifacts": 1})
    pd.DataFrame(rows).to_csv(root / "labels_brset.csv", index=False)
    return root, pd.DataFrame(rows)


def _mbrset_dir(tmp_path, n_patients=30):
    from PIL import Image
    root = tmp_path / "mbrset"
    (root / "images").mkdir(parents=True)
    rows = []
    for pid in range(n_patients):
        for eye in ("1", "2"):
            f = f"{pid}.{eye}.jpg"
            Image.fromarray(np.full((64, 64, 3), 130, np.uint8)).save(root / "images" / f)
            rows.append({"file": f, "patient": pid, "age": 40 + pid, "sex": pid % 2,
                         "systemic_hypertension": float(pid % 3 == 0), "final_icdr": 0})
    pd.DataFrame(rows).to_csv(root / "labels_mbrset.csv", index=False)
    return root


def test_canon_only_cm_hypertension_run_end_to_end(tmp_path, monkeypatch):
    import train_mbrset
    root, raw = _brset_dir(tmp_path)
    mroot = _mbrset_dir(tmp_path)
    out = tmp_path / "res.json"
    monkeypatch.setattr(train_mbrset, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sys, "argv", [
        "train_mbrset.py", "--dataset", "brset", "--root", str(root), "--camera", "Canon CR",
        "--task", "cm_hypertension", "--covariate-baseline",
        "--external-test-root", str(mroot), "--external-test-dataset", "mbrset",
        "--backbone", "mobilenetv3_small", "--no-gcg", "--no-pretrained", "--image-size", "64",
        "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--no-gpu-aug",
        "--ckpt-dir", str(tmp_path / "ck"), "--run-name", "t", "--results-json", str(out)])
    assert train_mbrset.main() == 0
    r = json.loads(out.read_text())
    assert r["camera"] == "Canon CR" and r["task"] == "cm_hypertension"
    canon_recorded = ((raw["camera"] == "Canon CR") & (raw["comorbidities"] != "NA")).sum()
    assert r["n_train"] + r["n_val"] + r["n_test"] == canon_recorded
    assert r["external"] is not None and r["n_external"] == 60       # scored on mBRSET's column
    inc = r["incremental"]
    assert inc["features"] == ["age", "sex"] and inc["n_test"] == r["n_test"]
    assert "no_diabetes_dr0" in inc["subsets"]


def test_summarize_systemic_pairs_the_photo_gain_by_seed(tmp_path):
    from summarize_systemic import summarize
    d = tmp_path / "cm_hypertension"
    d.mkdir()
    for s, (comb, cov, sub) in enumerate([(0.76, 0.73, 0.70), (0.77, 0.73, 0.69), (0.75, 0.72, 0.71)]):
        rec = {"test": {"auroc": 0.70}, "test_pos": 40, "n_test": 100,
               "covariate_baseline": {"auroc": 0.73, "features": ["age", "sex"]},
               "incremental": {"test": {"auroc_combined": comb, "auroc_covariates": cov,
                                        "delta": comb - cov},
                               "subsets": {"no_diabetes_dr0": {"auroc_combined": sub,
                                                               "auroc_covariates": 0.70,
                                                               "delta": sub - 0.70}}}}
        (d / f"ctrl_seed{s}.json").write_text(json.dumps(rec))
    inc = summarize(str(tmp_path), None, control="ctrl")["cm_hypertension"]["paired_incremental"]["ctrl"]
    assert inc["all"]["mean_delta"] == pytest.approx((0.03 + 0.04 + 0.03) / 3)
    assert inc["all"]["significant"]
    assert inc["no_diabetes_dr0"]["n"] == 3 and not inc["no_diabetes_dr0"]["significant"]


def test_cm_task_needs_brset(monkeypatch, tmp_path):
    import train_mbrset
    mroot = _mbrset_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_mbrset.py", "--dataset", "mbrset", "--root", str(mroot),
                                      "--task", "cm_hypertension", "--ckpt-dir", str(tmp_path / "ck")])
    with pytest.raises(SystemExit, match="free-text target"):
        train_mbrset.main()
