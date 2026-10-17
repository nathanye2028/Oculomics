"""The 5-fold design of the systemic runs: dataset.fold_split tests every labelled
patient exactly once; train_mbrset.normal_eye_masks decides the before-lesions
subsets per patient (unknown is never clear); covariate_baseline.incremental_rows
writes the scores incremental_value scores; summarize_systemic pools fold runs out
of fold; and one --fold run end to end writes its test rows."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from covariate_baseline import incremental_rows, incremental_value
from dataset import fold_split


def _table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        htn = float(rng.random() < 0.6) if pid % 9 else np.nan        # every 9th: no history
        for eye in (1, 2):
            rows.append({"file": f"{pid}_{eye}.jpg", "patient": pid, "cm_hypertension": htn})
    return pd.DataFrame(rows)


def test_fold_split_tests_every_labelled_patient_once():
    df = _table()
    labelled = set(df.loc[df["cm_hypertension"].notna(), "patient"])
    tested = []
    for k in range(5):
        sp = fold_split(df, "cm_hypertension", fold=k, n_folds=5, seed=42)
        tr, va, te = (set(sp[p]["patient"]) for p in ("train", "val", "test"))
        assert not (tr & va or tr & te or va & te)                     # patient-grouped
        assert tr | va | te == labelled                                # no history: out
        assert 0.06 < len(va) / len(labelled) < 0.14
        tested += sorted(te)
    assert sorted(tested) == sorted(labelled)                          # each tested exactly once
    again = fold_split(df, "cm_hypertension", fold=2, seed=42)["test"]
    assert again.equals(fold_split(df, "cm_hypertension", fold=2, seed=42)["test"])
    assert not again.equals(fold_split(df, "cm_hypertension", fold=2, seed=7)["test"])
    with pytest.raises(ValueError, match="outside"):
        fold_split(df, "cm_hypertension", fold=5, n_folds=5)


def test_normal_eye_masks_are_per_patient_and_unknown_is_not_clear():
    from brset_dataset import BRSET_EYE_FLAGS
    from train_mbrset import normal_eye_masks
    clear = {f"flag_{c}": 0 for c in BRSET_EYE_FLAGS}
    eye = lambda pid, dm, **kw: {**clear, "patient": pid, "final_icdr": 0, "final_edema": 0,
                                 "diabetes": dm, **kw}
    rows = pd.DataFrame([
        eye(1, "no"), eye(1, "no"),                                        # clear
        eye(2, "yes", flag_hypertensive_retinopathy=1), eye(2, "yes"),     # one eye HR: out
        eye(3, "yes", flag_drusens=1), eye(3, "yes"),                      # normal-looking, not all clear
        eye(4, "no", final_icdr=1), eye(4, "no"),                          # DR grade 1 in one eye
        eye(5, "no", flag_hemorrhage=np.nan), eye(5, "no"),                # unreadable flag
        eye(6, "no", final_edema=1), eye(6, "no"),                         # edema
    ])
    m = normal_eye_masks(rows)
    pats = lambda k: sorted(rows.loc[m[k], "patient"].unique().tolist())
    assert pats("normal_looking") == [1, 3]
    assert pats("normal_looking_no_dm") == [1]
    assert pats("all_flags_clear") == [1]
    # without one non-vascular flag only the strict subset becomes undecidable
    assert set(normal_eye_masks(rows.drop(columns=["flag_nevus"]))) == {"normal_looking",
                                                                       "normal_looking_no_dm"}
    assert normal_eye_masks(rows.drop(columns=["flag_hemorrhage", "flag_nevus"])) == {}


def _frame(n_patients, seed, signal=1.5):
    rng = np.random.default_rng(seed)
    rows, probs = [], []
    for pid in range(n_patients):
        age, latent = rng.uniform(30, 85), rng.normal()
        htn = rng.random() < 1 / (1 + np.exp(-((age - 60) / 8 + signal * latent)))
        for eye in (0, 1):
            rows.append({"file": f"{pid}_{eye}.jpg", "patient": pid, "age": age,
                         "sex": float(rng.integers(0, 2)), "cm_hypertension": float(htn)})
            probs.append(1 / (1 + np.exp(-(latent + rng.normal(0, 0.5)))))
    return pd.DataFrame(rows), np.array(probs)


def test_incremental_rows_score_what_incremental_value_scores():
    from sklearn.metrics import roc_auc_score
    val, pv = _frame(300, 0)
    test, pt = _frame(300, 1)
    r = incremental_value(val, pv, test, pt, "cm_hypertension", n_boot=20)["test"]
    rows = incremental_rows(val, pv, test, pt, "cm_hypertension")
    y = test["cm_hypertension"].to_numpy()
    assert roc_auc_score(y, rows["p_combined"]) == pytest.approx(r["auroc_combined"])
    assert roc_auc_score(y, rows["p_covariates"]) == pytest.approx(r["auroc_covariates"])
    gap = test.copy()
    gap.loc[0, "age"] = np.nan                                          # a missing covariate
    rows2 = incremental_rows(val, pv, gap, pt, "cm_hypertension")
    assert np.isnan(rows2.loc[0, "p_combined"]) and rows2["p_combined"].notna().sum() == len(test) - 1
    assert incremental_rows(val, pv, test, pt, "dr_grade")["p_combined"].isna().all()


def _write_fold_runs(root, n_folds=5, seed=0, skip=None, reuse_patient=False):
    """Five fold runs of a model whose image score carries signal beyond age."""
    d = root / "cm_hypertension"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pid = 0
    for k in range(n_folds):
        rows = []
        for _ in range(80):
            age, lat = rng.uniform(30, 85), rng.normal()
            y = float(rng.random() < 1 / (1 + np.exp(-((age - 60) / 8 + 1.5 * lat))))
            z = lat + rng.normal(0, 0.5)
            p_img, p_cov = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-(age - 60) / 8))
            p_comb = 1 / (1 + np.exp(-((age - 60) / 8 + 1.5 * z)))
            for e in (1, 2):
                rows.append({"file": f"{pid}_{e}.jpg", "patient": str(pid), "label": y,
                             "p_image": p_img, "p_covariates": p_cov, "p_combined": p_comb,
                             "in_normal_looking": pid % 2 == 0, "task": "cm_hypertension",
                             "fold": k, "seed": seed})
            pid += 1
        if reuse_patient and k == 1:
            rows[0]["patient"] = "0"                                    # patient 0 again, in fold 1
        if k == skip:
            continue
        pd.DataFrame(rows).to_csv(d / f"ctrl_fold{k}_seed{seed}_test.csv", index=False)
        (d / f"ctrl_fold{k}_seed{seed}.json").write_text(json.dumps(
            {"task": "cm_hypertension", "fold": k, "n_folds": n_folds, "fold_seed": 42, "seed": seed,
             "predictions_csv": "/box/path/that/is/not/here.csv",      # the sibling file is used
             "incremental": {"features": ["age", "sex"]}, "test": {"auroc": 0.7}}))


def test_summarize_pools_fold_runs_out_of_fold(tmp_path):
    from summarize_systemic import summarize
    _write_fold_runs(tmp_path)
    rep = summarize(str(tmp_path), ["cm_hypertension"], control="ctrl", n_boot=200)
    p = rep["cm_hypertension"]["pooled_out_of_fold"]["seeds"][0]
    assert p["patients"] == 400 and p["n_folds"] == 5
    allb, sub = p["blocks"]["all"], p["blocks"]["normal_looking"]
    assert allb["delta"] > 0.02 and allb["delta_ci"][0] > 0
    assert sub["n"] == allb["n"] // 2
    md = (tmp_path / "summary.md").read_text()
    assert "POOLED OUT OF FOLD" in md and "[normal_looking" in md


def test_pooling_refuses_incomplete_or_overlapping_folds(tmp_path):
    from summarize_systemic import summarize
    _write_fold_runs(tmp_path / "a", skip=3)
    rep = summarize(str(tmp_path / "a"), ["cm_hypertension"], control="ctrl", n_boot=20)
    pooled = rep["cm_hypertension"]["pooled_out_of_fold"]
    assert pooled["seeds"] == {} and pooled["incomplete"][0] == [0, 1, 2, 4]
    _write_fold_runs(tmp_path / "b", reuse_patient=True)
    with pytest.raises(SystemExit, match="two folds"):
        summarize(str(tmp_path / "b"), ["cm_hypertension"], control="ctrl", n_boot=20)


def _brset_with_flags(tmp_path, n_patients=120, seed=0):
    from PIL import Image
    from brset_dataset import BRSET_EYE_FLAGS
    rng = np.random.default_rng(seed)
    root = tmp_path / "BRSET"
    (root / "fundus_photos").mkdir(parents=True)
    rows = []
    for pid in range(1, n_patients + 1):
        cam = "NIKON NF5050" if pid % 4 == 0 else "Canon CR"
        age = float(rng.uniform(30, 85))
        htn = age > 58
        text = "NA" if (cam.startswith("NIKON") or pid % 7 == 0) else ("hypertension" if htn else "0")
        for e in (1, 2):
            iid = f"img{pid:05d}_{e}"
            Image.fromarray(np.full((64, 64, 3), 120 + (40 if htn else 0), np.uint8)
                            ).save(root / "fundus_photos" / f"{iid}.jpg")
            r = {"image_id": iid, "patient_id": pid, "camera": cam, "patient_age": age,
                 "patient_sex": int(rng.integers(1, 3)), "exam_eye": e, "comorbidities": text,
                 "diabetes": "No", "DR_ICDR": 0, "macular_edema": 0, "quality": "Adequate",
                 "artifacts": 1}
            r.update({c: int(c == "hypertensive_retinopathy" and pid % 5 == 0 and e == 1)
                      for c in BRSET_EYE_FLAGS})
            rows.append(r)
    pd.DataFrame(rows).to_csv(root / "labels_brset.csv", index=False)
    return root


def test_one_fold_run_writes_its_test_rows(tmp_path, monkeypatch):
    import train_mbrset
    root = _brset_with_flags(tmp_path)
    out = tmp_path / "res" / "ctrl_fold1_seed0.json"
    monkeypatch.setattr(train_mbrset, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sys, "argv", [
        "train_mbrset.py", "--dataset", "brset", "--root", str(root), "--camera", "Canon CR",
        "--task", "cm_hypertension", "--covariate-baseline", "--fold", "1", "--n-folds", "3",
        "--backbone", "mobilenetv3_small", "--no-gcg", "--no-pretrained", "--image-size", "64",
        "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--no-gpu-aug",
        "--ckpt-dir", str(tmp_path / "ck"), "--run-name", "ctrl_fold1_seed0", "--results-json", str(out)])
    assert train_mbrset.main() == 0
    r = json.loads(out.read_text())
    assert r["fold"] == 1 and r["n_folds"] == 3 and r["fold_seed"] == 42
    pr = pd.read_csv(tmp_path / "res" / "ctrl_fold1_seed0_test.csv")
    assert r["predictions_csv"].endswith("ctrl_fold1_seed0_test.csv") and len(pr) == r["n_test"]
    assert {"label", "p_image", "p_covariates", "p_combined", "in_no_diabetes_dr0",
            "in_normal_looking", "in_normal_looking_no_dm", "in_all_flags_clear"} <= set(pr.columns)
    assert (pr["fold"] == 1).all()
    # an eye flagged with hypertensive retinopathy takes its patient out of normal_looking
    hr = {f"{p}" for p in range(1, 121) if p % 5 == 0}
    assert pr["patient"].astype(str).isin(hr).any() and pr["in_normal_looking"].any()
    assert not pr.loc[pr["in_normal_looking"], "patient"].astype(str).isin(hr).any()
    assert set(r["incremental"]["subsets"]) >= {"normal_looking", "all_flags_clear"}
    with pytest.raises(SystemExit, match="outside"):
        monkeypatch.setattr(sys, "argv", ["train_mbrset.py", "--dataset", "brset", "--root", str(root),
                                          "--task", "cm_hypertension", "--fold", "3", "--n-folds", "3"])
        train_mbrset.main()
