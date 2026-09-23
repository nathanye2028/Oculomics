"""CPU-only synthetic tests for metadata_model.py: feature-set leakage rules,
column typing, the operating-point helpers, and an end-to-end run that must
find a planted signal and must NOT find one in the shuffled-label control."""
import json

import numpy as np
import pandas as pd

from metadata_model import (FEATURE_SETS, _column_kind, encode, evaluate_target, features_for,
                            main, operating_point, threshold_at_sensitivity)


def _mbrset_like(n_patients=300, seed=0):
    """Two eyes per patient; hypertension driven by age, nephropathy by dm_time,
    DR referable by dm_time. Encodings mimic mBRSET (yes/no strings, sex as text)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        age = rng.uniform(30, 85)
        dm = rng.uniform(0, 30)
        sex = rng.choice(["female", "male"])
        htn = rng.random() < 1 / (1 + np.exp(-(age - 60) / 6))
        neph = rng.random() < 1 / (1 + np.exp(-(dm - 22) / 3))
        edu = rng.choice(["1", "2", "3", None])
        for eye in ("right", "left"):
            ref = rng.random() < 1 / (1 + np.exp(-(dm - 18) / 4))
            rows.append({
                "file": f"{pid}_{eye}.jpg", "patient": pid, "laterality": eye,
                "age": age, "sex": sex, "dm_time": dm,
                "insulin": rng.choice(["yes", "no"]), "insulin_time": rng.uniform(0, 5),
                "educational_level": edu,
                "systemic_hypertension": "yes" if htn else "no",
                "nephropathy": "yes" if neph else "no",
                "obesity": rng.choice(["yes", "no"]),
                "final_icdr": int(ref) * 2, "final_edema": rng.choice(["yes", "no"]),
            })
    return pd.DataFrame(rows)


def test_target_and_proxies_never_features():
    cols = list(FEATURE_SETS["full"])
    for task, col in [("hypertension", "systemic_hypertension"), ("nephropathy", "nephropathy")]:
        assert col not in features_for(task, "full", cols)
    ins = features_for("insulin", "full", cols)
    assert "insulin" not in ins and "insulin_time" not in ins
    # Image-derived columns are never in any feature set.
    for fs in FEATURE_SETS.values():
        assert not {"final_icdr", "final_edema", "final_quality", "final_artifacts",
                    "laterality", "camera"} & set(fs)


def test_column_kinds_and_encoding():
    df = pd.DataFrame({"a": ["yes", "no", None], "b": [1.5, 2, None],
                       "c": ["M", "F", None], "d": [1, 2, 1]})
    assert _column_kind(df["a"]) == "binary"
    assert _column_kind(df["b"]) == "numeric"
    assert _column_kind(df["c"]) == "categorical"
    assert _column_kind(df["d"]) == "numeric"          # a 1/2 column is not yes/no
    X, kinds = encode(df, ["a", "b", "c"])
    assert X["a"].tolist()[:2] == [1.0, 0.0] and np.isnan(X["a"].iloc[2])
    assert X["c"].tolist()[:2] == ["m", "f"] and pd.isna(X["c"].iloc[2])
    # External table reuses the training kinds even when its values look different.
    X2, _ = encode(pd.DataFrame({"a": [1, 0, 0], "b": [1, 2, 3], "c": ["m", "f", "x"]}),
                   ["a", "b", "c"], kinds)
    assert X2["a"].tolist() == [1.0, 0.0, 0.0]


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = y * 0.5 + rng.random(2000)
    thr = threshold_at_sensitivity(y, p, 0.85)
    op = operating_point(y, p, thr)
    assert 0.85 <= op["sensitivity"] <= 0.86
    assert 0 < op["flagged_fraction"] < 1


def test_finds_planted_signal_and_shuffle_is_chance():
    df = _mbrset_like()
    for mk in ("logreg", "gbm"):
        r = evaluate_target(df, "hypertension", "age_sex", mk, n_boot=100, cv_repeats=2)
        assert r["reason"] is None
        assert r["test"]["auroc"] > 0.8, r["test"]
        lo, hi = r["test"]["auroc_ci"]
        assert lo <= r["test"]["auroc"] <= hi
        assert abs(r["shuffle_auroc"] - 0.5) < 3 * r["shuffle_auroc_se"] + 0.02
        assert r["cv"]["n_folds"] == 10
        # age drives it -> age is the top logistic term
        if mk == "logreg":
            assert r["top_coefficients"][0][0] == "age"
    # Nephropathy depends on dm_time, which age_sex does not carry.
    weak = evaluate_target(df, "nephropathy", "age_sex", "logreg", n_boot=50, cv_repeats=1)
    strong = evaluate_target(df, "nephropathy", "clinical", "logreg", n_boot=50, cv_repeats=1)
    assert strong["cv"]["auroc_mean"] > weak["cv"]["auroc_mean"] + 0.15


def test_external_scoring_uses_shared_features_only():
    src, ext = _mbrset_like(seed=1), _mbrset_like(seed=2).drop(columns=["educational_level"])
    r = evaluate_target(src, "dr_referable", "full", "logreg", n_boot=50, cv_repeats=1,
                        external=ext)
    assert "educational_level" in r["features"]
    assert "educational_level" not in r["external"]["features"]
    assert r["external"]["auroc"] > 0.75


def test_cli_end_to_end(tmp_path):
    root = tmp_path / "mbrset"
    root.mkdir()
    _mbrset_like(n_patients=150).to_csv(root / "labels_mbrset.csv", index=False)
    out = tmp_path / "out"
    assert main(["--dataset", "mbrset", "--root", str(root), "--out", str(out),
                 "--tasks", "hypertension", "dr_referable", "--n-boot", "50",
                 "--cv-repeats", "1"]) == 0
    table = pd.read_csv(out / "metadata_results.csv")
    assert len(table) == 2 * 3 * 2               # tasks x feature sets x models
    best = pd.read_csv(out / "metadata_best_per_task.csv")
    assert set(best["task"]) == {"hypertension", "dr_referable"}
    blob = json.loads((out / "metadata_results.json").read_text())
    assert blob["dataset"] == "mbrset"
