"""CPU-only synthetic tests for the systemic (oculomics) targets: the strict
yes/no normaliser, the registry entries, the age+sex covariate baseline, the
--init-from warm start, the pre-flight inspector and the per-task summariser."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from covariate_baseline import covariate_baseline, image_minus_covariate
from dataset import LABEL_REGISTRY, SYSTEMIC_TASKS, MBRSETDataset, _binary_flag, stratified_split
from inspect_mbrset import inspect
from model import MBRSETClassifier
from summarize_systemic import summarize
from train_mbrset import load_init_weights


# ---- (a) strict binary flag ------------------------------------------------- #
def test_binary_flag_tokens_and_nan():
    for v in ("yes", " Yes ", "Y", "true", "1", 1, 1.0, True, "Sim"):
        assert _binary_flag(v) == 1.0, v
    for v in ("no", "N", "false", "0", 0, 0.0, False, "Nao", "não"):
        assert _binary_flag(v) == 0.0, v
    # Unknown tokens and a 1/2 release become NaN (dropped), never a confident 0.
    for v in (None, np.nan, "unknown", "n/a", 2, 2.0, "2", "", object()):
        assert np.isnan(_binary_flag(v)), v


def test_systemic_tasks_registered_as_binary():
    for task, (col, _) in SYSTEMIC_TASKS.items():
        spec = LABEL_REGISTRY[task]
        assert spec.source_cols == (col,) and spec.num_classes == 2 and spec.dtype == torch.long
    # The factory binds each column: hypertension must not read nephropathy.
    r = {"systemic_hypertension": "yes", "nephropathy": "no"}
    assert LABEL_REGISTRY["hypertension"].fn(r) == 1.0
    assert LABEL_REGISTRY["nephropathy"].fn(r) == 0.0


def _systemic_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 85, n).astype(float)
    # hypertension driven by age; nephropathy independent of age (pure noise here)
    htn = (age + rng.normal(0, 8, n) > 60).astype(int)
    return pd.DataFrame({
        "file": [f"{i}.jpg" for i in range(n)],
        "patient": [f"p{i // 2}" for i in range(n)],
        "age": age,
        "sex": rng.choice([1, 2], n),
        "systemic_hypertension": np.where(htn == 1, "yes", "no"),
        # object array with real None (a np.nan in a str list becomes the string "nan")
        "nephropathy": rng.choice(np.array(["yes", "no", None], dtype=object), n, p=[0.15, 0.8, 0.05]),
        "final_icdr": rng.integers(0, 5, n),
    })


def test_dataset_drops_nan_systemic_rows():
    df = _systemic_df()
    ds = MBRSETDataset(csv=df, images_dir="/nonexistent", task="nephropathy", split="val",
                       image_size=32, drop_missing_files=False)
    n_nan = int(df["nephropathy"].isna().sum())
    assert len(ds) == len(df) - n_nan and n_nan > 0
    assert ds.num_classes == 2 and int(ds.class_counts().sum()) == len(ds)


# ---- (b) covariate baseline ------------------------------------------------- #
def test_covariate_baseline_learns_age_confound_and_is_paired_on_split():
    df = _systemic_df(n=300)
    sp = stratified_split(df, task="hypertension", val_frac=0.1, test_frac=0.2,
                          group_col="patient", seed=1)
    cb = covariate_baseline(sp["train"], sp["test"], "hypertension")
    assert cb["reason"] is None and cb["auroc"] > 0.8         # hypertension IS age here
    assert cb["n_test"] == len(sp["test"]) and cb["features"] == ["age", "sex"]
    assert image_minus_covariate(0.9, cb) == pytest.approx(0.9 - cb["auroc"])
    assert image_minus_covariate(float("nan"), cb) is None


def test_covariate_baseline_refuses_meaningless_cases():
    df = _systemic_df()
    tr, te = df.iloc[:80], df.iloc[80:]
    assert covariate_baseline(tr, te, "dr_grade")["reason"].startswith("task")       # not binary
    assert "missing" in covariate_baseline(tr.drop(columns=["sex"]), te, "hypertension")["reason"]
    te1 = te.copy(); te1["systemic_hypertension"] = "yes"
    cb = covariate_baseline(tr, te1, "hypertension")
    assert np.isnan(cb["auroc"]) and cb["reason"] == "single-class test split"
    # object-typed sex ('M'/'F') is factorised, not coerced to all-NaN
    tr2, te2 = tr.copy(), te.copy()
    tr2["sex"] = np.where(tr2["sex"] == 1, "M", "F"); te2["sex"] = np.where(te2["sex"] == 1, "M", "F")
    cb = covariate_baseline(tr2, te2, "hypertension")
    assert cb["n_train"] == len(tr2) and cb["auroc"] == cb["auroc"]


# ---- (c) --init-from warm start --------------------------------------------- #
def _cls(seed, n=2):
    torch.manual_seed(seed)
    return MBRSETClassifier(num_classes=n, pretrained=False)


def test_load_init_weights_copies_backbone_not_head(tmp_path):
    src, dst = _cls(0), _cls(1)
    ck = tmp_path / "dr.pt"
    torch.save({"model": src.state_dict(), "args": {"task": "dr_referable", "dataset": "brset",
                                                     "root": "/data/BRSET", "seed": 0,
                                                     "backbone": "mobilenetv3_small"}}, ck)
    head_before = {k: v.clone() for k, v in dst.state_dict().items() if k.startswith("head.")}
    info = load_init_weights(dst, str(ck), root="/data/mBRSET", external_root=None, seed=3)
    s, d = src.state_dict(), dst.state_dict()
    for k in s:
        if k.startswith("head."):
            assert torch.equal(d[k], head_before[k]), k          # head untouched
        else:
            assert torch.equal(d[k], s[k]), k                    # everything else copied
    assert info["n_loaded"] > 0 and info["n_head_skipped"] == len(head_before)
    assert info["n_shape_mismatch"] == 0 and info["n_missing"] == 0
    assert info["init_task"] == "dr_referable" and info["warning"] is None


def test_load_init_weights_guards(tmp_path):
    src = _cls(0)
    ck = tmp_path / "x.pt"
    torch.save({"model": src.state_dict(), "args": {"root": "/data/mBRSET", "seed": 0}}, ck)
    # same root, different seed -> overlapping splits -> warning, not fatal
    info = load_init_weights(_cls(1), str(ck), root="/data/mBRSET", seed=1)
    assert info["warning"] and "seed" in info["warning"]
    assert load_init_weights(_cls(1), str(ck), root="/data/mBRSET", seed=0)["warning"] is None
    # trained on the external test root -> refused
    with pytest.raises(SystemExit):
        load_init_weights(_cls(1), str(ck), root="/data/BRSET", external_root="/data/mBRSET", seed=0)
    # no 'model' key -> refused
    torch.save({"weights": {}}, tmp_path / "bad.pt")
    with pytest.raises(SystemExit):
        load_init_weights(_cls(1), str(tmp_path / "bad.pt"))
    # nothing matches -> refused (a different architecture)
    torch.save({"model": {"foo.weight": torch.zeros(3)}, "args": {}}, tmp_path / "other.pt")
    with pytest.raises(SystemExit):
        load_init_weights(_cls(1), str(tmp_path / "other.pt"))


# ---- (d) pre-flight inspector ------------------------------------------------ #
def test_inspect_reports_missing_single_class_and_ok():
    df = _systemic_df(n=200)
    df["obesity"] = "no"                                    # single-class as encoded
    df["smoking"] = np.random.default_rng(0).choice([1, 2], len(df))   # a 1/2 release -> empty
    rows = {r["task"]: r for r in inspect(df, ["hypertension", "obesity", "smoking", "diabetic_foot"])}
    assert rows["hypertension"]["ok"] and rows["hypertension"]["covariate_auroc"] > 0.8
    assert rows["hypertension"]["patients"] == df["patient"].nunique()
    assert not rows["obesity"]["ok"] and rows["obesity"]["pos"] == 0
    # 1/2 release: the 1s read as positives, the 2s drop -> single-class, flagged, raw values shown
    assert not rows["smoking"]["ok"] and rows["smoking"]["pos"] == rows["smoking"]["n"] > 0
    assert rows["smoking"]["raw_values"] == ["1", "2"]
    assert not rows["diabetic_foot"]["present"]


# ---- (e) per-task summariser ------------------------------------------------ #
def _write_run(d, cond, seed, auroc, cov_auroc, task="hypertension"):
    d.mkdir(parents=True, exist_ok=True)
    rec = {"task": task, "seed": seed, "test": {"auroc": auroc, "acc": 0.7, "f1": 0.6, "kappa": 0.3, "n": 100},
           "test_pos": 40, "n_test": 100,
           "covariate_baseline": {"auroc": cov_auroc, "features": ["age", "sex"], "n_test": 100},
           "init_from": {"path": "/ck/dr.pt"} if cond == "drinit" else None}
    (d / f"{cond}_seed{seed}.json").write_text(json.dumps(rec))


def test_summarize_systemic_pairs_by_seed(tmp_path):
    t = tmp_path / "hypertension"
    for s, (c, dr, cov) in enumerate([(0.80, 0.83, 0.70), (0.81, 0.85, 0.71), (0.79, 0.82, 0.69)]):
        _write_run(t, "ctrl", s, c, cov); _write_run(t, "drinit", s, dr, cov)
    _write_run(t, "drinit", 7, 0.99, 0.5)                   # unpaired seed must be dropped
    rep = summarize(str(tmp_path), None, control="ctrl", treatment="drinit")
    r = rep["hypertension"]
    assert r["seeds"] == [0, 1, 2]
    tc = r["paired_treatment_minus_control"]
    assert tc["n"] == 3 and tc["mean_delta"] == pytest.approx((0.03 + 0.04 + 0.03) / 3)
    ic = r["paired_image_minus_covariate"]["ctrl"]
    assert ic["mean_delta"] == pytest.approx(0.10) and ic["significant"]
    assert r["conditions"]["ctrl"]["prevalence"] == pytest.approx(0.4)
    assert (tmp_path / "summary.md").exists() and (tmp_path / "summary.json").exists()
    # control-only mode answers question 1 only
    rep = summarize(str(tmp_path), ["hypertension"], control="ctrl", treatment=None)
    assert rep["hypertension"]["paired_treatment_minus_control"] is None


def test_summarize_systemic_fails_loudly_on_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        summarize(str(tmp_path), ["hypertension"], control="ctrl")
