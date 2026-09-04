"""CPU-only synthetic tests for the BRSET ophthalmic multi-label head: the adapter
carries the label columns, vector labels are vectorised / NaN-dropped / stratified,
the imbalance helpers, the multilabel evaluate() and KD term, and the summariser."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from brset_dataset import OPHTHALMIC_MAP, _normalise_flag, load_brset
from dataset import (LABEL_REGISTRY, OPHTHALMIC_LABELS, MBRSETDataset, _binary_flag,
                     _strat_key, stratified_split)
from model import MBRSETClassifier
from summarize_ophthalmic import summarize
from train_mbrset import evaluate, kd_loss_multilabel

L = len(OPHTHALMIC_LABELS)


def _eq_nan(a, b):
    return list(pd.Series(a).fillna(-99)) == list(pd.Series(b).fillna(-99))


# ---- (a) adapter ------------------------------------------------------------- #
def test_normalise_flag_keeps_01_and_nans_the_rest():
    assert _eq_nan(_normalise_flag(pd.Series([0, 1, np.nan, 2, -1], dtype=float)), [0.0, 1.0, np.nan, np.nan, np.nan])
    assert _eq_nan(_normalise_flag(pd.Series(["yes", "no", "1", "0", "maybe", None])), [1.0, 0.0, 1.0, 0.0, np.nan, np.nan])


def test_load_brset_carries_ophthalmic_columns_and_drops_redundant_ones():
    raw = pd.DataFrame({"image_id": ["a", "b", "c"], "patient_id": ["p1", "p2", "p3"],
                        "DR_ICDR": [0, 2, 4], "amd": [0, 1, np.nan], "drusens": [1, 0, 0],
                        "diabetic_retinopathy": [0, 1, 1], "other": [0, 0, 1]})
    df = load_brset(raw)
    assert _eq_nan(df["amd"], [0.0, 1.0, np.nan]) and list(df["drusens"]) == [1.0, 0.0, 0.0]
    assert "diabetic_retinopathy" not in df.columns and "other" not in df.columns
    assert set(OPHTHALMIC_MAP) == set(OPHTHALMIC_LABELS)


# ---- (b) vector labels in the dataset ---------------------------------------- #
def _oph_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    d = {"file": [f"{i}.jpg" for i in range(n)], "patient": [f"p{i // 2}" for i in range(n)]}
    prev = np.linspace(0.02, 0.4, L)                       # rare ... common labels
    for j, c in enumerate(OPHTHALMIC_LABELS):
        d[c] = (rng.random(n) < prev[j]).astype(float)
    df = pd.DataFrame(d)
    df.loc[0, "amd"] = np.nan                                # one partially labelled row
    return df


def test_multilabel_dataset_vectorises_and_drops_partial_rows():
    df = _oph_df()
    ds = MBRSETDataset(csv=df, images_dir="/nonexistent", task="ophthalmic", split="val",
                       image_size=32, drop_missing_files=False)
    assert ds.multilabel and ds.num_classes == L and ds.label_names == OPHTHALMIC_LABELS
    assert len(ds) == len(df) - 1                           # the NaN-component row is gone
    assert ds.labels.shape == (len(ds), L) and ds.labels.dtype == torch.float32
    assert ds.class_counts() is None                        # CE helpers do not apply
    pos = ds.label_pos_counts()
    assert pos.shape == (L,) and int(pos[0]) < int(pos[-1])   # rare < common as generated
    pw = ds.pos_weight()
    assert pw.shape == (L,) and torch.all(pw > 0) and pw[0] > pw[-1]
    w = ds.sample_weights()
    assert w.shape == (len(ds),) and torch.isfinite(w).all() and w.mean() == pytest.approx(1.0)
    rare_rows = ds.labels[:, 0] > 0.5
    assert w[rare_rows].mean() > w[~rare_rows].mean()       # rarest label drawn more often
    # each single-label task reads its own column
    da = MBRSETDataset(csv=df, images_dir="/nonexistent", task="amd", split="val",
                       image_size=32, drop_missing_files=False)
    assert da.num_classes == 2 and len(da) == len(df) - 1


def test_strat_key_is_rarest_positive_label():
    y = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=float)   # label0 rare(2), label1 (2), label2 (1)
    assert _strat_key(y).tolist() == [0, 1, -1, 2]
    assert _strat_key(np.array([0.0, 1.0])).tolist() == [0.0, 1.0]          # scalar passthrough


def test_stratified_split_multilabel_keeps_patients_disjoint_and_covers_rows():
    df = _oph_df(n=300)
    sp = stratified_split(df, task="ophthalmic", val_frac=0.1, test_frac=0.2, group_col="patient", seed=0)
    import itertools
    psets = {k: set(v["patient"]) for k, v in sp.items()}
    for a, b in itertools.combinations(psets, 2):
        assert not (psets[a] & psets[b])
    assert sum(len(v) for v in sp.values()) == len(df) - 1
    # the rarest label is represented in every partition (that is what the key is for)
    assert all((sp[k][OPHTHALMIC_LABELS[0]] == 1).any() for k in ("train", "test"))
    # ungrouped multilabel path also works
    sp2 = stratified_split(df, task="ophthalmic", group_col=None, val_frac=0.2, test_frac=0.2)
    assert sum(len(v) for v in sp2.values()) == len(df) - 1


# ---- (c) model / evaluate / KD ---------------------------------------------- #
def test_classifier_emits_one_logit_per_label():
    torch.manual_seed(0)
    m = MBRSETClassifier(num_classes=L, multilabel=True, pretrained=False).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, L) and m.multilabel


class _Const(nn.Module):
    def __init__(self, logits):
        super().__init__(); self.logits = logits; self.i = 0

    def forward(self, x):
        out = self.logits[self.i: self.i + x.shape[0]]; self.i += x.shape[0]; return out


def test_evaluate_multilabel_scores_per_label_and_skips_constant_labels():
    torch.manual_seed(0)
    n = 40
    y = torch.zeros(n, 3); y[:20, 0] = 1; y[::3, 1] = 1                    # label 2 all-negative
    logits = torch.randn(n, 3); logits[:, 0] = y[:, 0] * 4 - 2              # label 0 perfectly separable
    loader = [{"image": torch.zeros(8, 1), "label": y[i:i + 8]} for i in range(0, n, 8)]
    m = _Const(logits)
    r = evaluate(m, loader, torch.device("cpu"), 3, multilabel=True, label_names=["a", "b", "c"])
    assert r["per_label_auroc"]["a"] == pytest.approx(1.0)
    assert np.isnan(r["per_label_auroc"]["c"]) and r["n_labels_scored"] == 2
    assert r["auroc"] == pytest.approx(np.mean([r["per_label_auroc"]["a"], r["per_label_auroc"]["b"]]))
    assert np.isnan(r["kappa"]) and r["n"] == n and 0 <= r["acc"] <= 1


def test_kd_loss_multilabel_matches_hand_computed_bce():
    torch.manual_seed(1)
    s, t, T = torch.randn(5, 4), torch.randn(5, 4) * 3, 2.0
    ref = F.binary_cross_entropy_with_logits(s / T, torch.sigmoid(t / T)) * T * T
    assert torch.allclose(kd_loss_multilabel(s, t, T), ref)
    assert kd_loss_multilabel(t, t, T) < kd_loss_multilabel(-t, t, T)     # agreeing student is cheaper


# ---- (d) summariser ---------------------------------------------------------- #
def _multi(d, seed, per):
    d.mkdir(exist_ok=True)
    (d / f"multi_seed{seed}.json").write_text(json.dumps({
        "task": "ophthalmic", "seed": seed, "multilabel": True, "label_names": list(per),
        "test": {"auroc": float(np.mean(list(per.values()))), "per_label_auroc": per, "acc": .9, "f1": .5, "kappa": None, "n": 100},
        "test_pos": {k: 10 for k in per}, "n_test": 100}))


def _single(d, lab, seed, auroc):
    (d / f"single_{lab}_seed{seed}.json").write_text(json.dumps({
        "task": lab, "seed": seed, "multilabel": False,
        "test": {"auroc": auroc, "acc": .9, "f1": .5, "kappa": .1, "n": 100}, "test_pos": 10, "n_test": 100}))


def test_summarize_ophthalmic_pairs_per_label(tmp_path):
    for s in range(3):
        _multi(tmp_path, s, {"amd": 0.90 + s * 0.01, "nevus": 0.70, "scar": float("nan")})
        _single(tmp_path, "amd", s, 0.85 + s * 0.01)
        _single(tmp_path, "nevus", s, 0.75)
    rep = summarize(str(tmp_path))
    amd = rep["labels"]["amd"]
    assert amd["paired_seeds"] == [0, 1, 2]
    assert amd["paired_multi_minus_single"]["mean_delta"] == pytest.approx(0.05) and amd["paired_multi_minus_single"]["significant"]
    assert rep["labels"]["nevus"]["paired_multi_minus_single"]["mean_delta"] == pytest.approx(-0.05)
    assert rep["labels"]["scar"]["paired_multi_minus_single"] is None      # NaN multi + no single -> unpaired
    assert rep["labels"]["amd"]["prevalence"] == pytest.approx(0.1)
    assert (tmp_path / "summary.md").exists() and len(rep["multi_macro_auroc"]) == 3


def test_summarize_ophthalmic_fails_loudly_on_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        summarize(str(tmp_path))


def test_binary_flag_basics():
    assert _binary_flag("yes") == 1.0 and _binary_flag(0) == 0.0 and np.isnan(_binary_flag(2))
    assert LABEL_REGISTRY["ophthalmic"].multilabel and LABEL_REGISTRY["amd"].num_classes == 2
