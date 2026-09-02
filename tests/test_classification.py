"""CPU-only, synthetic, network-free tests for the classification track:
brset_dataset value re-encoders, stratified_split guards, AdaBN, the KD loss,
and MBRSETClassifier's timm/GCG refusal. Everything uses pretrained=False and
tiny inputs so the whole file runs in a few seconds."""
import itertools

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from brset_dataset import (_flip_12, _normalise_artifacts, _normalise_quality,
                           _normalise_sex, load_brset)
from dataset import MBRSETDataset, stratified_split
from model import MBRSETClassifier
from train_mbrset import adapt_bn, kd_loss


# ---- (a) BRSET value re-encoders ------------------------------------------- #
def _eq_nan(a, b):
    """Elementwise equality that treats NaN == NaN."""
    return list(pd.Series(a).fillna(-99)) == list(pd.Series(b).fillna(-99))


def test_artifacts_polarity_inverted_and_nan_preserved():
    # BRSET artifacts: 1 = adequate (no artifacts), 2 = inadequate (artifacts PRESENT).
    out = _normalise_artifacts(pd.Series([1, 2, np.nan, 2, 1], dtype=float))
    assert _eq_nan(out, [0.0, 1.0, np.nan, 1.0, 0.0])
    assert out.dtype == float
    # yes/no strings (mBRSET style) go through the same function unchanged in polarity.
    out = _normalise_artifacts(pd.Series(["yes", "no", None]))
    assert _eq_nan(out, [1.0, 0.0, np.nan])


def test_flip_12_and_quality_preserve_nan():
    out = _flip_12(pd.Series([1, 2, np.nan], dtype=float))     # 1=adequate -> 1.0
    assert _eq_nan(out, [1.0, 0.0, np.nan])
    out = _normalise_quality(pd.Series(["Adequate", "Inadequate", None, " adequate "]))
    assert _eq_nan(out, [1.0, 0.0, np.nan, 1.0])
    out = _normalise_quality(pd.Series([1, 2, np.nan], dtype=float))   # 1/2 releases
    assert _eq_nan(out, [1.0, 0.0, np.nan])


def test_sex_reencoded_and_passthrough():
    assert _eq_nan(_normalise_sex(pd.Series([1, 2, np.nan], dtype=float)), [0.0, 1.0, np.nan])
    assert _eq_nan(_normalise_sex(pd.Series([0, 1], dtype=float)), [0.0, 1.0])   # already 0/1


def test_load_brset_nan_artifacts_row_is_dropped_by_dataset():
    raw = pd.DataFrame({"image_id": ["img1", "img2", "img3", "img4"],
                        "patient_id": ["p1", "p1", "p2", "p3"],
                        "DR_ICDR": [0, 2, 4, 1],
                        "artifacts": [1.0, 2.0, np.nan, 1.0],
                        "quality": ["Adequate", "Inadequate", "Adequate", np.nan]})
    df = load_brset(raw)
    assert list(df["file"]) == ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    assert _eq_nan(df["final_artifacts"], [0.0, 1.0, np.nan, 0.0])
    assert _eq_nan(df["final_quality"], [1.0, 0.0, 1.0, np.nan])
    # The NaN rows must reach MBRSETDataset as NaN so drop_missing_labels removes
    # them, instead of arriving as a confident 0.
    ds = MBRSETDataset(csv=df, images_dir="/nonexistent", task="artifacts", split="val",
                       image_size=32, drop_missing_files=False)
    assert len(ds) == 3 and ds.class_counts().tolist() == [2, 1]
    ds = MBRSETDataset(csv=df, images_dir="/nonexistent", task="quality", split="val",
                       image_size=32, drop_missing_files=False)
    assert len(ds) == 3 and ds.class_counts().tolist() == [1, 2]


# ---- (b) stratified_split --------------------------------------------------- #
def _label_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"file": [f"{i}.jpg" for i in range(n)],
                         "patient": [f"p{i // 2}" for i in range(n)],      # two eyes each
                         "final_icdr": rng.integers(0, 5, n)})


def test_stratified_split_raises_on_missing_group_col():
    df = _label_df().drop(columns=["patient"])
    with pytest.raises(ValueError, match="patient"):
        stratified_split(df, task="dr_referable", group_col="patient")
    # Explicitly ungrouped is still allowed.
    sp = stratified_split(df, task="dr_referable", group_col=None, val_frac=0.2, test_frac=0.2)
    assert sum(len(v) for v in sp.values()) == len(df)


def test_stratified_split_keeps_patients_disjoint():
    sp = stratified_split(_label_df(), task="dr_referable", val_frac=0.2, test_frac=0.2,
                          group_col="patient", seed=3)
    psets = {k: set(v["patient"]) for k, v in sp.items()}
    for a, b in itertools.combinations(psets, 2):
        assert not (psets[a] & psets[b])
    assert sum(len(v) for v in sp.values()) == 60


# ---- (c) adapt_bn ----------------------------------------------------------- #
class _Batches:
    """Minimal loader: a list of {"image": tensor} dicts."""
    def __init__(self, sizes, ch=3, hw=8, shift=5.0):
        self.items = [{"image": torch.randn(b, ch, hw, hw) + shift} for b in sizes]

    def __iter__(self):
        return iter(self.items)


def _bn_net():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                        nn.Dropout(0.5), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    return net.eval()


def test_adapt_bn_reestimates_running_stats_and_leaves_source_untouched():
    net = _bn_net()
    src_mean = net[1].running_mean.clone()
    adapted, n_bn = adapt_bn(net, _Batches([4, 4, 4]), torch.device("cpu"))
    assert n_bn == 1
    assert not adapted.training and adapted[1].momentum is None
    assert int(adapted[1].num_batches_tracked) == 3
    assert not torch.allclose(adapted[1].running_mean, src_mean)   # shifted target stats
    assert torch.equal(net[1].running_mean, src_mean)              # deepcopy: source untouched
    assert adapted[1].running_mean.abs().sum() > 0


def test_adapt_bn_skips_batch_size_one_and_honours_max_batches():
    net = _bn_net()
    adapted, n_bn = adapt_bn(net, _Batches([1, 1, 1]), torch.device("cpu"))
    assert n_bn == 1 and int(adapted[1].num_batches_tracked) == 0
    assert torch.equal(adapted[1].running_mean, torch.zeros(4))   # reset, nothing seen
    adapted, _ = adapt_bn(net, _Batches([1, 4, 4, 4]), torch.device("cpu"), max_batches=2)
    assert int(adapted[1].num_batches_tracked) == 2                # the size-1 batch did not count


def test_adapt_bn_flags_zero_bn_models():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.LayerNorm(3 * 8 * 8), nn.Linear(3 * 8 * 8, 2)).eval()
    adapted, n_bn = adapt_bn(net, _Batches([4]), torch.device("cpu"))
    assert n_bn == 0
    assert all(torch.equal(a, b) for a, b in zip(adapted.state_dict().values(),
                                                  net.state_dict().values()))


# ---- (d) KD loss ------------------------------------------------------------ #
def test_kd_loss_matches_hand_computed_kl_teacher_student():
    torch.manual_seed(1)
    s_logits = torch.randn(5, 4)
    t_logits = torch.randn(5, 4) * 3
    T = 4.0
    p_t = F.softmax(t_logits / T, dim=1)
    p_s = F.softmax(s_logits / T, dim=1)
    expected = (p_t * (p_t.log() - p_s.log())).sum(1).mean() * T * T   # T^2 * KL(teacher||student)
    got = kd_loss(s_logits, t_logits, T)
    assert torch.allclose(got, expected, atol=1e-6)
    # It is NOT the reverse KL: the two differ for these logits.
    reverse = (p_s * (p_s.log() - p_t.log())).sum(1).mean() * T * T
    assert not torch.allclose(got, reverse, atol=1e-4)
    # Identical logits -> zero; and fp16 inputs are upcast (no dtype error).
    assert kd_loss(t_logits, t_logits, T).abs() < 1e-6
    assert kd_loss(s_logits.half(), t_logits.half(), T).dtype == torch.float32


# ---- (e) MBRSETClassifier / GCG on timm ------------------------------------- #
def test_timm_backbone_refuses_use_gcg():
    pytest.importorskip("timm")
    with pytest.raises(ValueError, match="no-gcg"):
        MBRSETClassifier(num_classes=2, pretrained=False, use_gcg=True, backbone="timm:resnet10t")
    m = MBRSETClassifier(num_classes=2, pretrained=False, use_gcg=False, backbone="timm:resnet10t")
    assert m.use_gcg is False and m.features is None
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 2)


def test_mobilenet_probe_does_not_touch_bn_stats():
    # The stride probe runs in eval mode, so a freshly built (un-pretrained) model
    # still has pristine BN buffers and its training flag is restored.
    m = MBRSETClassifier(num_classes=2, pretrained=False, use_gcg=True)
    assert m.training and m.use_gcg is True
    bns = [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    assert bns and all(int(b.num_batches_tracked) == 0 for b in bns)
    assert all(torch.equal(b.running_mean, torch.zeros_like(b.running_mean)) for b in bns)
