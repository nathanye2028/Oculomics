"""Deployment-path unit tests: CPU, synthetic, no network, no coremltools.

(a) the baked-in normalisation in export_coreml's deploy wrappers matches
    dataset.py's constants exactly (the whole point of baking it in);
(b) evaluate_deploy's operating point is the HIGHEST threshold meeting the
    target sensitivity, and binary_report agrees with it;
(c) check_env's Python-version gate is a pure function of (major, minor).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import check_env
import dataset
import evaluate_deploy
import export_coreml


# --------------------------------------------------------------------------- #
# (a) deploy wrappers == Normalize(dataset constants) + activation
# --------------------------------------------------------------------------- #
def _tiny_net(out_ch: int, spatial: bool):
    torch.manual_seed(0)
    layers = [nn.Conv2d(3, 6, 3, padding=1), nn.ReLU()]
    if spatial:
        layers.append(nn.Conv2d(6, out_ch, 1))                       # [B,out,H,W]
    else:
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, out_ch)]
    return nn.Sequential(*layers).eval()


def _normalize_like_dataset(x01: torch.Tensor) -> torch.Tensor:
    from torchvision.transforms import Normalize
    return Normalize(mean=list(dataset.IMAGENET_MEAN), std=list(dataset.IMAGENET_STD))(x01)


def test_export_constants_match_dataset():
    assert export_coreml.IMAGENET_MEAN.flatten().tolist() == pytest.approx(list(dataset.IMAGENET_MEAN))
    assert export_coreml.IMAGENET_STD.flatten().tolist() == pytest.approx(list(dataset.IMAGENET_STD))
    from fundus_utils import crop_to_fov
    import inspect
    assert inspect.signature(crop_to_fov).parameters["tol"].default == export_coreml.FOV_TOL


@torch.no_grad()
def test_cls_wrapper_matches_normalized_softmax():
    net = _tiny_net(2, spatial=False)
    x01 = torch.rand(2, 3, 16, 16)                                   # what ToDtype(scale=True) yields
    expect = torch.softmax(net(_normalize_like_dataset(x01)), dim=1)
    got = export_coreml.ClsDeployWrapper(net).eval()(x01 * 255.0)   # raw 0-255 as the app feeds it
    assert torch.allclose(got, expect, atol=1e-5)
    assert torch.allclose(got.sum(1), torch.ones(2), atol=1e-6)


@torch.no_grad()
def test_seg_wrapper_matches_normalized_sigmoid():
    net = _tiny_net(4, spatial=True)
    x01 = torch.rand(1, 3, 16, 16)
    expect = torch.sigmoid(net(_normalize_like_dataset(x01)))
    got = export_coreml.DeployWrapper(net).eval()(x01 * 255.0)
    assert got.shape == (1, 4, 16, 16)
    assert torch.allclose(got, expect, atol=1e-5)


@torch.no_grad()
def test_reference_output_equals_wrapper_on_same_pixels():
    """verify() compares wrapper(uint8) against reference_output(net, normalised);
    on identical pixels those two paths must agree, or the check tests itself."""
    net = _tiny_net(2, spatial=False)
    wrapper = export_coreml.ClsDeployWrapper(net).eval()
    u8 = torch.randint(0, 256, (1, 3, 16, 16)).float()
    ref = export_coreml.reference_output(wrapper, _normalize_like_dataset(u8 / 255.0), "cls")
    assert np.allclose(ref, wrapper(u8).numpy(), atol=1e-5)


def test_preprocess_spec_states_the_contract():
    spec = export_coreml.preprocess_spec("cls", 224, "adapted", "ck.pt", "fp16")
    joined = " ".join(spec.values())
    assert "0.485" in joined and "0.229" in joined                   # constants spelled out
    assert f"> {export_coreml.FOV_TOL}" in spec["preprocess.1_fov_crop"]
    assert "224x224" in spec["preprocess.2_resize"] and "antialias" in spec["preprocess.2_resize"]
    assert spec["bn_stats"].startswith("adapted")
    assert "softmax" in spec["postprocess"]
    assert all(isinstance(v, str) for v in spec.values())            # Core ML metadata is str->str
    seg = export_coreml.preprocess_spec("seg", 512, "source", None, "none")
    assert "sigmoid" in seg["postprocess"] and "RANDOM" in seg["checkpoint"]


def test_extract_state_bn_stats():
    ck = {"model": {"w": torch.zeros(1)}, "model_bnadapt": {"w": torch.ones(1)}, "args": {}}
    assert export_coreml.extract_state(ck, "source")["w"].item() == 0.0
    assert export_coreml.extract_state(ck, "adapted")["w"].item() == 1.0
    with pytest.raises(SystemExit, match="model_bnadapt"):
        export_coreml.extract_state({"model": {}}, "adapted")
    assert export_coreml.gcg_from_args({"no_gcg": True}) is False
    assert export_coreml.gcg_from_args({"use_gcg": False, "no_gcg": False}) is False
    assert export_coreml.gcg_from_args({}) is True
    # Pre-2026-09-01 timm checkpoints (incl. the deployed V4-Small students) were
    # trained without --no-gcg but never had a gate: the old model.py dropped it
    # silently. They must export as gcg=off, not raise in MBRSETClassifier.
    assert export_coreml.gcg_from_args({"no_gcg": False, "backbone": "timm:mobilenetv4_conv_small"}) is False
    assert export_coreml.gcg_from_args({"use_gcg": True, "backbone": "timm:x"}) is True  # explicit key wins


# --------------------------------------------------------------------------- #
# (b) operating point + binary_report
# --------------------------------------------------------------------------- #
def test_operating_point_is_highest_threshold_meeting_target():
    pytest.importorskip("sklearn")
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    labels = np.array([1, 1, 0, 1, 1, 0, 0, 0])
    target = 0.75
    thr, sens, spec = evaluate_deploy.operating_point(scores, labels, target_sens=target)

    # Brute force over every achievable threshold: the chosen one must be the
    # LARGEST whose sensitivity reaches the target.
    def sens_at(t):
        pred = scores >= t
        return (pred & (labels == 1)).sum() / (labels == 1).sum()
    feasible = [t for t in np.unique(scores) if sens_at(t) >= target]
    # 0.6 lies on a vertical ROC segment; sklearn's default drop_intermediate
    # would prune it and return 0.5 — this is the regression guard for that.
    assert thr == pytest.approx(max(feasible))
    assert thr == pytest.approx(0.6)
    assert sens == pytest.approx(0.75) and spec == pytest.approx(0.75)

    rep = evaluate_deploy.binary_report(scores, labels, thr)
    assert rep["sensitivity"] == pytest.approx(sens)
    assert rep["specificity"] == pytest.approx(spec)
    assert (rep["tp"], rep["fp"], rep["tn"], rep["fn"]) == (3, 1, 3, 1)
    assert rep["tp"] + rep["fp"] + rep["tn"] + rep["fn"] == len(labels)
    assert rep["ppv"] == pytest.approx(3 / 4) and rep["npv"] == pytest.approx(3 / 4)


def test_operating_point_falls_back_to_youden_when_unreachable():
    pytest.importorskip("sklearn")
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 1, 0, 0])
    # sensitivity 1.0 is reachable only at thr<=0.1 where spec=0; target 1.0 is
    # still "met" so we get that; target > 1 is unreachable -> Youden's J.
    thr_j, s, sp = evaluate_deploy.operating_point(scores, labels, target_sens=1.5)
    thr_y, s2, sp2 = evaluate_deploy.operating_point(scores, labels, target_sens=None)
    assert thr_j == thr_y and (s, sp) == (s2, sp2)


# --------------------------------------------------------------------------- #
# (c) Python version gate
# --------------------------------------------------------------------------- #
def test_python_supported_gate():
    assert check_env.python_supported((3, 11)) is True
    assert check_env.python_supported((3, 9, 6, "final", 0)) is True
    assert check_env.python_supported((3, 12)) is True
    assert check_env.python_supported((3, 14)) is False
    assert check_env.python_supported((3, 13)) is True
    assert check_env.python_supported((3, 8)) is False
    assert check_env.python_supported((2, 7)) is False
