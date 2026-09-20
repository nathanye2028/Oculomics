"""Core ML export path for the retinal-age clock: the deploy wrapper returns years,
the checkpoint kind is detected, and both the checkpoint and the random-weight
probe paths build (no coremltools conversion here; that is the script's own run)."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_coreml import (AgeDeployWrapper, DEFAULT_TOL, IMAGENET_MEAN, IMAGENET_STD,   # noqa: E402
                           OUT_NAMES, build_wrapped_age, detect_model_kind, preprocess_spec,
                           reference_output)
from train_retinal_age import RetinalAgeModel                  # noqa: E402


def _ns(**kw):
    base = dict(model="auto", checkpoint=None, bn_stats="source", backbone=None, image_size=None,
                head="linear", ldl_bins=100, no_gcg=False, classes=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_age_wrapper_folds_normalisation_and_head_into_years():
    for head, centers in (("linear", None), ("ldl", np.arange(0, 100, dtype=np.float32))):
        m = RetinalAgeModel("mobilenetv3_small", 64, head, centers, 50.0, 15.0, pretrained=False).eval()
        m.tta = True                                            # the wrapper must still run one view
        w = AgeDeployWrapper(m).eval()
        m.tta = False
        x255 = torch.rand(2, 3, 64, 64) * 255.0
        with torch.no_grad():
            y = w(x255)
            ref = m((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0))
        assert y.shape == (2, 1)
        torch.testing.assert_close(y.squeeze(1), ref, atol=1e-4, rtol=0)
        # reference_output() (the verify path) reads years off the unwrapped net
        with torch.no_grad():
            r = reference_output(w, ((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0))[:1], "age")
        assert r.shape == (1, 1) and abs(float(r[0, 0]) - float(ref[0])) < 1e-4


def test_kind_detection_and_spec():
    assert detect_model_kind(_ns(), {"task": "retinal_age", "args": {"backbone": "x"}}) == "age"
    assert detect_model_kind(_ns(), {"args": {"backbone": "x", "task": "dr_referable"}}) == "cls"
    assert detect_model_kind(_ns(model="age"), None) == "age"
    assert OUT_NAMES["age"] == "age_years" and DEFAULT_TOL["age"] > DEFAULT_TOL["cls"]
    spec = preprocess_spec("age", 384, "source", None, "fp16")
    assert "YEARS" in spec["postprocess"] and "preprocess.2_resize" in spec


def test_build_wrapped_age_from_checkpoint_and_random(tmp_path):
    m = RetinalAgeModel("mobilenetv3_small", 64, "ldl", np.arange(20, 90, dtype=np.float32), 55.0, 12.0,
                        pretrained=False).eval()
    ck = {"task": "retinal_age", "model": m.net.state_dict(), "head": m.head_spec(),
          "target_norm": {"mean": 55.0, "std": 12.0},
          "args": {"backbone": "mobilenetv3_small", "image_size": 64, "tta": True, "dropout": 0.2}}
    torch.save(ck, tmp_path / "age.pt")
    w, trained, size = build_wrapped_age(_ns(checkpoint=str(tmp_path / "age.pt"), image_size=96), ck)
    assert trained and size == 64 and w.net.head_type == "ldl" and w.net.tta is False
    with torch.no_grad():
        y = w(torch.rand(1, 3, 64, 64) * 255.0)
        ref = m((torch.zeros(1, 3, 64, 64)))
    assert y.shape == (1, 1) and 20.0 <= float(y) <= 90.0 and ref.shape == (1,)
    # random-weight probe: architecture only
    w2, trained2, size2 = build_wrapped_age(_ns(backbone="mobilenetv3_small", image_size=64, head="ldl",
                                                ldl_bins=50), None)
    assert not trained2 and size2 == 64 and w2.net.n_bins == 50
    assert w2(torch.rand(1, 3, 64, 64) * 255.0).shape == (1, 1)
