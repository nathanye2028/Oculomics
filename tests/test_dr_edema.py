"""train_dr_edema.py: the four-way class arithmetic, the sampler, and one tiny end-to-end run."""
import json

import numpy as np

from test_retinareach import _make_brset, _make_mbrset
from train_dr_edema import CLASSES, four_way, joint_class, joint_weights, main, pick_threshold


def test_joint_class_and_missing():
    y = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [np.nan, 1], [1, np.nan]], dtype=float)
    assert joint_class(y).tolist() == [0, 1, 2, 3, -1, -1]


def test_joint_weights_favour_rare_classes_without_full_balancing():
    cls = np.array([0] * 100 + [1] * 25 + [3] * 4 + [-1] * 3)
    w = joint_weights(cls, 0.5).numpy()
    assert np.isclose(w[100] / w[0], 2.0) and np.isclose(w[125] / w[0], 5.0)
    assert np.allclose(w[-3:], w[0])                     # unlabelled rows = most common class
    assert np.allclose(joint_weights(cls, 0.0).numpy()[:129], 1.0)


def test_four_way_metrics_on_a_known_table():
    y = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [np.nan, 0]], dtype=float)
    p = np.array([[.1, .1], [.9, .1], [.9, .1], [.1, .1], [.9, .9], [.9, .1], [.1, .9], [.9, .1]])
    m = four_way(y, p, [0.5, 0.5])
    assert m["n"] == 7                                    # the half-labelled row is not a 4-way example
    assert m["confusion"] == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    assert np.isclose(m["accuracy"], 4 / 7)
    assert np.isclose(m["balanced_accuracy"], np.mean([.5, .5, 1, .5]))
    assert m["heads"]["edema"]["n"] == 8 and m["heads"]["dr"]["n"] == 7
    assert m["per_class"]["edema_only"]["support"] == 1


def test_pick_threshold_separates_a_separable_head():
    y = np.array([0, 0, 0, 1, 1, np.nan])
    p = np.array([.1, .2, .3, .7, .8, .5])
    assert 0.3 < pick_threshold(y, p, "youden", 0.85) <= 0.7
    assert pick_threshold(y, p, "sens", 0.85) == 0.7
    assert pick_threshold(np.zeros(4), np.linspace(0, 1, 4), "youden", 0.85) == 0.5


def test_end_to_end_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    _make_brset(tmp_path / "brset", 40, rng)
    _make_mbrset(tmp_path / "mbrset", 20, rng)
    out, ck = tmp_path / "out", tmp_path / "ck" / "s0.pt"
    rc = main(["--root", str(tmp_path / "brset"), "--external-root", str(tmp_path / "mbrset"),
               "--dr-def", "referable", "--backbone", "timm:mobilenetv4_conv_small.e2400_r224_in1k",
               "--no-pretrained", "--image-size", "64", "--epochs", "1", "--batch-size", "8",
               "--num-workers", "0", "--device", "cpu", "--out", str(out), "--ckpt", str(ck)])
    assert rc == 0 and ck.exists()
    res = json.loads((out / "results.json").read_text())
    assert res["classes"] == list(CLASSES) and set(res["thresholds"]) == {"dr", "edema"}
    for name, n in (("test", None), ("external", 40)):
        m = res[name]
        assert np.array(m["confusion"]).shape == (4, 4) and np.array(m["confusion"]).sum() == m["n"]
        assert n is None or m["n"] == n
        assert (out / f"confusion_{name}.csv").exists() and (out / f"predictions_{name}.csv").exists()
