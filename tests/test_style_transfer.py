"""Tabletop -> handheld synthesis: the stylizers' invariants and the experiment
end to end on the synthetic camera pair from test_retinareach."""
import numpy as np
import pandas as pd
import pytest
import torch

from style_transfer import (FDAStylizer, Generator, ReferenceBank, colour_transfer, fda, handheld_field,
                            match_sharpness, sharpness, train_translator)
from style_transfer_xfer import grade_matched, pairing
from test_retinareach import BACKBONE, _make_brset, _make_mbrset


def _disc(n, colour, size=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    m = ((yy ** 2 + xx ** 2) <= 0.9).float()
    x = torch.tensor(colour).view(1, 3, 1, 1) + 0.15 * torch.rand(n, 3, size, size, generator=g)
    return (x * m).clamp(0, 1)


def test_fda_takes_the_reference_colour_and_keeps_the_field_of_view():
    x, ref = _disc(2, (0.6, 0.3, 0.15)), _disc(2, (0.3, 0.3, 0.6), seed=1)
    y = fda(x, ref, beta=0.05)
    inside = x.max(dim=1, keepdim=True).values > 12 / 255
    assert torch.all(y[(~inside).expand_as(y)] == 0)
    mean = lambda t: (t * inside).sum(dim=(0, 2, 3)) / inside.sum()
    assert mean(y)[2] > mean(x)[2] + 0.1 and mean(y)[0] < mean(x)[0] - 0.1
    assert torch.allclose(fda(x, x, beta=0.05), x, atol=1e-4)          # own amplitude = no change
    z = colour_transfer(x, ref)
    assert torch.allclose(mean(z), mean(ref), atol=0.02) and torch.allclose(colour_transfer(x, x), x, atol=1e-4)


def test_sharpness_matching_only_removes_detail():
    x = _disc(3, (0.5, 0.3, 0.2))
    s = sharpness(x)
    assert torch.equal(match_sharpness(x, s * 10), x)                   # sharper reference: untouched
    y = match_sharpness(x, s / 50)
    assert torch.all(sharpness(y) < s)


def test_handheld_field_is_a_full_circle():
    y = handheld_field(torch.ones(1, 3, 41, 41), zoom=1.2)
    assert y[0, 0, 20, 20] == 1 and y[0, 0, 0, 0] == 0 and y[0, 0, 0, 20] == 1


def test_stylizer_is_reproducible_and_named():
    bank = ReferenceBank(_disc(6, (0.3, 0.3, 0.6), seed=2))
    x = _disc(4, (0.6, 0.3, 0.15))
    for kw, name in (({"colour_only": True}, "colour"), ({}, "fda"), ({"sharp": True}, "fda_sharp"), ({"sharp": True, "zoom": 1.15}, "fda_geo")):
        s = FDAStylizer(bank, **kw)
        a = s(x, torch.Generator().manual_seed(3))
        assert s.name == name and a.shape == x.shape and 0 <= a.min() and a.max() <= 1
        assert torch.equal(a, s(x, torch.Generator().manual_seed(3)))


def test_matched_pairing_shows_both_sides_the_same_grades_and_only_diabetics():
    rng = np.random.default_rng(0)
    g_src = rng.choice([0, 1, 2, 3, 4], 2000, p=[0.92, 0.02, 0.03, 0.02, 0.01]).astype(float)
    g_tgt = rng.choice([0, 1, 2, 3], 800, p=[0.7, 0.1, 0.15, 0.05]).astype(float)
    dm = (rng.random(2000) < 0.5).astype(float)
    pr = pairing("matched", g_src, g_tgt, dm, 4000, seed=1)
    s, t = np.array(pr["src"]), np.array(pr["tgt"])
    assert np.array_equal(g_src[s], g_tgt[t]) and np.all(dm[s] == 1) and pr["diabetic_only"]
    assert 4 not in pr["used_grades"]                                   # absent on the handheld side
    assert abs((g_tgt[t] >= 2).mean() - (g_tgt >= 2).mean()) < 0.03     # the handheld grade mix
    nv = pairing("naive", g_src, g_tgt, dm, 4000, seed=1)
    assert (g_src[nv["src"]] >= 2).mean() < 0.5 * (g_tgt[nv["tgt"]] >= 2).mean()   # the confound
    assert not pairing("matched", g_src, g_tgt, None, 100, seed=1)["diabetic_only"]
    ia, ib = grade_matched(g_src, g_tgt, 50, rng)
    assert np.array_equal(np.sort(g_src[ia]), np.sort(g_tgt[ib]))


def test_translator_trains_and_is_fully_convolutional():
    a, b = _disc(2, (0.6, 0.3, 0.15), size=32), _disc(2, (0.3, 0.3, 0.6), size=32, seed=1)
    st = train_translator(lambda: (a, b), 3, torch.device("cpu"), ngf=4, n_blocks=1, log_every=1,
                          log=lambda s: None)
    assert len(st["history"]) == 3 and all(np.isfinite(list(h.values())).all() for h in st["history"])
    g = Generator(4, 1)
    g.load_state_dict(st["G_ab"])
    assert g(torch.zeros(1, 3, 48, 40)).shape == (1, 3, 48, 40)


def test_end_to_end_synthetic(tmp_path):
    from style_transfer_xfer import main
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 45, rng)
    _make_mbrset(M, 45, rng)
    lab = pd.read_csv(B / "labels.csv")
    lab["diabetes"] = np.where(lab.patient_id % 2 == 0, "yes", "no")
    lab.to_csv(B / "labels.csv", index=False)

    def retrain(src, out, ck):
        return train(["--source-root", str(src), "--target-root", str(M),
                      "--probe-targets", "hypertension", "--backbone", BACKBONE, "--no-pretrained",
                      "--image-size", "64", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
                      "--device", "cpu", "--calib-batch", "4", "--calib-sizes", "4",
                      "--calib-repeats", "1", "--envelope-repeats", "2", "--prior-strengths", "0",
                      "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1",
                      "--min-test-pos-patients", "2", "--demo-n", "4", "--n-shuffle", "0",
                      "--ablation-frac", "0", "--out", str(out), "--ckpt", str(ck)])

    base = tmp_path / "base.pt"
    assert retrain(B, tmp_path / "out_base", base) == 0
    roots = ["--source-root", str(B), "--target-root", str(M), "--image-size", "64",
             "--num-workers", "0", "--device", "cpu"]
    tr = {}
    for mode in ("matched", "naive"):
        tr[mode] = tmp_path / f"{mode}.pt"
        assert main(["fit", "--mode", mode, *roots, "--out", str(tr[mode]), "--iters", "2",
                     "--crop", "32", "--ngf", "4", "--n-blocks", "1", "--min-per-stratum", "2"]) == 0
    assert torch.load(tr["matched"], weights_only=False)["pairing"]["diabetic_only"]
    named = ["--translator", f"cyc_matched={tr['matched']}", f"cyc_naive={tr['naive']}"]

    out = tmp_path / "check"
    assert main(["check", "--ckpt", str(base), *named, "--out", str(out), "--bank", "8",
                 "--calib-n", "8", "--calib-batch", "4", "--grid-n", "4", "--num-workers", "0",
                 "--batch-size", "8", "--device", "cpu"]) == 0
    df = pd.read_csv(out / "check.csv").set_index("method")
    assert list(df.index) == ["REAL handheld", "identity", "colour", "fda", "fda_sharp", "fda_geo",
                              "cyc_matched", "cyc_naive"]
    assert df.loc["identity", "vessel_r"] == pytest.approx(1.0, abs=1e-4)
    assert df.loc["identity", "dlogit_icdr0"] == 0
    assert df.loc["identity", "p_handheld_icdr0"] < 0.5 < df.loc["REAL handheld", "p_handheld_icdr0"]
    assert df.loc["fda", "cam_dist"] < df.loc["identity", "cam_dist"]       # the colour cast transfers
    assert df.loc["identity", "n_icdr0_nondm"] > 0 and (out / "grid.jpg").exists()

    arms = {}
    for method in ("identity", "fda_sharp", "cyc_matched"):
        root = tmp_path / "roots" / method
        assert main(["export", "--method", method, *roots, *named, "--out-root", str(root),
                     "--bank", "8", "--batch-size", "8"]) == 0
        assert len(list((root / "fundus_photos").iterdir())) == 90
        arms[method] = tmp_path / f"{method}.pt"
        assert retrain(root, tmp_path / f"out_{method}", arms[method]) == 0
    assert main(["score", *roots, "--ckpt", *[f"{k}={v}" for k, v in arms.items()],
                 "--out", str(tmp_path / "score"), "--batch-size", "8"]) == 0
    sc = pd.read_csv(tmp_path / "score" / "score.csv")
    assert len(sc) == 3 * 2 * 2 and set(sc.camera) == {"tabletop", "handheld"}
    assert sc.auroc.between(0, 1).all()
