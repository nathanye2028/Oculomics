"""train_phone_ceiling.py: phone-trained reference model on a RetinaReach run's own
handheld split, with that run's checkpoint scored on the same test rows."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_retinareach import BACKBONE, _make_brset, _make_mbrset


def test_phone_ceiling_on_a_retinareach_split(tmp_path):
    from retinareach import build_from_checkpoint
    from train_phone_ceiling import checkpoint_tag, family, main as ceiling
    from train_retinareach import main as retinareach
    rng = np.random.default_rng(3)
    _make_brset(tmp_path / "brset", 40, rng)
    _make_mbrset(tmp_path / "mbrset", 40, rng)
    rr, rr_ck = tmp_path / "rr", tmp_path / "ck_rr" / "seed0.pt"
    small = ["--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1",
             "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--n-boot", "10"]
    assert retinareach(["--source-root", str(tmp_path / "brset"),
                        "--target-root", str(tmp_path / "mbrset"), "--probe-targets", "hypertension",
                        *small, "--calib-batch", "4", "--calib-sizes", "4", "--calib-repeats", "1",
                        "--prior-strengths", "0", "--min-probe-pos", "2", "--envelope-repeats", "2",
                        "--demo-n", "8", "--n-shuffle", "2", "--ablation-frac", "0",
                        "--cv-repeats", "1", "--min-test-pos-patients", "2",
                        "--out", str(rr), "--ckpt", str(rr_ck)]) == 0

    out = tmp_path / "ceiling" / "seed0"
    assert ceiling(["--root", str(tmp_path / "mbrset"), "--split-file", str(rr / "split_handheld.csv"),
                    "--compare", str(rr_ck), *small, "--ema-decay", "0",
                    "--out", str(out), "--ckpt", str(tmp_path / "ck_pc" / "seed0.pt")]) == 0

    # the RetinaReach run's handheld partition, unchanged; no patient in two splits
    mine, theirs = pd.read_csv(out / "split_handheld.csv"), pd.read_csv(rr / "split_handheld.csv")
    assert mine.set_index("file")["split"].to_dict() == theirs.set_index("file")["split"].to_dict()
    assert mine.groupby("patient")["split"].nunique().max() == 1
    res = json.loads((out / "results.json").read_text())
    assert res["split_file_agrees"] is True

    tag = checkpoint_tag(str(rr_ck))
    assert tag == "ck_rr_seed0"
    models = {"phone", f"{tag}_trained", f"{tag}_calibrated"}
    ceil = pd.read_csv(out / "ceiling.csv")
    assert set(ceil["model"]) == models
    assert {family(m) for m in models} == {"phone", "compare_trained", "compare_calibrated"}
    assert ceil[["auroc", "oracle_spec"]].notna().all().all()
    # the oracle point sits at (or just above) the target sensitivity by construction
    assert (ceil["oracle_sens"] >= 0.85 - 1e-9).all()

    # the compared checkpoint scores exactly the rows, and gives exactly the scores,
    # that its own run saved for the handheld test split
    ours = pd.read_csv(out / f"predictions_{tag}_trained.csv")
    saved = pd.read_csv(rr / "predictions_handheld_trained.csv")
    assert list(ours["file"]) == list(saved["file"])
    assert np.allclose(ours["p_dr_referable"], saved["p_dr_referable"], atol=1e-5)
    assert np.allclose(ours["threshold_dr_referable"], saved["threshold_dr_referable"])
    assert list(pd.read_csv(out / "predictions_phone.csv")["file"]) == list(saved["file"])

    gap = pd.read_csv(out / "transfer_gap.csv")
    assert set(gap["versus"]) == models - {"phone"}
    assert (gap["delta_lo"] <= gap["delta"]).all() and (gap["delta"] <= gap["delta_hi"]).all()
    ref = pd.read_csv(out / "referral.csv")
    assert set(ref["model"]) == models and set(ref["rule"]) == {"dr_referable", "dr_or_edema"}

    # the phone checkpoint rebuilds like any RetinaReach checkpoint
    ck = torch.load(tmp_path / "ck_pc" / "seed0.pt", map_location="cpu", weights_only=False)
    net = build_from_checkpoint(ck)
    assert net.targets == ["dr_referable", "edema"] and ck["thresholds"]["trained"]["dr_referable"] > 0

    assert ceiling(["--summarize", str(tmp_path / "ceiling")]) == 0
    summary = (tmp_path / "ceiling" / "summary.txt").read_text()
    assert "image level" in summary and "per patient" in summary and "transfer gap" in summary

    # mixed: the run's own tabletop train photos join the phone's; same phone test rows,
    # and the home-camera check on that run's own tabletop test rows
    mix = tmp_path / "mixed" / "seed0"
    assert ceiling(["--root", str(tmp_path / "mbrset"), "--split-file", str(rr / "split_handheld.csv"),
                    "--mix-root", str(tmp_path / "brset"),
                    "--mix-split-file", str(rr / "split_tabletop.csv"),
                    "--compare", str(rr_ck), *small, "--ema-decay", "0",
                    "--out", str(mix), "--ckpt", str(tmp_path / "ck_mix" / "seed0.pt")]) == 0
    res = json.loads((mix / "results.json").read_text())
    assert res["model"] == "mixed" and res["mix_split_file_agrees"] is True
    tab = pd.read_csv(rr / "split_tabletop.csv").set_index("file")["split"]
    assert pd.read_csv(mix / "split_tabletop.csv").set_index("file")["split"].to_dict() == tab.to_dict()
    assert res["n"]["tabletop_train"] == int((tab == "train").sum())
    mixed_models = {"mixed", f"{tag}_trained", f"{tag}_calibrated"}
    assert set(pd.read_csv(mix / "ceiling.csv")["model"]) == mixed_models
    assert list(pd.read_csv(mix / "predictions_mixed.csv")["file"]) == list(saved["file"])
    # the compared checkpoint at home gives exactly the tabletop scores its run saved
    home = pd.read_csv(mix / "home_camera.csv")
    assert set(home["model"]) == {"mixed", f"{tag}_trained"}
    ours = pd.read_csv(mix / f"predictions_home_{tag}_trained.csv")
    saved_home = pd.read_csv(rr / "predictions_tabletop_trained.csv")
    assert list(ours["file"]) == list(saved_home["file"])
    assert np.allclose(ours["p_dr_referable"], saved_home["p_dr_referable"], atol=1e-5)
    gap = pd.read_csv(mix / "transfer_gap.csv")
    assert set(gap["model"]) == {"mixed"} and set(gap["test"]) == {"handheld", "tabletop"}
    assert set(gap.loc[gap["test"] == "tabletop", "versus"]) == {f"{tag}_trained"}
    assert ceiling(["--summarize", str(tmp_path / "mixed")]) == 0
    summary = (tmp_path / "mixed" / "summary.txt").read_text()
    assert "home camera" in summary and "mixed minus" in summary and "mixed" in summary


def test_mixed_weights_split_the_draws_by_camera_then_class():
    from train_phone_ceiling import mixed_weights
    y_tab = np.array([1, 0, 0, 0, 0, 0, 0, 0, np.nan, 0], dtype=float)
    y_phone = np.array([1, 1, 0, 0], dtype=float)
    w = mixed_weights(y_tab, y_phone, 0.3).numpy()
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[10:].sum(), 0.3) and np.isclose(w[:10].sum(), 0.7)
    tab, ph = w[:10], w[10:]
    assert np.isclose(tab[y_tab == 1].sum(), tab[y_tab == 0].sum())      # balanced within a camera
    assert np.isclose(ph[y_phone == 1].sum(), ph[y_phone == 0].sum())
    flat = mixed_weights(y_tab, y_phone, 0.3, balance=False).numpy()
    assert np.allclose(flat[:10], 0.07) and np.allclose(flat[10:], 0.075)


def test_mixed_run_keeps_one_tabletop_camera(tmp_path):
    """--mix-camera without a split file: the tabletop split is recomputed on that
    camera's rows only, and no other camera's photo is trained on or tested."""
    from capability_gate import multitask_split
    from metadata_model import load_table
    from brset_dataset import filter_camera
    from train_phone_ceiling import main as ceiling
    rng = np.random.default_rng(5)
    _make_brset(tmp_path / "brset", 30, rng, camera_of=lambda pid: "Canon CR" if pid % 4 else "NIKON NF5050")
    _make_mbrset(tmp_path / "mbrset", 30, rng)
    out = tmp_path / "mix" / "seed0"
    assert ceiling(["--root", str(tmp_path / "mbrset"), "--mix-root", str(tmp_path / "brset"),
                    "--mix-camera", "Canon CR", "--phone-share", "0.6",
                    "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1",
                    "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--n-boot", "10",
                    "--out", str(out), "--ckpt", str(tmp_path / "ck.pt")]) == 0
    canon = filter_camera(load_table(str(tmp_path / "brset"), "brset")[0], "Canon CR")
    want = multitask_split(canon, "dr_referable", seed=42)
    split = pd.read_csv(out / "split_tabletop.csv")
    assert set(split["file"]) == set(canon["file"])
    for k in ("train", "test"):
        assert set(split.loc[split["split"] == k, "file"]) == set(want[k]["file"])
    assert set(pd.read_csv(out / "home_camera.csv")["model"]) == {"mixed"}
    assert set(pd.read_csv(out / "ceiling.csv")["model"]) == {"mixed"}
    with pytest.raises(SystemExit):        # a camera BRSET does not have
        ceiling(["--root", str(tmp_path / "mbrset"), "--mix-root", str(tmp_path / "brset"),
                 "--mix-camera", "Topcon", "--device", "cpu", "--out", str(tmp_path / "x"),
                 "--ckpt", str(tmp_path / "x.pt")])


def test_phone_ceiling_without_a_split_file_or_comparison(tmp_path):
    """No sweep to lean on: the split is recomputed exactly as the trainer would, and
    the summary still runs with an empty transfer_gap.csv."""
    from capability_gate import multitask_split
    from metadata_model import load_table
    from train_phone_ceiling import main as ceiling
    rng = np.random.default_rng(4)
    _make_mbrset(tmp_path / "mbrset", 30, rng)
    out = tmp_path / "pc" / "seed1"
    assert ceiling(["--root", str(tmp_path / "mbrset"), "--backbone", BACKBONE, "--no-pretrained",
                    "--image-size", "64", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
                    "--device", "cpu", "--n-boot", "10", "--seed", "1",
                    "--out", str(out), "--ckpt", str(tmp_path / "ck.pt")]) == 0
    test = multitask_split(load_table(str(tmp_path / "mbrset"), "mbrset")[0], "dr_referable",
                           seed=42)["test"]
    split = pd.read_csv(out / "split_handheld.csv")
    assert set(split.loc[split["split"] == "test", "file"]) == set(test["file"])
    assert set(pd.read_csv(out / "ceiling.csv")["model"]) == {"phone"}
    assert ceiling(["--summarize", str(tmp_path / "pc")]) == 0
