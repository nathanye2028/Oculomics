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


def test_views_collate_and_simulate_tabletop_touch_only_the_tabletop_photos():
    """A --mix-phone-sim batch: tabletop items carry two views, handheld items one; the
    simulator gets the tabletop crops with their own whole-FOV views, in order, and its
    renders go back into those slots (a mix-up would train a render under another
    photo's label, or re-render a real handheld photo)."""
    from train_phone_ceiling import simulate_tabletop, views_collate
    item = lambda k, tab: {"image": ({"crop": torch.full((3, 4, 4), float(k)), "full": torch.full((3, 6, 6), float(k))}
                                     if tab else torch.full((3, 4, 4), float(k))),
                           "label": torch.tensor([float(k), 0.0]), "file": f"f{k}"}
    kinds = [True, False, False, True, True, False]
    b = views_collate([item(k, t) for k, t in enumerate(kinds)])
    assert b["image"]["crop"].shape == (6, 3, 4, 4) and b["image"]["full"].shape == (3, 3, 6, 6)
    assert b["image"]["tabletop"].tolist() == kinds and b["label"][:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert b["image"]["full"][:, 0, 0, 0].tolist() == [0, 3, 4] and b["file"] == [f"f{k}" for k in range(6)]

    class Sim:                                    # "renders" a photo as 100 + its whole-FOV view's value
        def augment_views(self, crop, full, p, gen):
            assert crop.shape[0] == full.shape[0] and torch.equal(crop[:, 0, 0, 0], full[:, 0, 0, 0])
            self.p = p
            return 100 + full[:, :, :4, :4]

    sim = Sim()
    x = simulate_tabletop(sim, b["image"], 0.7, torch.Generator(), torch.device("cpu"), False)
    assert x[:, 0, 0, 0].tolist() == [100, 1, 2, 103, 104, 5] and sim.p == 0.7
    none = views_collate([item(k, False) for k in range(3)])               # a batch without a tabletop photo
    assert none["image"]["full"].shape[0] == 0
    assert torch.equal(simulate_tabletop(sim, none["image"], 1.0, torch.Generator(), torch.device("cpu"), False),
                       none["image"]["crop"])


def test_mixed_with_simulated_tabletop_photos(tmp_path, monkeypatch):
    """--mix-phone-sim: the mixed run with its tabletop photos re-rendered as handheld
    ones (model mixed_sim), read against the mixed run of the same seed."""
    import phone_sim as ps
    from train_phone_ceiling import family, main as ceiling
    rng = np.random.default_rng(5)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 40, rng)
    _make_mbrset(M, 40, rng)
    prof = tmp_path / "ck_phone_sim" / "profile.json"
    assert ps.main(["fit", "--target-root", str(M), "--source-root", str(B), "--image-size", "64", "--validate", "0",
                    "--num-workers", "0", "--device", "cpu", "--out", str(prof)]) == 0
    small = ["--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1", "--batch-size", "8",
             "--num-workers", "0", "--device", "cpu", "--n-boot", "10", "--ema-decay", "0"]
    mix = ["--root", str(M), "--mix-root", str(B)]
    with pytest.raises(SystemExit):                                       # nothing to re-render without tabletop photos
        ceiling(["--root", str(M), "--mix-phone-sim", str(prof), *small])
    seen = []
    orig = ps.PhoneSimulator.augment_views

    def views(self, crop, full, p, gen):
        out = orig(self, crop, full, p, gen)
        seen.append((tuple(crop.shape), tuple(full.shape), p, bool((out != crop).flatten(1).any(1).all())))
        return out

    monkeypatch.setattr(ps.PhoneSimulator, "augment_views", views)
    out = tmp_path / "sim" / "seed0"
    assert ceiling([*mix, "--mix-phone-sim", str(prof), *small, "--out", str(out),
                    "--ckpt", str(tmp_path / "ck_sim" / "seed0.pt")]) == 0
    # every tabletop photo of every batch went through the simulator, from a 96 px whole-FOV view
    assert seen and all(c[0] == f[0] and c[1:] == (3, 64, 64) and f[1:] == (3, 96, 96) and p == 1.0 and changed
                        for c, f, p, changed in seen)
    assert 0 < sum(c[0] for c, *_ in seen) < 8 * len(seen)                 # the handheld photos were not
    res = json.loads((out / "results.json").read_text())
    assert res["model"] == "mixed_sim" and family("mixed_sim") == "mixed_sim" and res["args"]["gpu_aug"] is True
    assert len(res["phone_sim_profile"]["sha256"]) == 64 and res["phone_sim_profile"]["train_size"] == 96
    assert res["phone_sim_profile"]["format"] == ps.FORMAT and res["phone_sim_profile"]["p"] == 1.0
    assert set(pd.read_csv(out / "ceiling.csv")["model"]) == {"mixed_sim"}
    assert set(pd.read_csv(out / "home_camera.csv")["model"]) == {"mixed_sim"}
    ck = torch.load(tmp_path / "ck_sim" / "seed0.pt", map_location="cpu", weights_only=False)
    assert ck["phone_sim_profile"]["sha256"] == res["phone_sim_profile"]["sha256"]

    # a bank holding a handheld validation / test photo, or none listed, is refused
    blob = json.loads(prof.read_text())
    test_file = pd.read_csv(out / "split_handheld.csv").query("split == 'test'")["file"].iloc[0]
    for files, why in ((blob["source"]["files"] + [test_file], "validation / test split"), ([], "no bank photos"),
                       (blob["source"]["files"] + ["elsewhere.jpg"], "not in this run's handheld table")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({**blob, "source": {**blob["source"], "files": files}}))
        with pytest.raises(SystemExit, match=why):
            ceiling([*mix, "--mix-phone-sim", str(bad), *small, "--out", str(tmp_path / "bad_out"),
                     "--ckpt", str(tmp_path / "bad.pt")])

    # read against the mixed run of the same seed, paired
    monkeypatch.setattr(ps.PhoneSimulator, "augment_views", orig)
    assert ceiling([*mix, *small, "--out", str(tmp_path / "mixed" / "seed0"),
                    "--ckpt", str(tmp_path / "ck_mixed" / "seed0.pt")]) == 0
    assert list(pd.read_csv(out / "predictions_mixed_sim.csv")["file"]) == \
        list(pd.read_csv(tmp_path / "mixed" / "seed0" / "predictions_mixed.csv")["file"])
    assert ceiling(["--summarize", str(tmp_path / "sim"), "--versus", str(tmp_path / "mixed"),
                    str(tmp_path / "nowhere")]) == 0
    summary = (tmp_path / "sim" / "summary.txt").read_text()
    assert "mixed_sim" in summary and "minus mixed" in summary and "paired by seed" in summary
    assert "oracle_spec" in summary and "no seed in both" in summary
    # ... and its numbers: this sweep's mean, the other's, the difference, in that order
    first = lambda d, m: pd.read_csv(d / "seed0" / "ceiling.csv").query("model == @m and target == 'dr_referable'").iloc[0]
    sim_au, mix_au = first(tmp_path / "sim", "mixed_sim").auroc, first(tmp_path / "mixed", "mixed").auroc
    line = next(l.split() for l in summary[summary.index("minus mixed"):].splitlines()
                if l.split()[:2] == ["dr_referable", "auroc"])
    assert line[2:5] == [f"{sim_au:.3f}", f"{mix_au:.3f}", f"{sim_au - mix_au:.3f}"]
    assert line[-1] == f"{sim_au - mix_au:+.3f}"


def test_the_paired_summary_subtracts_the_other_sweep_seed_by_seed(tmp_path):
    """versus_table: this sweep minus the other, each seed against the same seed, per
    metric; a seed only one sweep has is left out; two sweeps of one model keep a column
    each."""
    from train_phone_ceiling import versus_table

    def sweep(name, model, by_seed):
        for seed, (au, pa, sp) in by_seed.items():
            d = tmp_path / name / f"seed{seed}"
            d.mkdir(parents=True)
            pd.DataFrame([{"model": model, "target": "dr_referable", "auroc": au, "patient_auroc": pa, "oracle_spec": sp},
                          {"model": "ck_rr_canon_seed0_trained", "target": "dr_referable", "auroc": 0.1,
                           "patient_auroc": 0.1, "oracle_spec": 0.1}]).to_csv(d / "ceiling.csv", index=False)
        return str(tmp_path / name)

    sim = sweep("sim", "mixed_sim", {0: (0.90, 0.80, 0.70), 1: (0.80, 0.85, 0.75), 2: (0.99, 0.99, 0.99)})
    mix = sweep("mix", "mixed", {1: (0.86, 0.80, 0.70), 0: (0.85, 0.82, 0.60)})          # no seed 2
    txt = "\n".join(versus_table(sim, [mix]))
    row = lambda metric, t=None: next(l.split() for l in (t or txt).splitlines() if l.split()[:2] == ["dr_referable", metric])
    assert "mixed_sim (" in txt and "minus mixed (" in txt
    #                             this   other  difference  seeds above 0, seeds, each seed
    assert row("auroc")[2:] == ["0.850", "0.855", "-0.005", "1", "2", "+0.050", "-0.060"]
    assert row("patient_auroc")[2:] == ["0.825", "0.810", "0.015", "1", "2", "-0.020", "+0.050"]
    assert row("oracle_spec")[2:] == ["0.725", "0.650", "0.075", "2", "2", "+0.100", "+0.050"]
    # the seed the other sweep lacks is the first: still each seed against the same seed (paired
    # by position, seed 0 here would meet the other sweep's seed 1)
    late = sweep("late", "mixed", {1: (0.86, 0.80, 0.70), 2: (0.90, 0.90, 0.90)})
    txt3 = "\n".join(versus_table(sim, [late]))
    assert row("auroc", txt3)[2:] == ["0.895", "0.880", "0.015", "1", "2", "-0.060", "+0.090"]
    other = sweep("sim_b", "mixed_sim", {0: (0.80, 0.80, 0.70), 1: (0.70, 0.85, 0.75)})
    txt2 = "\n".join(versus_table(sim, [other]))
    assert "mixed_sim (this)" in txt2 and "mixed_sim (other)" in txt2
    assert row("auroc", txt2)[2:5] == ["0.850", "0.750", "0.100"]
