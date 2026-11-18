"""patient_rule.py: photos -> patients (eye means, ungradable photos, retakes), every
threshold from tabletop validation and none from handheld labels, the shipped rule
equal to referral_outcomes.py's, the seed ensemble (photo by photo, the trainer's
threshold rule), and the run end to end on the synthetic camera pair."""
import numpy as np
import pandas as pd
import pytest

import patient_rule as pr
from metadata_model import threshold_at_sensitivity
from referral_outcomes import patient_outcomes


def _photos(rows):
    """rows: (patient, eye, gradable, truth, p)."""
    df = pd.DataFrame(rows, columns=["patient", "eye", "gradable", "truth", "p"])
    df["file"] = [f"f{i}" for i in range(len(df))]
    return df


def test_eye_mean_averages_within_an_eye_then_takes_the_higher_eye():
    ph = _photos([("a", "right", True, 0, 0.9), ("a", "right", True, 1, 0.1), ("a", "left", True, 0, 0.4),
                  ("a", "left", False, 0, 0.99),                     # ungradable: never scores
                  ("b", "right", False, 1, 0.8),                     # no gradable photo: a retake
                  ("c", "right", True, np.nan, 0.3)])                # not graded
    any_ = pr.patient_scores(ph, "any")
    mean = pr.patient_scores(ph, "eye_mean")
    assert any_.loc["a", "score"] == pytest.approx(0.9) and mean.loc["a", "score"] == pytest.approx(0.5)
    assert any_.loc["a", "truth"] == 1                                  # referable in any photo
    assert np.isnan(mean.loc["b", "score"]) and not mean.loc["b", "gradable"] and mean.loc["b", "truth"] == 1
    assert np.isnan(mean.loc["c", "truth"]) and mean.loc["c", "gradable"]
    with pytest.raises(ValueError):
        pr.patient_scores(ph, "median")


def test_photo_frame_reads_quality_and_laterality():
    df = pd.DataFrame({"file": ["x", "y", "z"], "patient": [1, 1, 2],
                       "final_quality": ["yes", "no", np.nan], "laterality": ["right", np.nan, "left"]})
    f = pr.photo_frame(df, np.array([1, 0, np.nan]), np.array([0.2, 0.3, 0.4]))
    assert f.gradable.tolist() == [True, False, True]                   # ungraded quality counts as gradable
    assert f.eye.tolist() == ["right", "file:y", "left"] and f.patient.tolist() == ["1", "1", "2"]


def _pred(rng, arms=("a",), seeds=(0,), n=60, parts=("val", "test")):
    """Photo rows as predict() writes them: two photos per eye, two eyes per patient."""
    rows = []
    for arm in arms:
        for seed in seeds:
            for proto in ("trained", "calibrated"):
                for cam, cparts in (("tabletop", parts), ("handheld", [p for p in parts if p != "val"] or ["val"])):
                    for part in cparts:
                        for i in range(n):
                            ref = i % 4 == 0
                            for eye in ("right", "left"):
                                for k in range(2):
                                    sick = ref and eye == "right"
                                    p = float(np.clip(rng.normal(0.65 if sick else 0.3, 0.15), 0, 1))
                                    rows.append({"arm": arm, "seed": seed, "camera": cam, "part": part,
                                                 "protocol": proto, "file": f"{cam}{part}{i}{eye}{k}",
                                                 "patient": f"{cam}{part}{i}", "eye": eye, "gradable": k == 0 or i % 7,
                                                 "truth": float(sick), "p": p, "shipped_threshold": 0.45})
    df = pd.DataFrame(rows)
    df["gradable"] = df["gradable"].astype(bool)
    return df


def test_thresholds_come_from_tabletop_validation_only():
    rng = np.random.default_rng(0)
    pred = _pred(rng)
    res = pr.evaluate(pred, "test", n_boot=20)
    assert len(res) == 2 * 2 * len(pr.RULES)                             # cameras x protocols x rules
    # the tabletop-validation patients reach the target at each re-set threshold
    val = pred[(pred.camera == "tabletop") & (pred.part == "val") & (pred.protocol == "trained")]
    for rule, agg in pr.RULES[1:]:
        thr = res.query("protocol == 'trained' and rule == @rule").threshold.iloc[0]
        per = pr.patient_scores(val, agg)
        y, s = per.truth.to_numpy(), per.score.to_numpy()
        assert (s[y == 1] >= thr).mean() >= 0.85
    # handheld labels never move a threshold; they only change what it scores
    shuffled = pred.copy()
    hh = shuffled.camera == "handheld"
    shuffled.loc[hh, "truth"] = rng.permutation(shuffled.loc[hh, "truth"].to_numpy())
    res2 = pr.evaluate(shuffled, "test", n_boot=20)
    assert np.allclose(res.threshold, res2.threshold)
    h, h2 = res[res.camera == "handheld"], res2[res2.camera == "handheld"]
    assert not np.allclose(h.sensitivity, h2.sensitivity)
    assert (res.query("rule == 'shipped'").threshold == 0.45).all() and not res.in_sample.any()


def test_the_shipped_rule_is_referral_outcomes_rule():
    rng = np.random.default_rng(1)
    pred = _pred(rng)
    res = pr.evaluate(pred, "test", n_boot=20).set_index(["camera", "protocol", "rule"])
    for cam in pr.CAMERAS:
        ev = pred[(pred.camera == cam) & (pred.part == "test") & (pred.protocol == "calibrated")]
        ref = patient_outcomes(ev.assign(flag=ev.p >= 0.45)[["patient", "gradable", "flag", "truth"]], n_boot=20)
        got = res.loc[(cam, "calibrated", "shipped")]
        assert [got[k] for k in ("tp", "fn", "fp", "tn")] == [ref[k] for k in ("tp", "fn", "fp", "tn")]
        assert got.sensitivity == pytest.approx(ref["sensitivity"]) and got.retakes == ref["retakes"]


def test_val_marks_tabletop_in_sample_and_two_arms_are_paired():
    rng = np.random.default_rng(2)
    pred = _pred(rng, arms=("base", "arm"), seeds=(0, 1), parts=("val",))
    res = pr.evaluate(pred, "val", n_boot=20)
    assert res[res.camera == "tabletop"].in_sample.all() and not res[res.camera == "handheld"].in_sample.any()
    txt = pr.summarise(res, 0.85, 0.825)
    assert "IN-SAMPLE" in txt and "PAIRED: arm minus base" in txt and "--- base ---" in txt
    assert pr.oracle_spec(pr.patient_scores(pred[pred.camera == "handheld"].query("arm == 'base' and seed == 0 "
                                                                                  "and protocol == 'trained'"),
                                            "eye_mean"), 0.85) > 0.5
    with pytest.raises(ValueError, match="tabletop validation"):
        pr.evaluate(pred[~((pred.camera == "tabletop") & (pred.part == "val"))], "val", n_boot=5)


def test_ensemble_averages_each_photo_over_the_seeds():
    rng = np.random.default_rng(3)
    pred = _pred(rng, arms=("a", "b"), seeds=(0, 1, 2), n=24)
    pred = pred[~((pred.arm == "b") & (pred.seed > 0))]                 # b: one seed, no ensemble
    ens = pr.ensemble(pred, 0.85)
    assert set(ens.arm) == {"a"} and (ens.seed == pr.ENSEMBLE_SEED).all() and (ens.members == 3).all()
    a = pred[pred.arm == "a"]
    assert len(ens) == len(a) // 3
    want = a.groupby(pr.PHOTO)["p"].mean().sort_index()
    assert np.allclose(ens.set_index(pr.PHOTO)["p"].sort_index(), want)
    for proto, q in ens.groupby("protocol"):                             # the trainer's rule, on the averages
        v = q[(q.camera == "tabletop") & (q.part == "val")]
        y, s = v.truth.to_numpy().astype(int), v.p.to_numpy()
        thr = q.shipped_threshold.iloc[0]
        assert (q.shipped_threshold == thr).all() and thr == threshold_at_sensitivity(y, s, 0.85)
        assert (s[y == 1] >= thr).mean() >= 0.85
    with pytest.raises(ValueError, match="same photos"):
        pr.ensemble(a.drop(a.index[a.seed == 1][:1]), 0.85)
    with pytest.raises(ValueError, match="repeat an arm and seed"):
        pr.ensemble(pd.concat([a, a.iloc[:3]]), 0.85)


def test_ensemble_ranks_better_than_its_seeds_from_saved_predictions(tmp_path):
    rng = np.random.default_rng(4)
    pred = _pred(rng, arms=("base", "arm"), seeds=(0, 1, 2, 3, 4), n=40)
    img = pr.image_metrics(pd.concat([pred, pr.ensemble(pred)], ignore_index=True), "test")
    h = img[img.camera == "handheld"]
    e = h[h.seed == pr.ENSEMBLE_SEED].set_index(["arm", "protocol"]).auroc
    s = h[h.seed != pr.ENSEMBLE_SEED].groupby(["arm", "protocol"]).auroc.mean()
    assert len(e) == 4 and (e > s.reindex(e.index)).all()               # independent seed noise averages out
    # two saved files pooled, as two score runs would leave them
    files = []
    for arm in ("base", "arm"):
        files.append(str(tmp_path / f"{arm}.csv"))
        pred[pred.arm == arm].to_csv(files[-1], index=False)
    out, plain = tmp_path / "ens", tmp_path / "plain"
    assert pr.main(["--from-predictions", *files, "--ensemble", "--n-boot", "20", "--out", str(out)]) == 0
    assert pr.main(["--from-predictions", *files, "--n-boot", "20", "--out", str(plain)]) == 0
    txt = (out / "patient_rule_summary.txt").read_text()
    assert "ENSEMBLE OF SEEDS" in txt and "ENSEMBLE PAIRED: arm minus base" in txt and "PAIRED: arm minus base" in txt
    res_e = pd.read_csv(out / "patient_rule_ensemble.csv")
    assert set(res_e.seed) == {pr.ENSEMBLE_SEED} and len(res_e) == 2 * 2 * 2 * len(pr.RULES)
    assert set(pd.read_csv(out / "image_level.csv").seed) == {0, 1, 2, 3, 4, pr.ENSEMBLE_SEED}
    # the seeds' own rows are what they are without the ensemble
    pd.testing.assert_frame_equal(pd.read_csv(out / "patient_rule.csv"), pd.read_csv(plain / "patient_rule.csv"))
    with pytest.raises(SystemExit, match="repeat an arm and seed"):
        pr.main(["--from-predictions", files[0], files[0], "--n-boot", "5", "--out", str(tmp_path / "dup")])


def test_end_to_end_on_a_checkpoint(tmp_path):
    from referral_outcomes import run as referral
    from style_transfer_xfer import load_device
    from test_retinareach import BACKBONE, _make_brset, _make_mbrset
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 40, rng)
    _make_mbrset(M, 40, rng)
    ck, run_dir = tmp_path / "s0.pt", tmp_path / "run"
    assert train(["--source-root", str(B), "--target-root", str(M), "--probe-targets", "hypertension",
                  "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1",
                  "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--calib-batch", "4",
                  "--calib-sizes", "4", "--calib-repeats", "1", "--envelope-repeats", "2", "--prior-strengths", "0",
                  "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1", "--min-test-pos-patients", "2",
                  "--demo-n", "4", "--n-shuffle", "0", "--ablation-frac", "0",
                  "--out", str(run_dir), "--ckpt", str(ck)]) == 0
    out = tmp_path / "patient"
    common = ["--source-root", str(B), "--target-root", str(M), "--device", "cpu", "--num-workers", "0",
              "--batch-size", "8", "--n-boot", "10"]
    assert pr.main(["--ckpt", f"canon_only={ck}", *common, "--out", str(out)]) == 0
    res = pd.read_csv(out / "patient_rule.csv")
    assert len(res) == 2 * 2 * len(pr.RULES) and set(res.part) == {"test"}
    pred = pd.read_csv(out / "predictions.csv")
    assert set(zip(pred.camera, pred.part)) == {("tabletop", "val"), ("tabletop", "test"), ("handheld", "test")}
    for proto, d in pred.groupby("protocol"):                             # the ensemble's threshold rule is the trainer's
        assert pr.image_threshold(d, 0.85) == pytest.approx(d.shipped_threshold.iloc[0], abs=1e-6)
    n_test = load_device(str(B), "brset", 64, 42)["test"].df["patient"].nunique()
    assert (res[res.camera == "tabletop"].patients == n_test).all()
    # the shipped rule on this script's photo scores = referral_outcomes on the trainer's own
    ro = referral(str(run_dir), n_boot=10).query("rule == 'dr_referable'")
    for cam in pr.CAMERAS:                                                # the trainer's device names
        for proto in ("trained", "calibrated"):
            a = res.query("camera == @cam and protocol == @proto and rule == 'shipped'").iloc[0]
            b = ro.query("device == @cam and protocol == @proto").iloc[0]
            assert [a[k] for k in ("tp", "fn", "fp", "tn")] == [b[k] for k in ("tp", "fn", "fp", "tn")]
    # the rules again from the photo scores, no checkpoint
    again = tmp_path / "again"
    assert pr.main(["--from-predictions", str(out / "predictions.csv"), "--n-boot", "10", "--out", str(again)]) == 0
    pd.testing.assert_frame_equal(pd.read_csv(again / "patient_rule.csv"), res)
    one = tmp_path / "one"
    assert pr.main(["--from-predictions", str(out / "predictions.csv"), "--ensemble", "--n-boot", "10",
                    "--out", str(one)]) == 0
    assert "no arm has two or more seeds" in (one / "patient_rule_summary.txt").read_text()
    with pytest.raises(SystemExit, match="handheld val"):
        pr.main(["--from-predictions", str(out / "predictions.csv"), "--part", "val", "--out", str(again)])
    with pytest.raises(SystemExit):
        pr.main(["--ckpt", f"a={ck}", "--from-predictions", str(out / "predictions.csv")])
