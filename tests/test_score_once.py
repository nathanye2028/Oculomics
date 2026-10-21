"""score_once.py, RetinaReach's once-only external test: a synthetic Messidor-2
release and small frozen checkpoints. One look writes every table and completes
the ledger; a second look is refused whatever the output dir; a crashed look may
be retried with the same checkpoints only; the preflight and every argument error
spend nothing."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

BACKBONE = "timm:mobilenetv4_conv_small"
PRIMARY = "calibrated:em:dr_or_edema"


def _m2_release(root, n_exams=16, seed=0):
    """Krause-style grades, the release's ';' pairing sheet, the last exam ungradable."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "IMAGES"))
    rows, pairs = [], []
    for e in range(n_exams):
        ref, pair = e % 3 == 0, []
        for side in (0, 1):
            name = f"IM{e:04d}{side}.png"
            img = Image.new("RGB", (72, 72), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.ellipse([4, 4, 68, 68], fill=(170, 80, 40))
            for _ in range(4 if ref else 0):
                x, y = rng.integers(15, 55, 2)
                d.ellipse([x, y, x + 5, y + 5], fill=(250, 240, 120))
            img.save(os.path.join(root, "IMAGES", name))
            last = e == n_exams - 1
            rows.append({"image_id": name, "adjudicated_dr_grade": np.nan if last else (2 if ref else 0),
                         "adjudicated_dme": np.nan if last else int(ref and side == 0),
                         "adjudicated_gradable": 0 if last else 1})
            pair.append(name)
        pairs.append(pair)
    pd.DataFrame(rows).to_csv(os.path.join(root, "messidor_data.csv"), index=False)
    with open(os.path.join(root, "messidor-2.csv"), "w") as f:
        f.write("left;right\n" + "\n".join(f"{a};{b}" for a, b in pairs) + "\n")


def _checkpoint(path, seed=0):
    """What train_retinareach.py saves, for a random small trunk."""
    from retinareach import (RetinaReachNet, ThresholdReference, get_bn_stats, self_calibrate,
                             stats_to_lists)
    torch.manual_seed(seed)
    net = RetinaReachNet(["dr_referable", "edema"], ["hypertension"], backbone=BACKBONE,
                         pretrained=False, probe_size=64)
    net, _ = self_calibrate(net, [{"image": torch.randn(8, 3, 64, 64)} for _ in range(2)],
                            torch.device("cpu"))
    ref = ThresholdReference(0.5, 0.85, 0.2, 1.0, 0.0, 0.25, -1.0).to_dict()
    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()},
                "config": {**net.config(), "image_size": 64},
                "thresholds": {p: {"dr_referable": 0.5, "edema": 0.5} for p in ("trained", "calibrated")},
                "profiles": {"tabletop": {"bn_stats": stats_to_lists(get_bn_stats(net))}},
                "threshold_reference": {"dr_referable": ref, "edema": ref},
                "args": {"seed": seed, "split_seed": 42, "calib_batch": 4, "source_name": "tabletop",
                         "source_camera": "Canon CR", "target_sens": 0.85, "sens_tolerance": 0.10,
                         "gate_margin": 0.02, "min_test_pos_patients": 2}}, path)
    return str(path)


def _args(tmp_path, cks, *extra):
    return ["--dataset", "messidor2", "--root", str(tmp_path / "m2"), "--checkpoints", *cks,
            "--ledger", str(tmp_path / "ledger.json"), "--out", str(tmp_path / "once"),
            "--folds", "2", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
            "--n-boot", "10", *extra]


def test_preflight_and_argument_errors_spend_nothing(tmp_path, capsys):
    from score_once import main
    _m2_release(str(tmp_path / "m2"))
    ck = _checkpoint(tmp_path / "s0.pt")
    assert main(_args(tmp_path, [ck], "--preflight")) == 0
    assert "look is unspent" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="name the primary"):
        main(_args(tmp_path, [ck]))
    with pytest.raises(SystemExit, match="PROTOCOL:METHOD:RULE"):
        main(_args(tmp_path, [ck], "--primary", "calibrated:em"))
    with pytest.raises(SystemExit, match="does not exist"):
        main(_args(tmp_path, [ck], "--primary", PRIMARY, "--plan", str(tmp_path / "nope.md")))
    assert not (tmp_path / "ledger.json").exists() and not (tmp_path / "once").exists()


def test_one_look_writes_everything_and_a_second_is_refused(tmp_path):
    from score_once import main
    _m2_release(str(tmp_path / "m2"))
    cks = [_checkpoint(tmp_path / f"s{s}.pt", seed=s) for s in (0, 1)]
    plan = tmp_path / "plan.md"
    plan.write_text("primary: calibrated:em:dr_or_edema, targets sens 0.85 / spec 0.825\n")
    assert main(_args(tmp_path, cks, "--primary", PRIMARY, "--plan", str(plan))) == 0

    look = json.loads((tmp_path / "ledger.json").read_text())["looks"][0]
    assert look["dataset"] == "messidor2" and look["completed_utc"] and look["attempt"] == 1
    assert look["primary"] == PRIMARY and len(look["checkpoints"]) == 2 and look["plan"]["sha256"]
    assert "sensitivity_mean" in look["primary_result"]
    out = tmp_path / "once"
    cap = pd.read_csv(out / "capability_messidor2.csv")
    assert set(cap["target"]) == {"dr_referable", "edema"} and set(cap["protocol"]) == {"trained", "calibrated"}
    assert set(cap["seed"]) == {0, 1} and (cap["meta_feature_set"] == "none (chance)").all()
    ops = pd.read_csv(out / "operating_point_messidor2.csv")
    assert set(zip(ops["protocol"], ops["method"])) == {("trained", "shipped"), ("calibrated", "shipped"),
                                                         ("calibrated", "em"), ("calibrated", "anchor")}
    ref = pd.read_csv(out / "referral_messidor2.csv")
    assert set(ref["rule"]) == {"dr_referable", "dr_or_edema"}
    every = ["per1000_correct_referrals", "per1000_missed", "per1000_unnecessary",
             "per1000_correct_clear", "per1000_retakes"]
    assert np.allclose(ref[every].sum(1), 1000.0)
    assert (ref["retakes"] == 1).all() and (ref["patients"] == 16).all()   # the ungradable exam
    pr = pd.read_csv(out / "seed0" / "predictions_messidor2_calibrated.csv")
    assert {"fold", "threshold_dr_referable__em", "threshold_dr_referable__anchor"} <= set(pr.columns)
    assert pr["p_dr_referable"].notna().all() and set(pr["fold"]) == {0, 1}
    # both eyes of an examination share a fold (the release's pairing sheet)
    pr["exam"] = pr["file"].str.extract(r"IM(\d{4})")[0]
    assert (pr.groupby("exam")["fold"].nunique() == 1).all()
    assert json.loads((out / "once.json").read_text())["primary"]["primary"] == PRIMARY

    # a second look is refused, whatever the output dir, and so is its preflight
    with pytest.raises(SystemExit, match="already scored"):
        main(_args(tmp_path, cks, "--primary", PRIMARY, "--out", str(tmp_path / "again")))
    with pytest.raises(SystemExit, match="already scored"):
        main(_args(tmp_path, cks, "--preflight"))
    assert not (tmp_path / "again").exists()
    assert len(json.loads((tmp_path / "ledger.json").read_text())["looks"]) == 1


def test_a_crashed_look_is_retried_with_the_same_checkpoints_only(tmp_path):
    from score_once import check_ledger, open_look, read_ledger
    path = str(tmp_path / "l.json")
    open_look(path, {"dataset": "messidor2", "opened_utc": "2026-11-08T00:00:00+00:00",
                     "checkpoints": {"a.pt": "h1"}})
    ok, why, attempt = check_ledger(read_ledger(path), "messidor2", ["h1"])
    assert ok and attempt == 2 and "never completed" in why
    ok, why, _ = check_ledger(read_ledger(path), "messidor2", ["h2"])
    assert not ok and "DIFFERENT" in why
    assert check_ledger(read_ledger(path), "aptos", ["h9"])[0]          # another dataset is unaffected


def test_shallow_calibration(tmp_path):
    from score_once import main
    _m2_release(str(tmp_path / "m2"))
    ck = _checkpoint(tmp_path / "s0.pt")
    assert main(_args(tmp_path, [ck], "--primary", "calibrated:shipped:dr_referable",
                      "--calibration", "shallow", "--shallow-frac", "0.5")) == 0
    look = json.loads((tmp_path / "ledger.json").read_text())["looks"][0]
    assert look["calibration"] == "shallow" and look["shallow_frac"] == 0.5
