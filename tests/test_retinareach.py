"""CPU-only, synthetic, network-free tests for RetinaReach: the deployable
calibration graphs reproduce AdaBN exactly, the prior and camera distance behave,
the gate's decision table and split are right, probes fold correctly, and one
end-to-end run on a synthetic two-camera pair produces a complete capability table."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

from capability_gate import encoding_audit, gate_status, multitask_split
from retinareach import (BatchStatCollector, DeviceProfile, RetinaReachNet, StatInputNet,
                         blend_stats, build_from_checkpoint, calibrate_with_collector,
                         camera_distance, device_capability, get_bn_stats, self_calibrate,
                         set_bn_stats, stats_to_vector, vector_to_stats)

BACKBONE = "timm:mobilenetv4_conv_small"


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    m = RetinaReachNet(["dr_referable", "edema"], ["hypertension"], backbone=BACKBONE,
                       pretrained=False, probe_size=64)
    # Give the random net non-trivial source statistics.
    m, _ = self_calibrate(m, [{"image": torch.randn(8, 3, 64, 64)} for _ in range(2)],
                          torch.device("cpu"))
    return m.eval()


def _batches(n_batches=3, b=4, shift=0.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(b, 3, 64, 64, generator=g) * 255.0 * (1 - shift) + 255.0 * shift * 0.5
            for _ in range(n_batches)]


def _norm(x):
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1) * 255
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1) * 255
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# Deployable graphs == PyTorch reference
# --------------------------------------------------------------------------- #
def test_collector_average_reproduces_adapt_bn(net):
    raw = _batches(shift=0.3)
    ref, info = self_calibrate(net, [{"image": _norm(x)} for x in raw], torch.device("cpu"))
    assert info.n_images == 12 and info.n_batches == 3
    order = [n for n, _ in __import__("retinareach").bn_layers(net)]
    want = stats_to_vector(get_bn_stats(ref), order)
    got, n = calibrate_with_collector(BatchStatCollector(net), raw)
    assert n == 12
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), float((got - want).abs().max())


def test_stat_input_net_equals_eval_model_with_those_stats(net):
    raw = _batches(n_batches=1, b=3, shift=0.2, seed=5)[0]
    cal, _ = self_calibrate(net, [{"image": _norm(x)} for x in _batches(shift=0.4)],
                            torch.device("cpu"))
    order = [n for n, _ in __import__("retinareach").bn_layers(net)]
    vec = stats_to_vector(get_bn_stats(cal), order)
    sin = StatInputNet(net)
    assert sin.stats_len == vec.numel()
    with torch.no_grad():
        want = torch.sigmoid(cal.eval()(_norm(raw)))
        got = sin(raw, vec)
    assert torch.allclose(got, want, atol=1e-5)
    # and the stats vector round-trips
    back = vector_to_stats(vec, net)
    assert all(torch.equal(back[k][0], get_bn_stats(cal)[k][0]) for k in back)


def test_graphs_leave_the_source_model_untouched(net):
    before = {k: v.clone() for k, v in net.state_dict().items()}
    BatchStatCollector(net)(_batches(n_batches=1)[0])
    StatInputNet(net)(_batches(n_batches=1)[0], torch.zeros(StatInputNet(net).stats_len) + 1)
    after = net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_prior_zero_is_adabn_and_large_prior_is_source(net):
    loader = [{"image": _norm(x)} for x in _batches(shift=0.5)]
    src = get_bn_stats(net)
    plain, _ = self_calibrate(net, loader, torch.device("cpu"), prior_strength=0.0)
    huge, _ = self_calibrate(net, loader, torch.device("cpu"), prior_strength=1e9)
    k = next(iter(src))
    assert not torch.allclose(get_bn_stats(plain)[k][0], src[k][0])
    assert torch.allclose(get_bn_stats(huge)[k][0], src[k][0], atol=1e-5)
    # the blend is linear in the pseudo-count
    half = blend_stats(src, get_bn_stats(plain), n_target=12, prior_strength=12)
    assert torch.allclose(half[k][0], 0.5 * (src[k][0] + get_bn_stats(plain)[k][0]))


def test_self_calibrate_counts_out_batches_of_one(net):
    loader = [{"image": torch.randn(4, 3, 64, 64)}, {"image": torch.randn(1, 3, 64, 64)}]
    _, info = self_calibrate(net, loader, torch.device("cpu"))
    assert info.n_images == 4 and info.n_batches == 1


def test_trunk_without_batchnorm_is_refused():
    with pytest.raises(ValueError):
        RetinaReachNet(["dr_referable"], backbone="timm:vit_tiny_patch16_224", pretrained=False)


# --------------------------------------------------------------------------- #
# Camera signature and the device lookup
# --------------------------------------------------------------------------- #
def _stats(mu, var, c=4, layers=3):
    return {f"l{i}": (torch.full((c,), float(mu)), torch.full((c,), float(var))) for i in range(layers)}


def test_camera_distance_properties():
    a, b = _stats(0, 1), _stats(1, 2)
    assert camera_distance(a, a) == pytest.approx(0.0)
    assert camera_distance(a, b) == pytest.approx(camera_distance(b, a))
    assert camera_distance(a, _stats(2, 2)) > camera_distance(a, b) > 0


def test_device_capability_never_promotes_and_rejects_unfamiliar_cameras():
    cap = [{"target": "dr_referable", "protocol": "calibrated", "status": "SUPPORTED"},
           {"target": "hypertension", "protocol": "calibrated", "status": "NO_IMAGE_EVIDENCE",
            "reason": "x"}]
    prof = [DeviceProfile("handheld", "mbrset", _stats(0, 1), 100, {16: 0.05, 64: 0.01}, cap),
            DeviceProfile("tabletop", "brset", _stats(3, 1), 100, {16: 0.05}, [])]
    near = device_capability(prof, _stats(0.1, 1), 16, ["dr_referable", "hypertension", "smoking"])
    assert near["nearest_profile"] == "handheld" and near["inside_envelope"]
    st = {r["target"]: r["status"] for r in near["targets"]}
    assert st == {"dr_referable": "SUPPORTED", "hypertension": "NO_IMAGE_EVIDENCE",
                  "smoking": "NOT_VALIDATED"}
    assert near["shown"] == ["dr_referable"]
    far = device_capability(prof, _stats(1.5, 1), 16, ["dr_referable"])
    assert not far["inside_envelope"] and far["targets"][0]["status"] == "NOT_VALIDATED"
    # more captures -> tighter envelope: the same camera offset can fail at N=64
    assert device_capability(prof, _stats(0.1, 1), 64, ["dr_referable"])["inside_envelope"] is False


# --------------------------------------------------------------------------- #
# Gate, split, audit, probes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("args,status", [
    ((False, 0, 20, .1, .05, .02, .9, .85, .1), "NOT_VALIDATED"),
    ((True, 5, 20, .1, .05, .02, .9, .85, .1), "UNDERPOWERED"),
    ((True, 50, 20, .01, .005, .02, .9, .85, .1), "NO_IMAGE_EVIDENCE"),     # below margin
    ((True, 50, 20, .10, -.01, .02, .9, .85, .1), "NO_IMAGE_EVIDENCE"),     # CI spans 0
    ((True, 50, 20, .10, .05, .02, .60, .85, .1), "THRESHOLD_DRIFT"),
    ((True, 50, 20, .10, .05, .02, .80, .85, .1), "SUPPORTED"),
])
def test_gate_status_table(args, status):
    assert gate_status(*args)[0] == status


def test_missing_threshold_is_drift_not_zero_sensitivity():
    from capability_gate import gate_cell
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"file": [f"{i}.jpg" for i in range(80)], "patient": np.arange(80) // 2,
                       "systemic_hypertension": (np.arange(80) // 2) % 2})
    y = df["systemic_hypertension"].to_numpy()
    comp = {"scores": np.full(80, 0.5), "reason": None, "feature_set": "age_sex",
            "model": "logreg", "cv_sd": 0.01}
    row = gate_cell("hypertension", "handheld", "calibrated", df, y + rng.normal(0, .1, 80),
                    float("nan"), comp, min_pos_patients=5, n_boot=50)
    assert row["status"] == "THRESHOLD_DRIFT" and "no shipped threshold" in row["reason"]
    assert np.isnan(row["sensitivity"])


def _table(n_pat=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pat):
        icdr = rng.choice([0, 0, 0, 2, np.nan])
        for eye in range(2):
            rows.append({"file": f"{pid}_{eye}.jpg", "patient": pid, "final_icdr": icdr,
                         "systemic_hypertension": rng.choice([0, 1])})
    return pd.DataFrame(rows)


def test_multitask_split_is_patient_disjoint_and_keeps_every_row():
    df = _table()
    sp = multitask_split(df, "dr_referable", seed=42)
    pats = [set(v["patient"]) for v in sp.values()]
    assert not (pats[0] & pats[1] or pats[0] & pats[2] or pats[1] & pats[2])
    assert sum(len(v) for v in sp.values()) == len(df)            # NaN-label rows kept
    assert sp["test"]["final_icdr"].isna().any() or df["final_icdr"].isna().sum() < 5
    again = multitask_split(df, "dr_referable", seed=42)
    assert again["test"]["file"].tolist() == sp["test"]["file"].tolist()


def test_encoding_audit_refuses_non_binary_decodes():
    # _yes_no passes numbers through: a 1/2-coded edema column decodes to {1, 2}
    au = encoding_audit(pd.DataFrame({"final_edema": [1, 2, 2, 1] * 10}), "edema")
    assert not au["ok"] and "not 0/1" in au["reason"]
    assert encoding_audit(pd.DataFrame({"final_edema": ["yes", "no"] * 10}), "edema")["ok"]


def test_blend_vector_matches_blend_stats(net):
    from retinareach import blend_vector, bn_layers
    order = [n for n, _ in bn_layers(net)]
    src = get_bn_stats(net)
    tgt = {k: (m + 1.0, v * 2.0) for k, (m, v) in src.items()}
    want = stats_to_vector(blend_stats(src, tgt, 16, 32.0), order)
    got = blend_vector(stats_to_vector(src, order), stats_to_vector(tgt, order), 16, 32.0)
    assert torch.allclose(got, want)
    assert torch.equal(blend_vector(want, got, 16, 0.0), got)


def test_encoding_audit_catches_a_one_two_column():
    df = pd.DataFrame({"systemic_hypertension": [1, 2, 2, 1, 2] * 10})
    au = encoding_audit(df, "hypertension")
    assert not au["ok"] and "encoding" in au["reason"]
    assert encoding_audit(pd.DataFrame({"systemic_hypertension": [0, 1] * 10}), "hypertension")["ok"]
    assert not encoding_audit(pd.DataFrame({"x": [0, 1]}), "hypertension")["ok"]


def test_fit_probe_folds_into_raw_units():
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from train_retinareach import fit_probe
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, (200, 6))
    y = (X[:, 0] + rng.normal(0, 1, 200) > 3).astype(int)
    fit = fit_probe(X[:150], y[:150], X[150:], y[150:], [0.01, 1.0])
    logit = X @ fit["weight"] + fit["bias"]
    mu, sd = X[:150].mean(0), X[:150].std(0) + 1e-6
    from sklearn.linear_model import LogisticRegression as LR
    ref = LR(C=fit["C"], class_weight="balanced", max_iter=5000, random_state=0) \
        .fit((X[:150] - mu) / sd, y[:150]).decision_function((X - mu) / sd)
    assert np.allclose(logit, ref, atol=1e-6)
    assert fit["val_auroc"] > 0.8


# --------------------------------------------------------------------------- #
# End to end on a synthetic tabletop -> handheld pair
# --------------------------------------------------------------------------- #
def _fundus(path, lesion, cast, dark, rng):
    img = Image.new("RGB", (72, 72), (0, 0, 0))
    d = ImageDraw.Draw(img)
    base = np.array([170, 80, 40]) * (0.7 if dark else 1.0) * np.array(cast)
    d.ellipse([4, 4, 68, 68], fill=tuple(int(v) for v in np.clip(base, 20, 255)))
    for _ in range(4 if lesion else 0):
        x, y = rng.integers(15, 55, 2)
        d.ellipse([x, y, x + 5, y + 5], fill=(250, 240, 120))
    img.save(path, quality=95)


def _make_brset(root, n_pat, rng):
    (root / "fundus_photos").mkdir(parents=True)
    rows = []
    for pid in range(n_pat):
        ref = pid % 3 == 0
        for eye in (1, 2):
            f = f"img{pid:03d}{eye}"
            _fundus(root / "fundus_photos" / f"{f}.jpg", ref, (1, 1, 1), False, rng)
            rows.append({"image_id": f, "patient_id": pid, "DR_ICDR": 2 if ref else 0,
                         "macular_edema": int(ref and eye == 1), "patient_age": 40 + pid % 30,
                         "patient_sex": 1 + pid % 2, "exam_eye": eye, "quality": "Adequate",
                         "diabetes_time_y": pid % 20, "insuline": pid % 2,
                         "diabetes": "yes", "nationality": "Brazil"})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)


def _make_mbrset(root, n_pat, rng):
    (root / "images").mkdir(parents=True)
    rows = []
    for pid in range(n_pat):
        ref, htn = pid % 3 == 0, pid % 2 == 0
        for k in (1, 2):
            f = f"{pid}.{k}.jpg"
            _fundus(root / "images" / f, ref, (0.8, 1.1, 1.3), htn, rng)   # camera colour cast
            rows.append({"patient": pid, "age": 40 + pid % 30, "sex": pid % 2,
                         "dm_time": pid % 20, "insulin": pid % 2, "insulin_time": np.nan,
                         "oraltreatment_dm": 1, "systemic_hypertension": int(htn),
                         "nephropathy": int(pid % 5 == 0), "file": f,
                         "laterality": "right" if k == 1 else "left", "final_artifacts": "no",
                         "final_quality": "yes", "final_icdr": 2 if ref else 0,
                         "final_edema": "yes" if (ref and k == 1) else "no"})
    pd.DataFrame(rows).to_csv(root / "labels_mbrset.csv", index=False)


def test_end_to_end_synthetic(tmp_path):
    from train_retinareach import main
    rng = np.random.default_rng(0)
    _make_brset(tmp_path / "brset", 45, rng)
    _make_mbrset(tmp_path / "mbrset", 45, rng)
    out, ck = tmp_path / "out", tmp_path / "ck" / "s0.pt"
    rc = main(["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
               "--probe-targets", "hypertension", "nephropathy", "smoking",
               "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64",
               "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
               "--calib-batch", "4", "--calib-sizes", "4", "8", "--calib-repeats", "2",
               "--prior-strengths", "8", "0", "--min-probe-pos", "2", "--n-boot", "20",
               "--envelope-repeats", "4", "--demo-n", "8",
               "--cv-repeats", "1", "--min-test-pos-patients", "2",
               "--out", str(out), "--ckpt", str(ck)])
    assert rc == 0
    cap = pd.read_csv(out / "capability.csv")
    targets = ["dr_referable", "edema", "hypertension", "nephropathy", "smoking"]
    assert set(cap["target"]) == set(targets)
    assert len(cap) == len(targets) * 2 * 2                 # x devices x protocols
    assert set(cap["status"]) <= {"SUPPORTED", "THRESHOLD_DRIFT", "NO_IMAGE_EVIDENCE",
                                  "UNDERPOWERED", "NOT_VALIDATED"}
    by = cap.set_index(["target", "device", "protocol"])["status"]
    # smoking has no column in the synthetic mBRSET/BRSET -> never validated anywhere
    assert (cap[cap["target"] == "smoking"]["status"] == "NOT_VALIDATED").all()
    # probes exist only under self-calibration; systemic labels exist only on handheld
    assert by[("hypertension", "handheld", "trained")] == "NOT_VALIDATED"
    assert by[("hypertension", "tabletop", "calibrated")] == "NOT_VALIDATED"
    sweep = pd.read_csv(out / "calibration_sweep.csv")
    assert {"trained", "full_pool", "subset"} <= set(sweep["kind"])
    sub = sweep[sweep["kind"] == "subset"]
    assert set(sub["prior_strength"]) == {0.0, 8.0}
    # prior 0 comes AFTER prior 8 here: it must not inherit the blended statistics
    metric = [c for c in sub.columns if c.startswith(("auroc_", "sens_", "flagged_"))]
    p0 = sub[sub["prior_strength"] == 0].sort_values(["n", "repeat"])[metric].to_numpy()
    p8 = sub[sub["prior_strength"] == 8].sort_values(["n", "repeat"])[metric].to_numpy()
    assert not np.allclose(p0, p8, equal_nan=True)
    res = json.loads((out / "results.json").read_text())
    assert set(res["envelope"]) == {"tabletop", "handheld"}
    assert set(res["device_false_accept"]) == {"tabletop", "handheld"}
    assert res["device_report"]["n_images"] == 8
    prof = json.loads((out / "profiles.json").read_text())
    p = DeviceProfile.from_dict(prof["profiles"]["handheld"])
    assert p.n_pool > 0 and p.capability
    for dev in ("tabletop", "handheld"):
        for proto in ("trained", "calibrated"):
            assert os.path.exists(out / f"predictions_{dev}_{proto}.csv")
    # the checkpoint rebuilds, with either statistics, and the probe row is filled
    ckd = torch.load(ck, map_location="cpu")
    m0 = build_from_checkpoint(ckd)
    m1 = build_from_checkpoint(ckd, stats="handheld")
    assert m1.head.weight[m1.index("hypertension")].abs().sum() > 0
    x = torch.randn(2, 3, 64, 64)
    assert not torch.allclose(m0(x), m1(x))

    # the multi-seed summary runs on the per-seed layout (the same run twice here)
    import shutil
    from summarize_retinareach import main as summarize
    exp = tmp_path / "seed99" / "exp"          # a seed-like parent must not confuse the loader
    shutil.copytree(out, exp / "seed0")
    shutil.copytree(out, exp / "seed1")
    assert summarize(["--dir", str(exp)]) == 0
    final = pd.read_csv(exp / "capability_final.csv")
    assert len(final) == len(cap) and set(final["final"]) <= set(cap["status"]) | {"NO_IMAGE_EVIDENCE"}
    assert (exp / "summary.md").read_text().startswith("# RetinaReach")


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="Core ML runs on macOS only")
def test_coreml_calibration_matches_pytorch(tmp_path):
    """The phone-side procedure (Core ML calibrator averaged over batches, then the
    statistics-input graph) reproduces PyTorch AdaBN on the same uint8 captures."""
    pytest.importorskip("coremltools")
    from export_retinareach import main as export
    rng = np.random.default_rng(0)
    caps = tmp_path / "caps"
    caps.mkdir()
    for i in range(8):
        _fundus(caps / f"{i}.jpg", i % 2 == 0, (0.8, 1.1, 1.3), i % 3 == 0, rng)
    out = tmp_path / "export"
    rc = export(["--backbone", BACKBONE, "--image-size", "64", "--calib-batch", "4",
                 "--verify-images", str(caps), "--verify-n", "8", "--verify-units", "CPU_ONLY",
                 "--precision", "fp32", "--out-dir", str(out)])
    assert rc == 0
    rep = json.loads((out / "export_report.json").read_text())["verify"]
    assert rep["pass"] and rep["stats_max_rel_err"] < 1e-3 and rep["prob_max_err_total"] < 1e-3
    prof = json.loads((out / "retinareach_profiles.json").read_text())
    assert len(prof["trained_stats"]) == 2 * sum(prof["stats_layout"]["channels"])
