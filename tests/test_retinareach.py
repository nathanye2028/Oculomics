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
                         ThresholdReference,
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


class _Imgs(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i]}


def _loader(n=12, b=4, shift=0.5, seed=1):
    x = torch.cat([_norm(t) for t in _batches(n_batches=n // b, b=b, shift=shift, seed=seed)])
    return torch.utils.data.DataLoader(_Imgs(x), batch_size=b, shuffle=False)


def test_prior_zero_is_adabn_and_large_prior_is_source(net):
    src = get_bn_stats(net)
    plain, _ = self_calibrate(net, _loader(), torch.device("cpu"), prior_strength=0.0)
    huge, info = self_calibrate(net, _loader(), torch.device("cpu"), prior_strength=1e9)
    k = next(iter(src))
    assert not torch.allclose(get_bn_stats(plain)[k][0], src[k][0])
    assert torch.allclose(get_bn_stats(huge)[k][0], src[k][0], atol=1e-5)
    assert info.alpha == pytest.approx(1e9 / (1e9 + 12)) and info.prior_strength == 1e9
    with pytest.raises(ValueError):                       # the prior needs a sized dataset
        self_calibrate(net, [{"image": torch.randn(4, 3, 64, 64)}], torch.device("cpu"), prior_strength=8)
    # the post-hoc vector blend (phone path) is still linear in the pseudo-count
    half = blend_stats(src, get_bn_stats(plain), n_target=12, prior_strength=12)
    assert torch.allclose(half[k][0], 0.5 * (src[k][0] + get_bn_stats(plain)[k][0]))


def test_calibrate_blended_endpoints_and_shallow_only_exactness(net):
    from retinareach import bn_layers, calibrate_blended
    dev = torch.device("cpu")
    names = [n for n, _ in bn_layers(net)]
    src = get_bn_stats(net)
    ada, _ = self_calibrate(net, _loader(), dev)
    a0, info0 = calibrate_blended(net, _loader(), dev, alpha=0.0)
    for n in names:                                        # alpha 0 == AdaBN
        assert torch.allclose(get_bn_stats(a0)[n][0], get_bn_stats(ada)[n][0], atol=1e-5)
        assert torch.allclose(get_bn_stats(a0)[n][1], get_bn_stats(ada)[n][1], rtol=1e-4, atol=1e-5)
    a1, _ = calibrate_blended(net, _loader(), dev, alpha=1.0)
    assert all(torch.allclose(get_bn_stats(a1)[n][0], src[n][0]) for n in names)   # alpha 1 == source
    k = len(names) // 3
    part, info = calibrate_blended(net, _loader(), dev, adapt=names[:k])
    assert info.n_bn_layers == k
    for n in names[:k]:                                    # first K identical to full AdaBN
        assert torch.allclose(get_bn_stats(part)[n][0], get_bn_stats(ada)[n][0], atol=1e-5)
    for n in names[k:]:                                    # the rest untouched
        assert torch.equal(get_bn_stats(part)[n][0], src[n][0])
    # no patched forward is left behind on the copy
    assert all("forward" not in vars(m) for _, m in bn_layers(part))
    other = {n: (m + 1.0, v * 2.0) for n, (m, v) in src.items()}
    moved, _ = calibrate_blended(net, _loader(), dev, adapt=[], source=other)
    assert torch.equal(get_bn_stats(moved)[names[0]][0], other[names[0]][0])


def test_em_prevalence_threshold_and_prevalence_pools():
    from redesign_calibration import prevalence_subset
    from retinareach import em_prevalence, expected_sensitivity_threshold
    rng = np.random.default_rng(0)
    pi_s, pi_t, n = 0.1, 0.3, 20000
    y = (rng.random(n) < pi_t).astype(int)
    s = rng.normal(np.where(y == 1, 1.5, 0.0), 1.0)
    lr = np.exp(1.5 * s - 1.125)                           # likelihood ratio of the two normals
    q = pi_s * lr / (pi_s * lr + (1 - pi_s))               # posterior calibrated at the SOURCE prior
    est, post = em_prevalence(q, pi_s)
    assert abs(est - pi_t) < 0.02
    thr = expected_sensitivity_threshold(s, y.astype(float), 0.85)
    assert abs(float((s[y == 1] >= thr).mean()) - 0.85) < 0.01
    thr_soft = expected_sensitivity_threshold(s, post, 0.85)
    assert abs(float((s[y == 1] >= thr_soft).mean()) - 0.85) < 0.03   # label-free, still on target
    yy = np.r_[np.ones(50), np.zeros(450), np.full(20, np.nan)]
    idx = prevalence_subset(yy, 0.2, 100, rng)
    assert len(idx) == 100 and yy[idx].sum() == 20
    assert prevalence_subset(yy, 0.5, 200, rng).shape[0] == 100      # shrinks to what the pool allows


def test_dr_proxy_check_separates_a_dr_proxy_from_independent_signal():
    from dr_proxy_check import proxy_rows, within_stratum_auroc
    rng = np.random.default_rng(0)
    n = 6000
    grade = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.6, 0.15, 0.12, 0.08, 0.05]).astype(float)
    other = rng.normal(size=n)                               # image signal unrelated to DR
    # "proxy" follows the DR grade only; "own" also follows the unrelated signal
    y_proxy = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * grade - 1.5)))).astype(float)
    y_own = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * other - 1.0)))).astype(float)
    table = pd.DataFrame({"file": [f"{i}.jpg" for i in range(n)], "patient": np.arange(n) // 2,
                          "final_icdr": grade, "insulin": y_proxy, "systemic_hypertension": y_own})
    table.loc[:9, "final_icdr"] = np.nan                     # ungradable eyes are left out of the strata
    pred = pd.DataFrame({"file": table["file"], "patient": table["patient"],
                         "p_dr_referable": 1 / (1 + np.exp(-(grade - 2 + rng.normal(0, 0.5, n)))),
                         "p_insulin": 1 / (1 + np.exp(-(grade + rng.normal(0, 0.5, n)))),
                         "p_hypertension": 1 / (1 + np.exp(-(other + rng.normal(0, 0.5, n))))})
    rows = {r["target"]: r for r in proxy_rows(pred, table, ["insulin", "hypertension", "smoking"],
                                               n_boot=200)}
    assert set(rows) == {"insulin", "hypertension"}          # no smoking column -> no row
    px, own = rows["insulin"], rows["hypertension"]
    assert px["auroc_probe"] > 0.62 and abs(px["auroc_dr_score"] - px["auroc_probe"]) < 0.03
    assert abs(px["auroc_within_grade"] - 0.5) < 0.03 and px["within_lo"] < 0.5 < px["within_hi"]
    assert abs(px["auroc_no_dr"] - 0.5) < 0.04
    assert own["auroc_within_grade"] > 0.7 and own["within_lo"] > 0.5
    assert abs(own["auroc_dr_score"] - 0.5) < 0.03 and own["n_no_dr"] > 3000
    y = np.array([1, 0, 1, 0]); sc = np.array([0.9, 0.1, 0.2, 0.8])
    assert within_stratum_auroc(y, sc, np.array([0, 0, 1, 1])) == 0.5      # 1.0 and 0.0, equal weight
    assert np.isnan(within_stratum_auroc(y, sc, np.array([0, 1, 0, 1])))   # no stratum has both classes


def _scores(rng, n, prev, shift=0.0, sep=2.5):
    """Labels at ``prev`` and model probabilities whose logits separate the classes
    by ``sep`` SD, all moved by ``shift`` (a camera / calibration shift)."""
    y = (rng.random(n) < prev).astype(float)
    z = rng.normal(np.where(y == 1, sep, 0.0), 1.0) - 2.5 + shift
    return y, 1.0 / (1.0 + np.exp(-z))


def test_label_free_threshold_methods():
    from metadata_model import threshold_at_sensitivity
    from retinareach import (THRESHOLD_METHODS, ThresholdReference, fit_threshold_reference,
                             label_free_threshold)
    rng = np.random.default_rng(1)
    y_v, q_v = _scores(rng, 20000, 0.055)
    thr = threshold_at_sensitivity(y_v.astype(int), q_v, 0.85)
    ref = fit_threshold_reference(q_v, y_v, thr, 0.85)
    assert ref.prior == pytest.approx(0.055, abs=0.005) and ref.platt_a > 0
    assert ThresholdReference.from_dict(ref.to_dict()) == ref
    with pytest.raises(ValueError):
        fit_threshold_reference(q_v, np.zeros_like(y_v), thr, 0.85)
    assert label_free_threshold(q_v, ref, "shipped") == (thr, {})
    assert label_free_threshold(np.array([np.nan]), ref, "em")[0] == thr      # nothing to estimate from
    with pytest.raises(ValueError):
        label_free_threshold(q_v, ref, "median")
    assert THRESHOLD_METHODS[0] == "shipped"
    sens = lambda y, q, t: float((q[y == 1] >= t).mean())
    # anchor: a pure logit shift at the SAME prevalence is undone exactly
    y_s, q_s = y_v, 1.0 / (1.0 + np.exp(-(np.log(q_v / (1 - q_v)) - 1.0)))
    t_anchor, info = label_free_threshold(q_s, ref, "anchor")
    assert info["shift"] == pytest.approx(-1.0, abs=1e-6)
    assert sens(y_s, q_s, t_anchor) == pytest.approx(sens(y_v, q_v, thr), abs=1e-9)
    # the handheld case: scores shifted down AND 2.8x the prevalence -- the shipped
    # threshold misses the floor; the anchor comes back near target
    y_h, q_h = _scores(rng, 20000, 0.156, shift=-1.0)
    assert sens(y_h, q_h, thr) < 0.75
    t_an, _ = label_free_threshold(q_h, ref, "anchor")
    assert abs(sens(y_h, q_h, t_an) - 0.85) < 0.05
    # EM assumes label shift only: a score shift reads as "almost no disease here",
    # and its threshold ends up worse than the shipped one (the limitation it has)
    t_em, est = label_free_threshold(q_h, ref, "em")
    assert est["est_prev"] < 0.10 and sens(y_h, q_h, t_em) < sens(y_h, q_h, thr)
    # label shift only (no score shift): EM recovers the prevalence and keeps the target
    y_l, q_l = _scores(rng, 20000, 0.156)
    t_l, est_l = label_free_threshold(q_l, ref, "em")
    assert est_l["est_prev"] == pytest.approx(0.156, abs=0.02)
    assert abs(sens(y_l, q_l, t_l) - 0.85) < 0.03
    # em2 / rate: a camera that COMPRESSES the scores (ranks kept) at 2.8x the prevalence.
    # EM read on the compressed scores misplaces the threshold; em2 estimates the
    # prevalence on the same captures under source statistics and places the threshold
    # on the compressed scores; rate flags the expected fraction by rank alone.
    y_c, q_r = _scores(rng, 20000, 0.156)
    q_c = 1.0 / (1.0 + np.exp(-(0.6 * np.log(q_r / (1 - q_r)) - 0.4)))
    t_em, _ = label_free_threshold(q_c, ref, "em")
    t_e2, e2 = label_free_threshold(q_c, ref, "em2", q_r)
    t_rt, rt = label_free_threshold(q_c, ref, "rate", q_r)
    assert e2["est_prev"] == pytest.approx(0.156, abs=0.02) and rt["est_prev"] == e2["est_prev"]
    assert abs(sens(y_c, q_c, t_e2) - 0.85) < 0.03 and abs(sens(y_c, q_c, t_rt) - 0.85) < 0.04
    assert abs(sens(y_c, q_c, t_em) - 0.85) > abs(sens(y_c, q_c, t_e2) - 0.85)
    assert abs(float((q_c >= t_rt).mean()) - rt["flag_rate"]) < 1e-3
    assert label_free_threshold(q_c, ref, "em2") == label_free_threshold(q_c, ref, "em")   # no reference: em
    with pytest.raises(ValueError):
        label_free_threshold(q_c, ref, "em2", q_r[:-1])
    old = ThresholdReference.from_dict({k: v for k, v in ref.to_dict().items() if k != "source_spec"})
    assert old.source_spec != old.source_spec                      # a pre-rate reference loads unset
    with pytest.raises(ValueError):
        label_free_threshold(q_c, old, "rate", q_r)
    assert ref.source_spec == pytest.approx(float((q_v[y_v == 0] < thr).mean()))
    assert set(THRESHOLD_METHODS) == {"shipped", "em", "anchor", "em2", "rate"}


def test_prevalence_matched_pool_lowers_the_rate_label_free():
    from metadata_model import threshold_at_sensitivity
    from retinareach import (fit_threshold_reference, prevalence_match_weights,
                             prevalence_matched_sample)
    rng = np.random.default_rng(2)
    y_v, q_v = _scores(rng, 20000, 0.055)
    ref = fit_threshold_reference(q_v, y_v, threshold_at_sensitivity(y_v.astype(int), q_v, 0.85), 0.85)
    y_h, q_h = _scores(rng, 20000, 0.156)                           # the clinic: 2.8x the disease
    w, info = prevalence_match_weights(q_h, ref)
    assert w.sum() == pytest.approx(1.0) and info["est_prev"] == pytest.approx(0.156, abs=0.02)
    assert float(np.sum(w * y_h)) < 0.10 < float(y_h.mean())        # the weighted pool has less disease
    assert info["expected_prev"] == pytest.approx(float(np.sum(w * y_h)), abs=0.02)   # and says so, label-free
    idx, _ = prevalence_matched_sample(q_h, ref, rng, size=4000)
    assert len(idx) == 4000 and float(y_h[idx].mean()) < 0.10
    # a clinic with LESS disease than the source: the positives are up-weighted instead
    y_l, q_l = _scores(rng, 20000, 0.02)
    w_l, info_l = prevalence_match_weights(q_l, ref)
    assert info_l["positive_weight"] > 1.0 and float(np.sum(w_l * y_l)) > float(y_l.mean())
    # pt = 0: predicted disease-free eyes only; a NaN score gets no weight
    w0, _ = prevalence_match_weights(np.r_[q_h, np.nan], ref, target_prevalence=0.0)
    assert w0[-1] == 0.0 and float(np.sum(w0[:-1] * y_h)) <= float(np.sum(w * y_h)) + 1e-9


def test_cross_fit_probe_scores_every_patient_once_and_controls_sit_at_chance():
    from metadata_model import _auc
    from probe_folds import cross_fit_probe
    rng = np.random.default_rng(3)
    n_pat, d = 300, 24
    g = np.repeat(np.arange(n_pat), 2)
    y = np.repeat((rng.random(n_pat) < 0.3).astype(float), 2)
    X = rng.normal(size=(2 * n_pat, d))
    X[:, 0] += 1.5 * y
    y[:6] = np.nan                                                   # unlabelled rows stay unscored
    r = cross_fit_probe(X, y, g, folds=5, Cs=(1e-2, 1e-1), seed=0, n_shuffle=3, inner_folds=3)
    ok = ~np.isnan(y)
    assert r["fitted"] and np.isfinite(r["scores"][ok]).all() and np.isnan(r["scores"][~ok]).all()
    assert (r["fold"][~ok] == -1).all() and set(r["fold"][ok]) == set(range(5))
    f, gg = r["fold"][ok], g[ok]
    assert all(len(set(f[gg == p])) == 1 for p in np.unique(gg))     # both eyes in one fold
    assert _auc(y[ok].astype(int), r["scores"][ok]) > 0.7
    assert abs(r["shuffle"]["auroc"] - 0.5) < 0.1 and r["shuffle"]["n"] == 15
    assert len(r["C"]) == 5 == len(r["thresholds"]) and 0 < r["threshold"] < 1
    assert not cross_fit_probe(X, y, g, folds=5, min_pos=1000)["fitted"]


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


def test_probe_shuffle_sits_at_chance_and_real_probe_does_not():
    from train_retinareach import probe_shuffle
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (400, 8))
    y = (X[:, 0] + rng.normal(0, .5, 400) > 0).astype(int)
    sh = probe_shuffle(X[:300], y[:300], X[300:], y[300:], C=1.0, n=20, seed=0)
    assert abs(sh["auroc"] - 0.5) < 4 * sh["se"] + 0.05


def test_quality_only_calibration_strata_and_decodability():
    from capability_gate import (calibration_metrics, decodability, quality_only_auroc,
                                 quality_strata)
    rng = np.random.default_rng(0)
    n = 400
    q = rng.random(n) < 0.8
    df = pd.DataFrame({"file": [f"{i}.jpg" for i in range(n)], "patient": np.arange(n) // 2,
                       "final_quality": np.where(q, "yes", "no"),
                       "final_artifacts": rng.choice(["yes", "no"], n),
                       # the target is carried by image quality only
                       "systemic_hypertension": (~q | (rng.random(n) < 0.1)).astype(int)})
    assert quality_only_auroc(df.iloc[:300], df.iloc[300:], "hypertension") > 0.8
    y = rng.integers(0, 2, 200).astype(float)
    cm = calibration_metrics(y, np.clip(y * 0.8 + 0.1, 0, 1))
    assert cm["ece"] < 0.2 and cm["brier"] < 0.05
    rows = quality_strata(df, "hypertension", rng.random(n), 0.5)
    assert [r["stratum"] for r in rows] == ["gradable", "ungradable"]
    X = rng.normal(0, 1, (200, 5))
    lab = np.r_[np.zeros(100), np.ones(100)]
    X[lab == 1, 0] += 3.0                                       # a "camera" direction
    g = np.arange(200) // 2
    assert decodability(X, lab, g)["auroc"] > 0.95
    assert abs(decodability(rng.normal(0, 1, (200, 5)), lab, g)["auroc"] - 0.5) < 0.15


def test_odir_adapter_dispatches_through_load_any(tmp_path):
    from brset_dataset import load_any
    (tmp_path / "preprocessed_images").mkdir()
    pd.DataFrame({"ID": [1, 1, 2, 2], "Patient Age": [60, 60, 1, 1],
                  "Patient Sex": ["Male", "Male", "Female", "Female"],
                  "Left-Diagnostic Keywords": ["moderate non proliferative retinopathy"] * 2
                  + ["normal fundus"] * 2,
                  "Right-Diagnostic Keywords": ["normal fundus"] * 2 + ["lens dust"] * 2,
                  "filename": ["1_left.jpg", "1_right.jpg", "2_left.jpg", "2_right.jpg"]}
                 ).to_csv(tmp_path / "full_df.csv", index=False)
    df = load_any(str(tmp_path), "odir")["df"].set_index("file")
    assert df.loc["1_left.jpg", "final_icdr"] == 2.0 and df.loc["1_right.jpg", "final_icdr"] == 0.0
    assert np.isnan(df.loc["2_left.jpg", "age"])            # placeholder age 1 -> unknown
    assert encoding_audit(df.reset_index(), "dr_referable")["ok"]
    # sex is 1 = male in every adapter (mBRSET's raw convention); the old rules made
    # ODIR and BRSET 1 = female, reversing any sex coefficient across datasets
    assert df.loc["1_left.jpg", "sex"] == 1.0 and df.loc["2_left.jpg", "sex"] == 0.0
    from brset_dataset import _normalise_sex
    assert _normalise_sex(pd.Series([1, 2], dtype=float)).tolist() == [1.0, 0.0]


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


def _make_brset(root, n_pat, rng, camera_of=None, shared=()):
    """``camera_of(pid)`` adds BRSET's camera column; every patient in ``shared``
    also gets one photo on the other camera (a held-out camera must drop them)."""
    (root / "fundus_photos").mkdir(parents=True)
    rows = []

    def row(f, pid, eye, ref):
        r = {"image_id": f, "patient_id": pid, "DR_ICDR": 2 if ref else 0,
             "macular_edema": int(ref and eye == 1), "patient_age": 40 + pid % 30,
             "patient_sex": 1 + pid % 2, "exam_eye": eye, "quality": "Adequate",
             "diabetes_time_y": pid % 20, "insuline": pid % 2,
             "diabetes": "yes", "nationality": "Brazil"}
        if camera_of is not None:
            r["camera"] = camera_of(pid)
        return r

    for pid in range(n_pat):
        ref = pid % 3 == 0
        for eye in (1, 2):
            f = f"img{pid:03d}{eye}"
            _fundus(root / "fundus_photos" / f"{f}.jpg", ref, (1, 1, 1), False, rng)
            rows.append(row(f, pid, eye, ref))
    cams = sorted({camera_of(p) for p in range(n_pat)}) if camera_of is not None else []
    for pid in shared:
        f = f"img{pid:03d}9"
        _fundus(root / "fundus_photos" / f"{f}.jpg", pid % 3 == 0, (1, 1, 1), False, rng)
        rows.append(dict(row(f, pid, 1, pid % 3 == 0),
                         camera=next(c for c in cams if c != camera_of(pid))))
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)


def _make_mbrset(root, n_pat, rng, cast=(0.8, 1.1, 1.3)):
    (root / "images").mkdir(parents=True)
    rows = []
    for pid in range(n_pat):
        ref, htn = pid % 3 == 0, pid % 2 == 0
        for k in (1, 2):
            f = f"{pid}.{k}.jpg"
            _fundus(root / "images" / f, ref, cast, htn, rng)   # camera colour cast
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
    _make_mbrset(tmp_path / "cam3", 30, rng, cast=(1.3, 0.9, 0.6))      # an unfamiliar camera
    out, ck = tmp_path / "out", tmp_path / "ck" / "s0.pt"
    rc = main(["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
               "--probe-targets", "hypertension", "nephropathy", "smoking",
               "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64",
               "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
               "--calib-batch", "4", "--calib-sizes", "4", "8", "--calib-repeats", "2",
               "--prior-strengths", "8", "0", "--min-probe-pos", "2", "--n-boot", "20",
               "--envelope-repeats", "4", "--demo-n", "8", "--n-shuffle", "3",
               "--unfamiliar", str(tmp_path / "cam3"), "mbrset", "cam3",
               "--cv-repeats", "1", "--min-test-pos-patients", "2", "--probe-folds", "2",
               "--out", str(out), "--ckpt", str(ck)])
    assert rc == 0
    cap_all = pd.read_csv(out / "capability.csv")
    targets = ["dr_referable", "edema", "hypertension", "nephropathy", "smoking"]
    oof = cap_all[cap_all["protocol"] == "calibrated_oof2"]
    cap = cap_all[cap_all["protocol"].isin(["trained", "calibrated"])].reset_index(drop=True)
    # cross-fitted probes: every handheld row scored once, gated like any other row
    assert len(oof) >= 1 and set(oof["target"]) <= {"hypertension", "nephropathy"}
    assert (oof["device"] == "handheld").all() and (oof["design"] == "oof2").all()
    pr_oof = pd.read_csv(out / "predictions_handheld_probes_oof2.csv")
    assert {"fold_hypertension", "p_hypertension", "threshold_hypertension"} <= set(pr_oof.columns)
    assert set(pr_oof["fold_hypertension"].dropna().astype(int)) <= {-1, 0, 1}
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
    assert set(res["camera_decodability"]) == {"trained", "calibrated"}
    assert 0.0 <= res["camera_decodability"]["trained"]["auroc"] <= 1.0
    # controls and strata
    hp = cap[(cap["target"] == "hypertension") & (cap["device"] == "handheld")
             & (cap["protocol"] == "calibrated")].iloc[0]
    assert "shuffle_auroc" in cap.columns and hp["shuffle_auroc"] == hp["shuffle_auroc"]
    assert {"quality_only_auroc", "brier", "ece", "mean_predicted"} <= set(cap.columns)
    strata = pd.read_csv(out / "quality_strata.csv")
    assert {"gradable", "ungradable"} == set(strata["stratum"])
    # the unfamiliar camera: gated with its own labels, and the phone's lookup recorded
    unf = pd.read_csv(out / "capability_unfamiliar.csv")
    assert set(unf["device"]) == {"cam3"} and len(unf) == len(targets) * 2
    look = pd.read_csv(out / "unfamiliar_lookup.csv")
    assert set(look["n"]) == {4, 8} and set(look["target"]) == set(targets)
    assert ((look["unsafe_rate"] == 0) | (look["observed"] != "SUPPORTED")).all()
    assert "cam3" in res["unfamiliar"]
    abl = pd.read_csv(out / "anatomy_ablation.csv")
    assert {"auroc_intact", "auroc_vessels", "auroc_control", "vessels_minus_control"} <= set(abl.columns)
    assert set(abl["device"]) <= {"tabletop", "handheld"}
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
    # label-free operating point: every device x method; the references ship with the profiles
    from retinareach import THRESHOLD_METHODS
    op = pd.read_csv(out / "operating_point.csv")
    assert set(op["method"]) == set(THRESHOLD_METHODS) and set(op["device"]) == {"tabletop", "handheld"}
    assert set(op["target"]) == set(res["threshold_reference"])
    assert len(op) == len(THRESHOLD_METHODS) * 2 * len(set(op["target"]))
    assert op.loc[op["method"].isin(["em", "em2", "rate"]), "est_prev"].between(0, 1).all()
    assert op.loc[op["method"].isin(["shipped", "anchor"]), "est_prev"].isna().all()
    assert op.loc[op["method"] == "rate", "flag_rate"].between(0, 1).all()
    for t in res["threshold_reference"]:
        assert 0 <= res["threshold_reference"][t]["source_spec"] <= 1
    for t, thr in res["thresholds"]["calibrated"].items():
        if t in res["threshold_reference"]:
            shipped = op.loc[(op["method"] == "shipped") & (op["target"] == t), "threshold"]
            assert shipped.to_numpy() == pytest.approx(np.full(len(shipped), thr))
            assert ThresholdReference.from_dict(prof["threshold_reference"][t]).threshold == thr
    assert ckd["threshold_reference"] == prof["threshold_reference"]
    t0 = next(iter(res["threshold_reference"]))
    assert {f"sens_{t0}__em", f"flagged_{t0}__anchor", f"prev_{t0}__em", f"sens_{t0}__em2",
            f"prev_{t0}__rate"} <= set(sweep.columns)
    assert sweep.loc[sweep["kind"] == "trained", f"sens_{t0}__em"].isna().all()
    assert sweep.loc[sweep["kind"] != "trained", f"sens_{t0}__anchor"].notna().all()

    # the multi-seed summary runs on the per-seed layout (the same run twice here)
    import shutil
    from summarize_retinareach import main as summarize
    exp = tmp_path / "seed99" / "exp"          # a seed-like parent must not confuse the loader
    shutil.copytree(out, exp / "seed0")
    shutil.copytree(out, exp / "seed1")
    # seed1 plays a run from before the in-forward prior: its prior rows must be dropped
    r1 = json.loads((exp / "seed1" / "results.json").read_text())
    r1.pop("prior_mode")
    (exp / "seed1" / "results.json").write_text(json.dumps(r1))
    for sd, auc in (("seed0", 0.71), ("seed1", 0.69)):          # stand-ins for PERTARGET runs
        (exp / sd / "pertarget_hypertension.json").write_text(json.dumps(
            {"task": "hypertension", "test": {"auroc": auc, "n": 10},
             "covariate_baseline": {"auroc": 0.6}}))
    assert summarize(["--dir", str(exp)]) == 0
    final = pd.read_csv(exp / "capability_final.csv")
    assert len(final) == len(cap_all) and set(final["final"]) <= set(cap_all["status"]) | {"NO_IMAGE_EVIDENCE"}
    md = (exp / "summary.md").read_text()
    assert md.startswith("# RetinaReach")
    assert "## 4. Controls and mechanism" in md and "## 5. Unfamiliar cameras" in md
    assert "cam3" in md and "Camera decodability" in md
    summ = json.loads((exp / "summary.json").read_text())
    assert summ["decodability"]["paired"]["n"] == 2 and summ["unfamiliar_lookup"]
    svp = summ["shared_vs_pertarget"]
    assert len(svp) == 1 and svp[0]["target"] == "hypertension" and svp[0]["seeds"] == 2
    assert "Shared trunk vs per-target" in md
    from dr_proxy_check import main as dr_proxy
    assert dr_proxy(["--dir", str(exp), "--n-boot", "20"]) == 0
    px = pd.read_csv(exp / "dr_proxy.csv")
    assert set(px["seed"]) == {0, 1} and set(px["target"]) <= {"hypertension", "nephropathy"}
    assert {"auroc_probe", "auroc_dr_score", "auroc_no_dr", "auroc_within_grade"} <= set(px.columns)
    assert "Label-free operating point" in md and summ["operating_point"]
    assert {r["method"] for r in summ["operating_point"]} == set(THRESHOLD_METHODS)
    final_all = pd.read_csv(exp / "capability_final.csv")
    assert (final_all["protocol"] == "calibrated_oof2").any()      # the cross-fitted rows summarise too
    assert any(k.startswith(f"sens_{t0}__em") for k in summ["sweep"][0])
    from summarize_retinareach import load as load_runs
    loaded = load_runs(str(exp))
    assert (loaded[0]["sweep"]["prior_strength"] > 0).any()          # in-forward run keeps them
    assert not (loaded[1]["sweep"]["prior_strength"] > 0).any()      # pre-fix run: dropped
    # poster figures: any Python with matplotlib + pandas (the repo .venv has no matplotlib)
    import shutil
    import subprocess
    import sys as _sys
    py = next((c for c in (_sys.executable, shutil.which("python3")) if c and subprocess.run(
        [c, "-c", "import matplotlib, pandas"], capture_output=True).returncode == 0), None)
    if py:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run([py, os.path.join(root, "plot_retinareach.py"), "--dir", str(exp)],
                       check=True, capture_output=True)
        for i, name in enumerate(("threshold_transfer", "capability", "calibration_size",
                                  "mechanism", "unfamiliar"), 1):
            assert (exp / "figures" / f"fig{i}_{name}.png").exists()
            assert (exp / "figures" / f"fig{i}_{name}.csv").exists()


def test_one_source_camera_with_the_other_held_out(tmp_path):
    """--source-camera / --holdout-camera: the model trains on one BRSET camera and
    the other is scored as an unfamiliar camera; a patient photographed on both is
    dropped from the held-out camera, so no held-out patient was in training."""
    from train_retinareach import camera_slug, main
    rng = np.random.default_rng(1)
    cam = lambda pid: "Canon CR" if pid % 2 == 0 else "NIKON NF5050"
    _make_brset(tmp_path / "brset", 60, rng, camera_of=cam, shared=(0,))
    _make_mbrset(tmp_path / "mbrset", 30, rng)
    out = tmp_path / "out"
    rc = main(["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
               "--source-camera", "Canon CR", "--holdout-camera", "NIKON NF5050",
               "--probe-targets", "hypertension",
               "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64",
               "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
               "--calib-batch", "4", "--calib-sizes", "4", "--calib-repeats", "1",
               "--prior-strengths", "0", "--min-probe-pos", "2", "--n-boot", "10",
               "--envelope-repeats", "2", "--demo-n", "8", "--n-shuffle", "2",
               "--ablation-frac", "0", "--cv-repeats", "1", "--min-test-pos-patients", "2",
               "--out", str(out), "--ckpt", str(tmp_path / "ck.pt")])
    assert rc == 0
    name = camera_slug("NIKON NF5050")
    assert name == "nikon_nf5050"
    src = pd.read_csv(out / "split_tabletop.csv")
    hold = pd.read_csv(out / f"split_{name}.csv")
    labels = pd.read_csv(tmp_path / "brset" / "labels.csv").set_index("image_id")
    assert (labels.loc[src["file"].str.replace(".jpg", "", regex=False), "camera"] == "Canon CR").all()
    assert (labels.loc[hold["file"].str.replace(".jpg", "", regex=False), "camera"] == "NIKON NF5050").all()
    assert not set(src["patient"]) & set(hold["patient"])          # patient 0's Nikon photo dropped
    res = json.loads((out / "results.json").read_text())
    assert res["holdout_camera"]["dropped_patients"] == 1 and res["holdout_camera"]["name"] == name
    assert res["holdout_camera"]["kept_patients"] == 30
    unf = pd.read_csv(out / "capability_unfamiliar.csv")
    assert set(unf["device"]) == {name}
    for proto in ("trained", "calibrated"):
        pr = pd.read_csv(out / f"predictions_{name}_{proto}.csv")
        assert set(pr["file"]) <= set(hold.loc[hold["split"] == "test", "file"]) and len(pr)
    # per-1,000 referral outcomes on every rung of the camera ladder, from the saved files
    from referral_outcomes import main as referral
    assert referral(["--dir", str(out), "--n-boot", "10"]) == 0
    ro = pd.read_csv(out / "referral_outcomes.csv")
    assert set(ro["device"]) == {"tabletop", "handheld", name}
    assert set(ro["rule"]) == {"dr_referable", "dr_or_edema"}
    assert set(ro["protocol"]) == {"trained", "calibrated"}
    every = ["per1000_correct_referrals", "per1000_missed", "per1000_unnecessary",
             "per1000_correct_clear", "per1000_retakes"]
    assert np.allclose(ro[every].sum(1), 1000.0)
    assert {"per1000_missed_at_0.055", "per1000_unnecessary_at_0.156"} <= set(ro.columns)

    # redesign_calibration rebuilds the run's own cameras: Canon filtered, Nikon held
    # out, with the trainer's exact splits -- so no training patient is a test patient
    import argparse
    from capability_gate import multitask_split
    from redesign_calibration import device_specs, main as redesign
    ckd = torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=False)
    specs = device_specs(ckd, argparse.Namespace(source_root=None, target_root=None,
                                                 no_unfamiliar=False, no_holdout=False))
    assert [s[0] for s in specs] == ["tabletop", name, "handheld"]
    for dev_name, _, _, table in specs[:2]:
        test = multitask_split(table, "dr_referable", seed=42)["test"]
        trainer = pd.read_csv(out / f"split_{dev_name}.csv")
        assert set(test["file"]) == set(trainer.loc[trainer["split"] == "test", "file"])
    rd = tmp_path / "redesign"
    assert redesign(["--ckpt", str(tmp_path / "ck.pt"), "--out", str(rd), "--pool-size", "8",
                     "--repeats", "1", "--depth-fractions", "0", "0.5", "1", "--alphas", "0.5",
                     "--small-n", "4", "--small-n-repeats", "1", "--num-workers", "0",
                     "--device", "cpu"]) == 0
    df = pd.read_csv(rd / "redesign.csv")
    assert {"tabletop", name, "handheld"} <= set(df["device"])
    a_rows = df[(df["experiment"] == "A_prevalence") & (df["device"] == name)]
    assert a_rows["source_prevalence"].any()                   # Nikon at Canon's disease rate
    assert ((df["experiment"] == "B1_shallow") & (df["device"] == name)).any()
    lad = pd.read_csv(rd / "ladder.csv")
    nik = set(lad.loc[lad["device"] == name, "setting"])
    assert {"as trained", "AdaBN, own prevalence", "AdaBN, source prevalence",
            "shallow, own prevalence", "shallow + EM threshold"} <= nik
    assert list(dict.fromkeys(lad["device"])) == ["tabletop", name, "handheld"]   # ladder order
    assert "camera ladder" in (rd / "redesign_summary.txt").read_text()


def test_referral_outcomes_count_per_patient():
    """Refer if any gradable eye fires; no gradable eye = retake; truth = any eye."""
    from referral_outcomes import patient_outcomes
    eyes = pd.DataFrame({
        "patient": ["1", "1", "2", "2", "3", "3", "4", "4"],
        "gradable": [True, True, True, True, False, False, True, False],
        "flag": [True, False, False, False, True, False, True, False],     # 3's flag is on an ungradable eye
        "truth": [1, 0, 0, 0, 1, 0, 0, np.nan]})
    o = patient_outcomes(eyes, n_boot=20, prevalences=[0.1])
    assert (o["tp"], o["fn"], o["fp"], o["tn"], o["retakes"]) == (1, 0, 1, 1, 1)
    assert o["referable_in_retakes"] == 1 and o["patients"] == 4
    assert o["per1000_retakes"] == 250 and o["per1000_unnecessary"] == 250 and o["per1000_missed"] == 0
    assert o["sensitivity"] == 1.0 and o["specificity"] == 0.5
    assert o["per1000_unnecessary_at_0.1"] == pytest.approx(1000 * 0.9 * 0.5)


def test_camera_and_once_only_guards(tmp_path):
    from train_retinareach import main
    base = ["--source-root", str(tmp_path / "a"), "--target-root", str(tmp_path / "b"),
            "--out", str(tmp_path / "o"), "--ckpt", str(tmp_path / "c.pt")]
    with pytest.raises(SystemExit, match="needs --source-camera"):
        main(base + ["--holdout-camera", "NIKON NF5050"])
    with pytest.raises(SystemExit, match="is the --source-camera"):
        main(base + ["--source-camera", "Canon CR", "--holdout-camera", "Canon CR"])
    with pytest.raises(SystemExit, match="once-only external test"):
        main(base + ["--unfamiliar", str(tmp_path / "m2"), "messidor2"])
    assert not (tmp_path / "o").exists()                 # refused before anything was written


def _vessel_image(size=192, seed=0):
    """A fundus-like disc with a dark branching 'vessel' tree; returns (x01, vessel_truth)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, size - 4, size - 4], fill=(190, 95, 50))
    truth = Image.new("L", (size, size), 0)
    dt = ImageDraw.Draw(truth)
    c = size // 2
    for k in range(7):                                     # radial vessels from a 'disc'
        ang = 2 * np.pi * k / 7 + rng.uniform(-.2, .2)
        pts = [(c + r * np.cos(ang + 0.3 * np.sin(r / 20)), c + r * np.sin(ang + 0.3 * np.sin(r / 20)))
               for r in range(8, size // 2 - 12, 2)]
        d.line(pts, fill=(120, 40, 25), width=3)
        dt.line(pts, fill=255, width=3)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 255
    return x, torch.from_numpy(np.asarray(truth) > 0)


def test_vessel_mask_finds_vessels_and_inpainting_removes_them():
    from vessel_ablation import control_mask, fov_mask, inpaint, vessel_mask
    x, truth = _vessel_image()
    # quota = the true vessel fraction of the FOV, so precision is measurable
    frac = float(truth.sum() / fov_mask(x).sum())
    m = vessel_mask(x, frac=frac)[0, 0]
    recall = float((m & truth).sum() / truth.sum())
    precision = float((m & truth).sum() / m.sum())
    assert recall > 0.6 and precision > 0.4, (recall, precision)
    c = control_mask(m[None, None], fov_mask(x))[0, 0]
    assert float((c & truth).sum() / c.sum()) < precision          # the control is mostly off-vessel
    out = inpaint(x, m[None, None])
    bg = x[0, :, fov_mask(x)[0, 0] & ~m].mean(1)
    before = (x[0][:, truth & m].mean(1) - bg).abs().sum()
    after = (out[0][:, truth & m].mean(1) - bg).abs().sum()
    assert after < 0.3 * before                                      # vessel pixels now look like background
    assert torch.equal(out[0][:, ~m], x[0][:, ~m])                  # nothing else changes


def test_ablate_normalised_keeps_shape_and_range():
    from vessel_ablation import ablate_normalised
    x, _ = _vessel_image(96)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    xn = (x - mean) / std
    for kind in ("vessels", "control"):
        y = ablate_normalised(xn, kind)
        assert y.shape == xn.shape and torch.isfinite(y).all() and not torch.equal(y, xn)
    with pytest.raises(ValueError):
        ablate_normalised(xn, "nope")


def test_preflight_passes_and_catches_a_broken_unfamiliar_root(tmp_path, capsys):
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    _make_brset(tmp_path / "brset", 30, rng)
    _make_mbrset(tmp_path / "mbrset", 30, rng)
    base = ["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
            "--probe-targets", "hypertension", "--backbone", BACKBONE, "--no-pretrained",
            "--image-size", "64", "--epochs", "2", "--batch-size", "8", "--num-workers", "0",
            "--device", "cpu", "--calib-batch", "4", "--calib-sizes", "4", "8",
            "--out", str(tmp_path / "pf"), "--ckpt", str(tmp_path / "pf.pt"), "--preflight"]
    assert train(base) == 0
    out = capsys.readouterr().out
    assert "PREFLIGHT OK" in out and "positive patients" in out and "estimate:" in out
    assert not (tmp_path / "pf" / "results.json").exists()          # nothing was trained
    assert train(base + ["--unfamiliar", str(tmp_path / "nowhere"), "odir"]) == 1
    assert "PREFLIGHT FAILED" in capsys.readouterr().out


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


@pytest.mark.skipif(__import__("sys").platform != "darwin" or not os.environ.get("RR_SWIFT"),
                    reason="Swift reference parity: macOS + a Swift toolchain, opt in with RR_SWIFT=1")
def test_swift_reference_matches_python(tmp_path):
    """app/RetinaReachKit (the phone-side code) against Python on the same captures:
    identical calibration statistics, capability lookup and screening probabilities;
    preprocessing close (JPEG decoders differ)."""
    import shutil
    import subprocess
    pytest.importorskip("coremltools")
    if shutil.which("swift") is None:
        pytest.skip("no swift toolchain")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, "app", "RetinaReachKit")
    subprocess.run(["swift", "build", "-c", "release", "--package-path", pkg], check=True,
                   capture_output=True)
    cli = os.path.join(pkg, ".build", "release", "retinareach-cli")
    from export_retinareach import main as export
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    _make_brset(tmp_path / "brset", 45, rng)
    _make_mbrset(tmp_path / "mbrset", 45, rng)
    _make_mbrset(tmp_path / "cam3", 12, rng, cast=(1.3, 0.9, 0.6))
    assert train(["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
                  "--probe-targets", "hypertension", "--backbone", BACKBONE, "--no-pretrained",
                  "--image-size", "64", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
                  "--device", "cpu", "--calib-batch", "4", "--calib-sizes", "4", "8",
                  "--calib-repeats", "2", "--envelope-repeats", "4", "--prior-strengths", "0",
                  "--min-probe-pos", "2", "--n-boot", "20", "--cv-repeats", "1",
                  "--min-test-pos-patients", "2", "--demo-n", "8", "--n-shuffle", "2",
                  "--out", str(tmp_path / "out"), "--ckpt", str(tmp_path / "ck.pt")]) == 0
    out = tmp_path / "export"
    assert export(["--checkpoint", str(tmp_path / "ck.pt"), "--precision", "fp32",
                   "--verify-images", str(tmp_path / "cam3" / "images"), "--verify-n", "16",
                   "--verify-units", "CPU_ONLY", "--swift-cli", cli, "--out-dir", str(out)]) == 0
    par = json.loads((out / "export_report.json").read_text())["swift_parity"]
    cal = par["calibration"]
    assert cal["camera_distance_to_pytorch"] < 1e-6 * max(1.0, cal["camera_distance_trained_to_pytorch"])
    assert par["lookup"]["statuses_equal"] and par["lookup"]["nearest_equal"]
    assert par["lookup"]["inside_equal"]
    assert par["screen"]["max_abs"] < 1e-4
    assert par["preprocessing"]["mean_abs"] < 1.0                   # JPEG decoders differ slightly
    assert par["end_to_end"]["camera_distance_to_python"] < 0.1 * cal["camera_distance_trained_to_pytorch"]
    # the redesign's device path: shallow mode builds exactly Python's shallow vector, and
    # the em2 / rate thresholds re-set on the phone equal Python's on the same captures
    dp = par["device_path"]
    assert set(dp) == {"em2", "rate"}
    for d in dp.values():
        assert d["mode"] == "shallow" and d["shallow_vector_max_abs"] < 1e-6
        assert d["thresholds"] and all(v["abs_diff"] < 1e-3 for v in d["thresholds"].values())
        assert all(abs(v["est_prev_swift"] - v["est_prev_python"]) < 1e-3 for v in d["thresholds"].values())
    prof = json.loads((out / "retinareach_profiles.json").read_text())
    assert prof["calibration"]["source_profile"] == "tabletop" and prof["calibration"]["shallow_layers"] > 0
    assert set(prof["threshold_reference"]) == {"dr_referable", "edema"}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple-GPU (MPS) regression")
def test_mps_run_with_spawned_workers_trains_on_real_labels(tmp_path):
    """Two Mac-only failures, both silent: (1) a non_blocking copy of the strided
    label slice to MPS delivered garbage labels (loss 1e15, then negative);
    (2) with spawn-started workers, a numpy view of the dataset labels taken
    before the first loader ran pointed at freed memory after torch moved the
    storage to shared memory (validation AUROC undefined every epoch)."""
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    _make_brset(tmp_path / "brset", 45, rng)
    _make_mbrset(tmp_path / "mbrset", 45, rng)
    out = tmp_path / "out"
    assert train(["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
                  "--probe-targets", "hypertension", "--backbone", BACKBONE, "--no-pretrained",
                  "--image-size", "64", "--epochs", "2", "--batch-size", "8", "--num-workers", "2",
                  "--device", "mps", "--calib-batch", "4", "--calib-sizes", "4",
                  "--calib-repeats", "1", "--envelope-repeats", "2", "--prior-strengths", "0",
                  "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1",
                  "--min-test-pos-patients", "2", "--demo-n", "4", "--n-shuffle", "0",
                  "--out", str(out), "--ckpt", str(tmp_path / "ck.pt")]) == 0
    res = json.loads((out / "results.json").read_text())
    assert res["best_epoch"] > 0 and 0.0 <= res["best_val_mean_auroc"] <= 1.0


def test_split_file_pairs_a_per_target_model_with_the_retinareach_split(tmp_path, monkeypatch):
    import sys as _sys
    import train_mbrset
    from brset_dataset import load_any
    from metadata_model import target_vector
    rng = np.random.default_rng(0)
    _make_mbrset(tmp_path / "m", 30, rng)
    df = load_any(str(tmp_path / "m"), "mbrset")["df"]
    sp = multitask_split(df, "dr_referable", seed=42)
    split_csv = tmp_path / "split_handheld.csv"
    pd.concat([v.assign(split=k)[["file", "patient", "split"]] for k, v in sp.items()]).to_csv(
        split_csv, index=False)
    got = train_mbrset.splits_from_file(df, str(split_csv))
    assert got["test"]["file"].tolist() == sp["test"]["file"].tolist()
    bad = pd.read_csv(split_csv)
    bad.loc[bad["file"] == sp["test"]["file"].iloc[0], "split"] = "train"   # breaks patient grouping
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        train_mbrset.splits_from_file(df, str(tmp_path / "bad.csv"))
    out = tmp_path / "pt.json"
    monkeypatch.setattr(_sys, "argv", [
        "train_mbrset.py", "--dataset", "mbrset", "--root", str(tmp_path / "m"), "--task", "hypertension",
        "--split-file", str(split_csv), "--backbone", BACKBONE, "--no-gcg", "--no-pretrained",
        "--image-size", "64", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
        "--covariate-baseline", "--ckpt-dir", str(tmp_path / "ck"), "--run-name", "hypertension_seed0",
        "--results-json", str(out)])
    assert train_mbrset.main() == 0
    res = json.loads(out.read_text())
    n_labelled = int((~np.isnan(target_vector(sp["test"], "hypertension"))).sum())
    assert res["split_file"].endswith("split_handheld.csv") and res["test"]["n"] == n_labelled


def test_redesign_calibration_runs_on_a_checkpoint(tmp_path):
    from redesign_calibration import main as redesign
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    _make_brset(tmp_path / "brset", 40, rng)
    _make_mbrset(tmp_path / "mbrset", 40, rng)
    _make_mbrset(tmp_path / "cam3", 30, rng, cast=(1.3, 0.9, 0.6))
    ck = tmp_path / "ck.pt"
    assert train(["--source-root", str(tmp_path / "brset"), "--target-root", str(tmp_path / "mbrset"),
                  "--probe-targets", "hypertension", "--backbone", BACKBONE, "--no-pretrained",
                  "--image-size", "64", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
                  "--device", "cpu", "--calib-batch", "4", "--calib-sizes", "4",
                  "--calib-repeats", "1", "--envelope-repeats", "2", "--prior-strengths", "0",
                  "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1",
                  "--min-test-pos-patients", "2", "--demo-n", "4", "--n-shuffle", "0",
                  "--ablation-frac", "0", "--unfamiliar", str(tmp_path / "cam3"), "mbrset", "cam3",
                  "--out", str(tmp_path / "out"), "--ckpt", str(ck)]) == 0
    out = tmp_path / "redesign"
    assert redesign(["--ckpt", str(ck), "--out", str(out), "--pool-size", "12", "--repeats", "1",
                     "--depth-fractions", "0", "1", "--alphas", "0.5", "--num-workers", "0",
                     "--small-n", "4", "999999", "--small-n-repeats", "2",
                     "--device", "cpu"]) == 0
    from retinareach import THRESHOLD_METHODS
    df = pd.read_csv(out / "redesign.csv")
    assert {"baseline", "A_prevalence", "B1_shallow", "B2_prior", "B3_threshold",
            "D_matched"} <= set(df["experiment"])
    assert {"tabletop", "handheld", "cam3"} <= set(df["device"])
    a = df[df["experiment"] == "A_prevalence"]
    assert (a["device"] == "tabletop").any()                     # the no-camera-change control
    b3 = df[df["experiment"] == "B3_threshold"]
    assert set(b3["method"]) == set(THRESHOLD_METHODS) - {"shipped"} and b3["pool_prev"].notna().all()
    assert b3.loc[b3["method"].isin(["em", "em2", "rate"]), "est_prev"].between(0, 1).all()
    assert b3.loc[b3["method"] == "anchor", "est_prev"].isna().all()
    assert b3.loc[b3["method"] == "rate", "flag_rate"].between(0, 1).all()
    # D: the label-free prevalence-matched pool, full depth and shallow, then as B3 bases
    dm = df[df["experiment"] == "D_matched"]
    assert set(dm["device"]) == {"handheld", "cam3"} and dm["matched_expected_prev"].between(0, 1).all()
    assert {"matched pool, source prevalence (full depth)", "matched pool, source prevalence + shallow",
            "matched pool, zero prevalence (full depth)"} <= set(dm["setting"])
    assert {"matched source (full depth)", "matched source + shallow", "matched zero + shallow"
            } <= set(b3["base"])
    # C: N captures give both the statistics and the threshold, every method on every base
    c = df[df["experiment"] == "C_small_n"]
    assert set(c["method"]) == set(THRESHOLD_METHODS)
    shallow_base = [b for b in c["base"].unique() if b.startswith("shallow ")]
    assert len(shallow_base) == 1 and set(c["n_captures"]) == {4}
    assert set(c["base"]) == {"tabletop statistics", "AdaBN", shallow_base[0], "matched", "matched + shallow"}
    assert c.loc[c["base"] == "matched", "matched_expected_prev"].between(0, 1).all()
    assert any(b.startswith("shallow ") for b in b3["base"].unique())
    assert c["meets_floor"].isin([True, False]).all() and (c["repeat"] < 2).all()
    shipped = c[(c["method"] == "shipped") & (c["base"] == "tabletop statistics")]
    # no captures read: identical on every draw
    assert shipped.groupby(["device", "target"])["sensitivity"].nunique().eq(1).all()
    # K = 0 layers is the tabletop-statistics baseline, K = all is full AdaBN
    b1 = df[(df["experiment"] == "B1_shallow") & (df["device"] == "handheld") & (df["target"] == "dr_referable")]
    base = df[(df["experiment"] == "baseline") & (df["device"] == "handheld") & (df["target"] == "dr_referable")]
    k0 = b1[b1["layers"] == 0]["auroc"].iloc[0]
    assert k0 == pytest.approx(base[base["setting"] == "tabletop statistics"]["auroc"].iloc[0])
    summary = (out / "redesign_summary.txt").read_text()
    assert "C_small_n" in summary and "meets_floor" in summary and "D_matched" in summary
    assert "shallow + EM2 threshold" in summary and "specificity:" in summary
    # the cross-fitted probes, post hoc on the same checkpoint
    from probe_folds import main as probe_folds
    assert probe_folds(["--ckpt", str(ck), "--folds", "2", "--n-shuffle", "2", "--n-boot", "10",
                        "--num-workers", "0", "--device", "cpu", "--out", str(tmp_path / "out")]) == 0
    oof = pd.read_csv(tmp_path / "out" / "capability_probes_oof.csv")
    assert set(oof["protocol"]) == {"calibrated_oof2"} and set(oof["target"]) <= {"hypertension"}
    assert (oof["device"] == "handheld").all() and oof["pos_patients"].ge(1).all()
    assert os.path.exists(tmp_path / "out" / "predictions_handheld_probes_oof2.csv")
    assert json.load(open(tmp_path / "out" / "probe_folds.json"))["settings"]["probe_folds"] == 2
    # the summarizer picks the post-hoc rows up next to the run's own
    import shutil
    from summarize_retinareach import load as load_runs
    exp = tmp_path / "exp_pf"
    shutil.copytree(tmp_path / "out", exp / "seed0")
    runs = load_runs(str(exp))
    assert (runs[0]["cap"]["protocol"] == "calibrated_oof2").sum() == len(oof)
