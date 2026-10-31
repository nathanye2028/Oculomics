"""phone_sim.py: the tabletop -> handheld simulator's primitives, its label safety
on synthetic fundi (no invented lesions, planted lesions kept), the fit / render
CLI, and the training / check / score wiring on the synthetic camera pair."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import phone_sim as ps


def _fundus(n, size=96, phone=False, seed=0, lesions=None, info=False):
    """Synthetic model-view fundi: a phone view is a full circle in the frame, a
    tabletop view the loader's squashed, truncated circle (b/a = 1.25). Disc with
    vessels converging on it, a darker fovea, shading, mild grain. ``lesions`` =
    list of (x, y, radius_px, rgb_delta) in frame coordinates, drawn on every image."""
    g = torch.Generator().manual_seed(seed)
    xx, yy = ps.frame_coords(n, size, size, "cpu")
    a, b = (1.0, 1.0) if phone else (1.0, 1.25)
    fov = ((xx / a) ** 2 + (yy / b) ** 2 <= 1.0).float()
    base = torch.tensor((0.55, 0.2, 0.07) if phone else (0.66, 0.30, 0.10)).view(1, 3, 1, 1)
    r2 = (xx / a) ** 2 + (yy / b) ** 2
    img = base * (1 - 0.25 * r2).unsqueeze(1)
    side = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    dx, dy = 0.5 * side * a, 0.05 * torch.randn(n, generator=g)
    dd = torch.sqrt((xx - dx.view(n, 1, 1)) ** 2 + ((yy - dy.view(n, 1, 1)) * a / b) ** 2)
    img = img + 0.45 * torch.exp(-(dd / 0.09) ** 4).unsqueeze(1) * torch.tensor((0.5, 0.7, 0.8)).view(1, 3, 1, 1)
    ang = torch.atan2((yy - dy.view(n, 1, 1)), xx - dx.view(n, 1, 1))
    for k in range(6):                                  # vessels: dark rays leaving the disc
        th = k * math.pi / 3 + 0.3
        line = torch.exp(-((torch.sin(ang - th) * dd) / 0.03) ** 2) * (dd > 0.07) * (torch.cos(ang - th) > 0)
        img = img * (1 - 0.35 * line.unsqueeze(1))
    fd = torch.sqrt((xx + 0.1 * side.view(n, 1, 1)) ** 2 + (yy * a / b) ** 2)
    img = img * (1 - 0.25 * torch.exp(-(fd / 0.12) ** 2)).unsqueeze(1)
    for (lx, ly, rad, rgb) in lesions or []:
        d = torch.sqrt(((xx - lx) * size / 2) ** 2 + ((yy - ly) * size / 2) ** 2)
        img = img + torch.tensor(rgb).view(1, 3, 1, 1) * torch.exp(-(d / rad) ** 2).unsqueeze(1)
    img = img + (0.012 if phone else 0.003) * torch.randn(img.shape, generator=g)
    out = (img.clamp(0.0, 1.0) * fov.unsqueeze(1)).clamp(0, 1)
    return (out, {"dx": dx, "dy": dy}) if info else out


def _profile(size=64, n=12, seed=0):
    prof, _ = ps.fit_profile([_fundus(n, size, phone=True, seed=seed)], [_fundus(n, size, seed=seed + 1)],
                             size, log=lambda s: None)
    return prof


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_blur_warp_and_aperture_primitives():
    x = torch.rand(3, 3, 40, 40)
    s = torch.tensor([0.0, 0.8, 1.7])
    y = ps.blur(x, s)
    assert torch.equal(y[0], x[0])
    for i in (1, 2):
        r = int(math.ceil(3 * float(s.max())))
        assert torch.allclose(y[i:i + 1], TF.gaussian_blur(x[i:i + 1], 2 * r + 1, float(s[i])), atol=1e-5)
    one = torch.ones(3)
    assert torch.allclose(ps.warp(x, one, one, one, torch.zeros(3, 2), torch.zeros(3), (40, 40)), x, atol=1e-6)
    ap = ps.aperture(torch.ones(1), 0.0023, 41, 41)
    assert ap[0, 0, 20, 20] > 0.99 and ap[0, 0, 0, 0] < 1e-3 and ap[0, 0, 0, 20] > 0.5


def test_fov_shape_reads_the_loaders_squash():
    tab, phone = ps.fov_shape(_fundus(4, 96)), ps.fov_shape(_fundus(4, 96, phone=True))
    assert tab["reliable"].all() and phone["reliable"].all()
    assert torch.allclose(tab["aspect"], torch.full((4,), 0.8), atol=0.03)
    assert torch.allclose(phone["aspect"], torch.ones(4), atol=0.03)
    assert torch.allclose(tab["a_f"], torch.ones(4), atol=0.05)


def test_disc_finder_takes_the_vessel_hub_not_a_rim_reflection():
    x, where = _fundus(4, 96, phone=True, seed=3, info=True)
    xx, yy = ps.frame_coords(4, 96, 96, "cpu")
    band = ((torch.sqrt(xx ** 2 + yy ** 2) > 0.8) & (torch.sqrt(xx ** 2 + yy ** 2) < 0.95) & (yy > 0.3)).float()
    x = (x + 0.45 * band.unsqueeze(1)).clamp(0, 1) * ps.fov_mask(x)        # a bright reflection crescent
    d = ps.find_disc(x)
    assert ((d["cx"] - where["dx"]).abs() < 0.1).all() and ((d["cy"] - where["dy"]).abs() < 0.1).all()
    assert (ps.disc_confidence(d) > 0).all()


def test_zoom_and_shift_keep_the_aperture_inside_a_truncated_field():
    fov = ps.fov_mask(_fundus(2, 96))
    one = torch.ones(2)
    a_f, b_f, centre, rot = one, one * 1.25, torch.zeros(2, 2), torch.zeros(2)
    z = ps.min_zoom(fov, torch.full((2,), 1.1), a_f, b_f, centre, rot)
    assert (z >= 1.2).all() and (z <= 1.35).all()                          # clears the flat top/bottom
    want = torch.tensor([[0.6, 0.0], [0.05, 0.0]])
    got = ps.safe_shift(fov, want, z, a_f, b_f, centre, rot)
    assert got[0, 0] < 0.6 and torch.allclose(got[1], want[1])
    inside = ps._rim_inside(fov, z.view(2, 1), a_f, b_f, (centre + got).view(2, 1, 2), rot)
    assert (inside >= 0.98).all()


# --------------------------------------------------------------------------- #
# profile and render
# --------------------------------------------------------------------------- #
def test_profile_round_trip_and_guards(tmp_path):
    prof = _profile()
    path = tmp_path / "p.json"
    prof.save(str(path))
    back = ps.PhoneProfile.load(str(path))
    assert len(back) == len(prof) and back.image_size == 64
    assert all(np.allclose(back.bank[k], prof.bank[k], atol=1e-6, equal_nan=True) for k in ps.BANK_KEYS)
    blob = json.loads(path.read_text())
    blob["format"] = "other"
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError):
        ps.PhoneProfile.load(str(path))
    with pytest.raises(ValueError):
        ps.PhoneProfile(64, {k: v for k, v in prof.bank.items() if k != "med"})
    with pytest.raises(ValueError):
        ps.PhoneSimulator(prof, out_size=96).render(_fundus(2, 96), ps.PhoneSimulator(prof).sample(2, torch.Generator()))
    with pytest.raises(ValueError):
        ps.PhoneSimulator(prof, stages=("field", "sparkle"))


def test_render_is_reproducible_bounded_and_phone_shaped():
    prof = _profile()
    sim = ps.PhoneSimulator(prof, out_size=64)
    x = _fundus(4, 96, seed=5)                                            # supersampled tabletop view
    y1 = sim(x, torch.Generator().manual_seed(1))
    y2 = sim(x, torch.Generator().manual_seed(1))
    y3 = sim(x, torch.Generator().manual_seed(2))
    assert y1.shape == (4, 3, 64, 64) and torch.equal(y1, y2) and not torch.equal(y1, y3)
    assert torch.isfinite(y1).all() and y1.min() >= 0 and y1.max() <= 1
    assert y1[..., :3, :3].abs().max() == 0                               # corners outside the aperture
    sh = ps.fov_shape(y1)
    assert torch.allclose(sh["aspect"], torch.ones(4), atol=0.06)          # the squash is undone
    g = sim.geometry(x)
    assert g.shape == y1.shape
    for k in range(1, len(ps.STAGES) + 1):                                 # every prefix of the chain runs
        z = ps.PhoneSimulator(prof, stages=ps.STAGES[:k], out_size=64)(x, torch.Generator().manual_seed(1))
        assert torch.isfinite(z).all()
    weak = ps.PhoneSimulator(prof, strength=0.0, out_size=64)(x, torch.Generator().manual_seed(1))
    inner = ps.erode(ps.fov_mask(weak), 9)
    ref = torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear", antialias=True, align_corners=False)
    ref = ps.warp(ref, *[sim._last[k] for k in ("zoom", "a_f", "b_f", "offset", "rot")], (64, 64))
    assert (weak - ref).abs()[inner.expand_as(weak)].mean() < 0.12        # strength 0 ~ geometry only


def test_augment_resizes_everything_and_renders_a_share():
    sim = ps.PhoneSimulator(_profile(), out_size=64)
    x = _fundus(8, 96, seed=6)
    plain = torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear", antialias=True, align_corners=False)
    assert torch.allclose(sim.augment(x, 0.0, torch.Generator().manual_seed(0), 64), plain)
    y = sim.augment(x, 1.0, torch.Generator().manual_seed(0), 64)
    assert y.shape == (8, 3, 64, 64) and all(not torch.allclose(y[i], plain[i]) for i in range(8))


# --------------------------------------------------------------------------- #
# label safety
# --------------------------------------------------------------------------- #
def _spots(x, frac=0.35, floor=0.02, r=3):
    """Lesion-scale spots per image: green-channel points darker (or brighter) than
    ALL eight neighbours ``r`` px away -- a dot, which a straight vessel never is --
    by more than ``frac`` of the background's own spread (so a global contrast change
    does not count) and than ``floor`` (microaneurysms are several percent deep),
    inside the FOV, away from the rim and the optic disc."""
    B, _, H, W = x.shape
    g = ps.blur(x[:, 1:2], 0.7)
    p = torch.nn.functional.pad(g, (r, r, r, r), mode="replicate")
    ring = torch.stack([p[..., r + dy:r + dy + H, r + dx:r + dx + W]
                        for dy in (-r, 0, r) for dx in (-r, 0, r) if (dy, dx) != (0, 0)])
    depth = torch.maximum(ring.amin(dim=0) - g, g - ring.amax(dim=0))
    m = ps.erode(ps.fov_mask(x), 9)
    disc = ps.find_disc(x)
    xx, yy = ps.frame_coords(B, H, W, "cpu")
    far = (((xx - disc["cx"].view(B, 1, 1)) ** 2 + (yy - disc["cy"].view(B, 1, 1)) ** 2).sqrt()
           > 2.0 * disc["r"].clamp_min(0.08).view(B, 1, 1)).unsqueeze(1)
    spread = ps.robust_colour(x, m)["scale"][:, 1].view(B, 1, 1, 1)
    peak = depth == torch.nn.functional.max_pool2d(depth, 5, 1, 2)
    return ((depth > torch.clamp(frac * spread, min=floor)) & peak & m & far).sum(dim=(1, 2, 3))


def test_no_lesion_like_spots_on_a_clean_fundus():
    """Every stage but the grain: illumination, colour, disc bloom, detail and rim
    may not draw a lesion-scale spot the tabletop photo did not have."""
    prof = _profile(size=96)
    x = _fundus(6, 96, seed=8)
    stages = tuple(s for s in ps.STAGES if s not in ("field", "noise"))
    for seed in (0, 1, 2):
        y = ps.PhoneSimulator(prof, stages=stages, out_size=96)(x, torch.Generator().manual_seed(seed))
        assert (_spots(y) <= _spots(x) + 1).all()


def test_grain_lands_on_the_drawn_photos_noise_level():
    prof = _profile(size=96)
    sim = ps.PhoneSimulator(prof, stages=("noise",), out_size=96)
    x = ps.blur(_fundus(6, 96, seed=12), 1.5) * ps.fov_mask(_fundus(6, 96, seed=12))      # nearly grain-free
    prm = sim.sample(6, torch.Generator().manual_seed(4))
    y = sim.render(x, prm)
    m = ps.fov_mask(x)
    got = ps.noise_level(y.mean(dim=1, keepdim=True), m)
    want = torch.from_numpy(prof.bank["noise"])[prm["idx"]]
    have = ps.noise_level(x.mean(dim=1, keepdim=True), m)
    ok = want > have * 1.2                                    # rows that ask for more grain than x has
    assert ok.any() and ((got[ok] / want[ok]) - 1).abs().max() < 0.3


def test_planted_lesions_keep_their_sign_and_contrast():
    lesions = [(0.0, 0.35, 1.6, (-0.10, -0.12, -0.03)),                    # dot haemorrhage / MA
               (-0.3, -0.35, 2.2, (-0.15, -0.16, -0.04)),
               (0.25, -0.2, 1.8, (0.18, 0.15, -0.02))]                     # hard exudate (yellow)
    prof = _profile(size=96)
    stages = tuple(s for s in ps.STAGES if s != "field")                    # positions stay put
    sim = ps.PhoneSimulator(prof, stages=stages, out_size=96)
    clean = _fundus(4, 96, phone=True, seed=11)
    with_l = _fundus(4, 96, phone=True, seed=11, lesions=lesions)
    yc = sim.render(clean, sim.sample(4, torch.Generator().manual_seed(3)))
    yl = sim.render(with_l, sim.sample(4, torch.Generator().manual_seed(3)))
    for (lx, ly, rad, rgb) in lesions:
        i, j = int((ly + 1) * 48), int((lx + 1) * 48)
        before = (with_l - clean)[:, 1, i, j]
        after = (yl - yc)[:, 1, i, j]
        assert (torch.sign(after) == torch.sign(before)).all()
        assert (after.abs() >= 0.3 * before.abs()).all()


# --------------------------------------------------------------------------- #
# CLI and the trainers
# --------------------------------------------------------------------------- #
def _save_photos(folder, x, frame=None):
    folder.mkdir(parents=True)
    for k, im in enumerate(x):
        arr = (im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        img = Image.fromarray(arr)
        if frame is not None:                     # a tabletop frame: the circle cut top and bottom
            img = img.resize(frame)
        img.save(folder / f"p{k:02d}.jpg", quality=95)


def test_fit_and_render_cli(tmp_path):
    _save_photos(tmp_path / "phone", _fundus(10, 120, phone=True, seed=1))
    tab = _fundus(10, 120, phone=True, seed=2)[..., 12:108, :]              # truncated circle, 120 x 96
    _save_photos(tmp_path / "tab", tab)
    out = tmp_path / "ck" / "profile.json"
    assert ps.main(["fit", "--target-dir", str(tmp_path / "phone"), "--source-dir", str(tmp_path / "tab"),
                    "--image-size", "64", "--validate", "6", "--num-workers", "0", "--device", "cpu",
                    "--batch-size", "5", "--out", str(out)]) == 0
    prof = ps.PhoneProfile.load(str(out))
    assert len(prof) == 10 and prof.image_size == 64 and 0.7 < prof.params["source_aspect"] < 0.9
    assert (tmp_path / "ck" / "fit_report.txt").exists() and (tmp_path / "ck" / "fit_grid.jpg").exists()
    rep = pd.read_csv(tmp_path / "ck" / "fit_report.csv")
    assert {"statistic", "auc_tabletop", "auc_simulated"} <= set(rep.columns)
    r = tmp_path / "render"
    assert ps.main(["render", "--profile", str(out), "--source-dir", str(tmp_path / "tab"), "--target-dir",
                    str(tmp_path / "phone"), "--n", "4", "--stage-rows", "2", "--num-workers", "0",
                    "--device", "cpu", "--out", str(r)]) == 0
    assert (r / "grid.jpg").exists() and (r / "stages.jpg").exists()


def test_trainer_check_and_score_with_phone_sim(tmp_path):
    from style_transfer_xfer import main as xfer
    from test_retinareach import BACKBONE, _make_brset, _make_mbrset
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 30, rng)
    _make_mbrset(M, 30, rng)
    prof_path = tmp_path / "ck_phone_sim" / "profile.json"
    assert ps.main(["fit", "--target-root", str(M), "--source-root", str(B), "--image-size", "64",
                    "--validate", "0", "--num-workers", "0", "--device", "cpu", "--out", str(prof_path)]) == 0

    def run(out, ck, extra=()):
        return train(["--source-root", str(B), "--target-root", str(M), "--probe-targets", "hypertension",
                      "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1",
                      "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--calib-batch", "4",
                      "--calib-sizes", "4", "--calib-repeats", "1", "--envelope-repeats", "2",
                      "--prior-strengths", "0", "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1",
                      "--min-test-pos-patients", "2", "--demo-n", "4", "--n-shuffle", "0", "--ablation-frac", "0",
                      "--out", str(out), "--ckpt", str(ck), *extra])

    base, arm = tmp_path / "base.pt", tmp_path / "arm.pt"
    assert run(tmp_path / "out_base", base) == 0
    assert run(tmp_path / "out_arm", arm, ["--phone-sim", str(prof_path), "--phone-sim-p", "0.5"]) == 0
    res = json.loads((tmp_path / "out_arm" / "results.json").read_text())
    assert res["args"]["phone_sim"] == str(prof_path) and res["args"]["gpu_aug"] is True

    out = tmp_path / "check"
    assert xfer(["check", "--ckpt", str(base), "--methods", "colour", "--phone-profile", str(prof_path),
                 "--out", str(out), "--bank", "8", "--calib-n", "8", "--calib-batch", "4", "--grid-n", "4",
                 "--num-workers", "0", "--batch-size", "8", "--device", "cpu"]) == 0
    df = pd.read_csv(out / "check.csv").set_index("method")
    assert "phone_sim" in df.index and np.isfinite(df.loc["phone_sim", "tab_auroc"])
    assert 0 <= df.loc["phone_sim", "vessel_r"] <= 1 and (out / "grid.jpg").exists()

    sc = tmp_path / "score"
    assert xfer(["score", "--source-root", str(B), "--target-root", str(M), "--image-size", "64",
                 "--num-workers", "0", "--device", "cpu", "--batch-size", "8", "--ckpt", f"canon_only={base}",
                 f"canon_phonesim={arm}", "--out", str(sc)]) == 0
    s = pd.read_csv(sc / "score.csv")
    assert set(s.arm) == {"canon_only", "canon_phonesim"} and s.oracle_spec.between(0, 1).all()
