"""phone_sim.py: the tabletop -> handheld simulator's primitives, its label safety
on synthetic fundi (no invented lesions, planted lesions kept, the disc bloom
leaves lesions beside the disc alone), the trainer's views, the fit / render CLI,
and the training / check / score wiring on the synthetic camera pair."""
import copy
import functools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
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


@functools.lru_cache(maxsize=None)
def _profile_at(size, n=8):
    """A profile at the model's size or near it, fitted once per session (read-only:
    copy it before changing it)."""
    return _profile(size=size, n=n)


def _bloomed(prof):
    """A copy whose every bank row blows the disc out to near-white (the synthetic
    phone discs never clip, so the fitted bank's bloom is 0 and the stage idles)."""
    p = copy.deepcopy(prof)
    p.bank["bloom"][:] = 1.0
    p.bank["hcol"][:] = np.array([0.95, 0.9, 0.85], dtype=np.float32)
    return p


class _FixedGeometry(ps.PhoneSimulator):
    """The first render's geometry reused for the next (a lesion beside the disc can
    move the detected disc, and the comparison must be pixel for pixel)."""
    geo = None

    def _field(self, x01, prm, T):
        if self.geo is None:
            self.geo = super()._field(x01, prm, T)
        return self.geo


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
    yy, xx = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
    cb = (((xx + yy) % 2).float() * 0.6 + 0.2).expand(1, 3, 96, 96).contiguous()      # a 2 px checkerboard
    small = ps.warp(cb, one[:1], one[:1], one[:1], torch.zeros(1, 2), torch.zeros(1), (64, 64))
    assert small[..., 4:-4, 4:-4].std() < 0.04          # prefiltered: no aliased pattern (0.075 unfiltered)
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
    assert (ps.disc_confidence(d) > 0).all() and (ps.disc_gate(d) == 1).all()


def test_an_ambiguous_disc_pick_switches_the_disc_steps_off():
    """Two equally disc-like candidates (on the samples a PDR eye's laser-scar patch
    tied with its disc on every training view, and the bloom whitened the lesions
    there): no bloom, no shift toward the drawn photo's disc position, no mirroring."""
    assert torch.allclose(ps.disc_gate({"margin": torch.tensor([1.0, 1.2, 1.3, 1.4, 5.0])}),
                          torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    x = _fundus(2, 96, phone=True, seed=3)
    pair = 0.5 * (x + x.flip(-1))                                       # a mirror-image twin of the disc
    assert (ps.find_disc(pair)["margin"] < 1.05).all() and (ps.find_disc(x)["margin"] > 2).all()
    prof = _profile()
    sim = ps.PhoneSimulator(prof, out_size=64)
    sim(pair, torch.Generator().manual_seed(0))
    assert (sim._last["disc_conf"] == 0).all()
    sim(x, torch.Generator().manual_seed(0))
    assert (sim._last["disc_conf"] > 0).all()
    # every disc-keyed step reads the gated confidence: a bank whose rows bloom the disc
    # white, pull it to the centre and keep it on the other side renders the pair exactly
    # as the plain bank does -- and does change the render of the unambiguous photo
    keyed = _bloomed(prof)
    keyed.bank["disc_x"] = -keyed.bank["disc_x"]
    keyed.bank["disc_dist"][:] = 0.0
    for seed in (0, 1, 2):
        plain = ps.PhoneSimulator(prof, out_size=64)(pair, torch.Generator().manual_seed(seed))
        assert torch.equal(ps.PhoneSimulator(keyed, out_size=64)(pair, torch.Generator().manual_seed(seed)), plain)
    plain = ps.PhoneSimulator(prof, out_size=64)(x, torch.Generator().manual_seed(0))
    assert (ps.PhoneSimulator(keyed, out_size=64)(x, torch.Generator().manual_seed(0)) - plain).abs().max() > 0.05


def test_zoom_and_shift_keep_the_aperture_inside_a_truncated_field():
    fov = ps.fov_mask(_fundus(2, 96))
    one = torch.ones(2)
    a_f, b_f, centre, rot = one, one * 1.25, torch.zeros(2, 2), torch.zeros(2)
    z = ps.min_zoom(fov, torch.full((2,), 1.1), a_f, b_f, centre, rot)
    # clears the flat top/bottom (1.25 here), within one step of the 48-step grid
    assert (z >= 1.25).all() and (z <= 1.27).all()
    want = torch.tensor([[0.6, 0.0], [0.05, 0.0]])
    got = ps.safe_shift(fov, want, z, a_f, b_f, centre, rot)
    assert got[0, 0] < 0.6 and torch.allclose(got[1], want[1])
    inside = ps._rim_inside(fov, z.view(2, 1), a_f, b_f, (centre + got).view(2, 1, 2), rot, radius=1.0, n_pts=96)
    assert (inside >= 0.98).all()                                           # at the aperture's own edge
    z2, ok = ps.min_zoom(fov, torch.full((2,), 1.1), a_f, b_f, centre, rot, return_ok=True)
    assert torch.equal(z2, z) and ok.all()
    xx, yy = ps.frame_coords(2, 96, 96, "cpu")
    tiny = fov & (xx ** 2 + yy ** 2 < 0.09).unsqueeze(1)                     # no zoom up to 1.8 fits in it
    z3, ok3 = ps.min_zoom(tiny, torch.full((2,), 1.1), a_f, b_f, centre, rot, return_ok=True)
    assert not ok3.any() and torch.allclose(z3, torch.full((2,), 1.8))


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
    blob["format"] = "phone_sim/1"                     # an older format: its bank's estimators differ
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="not a phone_sim/2 profile"):
        ps.PhoneProfile.load(str(path))
    with pytest.raises(ValueError, match="missing"):
        ps.PhoneProfile(64, {k: v for k, v in prof.bank.items() if k != "med"})
    with pytest.raises(ValueError, match="measured at 64 px"):
        ps.PhoneSimulator(prof, out_size=96).render(_fundus(2, 96), ps.PhoneSimulator(prof).sample(2, torch.Generator()))
    with pytest.raises(ValueError, match="unknown phone_sim stages"):
        ps.PhoneSimulator(prof, stages=("field", "sparkle"))
    # a misspelt colour mode would silently run another ablation; reversed limits clamp every gain
    for bad in ({"colour_mode": "coupld"}, {"light_lim": [2.5, 0.25]}, {"kappa": [0.4]}, {"disc_margin": [1.4, 1.2]}):
        with pytest.raises(ValueError):
            ps.PhoneProfile(64, prof.bank, {**prof.params, **bad})
    with pytest.raises(ValueError, match="strength"):                   # above 1 a channel extrapolates past 0
        ps.PhoneSimulator(prof, strength=1.2)


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
    # the squash is undone: the fixture's FOV is 1.25x taller than wide in the loader's view
    assert sim._last["reliable"].all()
    assert torch.allclose(sim._last["b_f"] / sim._last["a_f"], torch.full((4,), 1.25), atol=0.05)
    g = sim.geometry(x)
    assert g.shape == y1.shape
    with pytest.raises(ValueError, match="batch"):                        # another batch's warp is refused
        sim.geometry(0.9 * x)
    for k in range(1, len(ps.STAGES) + 1):                                 # every prefix of the chain runs
        z = ps.PhoneSimulator(prof, stages=ps.STAGES[:k], out_size=64)(x, torch.Generator().manual_seed(1))
        assert torch.isfinite(z).all()
    # strength 0 ~ geometry only, against its own geometry (0.08 if strength were ignored)
    weak_sim = ps.PhoneSimulator(prof, strength=0.0, out_size=64)
    weak = weak_sim(x, torch.Generator().manual_seed(1))
    inner = ps.erode(ps.fov_mask(weak), 9)
    ref = F.interpolate(x, size=(64, 64), mode="bilinear", antialias=True, align_corners=False)
    ref = ps.warp(ref, *[weak_sim._last[k] for k in ("zoom", "a_f", "b_f", "offset", "rot")], (64, 64))
    assert (weak - ref).abs()[inner.expand_as(weak)].mean() < 0.01


def test_augment_resizes_everything_and_renders_a_share():
    sim = ps.PhoneSimulator(_profile(), out_size=64)
    x = _fundus(8, 96, seed=6)
    plain = F.interpolate(x, size=(64, 64), mode="bilinear", antialias=True, align_corners=False)
    assert torch.allclose(sim.augment(x, 0.0, torch.Generator().manual_seed(0), 64), plain)
    y = sim.augment(x, 1.0, torch.Generator().manual_seed(0), 64)
    assert y.shape == (8, 3, 64, 64) and all(not torch.allclose(y[i], plain[i]) for i in range(8))
    y = sim.augment(x, 0.5, torch.Generator().manual_seed(0), 64)
    pick = torch.rand(8, generator=torch.Generator().manual_seed(0)) < 0.5
    assert [not torch.allclose(y[i], plain[i]) for i in range(8)] == pick.tolist()      # the picked slots


def test_augment_views_puts_each_render_in_its_own_slot():
    """Training at p 0.5: exactly the drawn photos are re-rendered, each from its own
    whole-FOV view in its own slot (a mix-up would train a render under another photo's
    label); the rest keep their crop."""
    sim = ps.PhoneSimulator(_profile(), out_size=64)
    full = torch.cat([_fundus(1, 96, seed=40 + i) for i in range(8)])               # 8 different photos
    crop = F.interpolate(full, size=(64, 64), mode="bilinear", antialias=True, align_corners=False)
    y = sim.augment_views(crop, full, 0.5, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(0)
    pick = torch.rand(8, generator=g) < 0.5
    idx = pick.nonzero()[:, 0]
    assert 0 < len(idx) < 8 and not torch.equal(idx, torch.arange(len(idx)))       # not a prefix
    ref = sim.render(full[idx], sim.sample(len(idx), g), out_size=64)               # the same draws, one render
    assert torch.equal(y[idx], ref) and torch.equal(y[~pick], crop[~pick])


def _blob_aspect(y, xy, radius):
    """Width / height of the brightest blob (above half its peak over the local
    level) within ``radius`` of ``xy``, from its second moments."""
    B, _, H, W = y.shape
    xx, yy = ps.frame_coords(B, H, W, "cpu")
    lum = y.mean(dim=1)
    near = (xx - xy[:, 0].view(B, 1, 1)) ** 2 + (yy - xy[:, 1].view(B, 1, 1)) ** 2 <= radius.view(B, 1, 1) ** 2
    lo = torch.stack([lum[b][near[b]].median() for b in range(B)]).view(B, 1, 1)
    hi = torch.stack([lum[b][near[b]].max() for b in range(B)]).view(B, 1, 1)
    w = ((lum > lo + 0.5 * (hi - lo)) & near).float()
    n = w.sum(dim=(1, 2))
    mx, my = (w * xx).sum(dim=(1, 2)) / n, (w * yy).sum(dim=(1, 2)) / n
    vx = (w * (xx - mx.view(B, 1, 1)) ** 2).sum(dim=(1, 2)) / n
    vy = (w * (yy - my.view(B, 1, 1)) ** 2).sum(dim=(1, 2)) / n
    return torch.sqrt(vx / vy)


def test_training_views_render_the_whole_fov_with_round_retinas():
    """The trainer's views: a Canon-like frame's training crop (random resized crop,
    width capped at 1.1x the height) cuts the FOV on all four sides and its ellipse
    cannot be read -- un-squashed with the whole frame's aspect, simulated retinas came
    out 14 % wide. TrainViews hands the simulator the whole FOV instead, where it
    reads the squash; the photos it leaves alone keep the training crop. At 192 px,
    where the disc has enough pixels for its shape to be measured (+-2 %)."""
    frame = _fundus(1, 360, phone=True, seed=2)[..., 36:324, :]          # FOV cut top and bottom, W/H 1.25
    img = Image.fromarray((frame[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    tv = ps.TrainViews(192, 288)
    torch.manual_seed(0)
    views = [tv(img) for _ in range(4)]
    crop, full = torch.stack([v["crop"] for v in views]), torch.stack([v["full"] for v in views])
    assert crop.shape == (4, 3, 192, 192) and full.shape == (4, 3, 288, 288)
    sim = ps.PhoneSimulator(_profile_at(192), stages=("field",), out_size=192)
    y = sim.augment_views(crop, full, 1.0, torch.Generator().manual_seed(0))
    assert y.shape == crop.shape and sim._last["reliable"].all()
    assert torch.allclose(sim._last["b_f"] / sim._last["a_f"], torch.full((4,), 1.25), atol=0.05)
    asp = _blob_aspect(y, sim._last["disc_xy"], 2.5 * sim._last["disc_r"])
    assert ((asp > 0.93) & (asp < 1.07)).all(), asp                        # the round disc stays round
    assert torch.equal(sim.augment_views(crop, full, 0.0, torch.Generator().manual_seed(0)), crop)


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
    """Every stage but the field and the grain: illumination, colour, detail and rim
    may not draw a lesion-scale spot the tabletop photo did not have (a strong unsharp
    mask fails this). The bloom needs the field stage: see the tests below."""
    prof = _profile(size=96)
    x = _fundus(6, 96, seed=8)
    stages = tuple(s for s in ps.STAGES if s not in ("field", "noise"))
    for seed in (0, 1, 2):
        y = ps.PhoneSimulator(prof, stages=stages, out_size=96)(x, torch.Generator().manual_seed(seed))
        assert (_spots(y) <= _spots(x) + 1).all()


def _with_dots(y, geo, depth=0.03, rad=1.5, n=5):
    """``n`` microaneurysm-sized dark dots per image (green ``depth`` below the
    background, red and blue less), on a ring half the aperture out, at least 3 disc
    radii from the disc and apart from each other."""
    B, _, H, W = y.shape
    xx, yy = ps.frame_coords(B, H, W, "cpu")
    out = y.clone()
    for b in range(B):
        ang = torch.linspace(0, 2 * math.pi, 24 + 1)[:-1]
        px, py = 0.5 * torch.cos(ang), 0.5 * torch.sin(ang)
        far = torch.sqrt((px - geo["disc_xy"][b, 0]) ** 2 + (py - geo["disc_xy"][b, 1]) ** 2) > 3 * geo["disc_r"][b]
        for k in torch.nonzero(far)[:, 0][::3][:n].tolist():
            d2 = ((xx[b] - px[k]) ** 2 + (yy[b] - py[k]) ** 2) * (W / 2) ** 2
            out[b] = out[b] - depth * torch.exp(-d2 / (2 * rad ** 2)) * torch.tensor([0.8, 1.0, 0.3]).view(3, 1, 1)
    return out.clamp(0, 1) * (y.amax(dim=1, keepdim=True) > 0)


def test_no_lesion_like_spots_at_the_models_size():
    """The whole chain but the grain, field and bloom included, at the model's size from
    a supersampled view: no lesion-scale spot appears more than 2 disc radii from the
    disc, and 3 %-deep microaneurysm-sized dots added to its output are found (the
    detector can see what it guards against; at 96 px it cannot). Nearer the disc
    _spots is blind: the bloom's reach is pinned down by the next test."""
    prof = _bloomed(_profile_at(384))
    x = _fundus(4, 576, seed=8)
    kw = dict(frac=0.15, floor=0.01)
    base = _spots(F.interpolate(x, size=(384, 384), mode="bilinear", antialias=True, align_corners=False), **kw)
    sim = ps.PhoneSimulator(prof, stages=tuple(s for s in ps.STAGES if s != "noise"), out_size=384)
    for seed in (0, 1):
        y = sim(x, torch.Generator().manual_seed(seed))
        s = _spots(y, **kw)
        assert (s <= base + 1).all(), (s, base)
        found = _spots(_with_dots(y, sim._last), **kw) - s
        assert found.sum() >= 0.6 * 5 * len(y), found


def _bloom_pair(prof, x, size, seed=7):
    """The same fundi rendered through field, light and colour with and without the
    bloom, one geometry, one set of draws: (bloom on, bloom off, geometry, draws)."""
    on = _FixedGeometry(prof, stages=("field", "light", "colour", "bloom"), out_size=size)
    off = _FixedGeometry(prof, stages=("field", "light", "colour"), out_size=size)
    prm = on.sample(x.shape[0], torch.Generator().manual_seed(seed))
    return on.render(x, prm), off.render(x, prm), on, off, prm


def test_the_bloom_changes_nothing_beyond_the_disc():
    """The disc bloom's support ends at 1.1 detected disc radii: bloom on vs off, same
    draws and geometry, the green channel is identical beyond it. Reaching 1.3 radii
    (and 1.15 or 1.2) it was not."""
    prof = _bloomed(_profile())
    for fseed in (21, 22, 23):
        x = _fundus(4, 96, seed=fseed)
        y_on, y_off, on, _, _ = _bloom_pair(prof, x, 64)
        g = on.geo
        assert (g["disc_conf"] > 0).all() and (y_on - y_off).abs().max() > 0.1           # the bloom acts
        xx, yy = ps.frame_coords(4, 64, 64, "cpu")
        dist = torch.sqrt((xx - g["disc_xy"][:, 0].view(4, 1, 1)) ** 2 + (yy - g["disc_xy"][:, 1].view(4, 1, 1)) ** 2)
        beyond = (dist > 1.1 * g["disc_r"].view(4, 1, 1) * (1 + 1e-3)).unsqueeze(1)
        assert (y_on - y_off)[:, 1:2][beyond].abs().max() == 0


def test_the_bloom_leaves_lesions_beside_the_disc_alone():
    """A dot haemorrhage or an exudate whose core lies wholly outside the bloom's 1.1
    radii (centred at 1.25) keeps its contrast, bloom on vs off: at 192 px, where the
    disc's radius is ~9 px and a lesion core 1.2 px. With a 1.3-radii region
    and a weight steepest just above the FOV mean, such lesions came out up to 2x more
    conspicuous."""
    prof = _bloomed(_profile_at(192))
    for fseed in (21, 22):
        x = _fundus(4, 288, seed=fseed)
        yc_on, yc_off, on, off, prm = _bloom_pair(prof, x, 192)
        g = on.geo
        assert (g["disc_conf"] > 0).all()
        ang = torch.linspace(0, 2 * math.pi, 7)[:-1] + 0.3
        px = g["disc_xy"][:, :1] + 1.25 * g["disc_r"].view(-1, 1) * torch.cos(ang)   # [4,6], output frame
        py = g["disc_xy"][:, 1:] + 1.25 * g["disc_r"].view(-1, 1) * torch.sin(ang)
        sx, sy = ps._to_source(px, py, g["zoom"], g["a_f"], g["b_f"], g["offset"], g["rot"])
        xx, yy = ps.frame_coords(4, 288, 288, "cpu")
        xl = x.clone()
        for j in range(6):
            rgb = torch.tensor((-0.12, -0.12, -0.03) if j % 2 == 0 else (0.15, 0.13, -0.02)).view(1, 3, 1, 1)
            d = torch.sqrt(((xx - sx[:, j].view(4, 1, 1)) * 144) ** 2 + ((yy - sy[:, j].view(4, 1, 1)) * 144) ** 2)
            xl = xl + rgb * torch.exp(-(d / 1.6) ** 2).unsqueeze(1)
        xl = xl.clamp(0, 1) * ps.fov_mask(x)
        yl_on, yl_off = on.render(xl, prm), off.render(xl, prm)
        xo, yo = ps.frame_coords(4, 192, 192, "cpu")
        n = 0
        for j in range(6):
            core = (torch.sqrt((xo - px[:, j].view(4, 1, 1)) ** 2 + (yo - py[:, j].view(4, 1, 1)) ** 2) * 96
                    <= 1.2).unsqueeze(1)
            weber = lambda yl, yc: (((yl - yc)[:, 1:2] * core).sum(dim=(1, 2, 3))
                                    / (yc[:, 1:2] * core).sum(dim=(1, 2, 3)).clamp_min(1e-4))
            w_on, w_off = weber(yl_on, yc_on), weber(yl_off, yc_off)
            ok = (w_off.abs() > 0.02) & (torch.sqrt(px[:, j] ** 2 + py[:, j] ** 2) < 0.9)
            r = w_on[ok] / w_off[ok]
            n += int(ok.sum())
            assert ((r > 0.9) & (r < 1.1)).all(), r
        assert n >= 18                                                        # most lesions measured


def test_the_aperture_shows_no_black_from_outside_the_tabletop_fov():
    """At the model's size the aperture never reaches past the truncated tabletop FOV:
    tested at 0.97 of its radius, the coverage check left black chords in the circle
    (a simulator-only cue, kept through the whole chain)."""
    sim = ps.PhoneSimulator(_profile_at(384), stages=("field",), out_size=384)
    xx, yy = ps.frame_coords(4, 384, 384, "cpu")
    inner = (xx ** 2 + yy ** 2 < 0.98 ** 2).unsqueeze(1)
    for fseed, dseed in ((8, 0), (9, 1)):
        y = sim(_fundus(4, 576, seed=fseed), torch.Generator().manual_seed(dseed))
        assert int(((y.amax(dim=1, keepdim=True) < 1e-3) & inner).sum()) == 0


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
    r = got[ok] / want[ok]
    assert ok.any() and (r.median() - 1).abs() < 0.1 and (r - 1).abs().max() < 0.2       # x0.8 or x1.25 fail


def test_planted_lesions_keep_their_sign_and_contrast():
    """Lesions planted on a TABLETOP photo (where the chain does act: it changes the
    fixture's tabletop view ~25 %, its phone view ~3 %) keep their sign and their
    contrast relative to the local level, within 0.7-1.3x, in R, G and luminance (0.65
    when every row asks for the most blur, the 0.8 px cap: a soft phone photo softens
    microaneurysms too). A 3x3 colour map (mk) or a per-channel one fails this; so
    does an unsharp mask, or a blur past the cap."""
    lesions = [(0.0, 0.35, 1.6, (-0.10, -0.12, -0.03)),                    # dot haemorrhage / MA
               (-0.3, -0.35, 2.2, (-0.15, -0.16, -0.04)),
               (0.25, -0.2, 1.8, (0.18, 0.15, -0.02))]                     # hard exudate (yellow)
    prof = _profile(size=96)
    soft = copy.deepcopy(prof)
    soft.bank["detail"] = soft.bank["detail"] * 0.01
    stages = tuple(s for s in ps.STAGES if s != "field")                    # positions stay put
    xx, yy = ps.frame_coords(1, 96, 96, "cpu")

    def rel(a, b, c):                     # mean relative contrast over the lesion core; c 3 = luminance
        if c == 3:
            a, b, c = a.mean(dim=1, keepdim=True), b.mean(dim=1, keepdim=True), 0
        return ((a - b)[:, c][:, core] / b[:, c][:, core].clamp_min(1e-3)).mean(dim=1)

    for p, lo in ((prof, 0.7), (soft, 0.65)):
        sim = ps.PhoneSimulator(p, stages=stages, out_size=96)
        for fseed, dseed in ((11, 3), (12, 4)):
            clean = _fundus(4, 96, seed=fseed)
            with_l = _fundus(4, 96, seed=fseed, lesions=lesions)
            yc = sim.render(clean, sim.sample(4, torch.Generator().manual_seed(dseed)))
            yl = sim.render(with_l, sim.sample(4, torch.Generator().manual_seed(dseed)))
            for (lx, ly, rad, rgb) in lesions:
                core = torch.sqrt(((xx - lx) * 48) ** 2 + ((yy - ly) * 48) ** 2)[0] <= rad
                for c in (0, 1, 3):                                         # B: the tabletop's is nearly flat
                    before, after = rel(with_l, clean, c), rel(yl, yc, c)
                    if c < 3:
                        assert (torch.sign(after) == torch.sign(before)).all()
                    r = after / before
                    assert ((r > lo) & (r < 1.3)).all(), (c, r)


def test_blotch_reads_zero_on_an_evenly_lit_field():
    """The residual blotchiness of a smoothly lit circle (every phone photo and every
    render is one) is 0: smoothing it over cells outside the FOV read 0.10-0.13."""
    xx, yy = ps.frame_coords(1, 96, 96, "cpu")
    r2 = xx ** 2 + yy ** 2
    x = torch.stack([0.4 * torch.ones_like(r2), 0.4 * (1 - 0.3 * r2), 0.1 * torch.ones_like(r2)], 1) \
        * (r2 <= 0.96).unsqueeze(1)
    assert ps.illum_coeffs(x, with_residual=True)[1].item() < 0.01


def test_cross_fitted_report_uses_every_real_photo():
    """Real handheld photos outside the bank go to alternate folds (they were dropped),
    and a fold without one is left out instead of crashing on an empty batch."""
    prof = _profile()
    src_hi, real = _fundus(4, 96, seed=30), _fundus(6, 64, phone=True, seed=31)
    for rows in (np.array([0, 2, 4, 6, -1, -1]), np.full(6, -1)):
        df, y, diag, counts = ps.cross_fitted_report(prof, src_hi, real, rows, 0, 64, log=lambda s: None)
        assert df is not None and sum(nr for _, _, nr in counts) == 6 and y.shape == (4, 3, 64, 64)
        assert {"zoom", "clear", "reliable", "disc_margin"} <= set(diag)
    df, *_ = ps.cross_fitted_report(prof, src_hi, real[:0], np.zeros(0, dtype=int), 0, 64, log=lambda s: None)
    assert df is None


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                    reason="no CUDA or MPS device")
def test_fit_and_report_run_on_a_gpu_device():
    """fit's measurements and report on a GPU device, as on the box (every other test
    runs on the CPU, which missed a device mix-up that stopped the fit there)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    prof, _ = ps.fit_profile([_fundus(4, 192, phone=True, seed=1).to(dev)], [_fundus(4, 192, seed=2).to(dev)], 192,
                             log=lambda s: None)
    df, y, _, _ = ps.cross_fitted_report(prof, _fundus(4, 288, seed=3).to(dev),
                                         _fundus(4, 192, phone=True, seed=4).to(dev), np.arange(4), 0, 192,
                                         log=lambda s: None)
    assert df is not None and y.device.type == dev.type
    assert set(ps.BANK_KEYS) <= set(ps.describe(_fundus(2, 64, phone=True).to(dev)))  # 64 px: MPS pools on the CPU


# --------------------------------------------------------------------------- #
# CLI and the trainers
# --------------------------------------------------------------------------- #
def _save_photos(folder, x):
    folder.mkdir(parents=True)
    for k, im in enumerate(x):
        arr = (im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(folder / f"p{k:02d}.jpg", quality=95)


def test_fit_and_render_cli(tmp_path):
    from style_transfer_xfer import _check_phone_profile
    _save_photos(tmp_path / "phone", _fundus(10, 120, phone=True, seed=1))
    tab = _fundus(10, 120, phone=True, seed=2)[..., 12:108, :]              # truncated circle, 120 x 96
    _save_photos(tmp_path / "tab", tab)
    out = tmp_path / "ck" / "profile.json"
    fit = ["fit", "--target-dir", str(tmp_path / "phone"), "--source-dir", str(tmp_path / "tab"),
           "--image-size", "64", "--validate", "6", "--num-workers", "0", "--device", "cpu",
           "--batch-size", "5", "--out", str(out)]
    with pytest.raises(SystemExit):                     # a folder has no grades (nor a camera column)
        ps.main(fit + ["--select", "icdr0"])
    assert ps.main(fit) == 0
    prof = ps.PhoneProfile.load(str(out))
    assert len(prof) == 10 and prof.image_size == 64 and 0.7 < prof.params["source_aspect"] < 0.9
    # a folder is no split: the trainer must not take it for its own train split
    assert prof.source["split"] is None and prof.source["split_seed"] is None and len(prof.source["files"]) == 10
    txt = (tmp_path / "ck" / "fit_report.txt").read_text()
    assert "photos per fold" in txt and "FIELD on the 6 validation renders: zoom median" in txt
    assert (tmp_path / "ck" / "fit_grid.jpg").exists()
    rep = pd.read_csv(tmp_path / "ck" / "fit_report.csv")
    assert {"statistic", "auc_tabletop", "auc_simulated"} <= set(rep.columns)
    with pytest.raises(SystemExit, match="measured at 64 px"):            # check would find out last
        _check_phone_profile(str(out), 96)
    r = tmp_path / "render"
    assert ps.main(["render", "--profile", str(out), "--source-dir", str(tmp_path / "tab"), "--target-dir",
                    str(tmp_path / "phone"), "--n", "4", "--stage-rows", "2", "--num-workers", "0",
                    "--device", "cpu", "--out", str(r)]) == 0
    assert (r / "grid.jpg").exists() and (r / "stages.jpg").exists()


def test_trainer_check_and_score_with_phone_sim(tmp_path, monkeypatch):
    from style_transfer_xfer import load_device
    from style_transfer_xfer import main as xfer
    from test_retinareach import BACKBONE, _make_brset, _make_mbrset
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 30, rng, camera_of=lambda pid: "Canon CR" if pid % 4 else "NIKON NF5050")    # as on the box
    _make_mbrset(M, 30, rng)
    prof_path = tmp_path / "ck_phone_sim" / "profile.json"
    assert ps.main(["fit", "--target-root", str(M), "--source-root", str(B), "--source-camera", "Canon CR",
                    "--image-size", "64", "--validate", "0", "--num-workers", "0", "--device", "cpu",
                    "--out", str(prof_path)]) == 0

    def run(out, ck, extra=()):
        return train(["--source-root", str(B), "--target-root", str(M), "--source-camera", "Canon CR",
                      "--probe-targets", "hypertension",
                      "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1",
                      "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--calib-batch", "4",
                      "--calib-sizes", "4", "--calib-repeats", "1", "--envelope-repeats", "2",
                      "--prior-strengths", "0", "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1",
                      "--min-test-pos-patients", "2", "--demo-n", "4", "--n-shuffle", "0", "--ablation-frac", "0",
                      "--out", str(out), "--ckpt", str(ck), *extra])

    # what the training loop hands the simulator, and what the device-side augmentation
    # (the step before the model) then receives
    from dataset import DeviceAug
    calls, returned, augmented = [], [], []
    orig, orig_aug = ps.PhoneSimulator.augment_views, DeviceAug.__call__

    def views(self, crop, full, p, g):
        out = orig(self, crop, full, p, g)
        calls.append((tuple(crop.shape[-2:]), tuple(full.shape[-2:])))
        returned.append((out, not torch.equal(out, crop)))
        return out

    monkeypatch.setattr(ps.PhoneSimulator, "augment_views", views)
    monkeypatch.setattr(DeviceAug, "__call__", lambda self, x: (augmented.append(x), orig_aug(self, x))[1])
    base, arm = tmp_path / "base.pt", tmp_path / "arm.pt"
    assert run(tmp_path / "out_base", base) == 0
    assert not calls
    augmented.clear()
    assert run(tmp_path / "out_arm", arm, ["--phone-sim", str(prof_path), "--phone-sim-p", "0.5"]) == 0
    assert calls and all(c == ((64, 64), (96, 96)) for c in calls)        # model-size crop, supersampled FOV
    assert any(changed for _, changed in returned)                         # renders happened ...
    assert all(any(a is out for a in augmented) for out, _ in returned)    # ... and every batch went on
    res = json.loads((tmp_path / "out_arm" / "results.json").read_text())
    assert res["args"]["phone_sim"] == str(prof_path) and res["args"]["gpu_aug"] is True
    assert res["phone_sim_profile"]["train_size"] == 96 and len(res["phone_sim_profile"]["sha256"]) == 64

    # a profile from another split, holding a handheld val / test photo, or naming a photo
    # this run's handheld train split does not have, is refused
    blob = json.loads(prof_path.read_text())
    test_file = pd.read_csv(tmp_path / "out_arm" / "split_handheld.csv").query("split == 'test'")["file"].iloc[0]
    for change, why in (({"split_seed": 7}, "not fitted on this run's handheld train split"),
                        ({"files": blob["source"]["files"] + [test_file]}, "validation / test split"),
                        ({"files": blob["source"]["files"] + ["copy__" + test_file]}, "not in this run's")):
        bad = tmp_path / "bad_profile.json"
        bad.write_text(json.dumps({**blob, "source": {**blob["source"], **change}}))
        with pytest.raises(SystemExit, match=why):
            run(tmp_path / "out_bad", tmp_path / "bad.pt", ["--phone-sim", str(bad)])

    out = tmp_path / "check"
    assert xfer(["check", "--ckpt", str(base), "--methods", "colour", "--phone-profile", str(prof_path),
                 "--out", str(out), "--bank", "8", "--calib-n", "8", "--calib-batch", "4", "--grid-n", "4",
                 "--num-workers", "0", "--batch-size", "8", "--device", "cpu"]) == 0
    df = pd.read_csv(out / "check.csv").set_index("method")
    assert "phone_sim" in df.index and np.isfinite(df.loc["phone_sim", "tab_auroc"])
    assert 0 <= df.loc["phone_sim", "vessel_r"] <= 1 and (out / "grid.jpg").exists()

    sc = tmp_path / "score"
    score = ["score", "--source-root", str(B), "--target-root", str(M), "--num-workers", "0", "--device", "cpu",
             "--batch-size", "8", "--ckpt", f"canon_only={base}", f"canon_phonesim={arm}"]
    assert xfer(score + ["--out", str(sc)]) == 0
    s = pd.read_csv(sc / "score.csv")
    assert set(s.arm) == {"canon_only", "canon_phonesim"} and s.oracle_spec.between(0, 1).all()
    n_canon_test = len(load_device(str(B), "brset", 64, 42, camera="Canon CR")["test"])        # Canon rows only
    assert (s[s.camera == "tabletop"].n == n_canon_test).all()
    assert n_canon_test != len(load_device(str(B), "brset", 64, 42)["test"])
    summary = (sc / "score_summary.txt").read_text()
    assert "PAIRED: canon_phonesim minus canon_only" in summary and "phone_sim p 0.5, strength 1.0" in summary
    assert (s[s.arm == "canon_phonesim"].phone_sim_sha256 == res["phone_sim_profile"]["sha256"]).all()
    assert xfer(score + ["--part", "val", "--out", str(tmp_path / "score_val")]) == 0      # ablations choose here
    assert "val split" in (tmp_path / "score_val" / "score_summary.txt").read_text()
    v = pd.read_csv(tmp_path / "score_val" / "score.csv")
    tab = v[v.camera == "tabletop"]                           # the split that chose the epoch and threshold
    assert tab.in_sample.all() and tab.sensitivity.isna().all() and not v[v.camera == "handheld"].in_sample.any()
