#!/usr/bin/env python3
"""
style_transfer_xfer.py
======================
Can tabletop (BRSET) images be rendered as handheld (mBRSET) ones without the
translation carrying mBRSET's all-diabetic population along with its camera?
Three ways to try, one set of checks, one downstream test::

    # 1. the two learned translators (GPU, ~1-2 h each)
    python style_transfer_xfer.py fit --mode matched --source-root $B --target-root $M --out ck_style/matched.pt
    python style_transfer_xfer.py fit --mode naive   --source-root $B --target-root $M --out ck_style/naive.pt
    # 2. realism + label preservation of every method, on an existing checkpoint
    python style_transfer_xfer.py check --ckpt ck_retinareach/seed0.pt \\
        --translator cyc_matched=ck_style/matched.pt cyc_naive=ck_style/naive.pt --out exp_style/check_seed0
    # 3. a translated copy of BRSET to train on, then score the retrained models
    python style_transfer_xfer.py export --method fda_sharp --source-root $B --target-root $M --out-root data_style/fda_sharp
    python style_transfer_xfer.py score --ckpt fda_sharp=ck_style/fda_sharp_seed0.pt ... --out exp_style/score

``run_style_transfer.sh`` runs all of it.

Methods
-------
identity      the tabletop image untouched (the reference every number is read against)
colour        per-channel mean / SD matched to a random unlabelled handheld image
fda           Fourier low-frequency amplitude from that image (3x3 block by default)
fda_sharp     + blur until the Laplacian variance matches that reference
fda_geo       + the handheld field of view (centre zoom, full circular aperture)
phone_sim     the measured handheld imaging chain (``phone_sim.py``, ``--phone-profile``),
              rendered from a supersampled view as in training
cyc_naive     CycleGAN, all tabletop train images vs all handheld train images
cyc_matched   CycleGAN, pairs drawn at the SAME ICDR grade, tabletop side
              restricted to diabetic patients, both sides at the handheld set's
              grade mix: within a batch the two sets differ in camera only

Checks (``check``; tabletop TEST patients, translators never saw them)
----------------------------------------------------------------------
Realism (is it handheld-like?) -- measured among ICDR-0 eyes only, so a method
cannot score as "handheld" by adding disease:
  cam_dist     ``retinareach.camera_distance`` between the BN statistics of the
               translated tabletop images and of real handheld images; the
               floor is handheld vs handheld (two halves), the start is identity.
  p_handheld   a camera probe (logistic regression on the trunk embedding, real
               tabletop vs real handheld, grade-matched training set) -- its mean
               output on the translated images; 0 = reads as tabletop, 1 = handheld.
  sharp / illum / glare / red   image statistics next to the real handheld values.
Label preservation (did the translation edit the disease?):
  judge_*      a "handheld-native" referable-DR probe, trained on real handheld
               images, reading the translated tabletop images against their TRUE
               tabletop labels: AUROC, and the fraction of ICDR-0 eyes it flags
               next to its false-positive rate on real handheld ICDR-0 eyes.
               Flags above that rate = lesions the translation invented; the
               non-diabetic ICDR-0 row is the cell mBRSET does not have.
  tab_*        the shipped tabletop model on the translated images: AUROC,
               sensitivity at the shipped threshold, and the mean logit shift
               per true grade (ICDR-0 up = toward disease).
  vessel_r     correlation of the vessel line-detector response before and after.

``score`` (after retraining on a translated copy): the retrained model on the
REAL handheld and REAL tabletop test splits, at its own shipped threshold, and the
test-oracle specificity at ``--target-sens`` (``train_phone_ceiling.py``'s reading:
where the ROC curve itself crosses the target sensitivity).

A checkpoint trained on one tabletop camera (``--source-camera``) is checked and
scored on that camera's own split: the table is filtered before it is split, as
the trainer did, so its tabletop test rows hold neither the other camera nor any
of its own training patients.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brset_dataset import load_any  # noqa: E402
from capability_gate import multitask_split  # noqa: E402
from dataset import _binary_flag  # noqa: E402
from fundus_utils import seed_worker  # noqa: E402
from metadata_model import _auc, load_table, threshold_at_sensitivity  # noqa: E402
from retinareach import (build_from_checkpoint, calibrate_blended, camera_distance,  # noqa: E402
                         get_bn_stats, set_bn_stats, stats_from_lists)
from style_transfer import (CycleStylizer, FDAStylizer, ReferenceBank, Stylizer, denorm,  # noqa: E402
                            generator_from_state, renorm, sharpness, train_translator)
from train_mbrset import pick_device  # noqa: E402
from train_retinareach import MultiTargetDataset  # noqa: E402
from vessel_ablation import fov_mask, vessel_response, window_for  # noqa: E402

PRIMARY = "dr_referable"
STYLE_ONLY = ("colour", "fda", "fda_sharp", "fda_geo")
SIMULATED = ("phone_sim",)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_device(root: str, dataset: str, size: int, split_seed: int,
                targets: Sequence[str] = (PRIMARY,), camera: Optional[str] = None,
                strat: str = PRIMARY) -> Dict[str, object]:
    """The trainer's own patient split (``multitask_split`` on its first source
    target ``strat``, after the run's ``--source-camera`` filter when it had one),
    so 'train' here is 'train' there and the test patients are untouched."""
    df, csv = load_table(root, dataset)
    if camera:
        from brset_dataset import filter_camera
        df = filter_camera(df, camera.strip())
    split = multitask_split(df, strat, seed=split_seed)
    img = load_any(root, dataset)["images_dir"]
    out: Dict[str, object] = {"csv": csv, "images_dir": img}
    for part in ("train", "val", "test"):
        out[part] = MultiTargetDataset(split[part], img, targets, "val", size)
    return out


def grades(ds: MultiTargetDataset) -> np.ndarray:
    return pd.to_numeric(ds.df["final_icdr"], errors="coerce").to_numpy(dtype=float)


def diabetic(ds: MultiTargetDataset) -> Optional[np.ndarray]:
    """[N] 1/0/NaN from the table's ``diabetes`` column, or None when it has none
    (mBRSET: every patient is diabetic, there is no column)."""
    if "diabetes" not in ds.df.columns:
        return None
    d = np.array([_binary_flag(v) for v in ds.df["diabetes"]], dtype=float)
    return d if np.isfinite(d).any() else None


def loader(ds, batch: int, workers: int, sampler=None) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=False, sampler=sampler, num_workers=workers,
                      worker_init_fn=seed_worker, drop_last=sampler is not None)


def reference_bank(ds: MultiTargetDataset, n: int, seed: int, workers: int) -> ReferenceBank:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    imgs = [denorm(b["image"]) for b in loader(Subset(ds, idx.tolist()), 32, workers)]
    return ReferenceBank(torch.cat(imgs))


# --------------------------------------------------------------------------- #
# fit: the two translators differ only in which images are paired
# --------------------------------------------------------------------------- #
def pairing(mode: str, g_src: np.ndarray, g_tgt: np.ndarray, dm_src: Optional[np.ndarray],
            n: int, seed: int, min_per_stratum: int = 8) -> Dict[str, object]:
    """Index sequences (length ``n``) into the two train sets.

    naive    both uniform over every image.
    matched  draw a grade from the HANDHELD set's grade distribution, then one
             image of that grade from each side; the tabletop side only from
             diabetic patients when the table says who they are. Grades with
             fewer than ``min_per_stratum`` images on either side are left out
             of both, so no grade is shown on one side only.
    """
    rng = np.random.default_rng(seed)
    if mode == "naive":
        return {"src": rng.integers(0, len(g_src), n).tolist(),
                "tgt": rng.integers(0, len(g_tgt), n).tolist(),
                "strata": {}, "diabetic_only": False}
    if mode != "matched":
        raise ValueError(f"unknown pairing mode {mode!r}")
    use_dm = dm_src is not None and int(np.nansum(dm_src == 1)) >= min_per_stratum
    ok_src = np.isfinite(g_src) & ((dm_src == 1) if use_dm else True)
    strata, levels = {}, []
    for g in sorted(set(g_tgt[np.isfinite(g_tgt)].astype(int))):
        s, t = np.flatnonzero(ok_src & (g_src == g)), np.flatnonzero(g_tgt == g)
        strata[int(g)] = {"source": int(len(s)), "target": int(len(t))}
        if len(s) >= min_per_stratum and len(t) >= min_per_stratum:
            levels.append((g, s, t))
    if not levels:
        raise ValueError(f"matched pairing: no ICDR grade has {min_per_stratum}+ images on both "
                         f"sides: {strata}")
    p = np.array([len(t) for _, _, t in levels], dtype=float)
    pick = rng.choice(len(levels), size=n, p=p / p.sum())
    return {"src": [int(rng.choice(levels[k][1])) for k in pick],
            "tgt": [int(rng.choice(levels[k][2])) for k in pick],
            "strata": strata, "used_grades": [int(g) for g, _, _ in levels],
            "diabetic_only": bool(use_dm)}


def cmd_fit(a) -> int:
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    src = load_device(a.source_root, a.source_dataset, a.image_size, a.split_seed)["train"]
    tgt = load_device(a.target_root, a.target_dataset, a.image_size, a.split_seed)["train"]
    n = a.iters * a.batch_size
    pr = pairing(a.mode, grades(src), grades(tgt), diabetic(src), n, a.seed, a.min_per_stratum)
    print(f"[fit:{a.mode}] tabletop train {len(src)}, handheld train {len(tgt)}, {a.iters} iterations "
          f"x batch {a.batch_size} at {a.crop}px crops of {a.image_size}px on {dev}")
    if a.mode == "matched":
        print(f"  strata (images per ICDR grade): {pr['strata']}; used {pr['used_grades']}; "
              f"tabletop side diabetic only: {pr['diabetic_only']}")
        if not pr["diabetic_only"]:
            print("  WARNING: no usable 'diabetes' column on the tabletop side -- matched on grade only")
    it_s = iter(loader(src, a.batch_size, a.num_workers, sampler=pr["src"]))
    it_t = iter(loader(tgt, a.batch_size, a.num_workers, sampler=pr["tgt"]))
    gen = torch.Generator().manual_seed(a.seed)

    def crop(x01: torch.Tensor) -> torch.Tensor:
        h, w = x01.shape[-2:]
        c = min(a.crop, h, w)
        i = int(torch.randint(0, h - c + 1, (1,), generator=gen))
        j = int(torch.randint(0, w - c + 1, (1,), generator=gen))
        x01 = x01[..., i:i + c, j:j + c]
        return x01.flip(-1) if torch.rand(1, generator=gen).item() < 0.5 else x01

    def sample_pair():
        return (crop(denorm(next(it_s)["image"].to(dev))), crop(denorm(next(it_t)["image"].to(dev))))

    state = train_translator(sample_pair, a.iters, dev, ngf=a.ngf, n_blocks=a.n_blocks, seed=a.seed,
                             lambda_idt=a.lambda_idt, log_every=a.log_every)
    state.update({"mode": a.mode, "pairing": {k: v for k, v in pr.items() if k not in ("src", "tgt")},
                  "args": vars(a)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(state, a.out)
    print(f"[fit:{a.mode}] wrote {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# Stylizers by name
# --------------------------------------------------------------------------- #
def parse_named(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected name=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def build_stylizer(name: str, bank: Optional[ReferenceBank], translators: Dict[str, str],
                   dev: torch.device, beta: float, zoom: float, phone_profile: Optional[str] = None,
                   phone_strength: float = 1.0, out_size: Optional[int] = None) -> Stylizer:
    if name == "identity":
        return Stylizer()
    if name in STYLE_ONLY:
        if bank is None:
            raise ValueError(f"{name} needs handheld reference images (--target-root)")
        return FDAStylizer(bank, beta, sharp=name in ("fda_sharp", "fda_geo"),
                           zoom=zoom if name == "fda_geo" else 0.0, colour_only=name == "colour")
    if name in SIMULATED:
        if not phone_profile:
            raise ValueError(f"{name} needs --phone-profile (phone_sim.py fit)")
        from phone_sim import PhoneProfile, PhoneSimulator
        return PhoneSimulator(PhoneProfile.load(phone_profile), strength=phone_strength, out_size=out_size)
    if name in translators:
        state = torch.load(translators[name], map_location="cpu", weights_only=False)
        return CycleStylizer(generator_from_state(state).to(dev), name)
    raise ValueError(f"unknown method {name!r}; style-only: {STYLE_ONLY}; simulated: {SIMULATED}; "
                     f"translators given: {sorted(translators)}")


def supersample_of(phone_profile: Optional[str]) -> float:
    """How much larger a view ``phone_sim`` renders from (the profile's; 1 without one)."""
    if not phone_profile:
        return 1.0
    from phone_sim import PhoneProfile
    return float(PhoneProfile.load(phone_profile).params.get("supersample", 1.5))


def _check_phone_profile(phone_profile: str, size: int) -> None:
    """The profile loads and was measured at the model's size, or a clean exit."""
    from phone_sim import PhoneProfile, PhoneSimulator
    try:
        PhoneSimulator(PhoneProfile.load(phone_profile)).check_size(size)
    except (OSError, ValueError) as e:
        raise SystemExit(f"[fatal] --phone-profile: {e}")


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def image_stats(x01: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-image statistics inside the field of view: Laplacian variance, spread
    of the low-frequency luminance (std/mean), specular fraction (bright and
    desaturated, ``artifacts.detect_glare``'s rule), red share of the colour."""
    m = fov_mask(x01).float()
    n = m.sum(dim=(1, 2, 3)).clamp(min=1)
    lum = x01.mean(dim=1, keepdim=True)
    k = max(3, x01.shape[-1] // 8) | 1
    low = F.avg_pool2d(F.pad(lum, [k // 2] * 4, mode="reflect"), k, stride=1)
    mu = (low * m).sum(dim=(1, 2, 3)) / n
    sd = ((((low - mu.view(-1, 1, 1, 1)) ** 2) * m).sum(dim=(1, 2, 3)) / n).sqrt()
    mx, mn = x01.max(dim=1, keepdim=True).values, x01.min(dim=1, keepdim=True).values
    glare = ((mx > 0.85) & ((mx - mn) / mx.clamp_min(1e-6) < 0.25)).float()
    rgb = (x01 * m).sum(dim=(2, 3)) / n.view(-1, 1)
    return {"sharp": sharpness(x01), "illum": sd / mu.clamp_min(1e-6),
            "glare": (glare * m).sum(dim=(1, 2, 3)) / n,
            "red": rgb[:, 0] / rgb.sum(dim=1).clamp_min(1e-6)}


def vessel_corr(a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
    """[B] Pearson r of the vessel line response inside the shared field of view."""
    W = window_for(a01.shape[-1])
    ra, fa = vessel_response(a01.float(), W)
    rb, fb = vessel_response(b01.float(), W)
    m = (fa & fb).float()
    n = m.sum(dim=(1, 2, 3)).clamp(min=2)
    ma, mb = (ra * m).sum(dim=(1, 2, 3)) / n, (rb * m).sum(dim=(1, 2, 3)) / n
    da, db = (ra - ma.view(-1, 1, 1, 1)) * m, (rb - mb.view(-1, 1, 1, 1)) * m
    return (da * db).sum(dim=(1, 2, 3)) / ((da ** 2).sum(dim=(1, 2, 3)).sqrt()
                                           * (db ** 2).sum(dim=(1, 2, 3)).sqrt()).clamp_min(1e-9)


class _Styled:
    """A loader whose batches come out translated and on the device (what
    ``calibrate_blended`` and the passes below iterate)."""

    def __init__(self, dl: DataLoader, sty: Stylizer, dev: torch.device, seed: int) -> None:
        self.dl, self.sty, self.dev, self.seed = dl, sty, dev, seed

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)        # same references on every pass
        for b in self.dl:
            x01 = denorm(b["image"].to(self.dev))
            yield {"image": renorm(self.sty(x01, gen)), "orig01": x01}


@torch.no_grad()
def passes(models: Dict[str, torch.nn.Module], dl: DataLoader, sty: Stylizer, dev, seed: int,
           k: int) -> Dict[str, np.ndarray]:
    """One pass: every model's referable-DR logit and embedding, the image
    statistics and the vessel agreement with the (geometry-matched) original."""
    acc: Dict[str, List[torch.Tensor]] = {}
    put = lambda key, v: acc.setdefault(key, []).append(v.float().cpu())
    for b in _Styled(dl, sty, dev, seed):
        x, x01 = b["image"], denorm(b["image"])
        for name, m in models.items():
            put(f"logit_{name}", m(x)[:, k])
            put(f"emb_{name}", m.embed(x))
        for key, v in image_stats(x01).items():
            put(key, v)
        put("vessel_r", vessel_corr(sty.geometry(b["orig01"]), x01))
    return {key: torch.cat(v).numpy() for key, v in acc.items()}


def _fit_lr(X: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                              max_iter=5000, random_state=seed)).fit(X, y)


def grade_matched(g_a: np.ndarray, g_b: np.ndarray, cap: int, rng) -> tuple:
    """Indices into two sets with the same number of images at every ICDR grade."""
    ia, ib = [], []
    for g in sorted(set(g_a[np.isfinite(g_a)].astype(int)) & set(g_b[np.isfinite(g_b)].astype(int))):
        a, b = np.flatnonzero(g_a == g), np.flatnonzero(g_b == g)
        n = min(len(a), len(b), cap)
        ia += rng.choice(a, n, replace=False).tolist()
        ib += rng.choice(b, n, replace=False).tolist()
    return ia, ib


def _stats_of(model, ds, idx, sty, dev, a, seed) -> dict:
    dl = DataLoader(Subset(ds, list(idx)), batch_size=a.calib_batch, shuffle=False,
                    num_workers=a.num_workers, worker_init_fn=seed_worker)
    cal, _ = calibrate_blended(model, _Styled(dl, sty, dev, seed), dev)
    return get_bn_stats(cal)


def save_grid(path: str, rows: Dict[str, torch.Tensor]) -> None:
    from PIL import Image
    tiles = [torch.cat(list(v.cpu()), dim=2) for v in rows.values()]        # each [3,H,n*W]
    arr = (torch.cat(tiles, dim=1).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path, quality=92)


def cmd_check(a) -> int:
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    S, T = args["source_name"], args["target_name"]
    size, seed = int(ck["config"].get("image_size", 384)), int(args.get("seed", 0))
    if a.phone_profile:                           # a size mismatch fails now, not after every other method
        _check_phone_profile(a.phone_profile, size)
    rng = np.random.default_rng(seed)
    net = build_from_checkpoint(ck)
    k = net.targets.index(PRIMARY)
    thr = float(ck["thresholds"]["calibrated"][PRIMARY])
    models = {}
    for name, dname in (("tab", S), ("hh", T)):
        m = build_from_checkpoint(ck)
        set_bn_stats(m, stats_from_lists(ck["profiles"][dname]["bn_stats"]))
        models[name] = m.to(dev).eval()
    camera = args.get("source_camera")
    strat = (args.get("source_targets") or [PRIMARY])[0]          # the run's split was stratified on it
    src = load_device(a.source_root or args["source_root"], args["source_dataset"], size,
                      int(args["split_seed"]), camera=camera, strat=strat)
    tgt = load_device(a.target_root or args["target_root"], args["target_dataset"], size,
                      int(args["split_seed"]), strat=strat)
    translators = parse_named(a.translator)
    methods = ["identity"] + [m for m in a.methods if m != "identity"] + sorted(translators)
    if a.phone_profile and "phone_sim" not in methods:
        methods.append("phone_sim")
    # phone_sim renders from a supersampled view of the same test rows (as in training)
    src_hi = (load_device(a.source_root or args["source_root"], args["source_dataset"],
                          int(round(size * supersample_of(a.phone_profile))), int(args["split_seed"]),
                          camera=camera, strat=strat) if any(m in SIMULATED for m in methods) else src)
    bank = reference_bank(tgt["train"], a.bank, seed, a.num_workers)
    print(f"[check] {a.ckpt} (seed {seed}, {size}px) on {dev}; methods {methods}; "
          f"{len(bank)} handheld references; shipped threshold {thr:.3f}")

    cap = lambda ds: Subset(ds, rng.choice(len(ds), min(a.max_n, len(ds)), replace=False).tolist()) \
        if a.max_n and a.max_n < len(ds) else ds

    def keep(ds, sub):                                         # row arrays aligned to a (sub)set
        idx = np.array(sub.indices) if isinstance(sub, Subset) else np.arange(len(ds))
        y = ds.label_array()[idx, 0]
        dm = diabetic(ds)
        return idx, y, grades(ds)[idx], (dm[idx] if dm is not None else None)

    ident = Stylizer()
    run = lambda ds_sub, sty: passes(models, loader(ds_sub, a.batch_size, a.num_workers), sty, dev, seed, k)

    # real handheld: the judge, the camera probe's positive class, the reference values
    t_tr, t_va, t_te = cap(tgt["train"]), cap(tgt["val"]), cap(tgt["test"])
    _, y_ttr, g_ttr, _ = keep(tgt["train"], t_tr)
    _, y_tva, _, _ = keep(tgt["val"], t_va)
    i_tte, y_tte, g_tte, _ = keep(tgt["test"], t_te)
    R_ttr, R_tva, R_tte = run(t_tr, ident), run(t_va, ident), run(t_te, ident)
    ok = np.isfinite(y_ttr)
    judge = _fit_lr(R_ttr["emb_hh"][ok], y_ttr[ok].astype(int), seed)
    okv = np.isfinite(y_tva)
    judge_thr = threshold_at_sensitivity(y_tva[okv].astype(int),
                                         judge.decision_function(R_tva["emb_hh"][okv]), a.target_sens)
    jt = judge.decision_function(R_tte["emb_hh"])
    judge_ref = {"auroc": _auc(y_tte[np.isfinite(y_tte)].astype(int), jt[np.isfinite(y_tte)]),
                 "flag_icdr0": float((jt[g_tte == 0] >= judge_thr).mean()) if (g_tte == 0).any() else np.nan}

    # real tabletop train: the camera probe's negative class (grade-matched to handheld train)
    s_tr = cap(src["train"])
    _, _, g_str, _ = keep(src["train"], s_tr)
    R_str = run(s_tr, ident)
    ia, ib = grade_matched(g_str, g_ttr, a.max_n or 10 ** 9, rng)
    cam = _fit_lr(np.concatenate([R_str["emb_tab"][ia], R_ttr["emb_tab"][ib]]),
                  np.concatenate([np.zeros(len(ia)), np.ones(len(ib))]).astype(int), seed)
    p_cam = lambda E: cam.predict_proba(E)[:, 1]

    s_te = cap(src["test"])
    i_ste, y_ste, g_ste, dm_ste = keep(src["test"], s_te)
    z0s, z0t = np.flatnonzero(g_ste == 0), np.flatnonzero(g_tte == 0)
    n_cal = min(a.calib_n, len(z0s), len(z0t) // 2)
    z0s_cal = rng.choice(z0s, n_cal, replace=False) if n_cal >= 2 else z0s[:0]
    perm = rng.permutation(z0t)
    hh_a = _stats_of(models["tab"], tgt["test"], i_tte[perm[:n_cal]], ident, dev, a, seed) if n_cal >= 2 else None
    hh_b = _stats_of(models["tab"], tgt["test"], i_tte[perm[n_cal:2 * n_cal]], ident, dev, a, seed) if n_cal >= 2 else None

    def summarise(R, y, g, dm, extra=None) -> Dict[str, float]:
        fin = np.isfinite(y)
        yy = y[fin].astype(int)
        two = len(np.unique(yy)) == 2
        zt, jz = R["logit_tab"], judge.decision_function(R["emb_hh"])
        zero = g == 0
        row = {"n": int(len(y)), "prevalence": float(np.nanmean(y)),
               "tab_auroc": _auc(yy, zt[fin]) if two else np.nan,
               "tab_sens": float((zt[fin][yy == 1] >= np.log(thr / (1 - thr))).mean()) if yy.any() else np.nan,
               "hh_auroc": _auc(yy, R["logit_hh"][fin]) if two else np.nan,
               "judge_auroc": _auc(yy, jz[fin]) if two else np.nan,
               "judge_flag_icdr0": float((jz[zero] >= judge_thr).mean()) if zero.any() else np.nan,
               "p_handheld_icdr0": float(p_cam(R["emb_tab"][zero]).mean()) if zero.any() else np.nan,
               "vessel_r": float(np.mean(R["vessel_r"]))}
        if dm is not None:
            for tag, sel in (("nondm", zero & (dm == 0)), ("dm", zero & (dm == 1))):
                row[f"judge_flag_icdr0_{tag}"] = float((jz[sel] >= judge_thr).mean()) if sel.any() else np.nan
                row[f"n_icdr0_{tag}"] = int(sel.sum())
        for key in ("sharp", "illum", "glare", "red"):
            row[key] = float(np.median(R[key][zero])) if zero.any() else float(np.median(R[key]))
        row.update(extra or {})
        return row

    rows, base = [], None
    rows.append({"method": "REAL handheld", **summarise(R_tte, y_tte, g_tte, None, {
        "cam_dist": camera_distance(hh_a, hh_b) if hh_a else np.nan})})
    grid: Dict[str, torch.Tensor] = {}
    show = np.concatenate([np.flatnonzero(g_ste == 0)[:a.grid_n // 2],
                           np.flatnonzero(y_ste == 1)[:a.grid_n - a.grid_n // 2]])
    for name in methods:
        sty = build_stylizer(name, bank, translators, dev, a.beta, a.zoom, a.phone_profile, a.phone_strength,
                             out_size=size)
        ds_te = (src_hi if name in SIMULATED else src)["test"]
        R = run(Subset(ds_te, s_te.indices) if isinstance(s_te, Subset) else ds_te, sty)
        extra = {}
        if n_cal >= 2:
            st = _stats_of(models["tab"], ds_te, i_ste[z0s_cal], sty, dev, a, seed)
            extra["cam_dist"] = camera_distance(st, hh_a)
        if base is None:
            base = R
        d = R["logit_tab"] - base["logit_tab"]
        extra["dlogit_icdr0"] = float(d[g_ste == 0].mean()) if (g_ste == 0).any() else np.nan
        extra["dlogit_referable"] = float(d[y_ste == 1].mean()) if (y_ste == 1).any() else np.nan
        rows.append({"method": name, **summarise(R, y_ste, g_ste, dm_ste, extra)})
        print(f"  {name}: done")
        if len(show):
            show_x = torch.stack([ds_te[int(i_ste[j])]["image"] for j in show])
            grid[name] = sty(denorm(show_x.to(dev)), torch.Generator().manual_seed(seed))
    if len(show):
        tt = np.concatenate([np.flatnonzero(g_tte == 0)[:a.grid_n // 2],
                             np.flatnonzero(y_tte == 1)[:a.grid_n - a.grid_n // 2]])
        if len(tt) == len(show):
            grid["REAL handheld"] = denorm(torch.stack([tgt["test"][int(i_tte[j])]["image"] for j in tt]))
        save_grid(os.path.join(a.out, "grid.jpg"), grid)

    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "ckpt", os.path.basename(a.ckpt))
    df.to_csv(os.path.join(a.out, "check.csv"), index=False)
    txt = summarise_check(df, judge_ref, judge_thr, thr)
    print(txt)
    with open(os.path.join(a.out, "check_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nwrote {a.out}/check.csv, check_summary.txt, grid.jpg "
          f"(rows of grid.jpg, top to bottom: {list(grid)})")
    return 0


def summarise_check(df: pd.DataFrame, judge_ref: Dict[str, float], judge_thr: float, thr: float) -> str:
    f = lambda c: [c_ for c_ in c if c_ in df.columns]
    real = df[df.method == "REAL handheld"].iloc[0]
    out = ["", "REALISM among ICDR-0 eyes (handheld-like without adding disease)",
           f"  cam_dist floor = REAL handheld vs itself; p_handheld: 0 tabletop, 1 handheld",
           df[f(["method", "cam_dist", "p_handheld_icdr0", "sharp", "illum", "glare", "red"])]
           .to_string(index=False, float_format=lambda v: f"{v:.4f}"),
           "", "LABEL PRESERVATION on tabletop test images, against their true labels",
           f"  judge = referable-DR probe trained on real handheld images; on real handheld test: "
           f"AUROC {judge_ref['auroc']:.3f}, flags {judge_ref['flag_icdr0']:.3f} of ICDR-0 eyes",
           f"  shipped tabletop threshold {thr:.3f}; dlogit = mean shift vs identity (ICDR-0 up = toward disease)",
           df[df.method != "REAL handheld"][f(["method", "judge_auroc", "judge_flag_icdr0",
                                               "judge_flag_icdr0_nondm", "judge_flag_icdr0_dm",
                                               "tab_auroc", "tab_sens", "hh_auroc", "dlogit_icdr0",
                                               "dlogit_referable", "vessel_r"])]
           .to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# export: a translated copy of the tabletop set the unmodified trainer can read
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_export(a) -> int:
    from PIL import Image
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    if a.method in SIMULATED:
        _check_phone_profile(a.phone_profile, a.image_size)
    df, csv = load_table(a.source_root, a.source_dataset)
    img_dir = load_any(a.source_root, a.source_dataset)["images_dir"]
    size_in = (int(round(a.image_size * supersample_of(a.phone_profile))) if a.method in SIMULATED
               else a.image_size)
    ds = MultiTargetDataset(df, img_dir, [PRIMARY], "val", size_in)
    translators = parse_named(a.translator)
    bank = None
    if a.method in STYLE_ONLY:
        tgt = load_device(a.target_root, a.target_dataset, a.image_size, a.split_seed)
        bank = reference_bank(tgt["train"], a.bank, a.seed, a.num_workers)     # train patients only
    sty = build_stylizer(a.method, bank, translators, dev, a.beta, a.zoom, a.phone_profile, a.phone_strength,
                         out_size=a.image_size)
    out_img = os.path.join(a.out_root, os.path.basename(img_dir.rstrip("/")))
    os.makedirs(out_img, exist_ok=True)
    shutil.copy(csv, os.path.join(a.out_root, os.path.basename(csv)))
    gen = torch.Generator().manual_seed(a.seed)
    n = 0
    for b in loader(ds, a.batch_size, a.num_workers):
        y = (sty(denorm(b["image"].to(dev)), gen).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for arr, f in zip(y, b["file"]):
            dst = os.path.join(out_img, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.fromarray(arr).save(dst, quality=95, subsampling=0)
            n += 1
    with open(os.path.join(a.out_root, "style_transfer.json"), "w") as f:
        json.dump({"method": a.method, "n_images": n, "image_size": a.image_size, "source_csv": csv,
                   "translator": translators.get(a.method), "beta": a.beta, "zoom": a.zoom,
                   "phone_profile": a.phone_profile, "phone_strength": a.phone_strength,
                   "seed": a.seed}, f, indent=2)
    print(f"[export:{a.method}] {n} images -> {out_img} (labels: {os.path.basename(csv)})")
    return 0


# --------------------------------------------------------------------------- #
# score: retrained models on the REAL cameras
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_score(a) -> int:
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rows, cache = [], {}
    for arm, path in [it.split("=", 1) for it in a.ckpt]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        args = ck["args"]
        size, seed = int(ck["config"].get("image_size", 384)), int(args.get("seed", 0))
        net = build_from_checkpoint(ck).to(dev).eval()
        k = net.targets.index(PRIMARY)
        strat = (args.get("source_targets") or [PRIMARY])[0]
        for cam_name, root, dset in (("tabletop", a.source_root, args["source_dataset"]),
                                     ("handheld", a.target_root, args["target_dataset"])):
            camera = args.get("source_camera") if cam_name == "tabletop" else None
            key = (root, dset, size, int(args["split_seed"]), camera, strat, a.part)
            if key not in cache:
                cache[key] = load_device(root, dset, size, int(args["split_seed"]), camera=camera,
                                         strat=strat)[a.part]
            ds = cache[key]
            y = ds.label_array()[:, 0]
            profile = args["source_name"] if cam_name == "tabletop" else args["target_name"]
            for proto in ("trained", "calibrated"):
                m = net
                if proto == "calibrated":
                    m = build_from_checkpoint(ck).to(dev).eval()
                    set_bn_stats(m, stats_from_lists(ck["profiles"][profile]["bn_stats"]))
                p = torch.cat([torch.sigmoid(m(b["image"].to(dev))[:, k]).float().cpu()
                               for b in loader(ds, a.batch_size, a.num_workers)]).numpy()
                thr = ck["thresholds"][proto].get(PRIMARY)
                fin = np.isfinite(y)
                yy, pp = y[fin].astype(int), p[fin]
                orc = threshold_at_sensitivity(yy, pp, a.target_sens)
                # tabletop val chose the epoch and fixed the threshold: there sensitivity at
                # the threshold is >= the target and specificity == oracle_spec by construction
                in_sample = cam_name == "tabletop" and a.part == "val"
                at_thr = thr is not None and not in_sample
                sim = ck.get("phone_sim_profile") or {}
                rows.append({"arm": arm, "seed": seed, "camera": cam_name, "protocol": proto,
                             "ckpt": path, "n": int(fin.sum()), "in_sample": in_sample,
                             "phone_sim_p": args.get("phone_sim_p") if args.get("phone_sim") else None,
                             "phone_sim_strength": args.get("phone_sim_strength") if args.get("phone_sim") else None,
                             "phone_sim_sha256": sim.get("sha256"),
                             "auroc": _auc(yy, pp), "threshold": thr,
                             "sensitivity": float((pp[yy == 1] >= thr).mean()) if at_thr else np.nan,
                             "specificity": float((pp[yy == 0] < thr).mean()) if at_thr else np.nan,
                             "flagged": float((pp >= thr).mean()) if at_thr else np.nan,
                             "oracle_spec": float((pp[yy == 0] < orc).mean()) if orc == orc else np.nan,
                             "prevalence": float(yy.mean())})
        print(f"  scored {arm} seed {seed}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "score.csv"), index=False)
    g = (df.groupby(["camera", "protocol", "arm"])[["auroc", "sensitivity", "specificity", "flagged",
                                                     "oracle_spec"]]
         .agg(["mean", "min", "max"]).round(3))
    txt = ("\nRETRAINED ON A TRANSLATED / SIMULATED TABLETOP SET, SCORED ON THE REAL CAMERAS "
           f"(referable DR, {a.part} split, seeds {sorted(df.seed.unique().tolist())}; threshold fixed on "
           f"the arm's own tabletop validation split; oracle_spec = specificity where this split's ROC "
           f"crosses sensitivity {a.target_sens})\n"
           + ("IN-SAMPLE: the tabletop rows are the split that chose each arm's epoch and fixed its threshold, so "
              "their AUROC and oracle_spec are optimistic and their threshold columns are left empty; the "
              "handheld rows are out of sample.\n" if a.part == "val" else "")
           + "arms: " + "; ".join(f"{r.arm} = {os.path.dirname(r.ckpt) or '.'} ("
                                  + (f"phone_sim p {r.phone_sim_p}, strength {r.phone_sim_strength}, profile "
                                     f"{r.phone_sim_sha256[:12]}" if isinstance(r.phone_sim_sha256, str) else "no phone_sim")
                                  + ")" for r in df.drop_duplicates("arm").itertuples()) + "\n"
           + g.to_string())
    arms = list(dict.fromkeys(df.arm))
    if len(arms) == 2:                            # the second arm minus the first, seed by seed
        cols = ["auroc", "sensitivity", "specificity", "oracle_spec"]
        key = ["camera", "protocol", "seed"]
        d = (df[df.arm == arms[1]].set_index(key)[cols] - df[df.arm == arms[0]].set_index(key)[cols]).dropna(how="all")
        if len(d):
            pd_ = d.groupby(level=["camera", "protocol"]).agg(["mean", "min", "max", "count"]).round(3)
            txt += (f"\n\nPAIRED: {arms[1]} minus {arms[0]}, same seed and rows (count = seeds with both)\n"
                    + pd_.to_string())
    print(txt)
    with open(os.path.join(a.out, "score_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nwrote {a.out}/score.csv and score_summary.txt")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tabletop -> handheld image synthesis: three methods, one set of checks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, roots: bool = True):
        if roots:
            q.add_argument("--source-root", required=True, help="tabletop dataset (BRSET)")
            q.add_argument("--target-root", default=None, help="handheld dataset (mBRSET)")
            q.add_argument("--source-dataset", default="brset", choices=["brset", "mbrset"])
            q.add_argument("--target-dataset", default="mbrset", choices=["brset", "mbrset"])
            q.add_argument("--image-size", type=int, default=384)
            q.add_argument("--split-seed", type=int, default=42)
        q.add_argument("--batch-size", type=int, default=32)
        q.add_argument("--num-workers", type=int, default=4)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    def style(q):
        q.add_argument("--translator", nargs="*", default=[], metavar="NAME=CKPT",
                       help="fit checkpoints, e.g. cyc_matched=ck_style/matched.pt")
        q.add_argument("--bank", type=int, default=256, help="handheld reference images (train split)")
        q.add_argument("--beta", type=float, default=0.003,
                       help="FDA low-frequency window, fraction of the side (0.01 blotches at 384 px)")
        q.add_argument("--zoom", type=float, default=1.15, help="fda_geo centre zoom")
        q.add_argument("--phone-profile", default=None,
                       help="phone_sim.py profile; adds the phone_sim method to check")
        q.add_argument("--phone-strength", type=float, default=1.0, help="0-1")

    q = sub.add_parser("fit", help="train one CycleGAN translator")
    common(q)
    q.add_argument("--mode", required=True, choices=["matched", "naive"])
    q.add_argument("--out", required=True)
    q.add_argument("--iters", type=int, default=20000)
    q.add_argument("--crop", type=int, default=256)
    q.add_argument("--ngf", type=int, default=48)
    q.add_argument("--n-blocks", type=int, default=6)
    q.add_argument("--lambda-idt", type=float, default=5.0)
    q.add_argument("--min-per-stratum", type=int, default=8)
    q.add_argument("--log-every", type=int, default=200)
    q.set_defaults(fn=cmd_fit, batch_size=2)

    q = sub.add_parser("check", help="realism + label preservation of every method")
    common(q, roots=False)
    style(q)
    q.add_argument("--ckpt", required=True, help="a train_retinareach.py checkpoint (trained on the real tabletop set)")
    q.add_argument("--source-root", default=None, help="override the checkpoint's recorded roots")
    q.add_argument("--target-root", default=None)
    q.add_argument("--methods", nargs="+", default=list(STYLE_ONLY))
    q.add_argument("--out", default="exp_style/check")
    q.add_argument("--max-n", type=int, default=0, help="cap every split at this many images (0 = all)")
    q.add_argument("--calib-n", type=int, default=512, help="ICDR-0 images per camera-distance pool")
    q.add_argument("--calib-batch", type=int, default=16)
    q.add_argument("--target-sens", type=float, default=0.85, help="the judge's operating point")
    q.add_argument("--grid-n", type=int, default=8)
    q.set_defaults(fn=cmd_check)

    q = sub.add_parser("export", help="write a translated copy of the tabletop set")
    common(q)
    style(q)
    q.add_argument("--method", required=True)
    q.add_argument("--out-root", required=True)
    q.set_defaults(fn=cmd_export)

    q = sub.add_parser("score", help="retrained checkpoints on the real test splits")
    common(q)
    q.add_argument("--ckpt", nargs="+", required=True, metavar="ARM=CKPT")
    q.add_argument("--target-sens", type=float, default=0.85, help="the oracle operating point")
    q.add_argument("--part", default="test", choices=["val", "test"],
                   help="test: the pre-specified decision read; val: choosing among ablation arms "
                        "(the decision split is then read once, for the chosen arm)")
    q.add_argument("--out", default="exp_style/score")
    q.set_defaults(fn=cmd_score)

    a = p.parse_args(argv)
    if a.cmd in ("fit", "score") and not a.target_root:
        p.error(f"{a.cmd} needs --target-root")
    if a.cmd == "export" and a.method in STYLE_ONLY and not a.target_root:
        p.error(f"export --method {a.method} needs --target-root (handheld reference images)")
    if a.cmd in ("check", "export") and getattr(a, "method", None) in SIMULATED and not a.phone_profile:
        p.error(f"--method {a.method} needs --phone-profile")
    if a.cmd == "check" and any(m in SIMULATED for m in a.methods) and not a.phone_profile:
        p.error("--methods phone_sim needs --phone-profile")
    if a.cmd in ("check", "export") and not 0.0 <= a.phone_strength <= 1.0:
        p.error("--phone-strength must be in [0, 1]")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
