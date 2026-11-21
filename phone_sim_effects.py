#!/usr/bin/env python3
"""phone_sim_effects.py -- what each part of the phone simulator does to a trained model.

No training: finished ``train_retinareach.py`` checkpoints score one tabletop split as it
is, re-rendered by the whole simulator, and re-rendered by variants of it, next to the real
handheld split. Every variant draws the same bank rows and the same jitter for a photo, so
two rows of the table differ by the variant alone (under ``detail_by: rank`` read a re-read
row against ``full_cal``: *Detail ranks* below).

Variants (``--sets``):
  full        every stage, the phone field, strength 1 (what an arm trained with ``--profile``
              at strength 1 and the phone field sees; the log says, for every arm trained
              with a simulator, whether ``--profile`` is the file it was trained with)
  drop        the simulator without one stage (``drop_colour``, ...): a row that moves back
              toward ``tabletop`` names the stage responsible for the change
  only        one stage alone (``only_light``, ...; the bloom needs the field's disc, so it
              is ``field+bloom`` next to ``only_field``)
  strength    0, 0.25, 0.5, 0.75 of the photometric change (0 is the geometry alone)
  field       ``whole_field``: the whole tabletop FOV instead of the phone's narrower field
  render      the other generation of the render on the profile's own bank, one option at a
              time and all together: on a phone_sim/2 profile ``fill_0.18`` (the handheld's
              angular scale, the truncated frame's caps mirror-filled), ``prefilter_matched``
              and ``render_v3`` (both); on a /3 profile ``fill_0``, ``prefilter_gauss``,
              ``detail_by_band``, ``detail_by_edge`` and ``render_v2``. The summary names
              the parameters behind each of these rows. A /2 bank has no edge reading, so
              its render rows keep the band criterion, whose blur reacts to what reaches it
              (a render that arrives sharper is blurred more): there each row has a
              ``*_no_detail`` twin, and ``drop_detail`` is the profile's own render without
              that stage, so scale and resampling can be read apart from the blur

Detail ranks. Under ``detail_by: rank`` the detail stage ranks a render against ``fit``'s
reading of its own renders, so a variant that changes what reaches that stage (a stage
dropped, another prefilter or scale, a lower strength) would be ranked against the wrong
renders and blurred more or less than its criterion asks. Each such variant therefore
re-reads the ranks on ``--cal-n`` tabletop training photos first, as a fit with that setting
would have (``--cal-n 0``: every variant keeps the profile's reading). ``full`` keeps ``fit``'s
reading, taken on other photos; ``full_cal`` is the fitted chain ranked on the same ``--cal-n``
photos as the re-read rows, so a re-read row minus ``full_cal`` is the variant alone, and
``full`` minus ``full_cal`` is what the choice of photos does by itself. The trainers do not
re-read: ``train_retinareach.py --phone-sim-strength x`` ranks against the profile's own
reading, so ``strength_x`` is that arm's training render only with ``--cal-n 0``.

Arms. ``train_retinareach.py`` checkpoints (the Canon sweep, the simulator-trained sweep) are
read under their tabletop and their handheld statistics. ``train_phone_ceiling.py``
checkpoints -- trained on real handheld photos alone (``phone``) or on the tabletop train
split and handheld photos together (``mixed``) -- are read as trained, at the threshold their
own handheld validation split set. They are the simulator's referees: models that know what
a real handheld photo looks like, labels included. Phone-only training does worse on the
handheld test than the mixture (oracle specificity 0.868 against 0.889, five seeds), so the
tabletop photos carry something the handheld model can use; these rows ask whether a
simulated photo carries it too.

  * ``phone`` model: the ``tabletop`` row is another camera's photos. A faithful simulator
    brings them to the camera it knows: ``full`` at or above ``tabletop`` in AUROC, and its
    handheld threshold landing near its handheld sensitivity / specificity. ``full`` below
    ``tabletop`` says the simulator removes more than the camera difference costs.
  * ``mixed`` model: both cameras are home, so its ``tabletop`` -> ``full`` drop is not
    unfamiliarity. What it loses on ``full`` beyond its own tabletop-to-``REAL handheld``
    difference (other eyes, so a rough yardstick) is evidence the simulator destroys.
  * p_hh on the ``mixed`` model: its trunk has seen both cameras with labels, so the share
    of simulated ICDR-0 photos its camera probe calls handheld is the strictest realism
    reading here. On the ``phone`` model the probe sits on a trunk that never trained on a
    tabletop photo: a weaker reading.

Per arm, seed and variant, under each set of statistics the arm has
(referable DR, image level): AUROC; sensitivity, specificity and flagged share at
the shipped threshold; oracle specificity at ``--target-sens``; the mean logit shift against
the same photos unsimulated, among ICDR-0 and among referable photos; and how often the
trunk's own camera probe (ICDR-0 tabletop vs ICDR-0 handheld training photos) calls ICDR-0
photos handheld.

Reading it: on a Canon-only checkpoint, ``full`` should move the numbers the way ``REAL
handheld`` does; on a checkpoint trained with the simulator the same rows say how much of
that shift training absorbed. ``drop_field`` against ``full`` among referable photos is the
cost of the crop (peripheral lesions lost while the label stays).

  python phone_sim_effects.py --profile ck_phone_sim/profile.json --source-root <BRSET> \\
      --target-root <mBRSET> --ckpt canon_only=ck_rr_canon/seed0.pt canon_phonesim=ck_rr_phonesim/seed0.pt
  # the referees, and the new render against the old on the same bank (no training, no refit)
  python phone_sim_effects.py --profile ck_phone_sim/profile.json --source-root <BRSET> \\
      --target-root <mBRSET> --sets render field --out ck_phone_sim/effects_referees \\
      --ckpt canon_only=ck_rr_canon/seed0.pt phone=ck_phone_ceiling/seed0.pt mixed=ck_phone_mixed/seed0.pt

Writes ``effects.csv``, ``effects_summary.txt`` and ``grid.jpg`` (dataset photos: keep the
folder under ``ck_*``). The default split is validation, where simulator settings are chosen;
its tabletop rows chose each ``train_retinareach.py`` checkpoint's epoch and threshold (and
its handheld rows a ``train_phone_ceiling.py`` checkpoint's), so those rows are in-sample
there and the differences between rows are the reading. ``--part test`` reads the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadata_model import _auc, threshold_at_sensitivity  # noqa: E402
from phone_sim import (EDGE_SRC, LEGACY_PARAMS, RENDER_V3, STAGES, PhoneProfile, PhoneSimulator,  # noqa: E402
                       calibrate_detail)
from retinareach import build_from_checkpoint, set_bn_stats, stats_from_lists  # noqa: E402
from style_transfer import denorm, renorm  # noqa: E402
from style_transfer_xfer import (PRIMARY, _check_phone_profile, _fit_lr, grades, load_device,  # noqa: E402
                                 loader, save_grid)
from train_mbrset import pick_device  # noqa: E402

SETS = ("drop", "only", "strength", "field", "render")
STATS = ("tabletop", "handheld")                  # a train_retinareach.py checkpoint's two sets of BN statistics
TRAINED = "as trained"                            # a train_phone_ceiling.py checkpoint has the one
BASE, REAL = "tabletop", "REAL handheld"


def render_variants(profile: PhoneProfile, can_read: bool = False) -> Dict[str, Dict[str, object]]:
    """Every other value of each render option the profile's bank can serve (name ->
    parameter changes), then the other generation of the render as a whole: the field
    and the resampling (``fill``, ``prefilter``) with the detail criterion that goes
    with them. ``can_read``: the caller re-reads the detail ranks for its variants
    (:func:`variants` with ``cal``), so the rank criterion is on offer wherever the bank
    has an edge reading, not only where ``fit`` left its reading of the renders."""
    p = profile.params
    fill = RENDER_V3["fill"] if float(p["fill"]) == 0 else 0.0
    out: Dict[str, Dict[str, object]] = {f"fill_{fill:g}": {"fill": fill}}
    pre = "matched" if p["prefilter"] == "gauss" else "gauss"
    out[f"prefilter_{pre}"] = {"prefilter": pre}
    # the detail stage's criteria: edge needs the bank's edge reading (a /3 fit), rank also
    # a reading of the renders
    edge = "edge" in profile.bank
    by = ["band"] + (["edge"] if edge else []) + (["rank"] if edge and (can_read or p.get("edge_src") is not None)
                                                  else [])
    for d in by:
        if d != p["detail_by"]:
            out[f"detail_by_{d}"] = {"detail_by": d}
    if all(p[k] == v for k, v in RENDER_V3.items()):
        out["render_v2"] = dict(LEGACY_PARAMS)
    elif all(p[k] == LEGACY_PARAMS[k] for k in RENDER_V3):
        # the /3 render as far as the bank serves it: by rank, on a /2 bank (no edge
        # reading) still by band; a criterion in between is named in the row
        whole = {**RENDER_V3, **({"detail_by": by[-1]} if by[-1] != p["detail_by"] else {})}
        out["render_v3" if by[-1] == "rank" or not edge else f"render_v3_{by[-1]}"] = whole
    else:
        out["render_flipped"] = {"fill": fill, "prefilter": pre}      # neither generation: both options flipped
    return out


def chain_of(profile: PhoneProfile, stages: Sequence[str] = STAGES, field: str = "phone",
             strength: float = 1.0) -> tuple:
    """What a render's sharpness at the detail stage depends on: the field's scale and
    resampling, the stages before it (the rim stage among them: the light stage draws
    the photo's dark rim bands only with it), the framing and the strength."""
    return (float(profile.params["fill"]), str(profile.params["prefilter"]),
            tuple(s for s in STAGES if s in stages and s not in ("detail", "noise")), field, float(strength))


def variants(profile: PhoneProfile, size: int, sets: Sequence[str], cal: Optional[torch.Tensor] = None,
             seed: int = 0, log=print) -> Dict[str, PhoneSimulator]:
    """name -> simulator. ``cal`` (tabletop training photos at the supersampled size,
    in [0, 1]): a variant whose renders reach the detail stage differently from the
    fitted chain's has its detail ranks re-read on them (module docstring)."""
    fitted, read, reread, skipped = chain_of(profile), {}, [], []
    v: Dict[str, PhoneSimulator] = {}

    def add(name: str, change: Optional[Dict[str, object]] = None, **kw) -> None:
        prof = profile.with_params(**change) if change else profile
        stages, field, strength = kw.get("stages", STAGES), kw.get("field", "phone"), kw.get("strength", 1.0)
        key = EDGE_SRC[field]
        if prof.params["detail_by"] == "rank" and "detail" in stages:
            chain = chain_of(prof, stages, field, strength)
            own = chain[:3] == fitted[:3] and strength == 1.0 and prof.params.get(key) is not None
            if not own and cal is not None:
                if chain not in read:
                    read[chain] = calibrate_detail(prof, cal, size, seed, log=lambda t: None, stages=stages,
                                                   field=field, strength=strength)
                if read[chain] is not None:
                    prof = prof.with_params(**{key: read[chain]})
                    reread.append(name)
            if prof.params.get(key) is None and prof.params.get("edge_src") is None:
                skipped.append(name)                # nothing to rank against
                return
        v[name] = PhoneSimulator(prof, out_size=size, **kw)

    add("full")
    if "drop" in sets:
        for s in STAGES:
            add(f"drop_{s}", stages=[t for t in STAGES if t != s])
    if "only" in sets:
        add("only_field", stages=["field"])
        for s in STAGES[1:]:
            if s == "bloom":                       # keyed on the disc the field stage finds
                add("field+bloom", stages=["field", "bloom"])
            else:
                add(f"only_{s}", stages=[s])
    if "strength" in sets:
        for s in (0.0, 0.25, 0.5, 0.75):
            add(f"strength_{s:g}", strength=s)
    if "field" in sets:
        add("whole_field", field="whole")
    if "render" in sets:
        # a bank without an edge reading (a /2 fit) keeps the band criterion in every render
        # row, and that stage's blur reacts to what reaches it: a render that arrives sharper
        # or less magnified is blurred more. Each such row gets a twin without the detail
        # stage, next to ``drop_detail`` for the profile's own render
        twins = "edge" not in profile.bank
        for name, change in render_variants(profile, can_read=cal is not None).items():
            add(name, change)
            if twins:
                add(f"{name}_no_detail", change, stages=[t for t in STAGES if t != "detail"])
        if twins and "drop_detail" not in v:
            add("drop_detail", stages=[t for t in STAGES if t != "detail"])
    if reread and profile.params["detail_by"] == "rank":
        # the fitted chain ranked on the same photos as the re-read rows: their reference
        # (``full`` keeps fit's reading, which is what a training arm sees)
        r = calibrate_detail(profile, cal, size, seed, log=lambda t: None)
        if r is not None:
            v["full_cal"] = PhoneSimulator(profile.with_params(edge_src=r), out_size=size)
            reread.append("full_cal")
    if reread:
        log(f"[effects] detail ranks re-read on {len(cal)} tabletop training photos for: {', '.join(reread)}")
    if skipped:
        log(f"[effects] left out (no reading of the renders to rank against): {', '.join(skipped)}")
    return v


def is_referee(ck: dict) -> bool:
    """A ``train_phone_ceiling.py`` checkpoint (phone-only or mixed): trained with
    handheld labels, no per-camera statistics, its threshold from handheld validation."""
    return "calibrated" not in ck.get("thresholds", {})


def trained_with(ck: dict, profile_sha: str) -> Optional[str]:
    """For a checkpoint trained with a simulator: whether ``--profile`` is the file it was
    trained with, and which row (if any) is its training render. None for the others."""
    sim = ck.get("phone_sim_profile")
    if not sim:
        return None
    g = ck["args"]
    if is_referee(ck):                              # train_phone_ceiling.py --mix-phone-sim: strength 1, the phone field
        p, strength, field = sim.get("p", g.get("mix_sim_p")), 1.0, "phone"
    else:
        p, strength, field = (g.get("phone_sim_p"), float(g.get("phone_sim_strength") or 1.0),
                              g.get("phone_sim_field") or "phone")
    how = f"p {p}, strength {strength:g}, field {field}, profile {sim.get('path')} ({str(sim.get('sha256'))[:12]})"
    if sim.get("sha256") != profile_sha:
        return f"trained with phone_sim {how}: NOT --profile, so no row here is the render it was trained on"
    row = (("full" if strength == 1.0 else f"strength_{strength:g} under --cal-n 0") if field == "phone" else
           ("whole_field" if strength == 1.0 else None))
    return (f"trained with phone_sim {how}: --profile is that file; its training render is "
            + (f"the {row} row" if row else "not among the rows"))


def tabletop_camera(ck: dict) -> Optional[str]:
    """The tabletop camera a checkpoint's own tabletop train split was filtered to (None:
    every camera, or no tabletop photos at all)."""
    g = ck["args"]
    return ((g.get("mix_camera") if is_referee(ck) else g.get("source_camera")) or "").strip() or None


def data_key(ck: dict, a) -> dict:
    """Where a checkpoint's photos come from and how they were split (a referee trained
    on handheld photos alone names no tabletop set: ``None`` until :func:`main` fills
    it from ``--source-root`` / ``--source-camera`` or from another arm)."""
    g, size = ck["args"], int(ck["config"].get("image_size", 384))
    root = lambda r: os.path.normpath(str(r)) if r else None      # one spelling per folder: arms are grouped by it
    if is_referee(ck):
        return {"s_root": root(a.source_root or g.get("mix_root")), "t_root": root(a.target_root or g["root"]),
                "s_ds": g.get("mix_dataset") or "brset", "t_ds": g["dataset"], "split_seed": int(g["split_seed"]),
                "camera": a.source_camera or g.get("mix_camera"), "strat": g["targets"][0], "size": size}
    return {"s_root": root(a.source_root or g["source_root"]), "t_root": root(a.target_root or g["target_root"]),
            "s_ds": g["source_dataset"], "t_ds": g["target_dataset"], "split_seed": int(g["split_seed"]),
            "camera": a.source_camera or g.get("source_camera"), "strat": (g.get("source_targets") or [PRIMARY])[0],
            "size": size}


class Arm:
    """One checkpoint. A ``train_retinareach.py`` one: its trunk under the tabletop and
    the handheld statistics, at its shipped (calibrated) threshold. A referee
    (:func:`is_referee`): as trained, at its own handheld-validation threshold."""

    def __init__(self, arm: str, path: str, ck: dict, dev: torch.device) -> None:
        a = ck["args"]
        self.arm, self.path, self.seed = arm, path, int(a.get("seed", 0))
        self.referee = is_referee(ck)
        self.args = a
        self.models = {}
        if self.referee:
            self.thr = float(ck["thresholds"]["trained"][PRIMARY])
            self.models[TRAINED] = build_from_checkpoint(ck).to(dev).eval()
        else:
            self.thr = float(ck["thresholds"]["calibrated"][PRIMARY])
            for stats, dname in zip(STATS, (a["source_name"], a["target_name"])):
                m = build_from_checkpoint(ck)
                set_bn_stats(m, stats_from_lists(ck["profiles"][dname]["bn_stats"]))
                self.models[stats] = m.to(dev).eval()
        self.stats = tuple(self.models)
        self.probe = self.stats[0]                 # the camera probe's embedding: tabletop statistics / as trained
        self.k = self.models[self.probe].targets.index(PRIMARY)
        self.cam = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, out: Dict[str, List[np.ndarray]]) -> None:
        for stats, m in self.models.items():
            e = m.embed(x)
            out.setdefault(f"z_{stats}", []).append(m.head(m.drop(e))[:, self.k].float().cpu().numpy())
            if stats == self.probe:
                if self.cam is None:
                    out.setdefault("emb", []).append(e.float().cpu().numpy())
                else:
                    out.setdefault("p_cam", []).append(self.cam.predict_proba(e.float().cpu().numpy())[:, 1])


def run(arms: Sequence[Arm], ds, a, dev) -> List[Dict[str, np.ndarray]]:
    """Every arm on a dataset as it is -> one dict of row arrays per arm."""
    acc: List[Dict[str, List[np.ndarray]]] = [{} for _ in arms]
    for b in loader(ds, a.batch_size, a.num_workers):
        x = b["image"].to(dev)
        for arm, out in zip(arms, acc):
            arm.forward(x, out)
    return [{k: np.concatenate(v) for k, v in out.items()} for out in acc]


def metrics(R: Dict[str, np.ndarray], y: np.ndarray, g: np.ndarray, thr: float, target_sens: float,
            base: Optional[Dict[str, np.ndarray]] = None, stats_names: Sequence[str] = STATS) -> List[dict]:
    fin = np.isfinite(y)
    yy = y[fin].astype(int)
    two = len(np.unique(yy)) == 2
    rows = []
    for stats in stats_names:
        z = R[f"z_{stats}"]
        p = 1.0 / (1.0 + np.exp(-z[fin]))
        orc = threshold_at_sensitivity(yy, p, target_sens) if two else np.nan
        row = {"stats": stats, "n": int(fin.sum()), "prevalence": float(yy.mean()) if len(yy) else np.nan,
               "auroc": _auc(yy, p) if two else np.nan,
               "sensitivity": float((p[yy == 1] >= thr).mean()) if (yy == 1).any() else np.nan,
               "specificity": float((p[yy == 0] < thr).mean()) if (yy == 0).any() else np.nan,
               "flagged": float((p >= thr).mean()) if len(p) else np.nan,
               "oracle_spec": float((p[yy == 0] < orc).mean()) if orc == orc else np.nan,
               "dlogit_icdr0": np.nan, "dlogit_referable": np.nan, "p_handheld_icdr0": np.nan}
        if base is not None:
            d = z - base[f"z_{stats}"]
            row["dlogit_icdr0"] = float(d[g == 0].mean()) if (g == 0).any() else np.nan
            row["dlogit_referable"] = float(d[y == 1].mean()) if (y == 1).any() else np.nan
        if stats == stats_names[0] and "p_cam" in R and (g == 0).any():
            row["p_handheld_icdr0"] = float(R["p_cam"][g == 0].mean())
        rows.append(row)
    return rows


def _take(ds, idx: Optional[np.ndarray]):
    return ds if idx is None else Subset(ds, idx.tolist())


def _icdr0(ds, n: int, rng) -> Subset:
    z = np.flatnonzero(grades(ds) == 0)
    return Subset(ds, rng.choice(z, min(n, len(z)), replace=False).tolist())


def check_split(ds, split_file: Optional[str], part: str, what: str, own: bool = True) -> None:
    """A referee was trained on the split in ``split_file`` (a sweep's own); the photos
    scored here come from the recomputed split. If the file is at hand the two must
    agree on ``part``: a photo scored here that the file puts in another split could be
    one the referee was trained on. ``own``: the photos scored are of the set the file
    splits (the same camera filter), so every one of them must be in it; otherwise (the
    arm is scored on another camera's photos) a photo the file does not list was never
    one of its training photos."""
    if not split_file:
        return
    if not os.path.isfile(split_file):
        print(f"[effects] {what}: its split file {split_file} is not here; assuming the recomputed split is the "
              f"one it was trained on (its run printed whether they agree)")
        return
    sp = pd.read_csv(split_file)
    files = sp["file"].astype(str)
    known, theirs = set(files), set(files[sp["split"] == part])
    scored = ds.df["file"].astype(str).tolist()
    stray = [f for f in scored if f in known and f not in theirs]
    if stray:
        raise SystemExit(f"[fatal] {what}: {len(stray)} of the {part} photos scored here are in another split of "
                         f"{split_file} (e.g. {stray[0]}): the label table or the split changed since that run, "
                         f"and its training photos may be among them")
    absent = [f for f in scored if f not in known]
    if absent and own:
        raise SystemExit(f"[fatal] {what}: {len(absent)} of the {part} photos scored here are not in {split_file} "
                         f"at all (e.g. {absent[0]}): the label table changed since that run, so the split "
                         f"recomputed here is not the one it was trained on")
    if absent:
        print(f"[effects] {what}: {len(absent)} of the {len(scored)} {part} photos scored are not in {split_file} "
              f"(another camera's photos: none was a training photo); the rest are in its {part} split")


def group_effects(arms: Sequence[Arm], key: dict, a, dev) -> List[dict]:
    """The arms of one data key (roots, split, camera, size) through every variant."""
    size, camera, strat, split_seed = key["size"], key["camera"], key["strat"], key["split_seed"]
    _check_phone_profile(a.profile, size)
    profile = PhoneProfile.load(a.profile)
    kw = dict(camera=camera, strat=strat)
    s_root, t_root = key["s_root"], key["t_root"]
    src = load_device(s_root, key["s_ds"], size, split_seed, **kw)
    hi = load_device(s_root, key["s_ds"], int(round(size * float(profile.params.get("supersample", 1.5)))),
                     split_seed, **kw)
    tgt = load_device(t_root, key["t_ds"], size, split_seed, strat=strat)
    for arm in arms:
        if arm.referee:
            check_split(tgt[a.part], arm.args.get("split_file"), a.part, f"{arm.arm} seed {arm.seed} (handheld)")
            check_split(src[a.part], arm.args.get("mix_split_file"), a.part, f"{arm.arm} seed {arm.seed} (tabletop)",
                        own=((arm.args.get("mix_camera") or "").strip() or None) == camera)
    rng = np.random.default_rng(a.seed)

    # the camera probe: ICDR-0 training photos of the two cameras, on each arm's own embedding
    E_s, E_t = run(arms, _icdr0(src["train"], a.probe_n, rng), a, dev), run(arms, _icdr0(tgt["train"], a.probe_n, rng), a, dev)
    for arm, es, et in zip(arms, E_s, E_t):
        if len(es["emb"]) >= 2 and len(et["emb"]) >= 2:
            arm.cam = _fit_lr(np.concatenate([es["emb"], et["emb"]]),
                              np.concatenate([np.zeros(len(es["emb"])), np.ones(len(et["emb"]))]).astype(int), a.seed)
        else:
            print(f"[effects] {arm.arm} seed {arm.seed}: too few ICDR-0 training photos for the camera probe")
            arm.cam = _NoProbe()

    ds_s, ds_t = src[a.part], tgt[a.part]
    idx = rng.choice(len(ds_s), a.max_n, replace=False) if a.max_n and a.max_n < len(ds_s) else None
    idx_t = rng.choice(len(ds_t), a.max_n, replace=False) if a.max_n and a.max_n < len(ds_t) else None
    sel = lambda ds, i: (ds.label_array()[:, 0] if i is None else ds.label_array()[i, 0],
                         grades(ds) if i is None else grades(ds)[i])
    y, g = sel(ds_s, idx)
    y_t, g_t = sel(ds_t, idx_t)
    print(f"[effects] {a.part} split: {len(y)} tabletop photos ({camera or 'every camera'}), {len(y_t)} handheld; "
          f"{size} px; arms {[f'{m.arm}/seed{m.seed}' for m in arms]}")

    rows: List[dict] = []
    tag = lambda arm, variant, r: {"arm": arm.arm, "seed": arm.seed, "variant": variant, "threshold": arm.thr,
                                   "ckpt": arm.path, "part": a.part, "referee": arm.referee, **r}
    base = run(arms, _take(ds_s, idx), a, dev)
    real = run(arms, _take(ds_t, idx_t), a, dev)
    for arm, R0, Rt in zip(arms, base, real):
        rows += [tag(arm, REAL, r) for r in metrics(Rt, y_t, g_t, arm.thr, a.target_sens, stats_names=arm.stats)]
        rows += [tag(arm, BASE, r) for r in metrics(R0, y, g, arm.thr, a.target_sens, R0, arm.stats)]

    cal = None
    if a.cal_n and "edge" in profile.bank and len(hi["train"]):
        # tabletop training photos for the variants' detail ranks (their own generator: the
        # photos scored below stay the ones an earlier run drew)
        pick = np.random.default_rng(a.seed + 1).choice(len(hi["train"]), min(a.cal_n, len(hi["train"])), replace=False)
        cal = torch.cat([denorm(b["image"].to(dev))
                         for b in loader(Subset(hi["train"], pick.tolist()), a.batch_size, a.num_workers)])
    sims = variants(profile, size, a.sets, cal, a.seed)
    del cal
    gens = {name: torch.Generator().manual_seed(a.seed) for name in sims}       # the same draws for every variant
    acc = {name: [{} for _ in arms] for name in sims}
    grid: Dict[str, torch.Tensor] = {}
    t0, n_done = time.time(), 0
    with torch.no_grad():
        for i, b in enumerate(loader(_take(hi[a.part], idx), a.batch_size, a.num_workers)):
            x01 = denorm(b["image"].to(dev))
            if i == 0 and a.grid_n:
                grid[BASE] = F.interpolate(x01[:a.grid_n], size=(size, size), mode="bilinear", antialias=True,
                                           align_corners=False).cpu()
            for name, sim in sims.items():
                y01 = sim(x01, gens[name])
                if i == 0 and a.grid_n:
                    grid[name] = y01[:a.grid_n].cpu()
                x = renorm(y01)
                for arm, out in zip(arms, acc[name]):
                    arm.forward(x, out)
            n_done += x01.shape[0]
            if i % 10 == 0:
                print(f"  {n_done}/{len(y)} photos x {len(sims)} variants, {time.time() - t0:.0f} s", flush=True)
    for name in sims:
        for arm, out, R0 in zip(arms, acc[name], base):
            R = {k: np.concatenate(v) for k, v in out.items()}
            rows += [tag(arm, name, r) for r in metrics(R, y, g, arm.thr, a.target_sens, R0, arm.stats)]
    if grid and not os.path.exists(os.path.join(a.out, "grid.jpg")):
        save_grid(os.path.join(a.out, "grid.jpg"), grid)
        with open(os.path.join(a.out, "grid.txt"), "w") as f:
            f.write("rows of grid.jpg, top to bottom: " + ", ".join(grid) + "\n")
    return rows


class _NoProbe:
    def predict_proba(self, E: np.ndarray) -> np.ndarray:
        return np.full((len(E), 2), np.nan)


def summarise(df: pd.DataFrame, target_sens: float) -> str:
    order = list(dict.fromkeys(df.variant))
    part = df.part.iloc[0]
    out = ["", f"WHAT THE PHONE SIMULATOR DOES TO A TRAINED MODEL (referable DR, image level, {part} split; mean over "
               f"seeds {sorted(df.seed.unique().tolist())}; threshold = each checkpoint's shipped one; oracle_spec = "
               f"specificity where the row's own ROC crosses sensitivity {target_sens})",
           f"  {BASE} = the tabletop photos as they are; {REAL} = the real handheld {part} photos; every other row = "
           "the same tabletop photos re-rendered",
           "  d_ref / d_icdr0 = mean logit shift against the unsimulated photo, referable / ICDR-0 photos "
           "(d_icdr0 up = toward disease)",
           "  p_hh = share of ICDR-0 photos the trunk's camera probe calls handheld (0 tabletop, 1 handheld)"]
    referee = df.referee.astype(bool) if "referee" in df else pd.Series(False, index=df.index)
    if part == "val" and (~referee).any():
        out.append(f"  IN-SAMPLE: the {BASE} row is the split that chose each tabletop-trained checkpoint's epoch and "
                   "threshold; read the differences between rows")
    if referee.any():
        out += [f"  REFEREES (trained with handheld labels; read {TRAINED}, threshold = their own handheld-validation "
                f"one): {', '.join(dict.fromkeys(df.arm[referee]))}. A faithful simulator leaves a referee's AUROC "
                f"on the re-rendered rows at or above its {BASE} row when it knows the handheld camera alone, and "
                f"costs one that knows both cameras no more than its {BASE} -> {REAL} difference"]
        if part == "val":
            out.append(f"  IN-SAMPLE for the referees: the {REAL} row (it chose their epoch and threshold)")
    cols = {"auroc": "auroc", "sensitivity": "sens", "specificity": "spec", "flagged": "flagged",
            "oracle_spec": "oracle_spec", "dlogit_referable": "d_ref", "dlogit_icdr0": "d_icdr0",
            "p_handheld_icdr0": "p_hh"}
    for arm in dict.fromkeys(df.arm):
        stats_of = list(dict.fromkeys(df.stats[df.arm == arm]))
        for stats in stats_of:
            d = df[(df.arm == arm) & (df.stats == stats)]
            t = d.groupby("variant")[list(cols)].mean().reindex(order).rename(columns=cols)
            if stats != stats_of[0]:
                t = t.drop(columns="p_hh")
            what = stats if stats == TRAINED else f"{stats} statistics"
            out += ["", f"{arm}, {what} ({d.seed.nunique()} seed{'s' if d.seed.nunique() > 1 else ''})",
                    t.to_string(float_format=lambda v: f"{v:.3f}")]
    ref = df[referee & (df.stats == TRAINED)]
    if len(ref):
        # the referees' verdict in one table: AUROC on the re-rendered rows against the same
        # photos unsimulated (per seed, then averaged), and how handheld the probe finds them
        # (reindex: a column whose AUROC is undefined everywhere, one class in a small
        # sample, is dropped by the pivot)
        piv = ref.pivot_table(index=["arm", "seed"], columns="variant", values="auroc").reindex(columns=order)
        gain = piv.drop(columns=[c for c in (BASE, REAL) if c in piv]).sub(piv[BASE], axis=0)
        t = pd.DataFrame({"auroc": piv.groupby("arm").mean().stack(), "minus_tabletop": gain.groupby("arm").mean().stack(),
                          "seeds_above_tabletop": (gain > 0).groupby("arm").sum().stack(),
                          "of_seeds": gain.notna().groupby("arm").sum().stack()})
        t["p_hh"] = ref.pivot_table(index=["arm", "seed"], columns="variant", values="p_handheld_icdr0"
                                    ).groupby("arm").mean().stack()
        t = t.reindex([(arm, v) for arm in dict.fromkeys(ref.arm) for v in order if (arm, v) in t.index])
        out += ["", "REFEREES' VERDICT (AUROC; minus_tabletop = against the same photos unsimulated, paired per seed; "
                    "seeds_above_tabletop of the of_seeds that have both AUROCs)",
                t.to_string(float_format=lambda v: f"{v:.3f}", na_rep="")]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", nargs="+", required=True, metavar="ARM=CKPT",
                   help="train_retinareach.py checkpoints, e.g. canon_only=ck_rr_canon/seed0.pt")
    p.add_argument("--profile", required=True, help="phone_sim.py fit's profile.json")
    p.add_argument("--source-root", default=None, help="override the checkpoints' recorded roots")
    p.add_argument("--target-root", default=None)
    p.add_argument("--source-camera", default=None,
                   help="the tabletop camera for every arm; without it each arm keeps its own, and a phone-only "
                        "referee takes the camera of another arm on the same handheld split")
    p.add_argument("--part", default="val", choices=["val", "test"])
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--max-n", type=int, default=0, help="cap each split at this many photos (0 = all)")
    p.add_argument("--probe-n", type=int, default=1000, help="ICDR-0 training photos per camera for the camera probe")
    p.add_argument("--cal-n", type=int, default=64,
                   help="tabletop training photos each variant's detail ranks are re-read on (detail_by rank; 0 = "
                        "every variant keeps the profile's reading)")
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--grid-n", type=int, default=4, help="photos per row of grid.jpg (0 = none)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out", default="ck_phone_sim/effects")
    a = p.parse_args(argv)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    stale = os.path.join(a.out, "grid.jpg")
    if os.path.exists(stale):
        os.remove(stale)

    loaded = []
    for it in a.ckpt:
        if "=" not in it:
            p.error(f"--ckpt takes ARM=CKPT, got {it!r}")
        arm, path = it.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        loaded.append((arm, path, ck, data_key(ck, a)))
    # a referee trained on handheld photos alone names no tabletop set and no camera: it
    # takes those of another arm on the same handheld photos and split (--source-root given
    # or not: with the root alone it would be scored on every camera's photos, a different
    # split from the arms beside it), unless --source-camera names the camera
    side = lambda k: (k["t_root"], k["t_ds"], k["split_seed"], k["strat"], k["size"])
    alone = lambda c: is_referee(c) and not c["args"].get("mix_root")
    if a.source_camera:
        # the split is drawn after the camera filter: one camera's photos split afresh are not the
        # partition an every-camera arm was trained on, and most of them would be its training photos
        for arm, path, ck, _ in loaded:
            if not alone(ck) and tabletop_camera(ck) is None:
                p.error(f"{arm}={path} was trained on every camera's tabletop train split: --source-camera "
                        f"{a.source_camera!r} splits that camera's photos afresh, and most of the photos scored "
                        f"would be ones it was trained on. Score it without --source-camera, in a run of its own")
    for arm, path, ck, key in loaded:
        if key["s_root"] is None or (alone(ck) and key["camera"] is None):
            donors = [k for _, _, c, k in loaded if not alone(c) and k["s_root"] is not None and side(k) == side(key)]
            # what it would take from each: the tabletop set if it has none, the camera if none was named
            takes = list(dict.fromkeys((key["s_root"] or k["s_root"], k["s_ds"], key["camera"] or k["camera"])
                                       for k in donors))
            if len(takes) > 1:
                which = "; ".join(f"{r}, {c or 'every camera'}" for r, _, c in takes)
                p.error(f"{arm}={path} was trained on handheld photos alone, and the arms beside it read different "
                        f"tabletop photos ({which}): name the ones to score it on with --source-camera (and "
                        f"--source-root) when each of those arms was trained on one camera; an arm trained on "
                        f"every camera goes in a run of its own")
            if takes:
                key.update(s_root=takes[0][0], s_ds=takes[0][1], camera=takes[0][2])
                print(f"[effects] {arm} ({os.path.basename(path)}) was trained on handheld photos alone: scored on the "
                      f"tabletop photos of the arms beside it ({key['s_root']}, {key['camera'] or 'every camera'})")
            elif key["s_root"] is None:
                p.error(f"{arm}={path} was trained on handheld photos alone: give --source-root (and "
                        f"--source-camera), or another arm on the same handheld split")
            else:
                print(f"[effects] {arm} ({os.path.basename(path)}) was trained on handheld photos alone and no arm "
                      f"beside it shares its handheld split: scored on every camera's photos under {key['s_root']} "
                      f"(--source-camera names one)")
    with open(a.profile, "rb") as f:
        profile_sha = hashlib.sha256(f.read()).hexdigest()
    trained = [f"{arm} ({os.path.basename(path)}): {line}" for arm, path, ck, _ in loaded
               for line in [trained_with(ck, profile_sha)] if line]
    for line in trained:
        print(f"[effects] {line}")
    groups: Dict[tuple, List[tuple]] = {}
    for arm, path, ck, key in loaded:
        groups.setdefault(tuple(key.items()), []).append((arm, path, ck))
    rows: List[dict] = []
    for key, members in groups.items():
        arms = [Arm(arm, path, ck, dev) for arm, path, ck in members]
        rows += group_effects(arms, dict(key), a, dev)
        del arms
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "effects.csv"), index=False)
    txt = f"\nprofile {a.profile} ({profile_sha[:12]}); sets {list(a.sets)}"
    txt += "".join(f"\n{line}" for line in trained)
    prof = PhoneProfile.load(a.profile)
    if "render" in a.sets:
        made = set(df["variant"]) if len(df) else None                      # (a row left out for want of a reading is not named)
        rv = {n: ch for n, ch in render_variants(prof, can_read=a.cal_n > 0).items() if made is None or n in made}
        txt += "\nrender rows (changed from the profile's own " + ", ".join(
            f"{k} {prof.params[k]}" for k in ("fill", "prefilter", "detail_by")) + "): " + "; ".join(
            f"{name} = " + ", ".join(f"{k} {v}" for k, v in ch.items()) for name, ch in rv.items())
        if "edge" not in prof.bank:
            txt += ("\nthis bank has no edge reading (a phone_sim/2 fit): every render row keeps detail_by band, whose "
                    "blur reacts to the field and the resampling (a render that arrives sharper is blurred more), so "
                    "a row is not sharper for its new field or filter. Read the *_no_detail twins against drop_detail "
                    "for scale and resampling alone; the new render's own criterion needs a /3 fit")
    if prof.params["detail_by"] == "rank" or ("render" in a.sets and "edge" in prof.bank):
        txt += ("\ndetail ranks: " + (f"re-read on up to {a.cal_n} tabletop training photos for every row whose renders "
                                      "reach the detail stage differently from the fitted chain's; full_cal is the "
                                      "fitted chain ranked on those same photos (read a re-read row against it; full "
                                      "keeps fit's reading, and full minus full_cal is what the choice of photos "
                                      "does); a trainer's --phone-sim-strength x ranks against the profile's own "
                                      "reading, so strength_x is that arm's training render only with --cal-n 0"
                                      if a.cal_n else
                                      "every row ranks against the profile's own reading (--cal-n 0)"))
    txt += summarise(df, a.target_sens)
    print(txt)
    with open(os.path.join(a.out, "effects_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nwrote {a.out}/effects.csv, effects_summary.txt" + (", grid.jpg, grid.txt" if a.grid_n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
