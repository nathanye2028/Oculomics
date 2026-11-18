#!/usr/bin/env python3
"""phone_sim_effects.py -- what each part of the phone simulator does to a trained model.

No training: finished ``train_retinareach.py`` checkpoints score one tabletop split as it
is, re-rendered by the whole simulator, and re-rendered by variants of it, next to the real
handheld split. Every variant draws the same bank rows and the same jitter for a photo, so
two rows of the table differ by the variant alone.

Variants (``--sets``):
  full        every stage, the phone field, strength 1 (what the training arm sees)
  drop        the simulator without one stage (``drop_colour``, ...): a row that moves back
              toward ``tabletop`` names the stage responsible for the change
  only        one stage alone (``only_light``, ...; the bloom needs the field's disc, so it
              is ``field+bloom`` next to ``only_field``)
  strength    0, 0.25, 0.5, 0.75 of the photometric change (0 is the geometry alone)
  field       ``whole_field``: the whole tabletop FOV instead of the phone's narrower field

Per arm, seed and variant, under the tabletop statistics and under the checkpoint's handheld
statistics (referable DR, image level): AUROC; sensitivity, specificity and flagged share at
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

Writes ``effects.csv``, ``effects_summary.txt`` and ``grid.jpg`` (dataset photos: keep the
folder under ``ck_*``). The default split is validation, where simulator settings are chosen;
its tabletop rows chose each checkpoint's epoch and threshold, so the unsimulated row is
in-sample there and the differences between rows are the reading. ``--part test`` reads the
test split.
"""
from __future__ import annotations

import argparse
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
from phone_sim import STAGES, PhoneProfile, PhoneSimulator  # noqa: E402
from retinareach import build_from_checkpoint, set_bn_stats, stats_from_lists  # noqa: E402
from style_transfer import denorm, renorm  # noqa: E402
from style_transfer_xfer import (PRIMARY, _check_phone_profile, _fit_lr, grades, load_device,  # noqa: E402
                                 loader, save_grid)
from train_mbrset import pick_device  # noqa: E402

SETS = ("drop", "only", "strength", "field")
STATS = ("tabletop", "handheld")
BASE, REAL = "tabletop", "REAL handheld"


def variants(profile: PhoneProfile, size: int, sets: Sequence[str]) -> Dict[str, PhoneSimulator]:
    mk = lambda **kw: PhoneSimulator(profile, out_size=size, **kw)
    v = {"full": mk()}
    if "drop" in sets:
        for s in STAGES:
            v[f"drop_{s}"] = mk(stages=[t for t in STAGES if t != s])
    if "only" in sets:
        v["only_field"] = mk(stages=["field"])
        for s in STAGES[1:]:
            if s == "bloom":                       # keyed on the disc the field stage finds
                v["field+bloom"] = mk(stages=["field", "bloom"])
            else:
                v[f"only_{s}"] = mk(stages=[s])
    if "strength" in sets:
        for s in (0.0, 0.25, 0.5, 0.75):
            v[f"strength_{s:g}"] = mk(strength=s)
    if "field" in sets:
        v["whole_field"] = mk(field="whole")
    return v


class Arm:
    """One checkpoint: its trunk under the tabletop and the handheld statistics."""

    def __init__(self, arm: str, path: str, ck: dict, dev: torch.device) -> None:
        a = ck["args"]
        self.arm, self.path, self.seed = arm, path, int(a.get("seed", 0))
        self.thr = float(ck["thresholds"]["calibrated"][PRIMARY])
        self.models = {}
        for stats, dname in zip(STATS, (a["source_name"], a["target_name"])):
            m = build_from_checkpoint(ck)
            set_bn_stats(m, stats_from_lists(ck["profiles"][dname]["bn_stats"]))
            self.models[stats] = m.to(dev).eval()
        self.k = self.models[BASE].targets.index(PRIMARY)
        self.cam = None                            # the camera probe, on the tabletop-statistics embedding

    @torch.no_grad()
    def forward(self, x: torch.Tensor, out: Dict[str, List[np.ndarray]]) -> None:
        for stats, m in self.models.items():
            e = m.embed(x)
            out.setdefault(f"z_{stats}", []).append(m.head(m.drop(e))[:, self.k].float().cpu().numpy())
            if stats == BASE:
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
            base: Optional[Dict[str, np.ndarray]] = None) -> List[dict]:
    fin = np.isfinite(y)
    yy = y[fin].astype(int)
    two = len(np.unique(yy)) == 2
    rows = []
    for stats in STATS:
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
        if stats == BASE and "p_cam" in R and (g == 0).any():
            row["p_handheld_icdr0"] = float(R["p_cam"][g == 0].mean())
        rows.append(row)
    return rows


def _take(ds, idx: Optional[np.ndarray]):
    return ds if idx is None else Subset(ds, idx.tolist())


def _icdr0(ds, n: int, rng) -> Subset:
    z = np.flatnonzero(grades(ds) == 0)
    return Subset(ds, rng.choice(z, min(n, len(z)), replace=False).tolist())


def group_effects(arms: Sequence[Arm], cks: Sequence[dict], a, dev) -> List[dict]:
    """The arms of one data key (roots, split, camera, size) through every variant."""
    args = cks[0]["args"]
    size = int(cks[0]["config"].get("image_size", 384))
    _check_phone_profile(a.profile, size)
    profile = PhoneProfile.load(a.profile)
    camera = args.get("source_camera")
    strat = (args.get("source_targets") or [PRIMARY])[0]
    kw = dict(camera=camera, strat=strat)
    s_root, t_root = a.source_root or args["source_root"], a.target_root or args["target_root"]
    split_seed = int(args["split_seed"])
    src = load_device(s_root, args["source_dataset"], size, split_seed, **kw)
    hi = load_device(s_root, args["source_dataset"], int(round(size * float(profile.params.get("supersample", 1.5)))),
                     split_seed, **kw)
    tgt = load_device(t_root, args["target_dataset"], size, split_seed, strat=strat)
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
                                   "ckpt": arm.path, "part": a.part, **r}
    base = run(arms, _take(ds_s, idx), a, dev)
    real = run(arms, _take(ds_t, idx_t), a, dev)
    for arm, R0, Rt in zip(arms, base, real):
        rows += [tag(arm, REAL, r) for r in metrics(Rt, y_t, g_t, arm.thr, a.target_sens)]
        rows += [tag(arm, BASE, r) for r in metrics(R0, y, g, arm.thr, a.target_sens, R0)]

    sims = variants(profile, size, a.sets)
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
            rows += [tag(arm, name, r) for r in metrics(R, y, g, arm.thr, a.target_sens, R0)]
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
    if part == "val":
        out.append(f"  IN-SAMPLE: the {BASE} row is the split that chose each checkpoint's epoch and threshold; read "
                   "the differences between rows")
    cols = {"auroc": "auroc", "sensitivity": "sens", "specificity": "spec", "flagged": "flagged",
            "oracle_spec": "oracle_spec", "dlogit_referable": "d_ref", "dlogit_icdr0": "d_icdr0",
            "p_handheld_icdr0": "p_hh"}
    for arm in dict.fromkeys(df.arm):
        for stats in STATS:
            d = df[(df.arm == arm) & (df.stats == stats)]
            t = d.groupby("variant")[list(cols)].mean().reindex(order).rename(columns=cols)
            if stats != BASE:
                t = t.drop(columns="p_hh")
            out += ["", f"{arm}, {stats} statistics ({d.seed.nunique()} seed{'s' if d.seed.nunique() > 1 else ''})",
                    t.to_string(float_format=lambda v: f"{v:.3f}")]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", nargs="+", required=True, metavar="ARM=CKPT",
                   help="train_retinareach.py checkpoints, e.g. canon_only=ck_rr_canon/seed0.pt")
    p.add_argument("--profile", required=True, help="phone_sim.py fit's profile.json")
    p.add_argument("--source-root", default=None, help="override the checkpoints' recorded roots")
    p.add_argument("--target-root", default=None)
    p.add_argument("--part", default="val", choices=["val", "test"])
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--max-n", type=int, default=0, help="cap each split at this many photos (0 = all)")
    p.add_argument("--probe-n", type=int, default=1000, help="ICDR-0 training photos per camera for the camera probe")
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

    groups: Dict[tuple, List[tuple]] = {}
    for it in a.ckpt:
        if "=" not in it:
            p.error(f"--ckpt takes ARM=CKPT, got {it!r}")
        arm, path = it.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        g = ck["args"]
        key = (a.source_root or g["source_root"], a.target_root or g["target_root"], g["source_dataset"],
               g["target_dataset"], int(g["split_seed"]), g.get("source_camera"),
               (g.get("source_targets") or [PRIMARY])[0], int(ck["config"].get("image_size", 384)))
        groups.setdefault(key, []).append((arm, path, ck))
    rows: List[dict] = []
    for members in groups.values():
        arms = [Arm(arm, path, ck, dev) for arm, path, ck in members]
        rows += group_effects(arms, [ck for _, _, ck in members], a, dev)
        del arms
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "effects.csv"), index=False)
    txt = f"\nprofile {a.profile}; sets {list(a.sets)}" + summarise(df, a.target_sens)
    print(txt)
    with open(os.path.join(a.out, "effects_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nwrote {a.out}/effects.csv, effects_summary.txt" + (", grid.jpg, grid.txt" if a.grid_n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
