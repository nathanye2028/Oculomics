#!/usr/bin/env python3
"""
patient_rule.py
===============
Per-patient referral with the threshold chosen PER PATIENT on the home camera's
validation split, from finished ``train_retinareach.py`` checkpoints -- no
retraining, minutes per checkpoint on a GPU::

    python patient_rule.py --source-root $B --target-root $M --part test \\
        --ckpt canon_only=ck_rr_canon/seed0.pt canon_phonesim=ck_rr_phonesim/seed0.pt ... \\
        --out exp_phone_sim/patient

Why
---
The shipped threshold is fixed on single photos (tabletop validation, sensitivity
``--target-sens``), and ``referral_outcomes.py`` refers a patient when ANY gradable
photo reaches it. The maximum over a patient's photos lifts per-patient sensitivity
above the image-level target and pulls specificity below it: the phone-trained model
ranks phone photos at oracle specificity 0.868 (0.85 sensitivity, image level, five
seeds) but refers at 0.891 / 0.767 per patient at its shipped threshold (seed 0).

Rules (every threshold from the TABLETOP VALIDATION split -- the camera the device
ships from; no handheld label sets one)
------------------------------------------------------------------------------
shipped       any gradable photo >= the image-level shipped threshold
              (``referral_outcomes.py``'s rule: the reference)
any_patient   the same score (the patient's highest gradable photo), its threshold
              re-set so that tabletop-validation PATIENTS reach the target sensitivity
eye_mean      each eye's gradable photos averaged, the patient's score the higher eye,
              its threshold re-set the same way (the pre-specified new rule)
Next to each, ``oracle_spec``: specificity where the scored split's own patient ROC
crosses the target sensitivity -- threshold-free, how well that aggregation ranks
patients (it reads the scored split's labels, so it is a ceiling, not a device).

Protocols, as ``style_transfer_xfer.py score`` and the trainer: ``trained`` (the
tabletop BN statistics everywhere) and ``calibrated`` (each camera's own stored
statistics; the threshold from tabletop validation under the tabletop's).

Patients
--------
Gradable = the dataset's quality grade, as ``referral_outcomes.py``: a patient with no
gradable photo is a retake, outside sensitivity and specificity. Truth = referable DR in
any of the patient's photos. An eye = (patient, laterality); a photo without laterality
is its own eye. A checkpoint trained on one tabletop camera (``--source-camera``) is
scored on that camera's own split (``style_transfer_xfer.load_device``).

``--part val`` scores the handheld VALIDATION split, for choosing between arms; there
the tabletop rows are in-sample (that split set every threshold) and are marked so.

Outputs: ``<out>/predictions.csv`` (one row per photo x protocol x split x checkpoint;
``--from-predictions`` reruns the rules from it without a GPU), ``patient_rule.csv``
(one row per arm x seed x camera x protocol x rule) and ``patient_rule_summary.txt``
(mean over seeds, seeds meeting the targets, and the paired difference when two arms
are given).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadata_model import target_vector, threshold_at_sensitivity  # noqa: E402
from referral_outcomes import patient_outcomes  # noqa: E402

PRIMARY = "dr_referable"
AGGREGATIONS = ("any", "eye_mean")
RULES = (("shipped", "any"), ("any_patient", "any"), ("eye_mean", "eye_mean"))
CAMERAS = ("tabletop", "handheld")


# --------------------------------------------------------------------------- #
# Photos -> patients
# --------------------------------------------------------------------------- #
def photo_frame(df: pd.DataFrame, truth: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """One row per photo: patient, eye, gradable, truth, p (``df`` = a split's table)."""
    q = target_vector(df, "quality") if "final_quality" in df.columns else np.full(len(df), np.nan)
    files = df["file"].astype(str).to_numpy()
    lat = df["laterality"] if "laterality" in df.columns else pd.Series(np.nan, index=df.index)
    eye = np.where(lat.notna().to_numpy(), lat.astype(str).to_numpy(), "file:" + files)
    return pd.DataFrame({"file": files, "patient": df["patient"].astype(str).to_numpy(), "eye": eye,
                         "gradable": q != 0, "truth": np.asarray(truth, dtype=float),
                         "p": np.asarray(p, dtype=float)})


def patient_scores(photos: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """One row per patient (index): ``gradable`` (any gradable photo), ``truth`` (any
    referable photo; NaN when no photo is graded) and ``score`` (NaN without a gradable
    photo): the highest gradable photo (``any``) or the higher eye's mean over its
    gradable photos (``eye_mean``)."""
    g = photos[photos["gradable"].astype(bool)]
    if aggregation == "any":
        score = g.groupby("patient")["p"].max()
    elif aggregation == "eye_mean":
        score = g.groupby(["patient", "eye"])["p"].mean().groupby(level="patient").max()
    else:
        raise ValueError(f"aggregation {aggregation!r}; known: {AGGREGATIONS}")
    per = photos.groupby("patient").agg(gradable=("gradable", "any"), truth=("truth", "max"))
    per["score"] = score.reindex(per.index)
    return per


def _scored(per: pd.DataFrame):
    ok = per["gradable"].astype(bool) & per["truth"].notna()
    return per.loc[ok, "truth"].astype(int).to_numpy(), per.loc[ok, "score"].to_numpy()


def patient_threshold(per: pd.DataFrame, target_sens: float) -> float:
    """The highest threshold at which these patients reach ``target_sens``."""
    y, s = _scored(per)
    return threshold_at_sensitivity(y, s, target_sens) if (y == 1).any() else float("nan")


def oracle_spec(per: pd.DataFrame, target_sens: float) -> float:
    y, s = _scored(per)
    if not (y == 1).any() or not (y == 0).any():
        return float("nan")
    return float((s[y == 0] < threshold_at_sensitivity(y, s, target_sens)).mean())


def refer(per: pd.DataFrame, threshold: float, n_boot: int, seed: int) -> Dict[str, object]:
    """``referral_outcomes.patient_outcomes`` on one row per patient."""
    flag = (per["score"] >= threshold).fillna(False).to_numpy()
    rows = pd.DataFrame({"patient": per.index.to_numpy(), "gradable": per["gradable"].to_numpy(dtype=bool),
                         "flag": flag, "truth": per["truth"].to_numpy()})
    o = patient_outcomes(rows, n_boot=n_boot, seed=seed)
    scored = o.get("tp", 0) + o.get("fp", 0) + o.get("tn", 0) + o.get("fn", 0)
    o["referred"] = (o.get("tp", 0) + o.get("fp", 0)) / scored if scored else float("nan")
    return o


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def evaluate(pred: pd.DataFrame, part: str, target_sens: float = 0.85, target_spec: float = 0.825,
             n_boot: int = 1000) -> pd.DataFrame:
    """``pred``: :func:`predict`'s rows. Per arm x seed x protocol, the patient
    thresholds from tabletop validation, then every rule on each camera's ``part``."""
    rows = []
    for (arm, seed, proto), d in pred.groupby(["arm", "seed", "protocol"], sort=False):
        val = d[(d["camera"] == "tabletop") & (d["part"] == "val")]
        if val.empty:
            raise ValueError(f"{arm} seed {seed} {proto}: no tabletop validation rows to set the threshold")
        thr = {agg: patient_threshold(patient_scores(val, agg), target_sens) for agg in AGGREGATIONS}
        shipped = float(d["shipped_threshold"].iloc[0])
        for cam in CAMERAS:
            ev = d[(d["camera"] == cam) & (d["part"] == part)]
            if ev.empty:
                continue
            per = {agg: patient_scores(ev, agg) for agg in AGGREGATIONS}
            for rule, agg in RULES:
                t = shipped if rule == "shipped" else thr[agg]
                o = refer(per[agg], t, n_boot, int(seed)) if t == t else {}
                sens, spec = o.get("sensitivity", float("nan")), o.get("specificity", float("nan"))
                rows.append({"arm": arm, "seed": int(seed), "camera": cam, "protocol": proto, "rule": rule,
                             "aggregation": agg, "part": part, "threshold": t,
                             "in_sample": cam == "tabletop" and part == "val",
                             **{k: v for k, v in o.items() if k not in ("sensitivity", "specificity")},
                             "sensitivity": sens, "specificity": spec,
                             "oracle_spec": oracle_spec(per[agg], target_sens),
                             "meets_targets": bool(sens >= target_sens and spec >= target_spec)})
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame, target_sens: float, target_spec: float) -> str:
    part = res["part"].iloc[0]
    key = ["camera", "protocol", "rule"]
    order = {r: i for i, (r, _) in enumerate(RULES)}
    lines = []
    for arm, r in res.groupby("arm", sort=False):
        ga = r.groupby(key, sort=False)
        t = ga[["sensitivity", "specificity", "referred", "oracle_spec"]].mean()
        t.insert(0, "seeds", ga["seed"].nunique())
        t["meet"] = ga["meets_targets"].sum().astype(int)
        t = t.reset_index()
        t = t.sort_values(["camera", "protocol", "rule"], key=lambda c: c.map(order) if c.name == "rule" else c)
        lines += [f"\n--- {arm} ---", t.to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    txt = ("\nPER-PATIENT REFERRAL, referable DR in any eye "
           f"({part} split, seeds {sorted(res.seed.unique().tolist())}; targets sens >= {target_sens}, "
           f"spec >= {target_spec}; meet = seeds meeting both)\n"
           "rules: shipped = any gradable photo >= the image-level shipped threshold; any_patient = the same "
           "score, threshold re-set on tabletop-validation patients; eye_mean = each eye's photos averaged, "
           "the higher eye, threshold re-set the same way. Every threshold comes from the tabletop "
           "validation split. oracle_spec = specificity where this split's own patient ROC crosses "
           f"sensitivity {target_sens} (a ranking ceiling, not a device). referred = share of scored "
           "patients referred.\n"
           + ("IN-SAMPLE: on the val split the tabletop rows are the patients that set every threshold.\n"
              if part == "val" else "")
           + "\n".join(lines))
    arms = list(dict.fromkeys(res["arm"]))
    if len(arms) == 2:
        cols = ["sensitivity", "specificity", "oracle_spec"]
        k = key + ["seed"]
        d = (res[res.arm == arms[1]].set_index(k)[cols] - res[res.arm == arms[0]].set_index(k)[cols]).dropna(how="all")
        if len(d):
            pd_ = d.groupby(level=key, sort=False).agg(["mean", "min", "max", "count"]).round(3)
            txt += (f"\n\nPAIRED: {arms[1]} minus {arms[0]}, same seed and patients (count = seeds with both)\n"
                    + pd_.to_string())
    return txt


# --------------------------------------------------------------------------- #
# Checkpoints -> photo scores (GPU)
# --------------------------------------------------------------------------- #
def predict(ckpts: Sequence[str], source_root: Optional[str], target_root: Optional[str], part: str,
            device: str = "auto", batch_size: int = 32, num_workers: int = 4) -> pd.DataFrame:
    """Every checkpoint's referable-DR probability on tabletop validation (the
    threshold) and on each camera's ``part``, under both protocols."""
    import torch
    from retinareach import build_from_checkpoint, set_bn_stats, stats_from_lists
    from style_transfer_xfer import load_device, loader
    from train_mbrset import pick_device
    dev = pick_device() if device == "auto" else torch.device(device)
    out, cache = [], {}
    for arm, path in [it.split("=", 1) for it in ckpts]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        args = ck["args"]
        size, seed = int(ck["config"].get("image_size", 384)), int(args.get("seed", 0))
        strat = (args.get("source_targets") or [PRIMARY])[0]
        k = build_from_checkpoint(ck).targets.index(PRIMARY)
        for cam in CAMERAS:
            root = (source_root or args["source_root"]) if cam == "tabletop" else (target_root or args["target_root"])
            dset = args["source_dataset"] if cam == "tabletop" else args["target_dataset"]
            camera = args.get("source_camera") if cam == "tabletop" else None
            profile = args["source_name"] if cam == "tabletop" else args["target_name"]
            for sp in sorted({part, "val"} if cam == "tabletop" else {part}):
                key = (root, dset, size, int(args["split_seed"]), camera, strat)
                if key not in cache:
                    cache[key] = load_device(root, dset, size, int(args["split_seed"]), camera=camera, strat=strat)
                ds = cache[key][sp]
                truth = ds.label_array()[:, 0]
                for proto in ("trained", "calibrated"):
                    m = build_from_checkpoint(ck)
                    if proto == "calibrated":
                        set_bn_stats(m, stats_from_lists(ck["profiles"][profile]["bn_stats"]))
                    m = m.to(dev).eval()
                    with torch.no_grad():
                        p = torch.cat([torch.sigmoid(m(b["image"].to(dev))[:, k]).float().cpu()
                                       for b in loader(ds, batch_size, num_workers)]).numpy()
                    del m
                    f = photo_frame(ds.df, truth, p)
                    f.insert(0, "protocol", proto)
                    f.insert(0, "part", sp)
                    f.insert(0, "camera", cam)
                    f.insert(0, "seed", seed)
                    f.insert(0, "arm", arm)
                    f["shipped_threshold"] = ck["thresholds"][proto].get(PRIMARY, np.nan)
                    f["ckpt"] = path
                    out.append(f)
        print(f"  scored {arm} seed {seed}")
    return pd.concat(out, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-patient referral, threshold chosen per patient on tabletop "
                                            "validation, from finished checkpoints.")
    p.add_argument("--ckpt", nargs="+", default=[], metavar="ARM=CKPT")
    p.add_argument("--from-predictions", default=None, metavar="CSV",
                   help="an earlier run's predictions.csv: the rules again, no GPU")
    p.add_argument("--source-root", default=None, help="override the checkpoints' recorded tabletop root")
    p.add_argument("--target-root", default=None, help="override the recorded handheld root")
    p.add_argument("--part", default="test", choices=["val", "test"],
                   help="the split scored (val: choosing between arms; tabletop rows in-sample there)")
    p.add_argument("--target-sens", type=float, default=0.85)
    p.add_argument("--target-spec", type=float, default=0.825)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out", default="exp_phone_sim/patient")
    a = p.parse_args(argv)
    if bool(a.ckpt) == bool(a.from_predictions):
        p.error("give --ckpt ARM=CKPT ... or --from-predictions, not both")
    os.makedirs(a.out, exist_ok=True)
    if a.from_predictions:
        pred = pd.read_csv(a.from_predictions, dtype={"patient": str, "eye": str, "file": str})
        have = set(zip(pred["camera"], pred["part"]))
        missing = [f"{c} {s}" for c, s in (("tabletop", "val"), ("handheld", a.part)) if (c, s) not in have]
        if missing:
            raise SystemExit(f"[fatal] {a.from_predictions} holds no {' / '.join(missing)} rows: score the "
                             f"checkpoints with --part {a.part}")
    else:
        pred = predict(a.ckpt, a.source_root, a.target_root, a.part, a.device, a.batch_size, a.num_workers)
        pred.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    res = evaluate(pred, a.part, a.target_sens, a.target_spec, a.n_boot)
    res.to_csv(os.path.join(a.out, "patient_rule.csv"), index=False)
    txt = summarise(res, a.target_sens, a.target_spec)
    print(txt)
    with open(os.path.join(a.out, "patient_rule_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nwrote {a.out}/patient_rule.csv and patient_rule_summary.txt"
          + ("" if a.from_predictions else " (photo scores: predictions.csv)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
