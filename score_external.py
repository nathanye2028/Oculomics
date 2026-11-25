"""
score_external.py
=================
Score FINISHED models on another dataset, with no refitting: the BRSET hypertension
probes (probe_systemic.py ``*_probe.npz``) and / or CNN checkpoints (train_mbrset.py
``.pt``) against AI-READI's MEASURED hypertension (or any dataset with an
equivalent column, probe_systemic.external_table).

Per model and for the fold ensemble (mean probability over the folds' models):
AUROC per image and per patient (images averaged), with a patient-cluster
bootstrap 95 % CI, overall and by ``device_class`` (handheld Optomed Aurora vs
tabletop) and ``camera``. For context, the AUROC of age alone on the same rows:
a photo model that does not beat it on measured labels has not shown more than age.

    python score_external.py --task cm_hypertension --dataset aireadi --root <AIREADI> \\
        --probes "exp_probe/brset_canon/cm_hypertension/probe_fold*_seed0_probe.npz" \\
        --features features/aireadi_retfound.npz \\
        --checkpoints "ck_cm_folds/cm_hypertension/ctrl_fold*_seed0.pt" --out exp_external/aireadi_htn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def auroc_ci(y, p, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    from sklearn.metrics import roc_auc_score
    y, p, g = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(groups).astype(str)
    ok = np.isfinite(p)
    y, p, g = y[ok], p[ok], g[ok]
    out = {"n": int(len(y)), "pos": int(y.sum()), "patients": int(len(np.unique(g)))}
    if len(np.unique(y)) < 2:
        out["reason"] = "single-class"
        return out
    out["auroc"] = float(roc_auc_score(y, p))
    d = pd.DataFrame({"g": g, "y": y, "p": p}).groupby("g").agg(y=("y", "max"), p=("p", "mean"))
    if d["y"].nunique() == 2:
        out["auroc_patient"] = float(roc_auc_score(d["y"], d["p"]))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    if boots:
        out["ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def refuse_own_data(probes: List[str], cks: List[str], task: str, dataset: str, csv: str,
                    image_ext: str) -> None:
    """An external score must come from another dataset than the model trained on,
    and from a model of this task: refuse otherwise (a same-dataset score would
    include every fold model's training patients)."""
    import torch
    from brset_dataset import load_any
    from retfound import load_probe_head
    target = os.path.realpath(csv)
    models = [(p, load_probe_head(p)[2]) for p in probes]
    models += [(p, torch.load(p, map_location="cpu", weights_only=False).get("args", {})) for p in cks]
    for p, m in models:
        if m.get("task") and m["task"] != task:
            raise SystemExit(f"[fatal] {p} was trained for {m['task']!r}, not --task {task!r}")
        if m.get("dataset") == dataset:
            raise SystemExit(f"[fatal] {p} was trained on --dataset {dataset}: score it on another dataset")
        own = m.get("label_csv")
        if own is None and m.get("root") and m.get("dataset"):
            try:
                own = load_any(m["root"], m["dataset"], image_ext=image_ext)["csv"]
            except (FileNotFoundError, ValueError, StopIteration):
                own = None
        if own is not None and os.path.realpath(own) == target:
            raise SystemExit(f"[fatal] {p} was trained on {own}, the table being scored")


def score_probes(paths: List[str], features: str, rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    from extract_retfound import load_features
    from probe_systemic import feature_matrix
    from retfound import load_probe_head
    files, feats, fmeta = load_features(features)
    X = feature_matrix(rows["file"].astype(str).tolist(), {f: i for i, f in enumerate(files)}, feats, "external")
    out = {}
    for p in paths:
        w, b, meta = load_probe_head(p)
        if meta.get("kind") != "classifier":
            raise SystemExit(f"[fatal] {p} is not a classifier probe")
        fm = meta.get("features_meta") or {}
        if fm.get("weights_sha256") and fmeta.get("weights_sha256") and fm["weights_sha256"] != fmeta["weights_sha256"]:
            raise SystemExit(f"[fatal] {p} was fit on embeddings from other weights than {features}")
        if fm.get("pool") and fmeta.get("pool") and fm["pool"] != fmeta["pool"]:
            raise SystemExit(f"[fatal] {p} was fit on {fm['pool']} embeddings, {features} holds {fmeta['pool']}")
        out[os.path.basename(p)[:-len("_probe.npz")]] = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    return out


def score_checkpoints(paths: List[str], task: str, split_df: pd.DataFrame, images_dir: str,
                      batch_size: int, workers: int) -> Dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader
    from dataset import MBRSETDataset
    from train_mbrset import evaluate, load_teacher, pick_device
    device = pick_device()
    out = {}
    for p in paths:
        model, bk, a, _ = load_teacher(p, device, 2, task)
        ds = MBRSETDataset(csv=split_df, images_dir=images_dir, task=task, split="val",
                           image_size=int(a.get("image_size", 224)), drop_missing_files=True, fov_crop=True)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        _, prob = evaluate(model, dl, device, 2, return_probs=True)
        out[os.path.basename(p)[:-len(".pt")]] = np.asarray(prob, float)
        print(f"[info] {p}: {bk}, {len(ds)} images")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    from brset_dataset import DATASETS, load_any
    from probe_systemic import external_table, materialise
    p = argparse.ArgumentParser(description="Score finished probes / CNNs on another dataset.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--task", required=True, help="the models' task (its label is mapped onto --dataset)")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--root", required=True)
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--probes", default=None, help="glob of *_probe.npz")
    p.add_argument("--features", default=None, help="extract_retfound.py file for --root (with --probes)")
    p.add_argument("--checkpoints", default=None, help="glob of train_mbrset.py .pt files")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    probes = sorted(glob.glob(a.probes)) if a.probes else []
    cks = sorted(glob.glob(a.checkpoints)) if a.checkpoints else []
    if not probes and not cks:
        raise SystemExit("[fatal] nothing to score: --probes and/or --checkpoints matched no file")
    if probes and not a.features:
        raise SystemExit("[fatal] --probes needs --features")
    src = load_any(a.root, a.dataset, image_ext=a.image_ext)
    refuse_own_data(probes, cks, a.task, a.dataset, src["csv"], a.image_ext)
    df = external_table(src["df"], a.task, a.dataset)
    rows, y = materialise(df, src["images_dir"], a.task)
    scores = {}
    if probes:
        scores.update({f"probe:{k}": v for k, v in score_probes(probes, a.features, rows).items()})
    if cks:
        scores.update({f"cnn:{k}": v for k, v in
                       score_checkpoints(cks, a.task, df, src["images_dir"], a.batch_size, a.num_workers).items()})
    for kind in ("probe", "cnn"):
        ks = [k for k in scores if k.startswith(kind + ":")]
        if len(ks) > 1:
            scores[f"{kind}:fold_ensemble"] = np.mean([scores[k] for k in ks], axis=0)
    g = rows["patient"].astype(str).to_numpy()
    groups = {"all": np.ones(len(rows), bool)}
    for col in ("device_class", "camera"):
        if col in rows.columns:
            for v in sorted(rows[col].dropna().astype(str).unique()):
                groups[f"{col}={v}"] = (rows[col].astype(str) == v).to_numpy()
    res, L = {}, [f"EXTERNAL SCORE  --  {a.task} models on {a.dataset} @ {a.root}",
                  f"label: {'measured' if a.dataset == 'aireadi' else 'as recorded'} "
                  f"({len(y)} images, {int(np.nansum(y))} positive, {len(np.unique(g))} participants)"]
    if "age" in rows.columns:
        scores["context:age_alone"] = pd.to_numeric(rows["age"], errors="coerce").to_numpy()
    for name, m in groups.items():
        L.append(f"\n[{name}]")
        for k, s in scores.items():
            r = auroc_ci(y[m], s[m], g[m], n_boot=a.n_boot)
            res.setdefault(name, {})[k] = r
            if "auroc" in r:
                lo, hi = r.get("ci", [float("nan")] * 2)
                L.append(f"  {k:<36} AUROC {r['auroc']:.4f} [{lo:.4f}, {hi:.4f}]  per patient "
                         f"{r.get('auroc_patient', float('nan')):.4f}  ({r['pos']}/{r['n']} images)")
            else:
                L.append(f"  {k:<36} {r.get('reason')}")
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"file": rows["file"].astype(str), "patient": g, "label": y,
                  **{c: rows[c].to_numpy() for c in ("device_class", "camera", "age") if c in rows.columns},
                  **{k.replace(":", "__"): v for k, v in scores.items()}}).to_csv(
        os.path.join(a.out, "scores.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"task": a.task, "dataset": a.dataset, "probes": probes, "checkpoints": cks, "results": res},
                  f, indent=2)
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("```\n" + txt + "\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
