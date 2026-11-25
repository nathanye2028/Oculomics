"""
probe_systemic.py
=================
A small classifier on frozen RETFound embeddings, on EXACTLY the CNN runs' folds,
writing the same per-fold files, so ``summarize_systemic.py`` pools it unchanged.

Why: on mBRSET nephropathy a CNN fits millions of weights from ~32 positive
patients per fold. Here the trainable part is StandardScaler -> PCA(32) ->
L2 logistic regression with balanced class weights: 33 numbers.

Per fold K (``dataset.fold_split`` on the table ``train_mbrset.load_table`` builds,
same ``--fold-seed``, so fold K holds the same patients as the CNN's fold K):
  * rows = the trainer's rows (label present, image on disk), each looked up in the
    ``extract_retfound.py`` file (a missing embedding is fatal, never skipped);
  * scaler + PCA fit on the train rows; C picked on the val rows (AUROC; ties go to
    the smaller C); the model is refit on train only, so the val scores stay
    honest for the covariate models below, as the CNN's checkpoint selection does;
  * ``covariate_baseline.incremental_value`` (covariates vs covariates + probe logit,
    fit on val, scored on test, with the before-lesions subsets) and
    ``incremental_rows``;
  * ``<out>/<task>/<cond>_fold<K>_seed<S>.json`` + ``_test.csv`` with the columns
    train_mbrset.py writes (file, patient, label, p_image, the covariates,
    p_covariates, p_combined, in_<subset>, task, fold, seed), and ``_probe.npz``
    (the fitted probe folded into one linear map on the embedding, for
    ``train_mbrset.py --teacher`` and ``score_external.py``).

Then ``summarize_systemic.py --dir <out> --control <cond>`` pools the folds: the
incremental delta and the matched-pair concordance, out of fold.

A continuous target (an AI-READI measurement: ``sbp``, ``log_uacr``, ``moca``)
switches to ridge regression with alpha picked on val; its pooled summary (R2 of
covariates vs covariates + photo) is written by this script, because the
summariser's numbers are AUROCs.

    python probe_systemic.py --dataset mbrset --root <mBRSET> --task nephropathy \\
        --features features/mbrset_retfound.npz --out exp_probe/mbrset
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

C_GRID = tuple(float(c) for c in np.logspace(-4, 2, 13))
ALPHA_GRID = tuple(float(a) for a in np.logspace(-2, 4, 13))

# Columns that mean the same target on another dataset, for scoring a fold's probe
# out of domain. mBRSET: brset_comorbidities.MBRSET_EQUIVALENTS (hypertension only;
# the kidney / MI mappings failed on metadata). AI-READI: the measured flag
# (aireadi_dataset.py: SBP >= 130 or DBP >= 80 at the visit).
AIREADI_EQUIVALENTS = {"cm_hypertension": "htn_measured", "hypertension": "htn_measured"}


def external_table(df: pd.DataFrame, task: str, dataset: str) -> pd.DataFrame:
    """``df`` with the target under ``task``'s column name, or SystemExit when the
    target has no equivalent there."""
    from dataset import LABEL_REGISTRY
    col = LABEL_REGISTRY[task].source_cols[0]
    if dataset == "mbrset":
        from brset_comorbidities import MBRSET_EQUIVALENTS, mbrset_equivalents
        if task.startswith("cm_"):
            if task not in MBRSET_EQUIVALENTS:
                raise SystemExit(f"[fatal] {task} has no mBRSET equivalent (MBRSET_EQUIVALENTS)")
            return mbrset_equivalents(df)
        return df
    if dataset == "aireadi":
        if col in df.columns:
            return df
        if task not in AIREADI_EQUIVALENTS or AIREADI_EQUIVALENTS[task] not in df.columns:
            raise SystemExit(f"[fatal] {task} has no AI-READI equivalent (AIREADI_EQUIVALENTS) "
                             f"or labels_aireadi.csv lacks it: run aireadi_dataset.py targets")
        return df.assign(**{col: df[AIREADI_EQUIVALENTS[task]]})
    if col not in df.columns:
        raise SystemExit(f"[fatal] {dataset} has no column {col!r} for {task}")
    return df


def feature_matrix(files: Sequence[str], index: Dict[str, int], feats: np.ndarray,
                   what: str) -> np.ndarray:
    miss = [f for f in files if f not in index]
    if miss:
        raise SystemExit(f"[fatal] {len(miss)} of {len(files)} {what} images have no embedding "
                         f"(e.g. {miss[:3]}): the features were extracted from another table or "
                         f"root. Re-run extract_retfound.py on this --root (and no --camera / "
                         f"--limit narrower than this run).")
    return feats[[index[f] for f in files]]


def materialise(split_df: pd.DataFrame, images_dir: str, task: str):
    """The rows the CNN trains / scores on for this split (MBRSETDataset drops rows
    with no label or no image file), in the dataset's order, and their labels."""
    from dataset import MBRSETDataset
    from train_mbrset import rows_for
    ds = MBRSETDataset(csv=split_df, images_dir=images_dir, task=task, split="val",
                       drop_missing_files=True, fov_crop=True)
    lab = ds.labels.numpy() if hasattr(ds.labels, "numpy") else np.asarray(ds.labels)
    return rows_for(ds, split_df), lab.astype(float)


def _auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def fit_probe(Xtr, ytr, Xva, yva, kind: str, n_pca: int, grid: Sequence[float], seed: int):
    """Scaler + PCA fit once on train, then the linear model over ``grid``, picked on
    val (AUROC for a classifier, R2 for a regressor; ties -> stronger penalty)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    k = int(min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(scaler.transform(Xtr))
    Ztr, Zva = pca.transform(scaler.transform(Xtr)), pca.transform(scaler.transform(Xva))
    table, best = {}, None
    # strongest penalty first, so a tie keeps it: C ascending / alpha descending
    for g in (sorted(grid) if kind == "classifier" else sorted(grid, reverse=True)):
        if kind == "classifier":
            m = LogisticRegression(C=g, class_weight="balanced", max_iter=5000).fit(Ztr, ytr.astype(int))
            score = _auroc(yva, m.decision_function(Zva))
        else:
            m = Ridge(alpha=g).fit(Ztr, ytr)
            score = float(r2_score(yva, m.predict(Zva)))
        table[f"{g:.3g}"] = score
        if score == score and (best is None or score > best[0] + 1e-12):
            best = (score, g, m)
    if best is None:
        raise SystemExit("[fatal] the probe could not be scored on val (single-class val rows?)")
    return scaler, pca, best[2], {"param": best[1], "val_score": best[0], "by_param": table,
                                  "n_components": k,
                                  "explained_variance": float(pca.explained_variance_ratio_.sum())}


def predict(scaler, pca, clf, X, kind: str) -> np.ndarray:
    Z = pca.transform(scaler.transform(X))
    return clf.predict_proba(Z)[:, 1] if kind == "classifier" else clf.predict(Z)


def run_fold(args, src: dict, index: Dict[str, int], feats: np.ndarray, fmeta: dict,
             fold: int) -> Optional[str]:
    from covariate_baseline import (covariate_baseline, image_minus_covariate, incremental_rows,
                                    incremental_value, incremental_value_regression)
    from dataset import LABEL_REGISTRY, fold_split
    from retfound import collapse_probe, save_probe_head
    from train_mbrset import nodm_dr0_mask, normal_eye_masks

    name = f"{args.cond}_fold{fold}_seed{args.seed}"
    out_dir = os.path.join(args.out, args.task)
    os.makedirs(out_dir, exist_ok=True)          # before the external CSV, the first file written
    js = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(js) and not args.overwrite:
        print(f"[skip] {args.task} {name} (exists)")
        return js
    spec = LABEL_REGISTRY[args.task]
    kind = "classifier" if spec.num_classes == 2 else "regressor"
    if spec.num_classes not in (2, None) or spec.multilabel:
        raise SystemExit(f"[fatal] {args.task}: only binary or continuous targets can be probed")
    splits = fold_split(src["df"], task=args.task, fold=fold, n_folds=args.n_folds, val_frac=0.10,
                        group_col="patient", seed=args.fold_seed)
    rows, y = {}, {}
    for part in ("train", "val", "test"):
        rows[part], y[part] = materialise(splits[part], src["images_dir"], args.task)
    X = {p: feature_matrix(rows[p]["file"].astype(str).tolist(), index, feats, p) for p in rows}
    grid = args.c_grid if kind == "classifier" else args.alpha_grid
    scaler, pca, clf, sel = fit_probe(X["train"], y["train"], X["val"], y["val"], kind, args.pca,
                                      grid, args.seed)
    p_val, p_test = (predict(scaler, pca, clf, X[p], kind) for p in ("val", "test"))
    feats_cov = tuple(args.covariate_features)
    print(f"\n=== {args.task} {name}: train {len(y['train'])} / val {len(y['val'])} / test "
          f"{len(y['test'])} images ({rows['test']['patient'].nunique()} test patients) ===")
    print(f"  PCA {sel['n_components']} components ({sel['explained_variance']:.1%} of the variance); "
          f"{'C' if kind == 'classifier' else 'alpha'} = {sel['param']:.3g} (val "
          f"{'AUROC' if kind == 'classifier' else 'R2'} {sel['val_score']:.4f})")

    test_rows, val_rows = rows["test"], rows["val"]
    subsets = {}
    m = nodm_dr0_mask(test_rows)
    if m is not None:
        subsets["no_diabetes_dr0"] = m
    subsets.update(normal_eye_masks(test_rows))
    pr = pd.DataFrame({"file": test_rows["file"].astype(str), "patient": test_rows["patient"].astype(str),
                       "label": y["test"], "p_image": p_test})
    for c in dict.fromkeys(["age", "sex"] + list(feats_cov)):
        if c in test_rows.columns:
            pr[c] = test_rows[c].to_numpy()

    rec = {"task": args.task, "seed": args.seed, "num_classes": spec.num_classes,
           "train_dataset": args.dataset, "camera": args.camera, "model": "retfound_probe",
           "fold": fold, "n_folds": args.n_folds, "fold_seed": args.fold_seed,
           "n_train": int(len(y["train"])), "n_val": int(len(y["val"])), "n_test": int(len(y["test"])),
           "probe": {"kind": kind, "pca": sel["n_components"], "selected": sel["param"],
                     "val_score": sel["val_score"], "by_param": sel["by_param"],
                     "explained_variance": sel["explained_variance"],
                     "features": os.path.abspath(args.features)},
           "features_meta": {k: fmeta.get(k) for k in ("weights", "weights_sha256", "timm_name",
                                                       "image_size", "pool", "root", "created")},
           "args": vars(args)}
    if kind == "classifier":
        tm = {"auroc": _auroc(y["test"], p_test), "n": int(len(y["test"]))}
        cb = covariate_baseline(splits["train"], splits["test"], args.task, features=feats_cov,
                                seed=args.seed)
        inc = incremental_value(val_rows, p_val, test_rows, p_test, args.task, features=feats_cov,
                                subsets=subsets, seed=args.seed, n_boot=args.n_boot)
        if inc.get("reason") is None:
            pr = pd.concat([pr, incremental_rows(val_rows, p_val, test_rows, p_test, args.task,
                                                 features=feats_cov)], axis=1)
            for k, m_ in subsets.items():
                pr[f"in_{k}"] = np.asarray(m_, dtype=bool)
        rec.update(test=tm, best_val_auroc=sel["val_score"], test_pos=int(y["test"].sum()),
                   covariate_baseline=cb, incremental=inc,
                   image_minus_covariate_auroc=image_minus_covariate(tm["auroc"], cb))
        print(f"  test AUROC {tm['auroc']:.4f}   covariates alone {cb.get('auroc', float('nan')):.4f}")
        b = (inc.get("test") or {})
        if "delta" in b:
            lo, hi = b.get("delta_ci", [float("nan")] * 2)
            print(f"  +probe vs {'+'.join(feats_cov)}: {b['auroc_covariates']:.4f} -> "
                  f"{b['auroc_combined']:.4f}  delta {b['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        elif inc.get("reason"):
            print(f"  incremental: not computed ({inc['reason']})")
    else:
        from sklearn.metrics import mean_absolute_error, r2_score
        yt = y["test"]
        tm = {"r2": float(r2_score(yt, p_test)), "mae": float(mean_absolute_error(yt, p_test)),
              "r": float(np.corrcoef(yt, p_test)[0, 1]) if len(yt) > 2 else float("nan"),
              "n": int(len(yt))}
        inc = incremental_value_regression(val_rows, p_val, test_rows, p_test, args.task,
                                           features=feats_cov, subsets=subsets, seed=args.seed,
                                           n_boot=args.n_boot)
        if inc.get("reason") is None:
            pr["p_covariates"], pr["p_combined"] = inc.pop("_p_cov_rows"), inc.pop("_p_comb_rows")
            for k, m_ in subsets.items():
                pr[f"in_{k}"] = np.asarray(m_, dtype=bool)
        rec.update(test=tm, incremental_regression=inc)
        print(f"  test R2 {tm['r2']:.4f}  r {tm['r']:.3f}  MAE {tm['mae']:.3f}")

    if args.external_root:
        rec["external"] = score_external_fold(args, scaler, pca, clf, kind, os.path.join(out_dir, name), src)
    pr = pr.assign(task=args.task, fold=fold, seed=args.seed)
    csv = os.path.join(out_dir, f"{name}_test.csv")
    pr.to_csv(csv, index=False)
    rec["predictions_csv"] = os.path.abspath(csv)
    w, b = collapse_probe(scaler, pca, clf)
    head_meta = {"kind": kind, "task": args.task, "dataset": args.dataset,
                 "root": os.path.abspath(src["root"]), "label_csv": os.path.realpath(src["csv"]),
                 "camera": args.camera or None, "fold": fold,
                 "n_folds": args.n_folds, "fold_seed": args.fold_seed, "seed": args.seed,
                 "val_auroc" if kind == "classifier" else "val_r2": sel["val_score"],
                 "features_meta": {**rec["features_meta"], "weights_path": fmeta.get("weights_path")},
                 "created": datetime.datetime.now().isoformat(timespec="seconds")}
    save_probe_head(os.path.join(out_dir, f"{name}_probe.npz"), w, b, head_meta)
    with open(js, "w") as f:                                     # last: the completion marker
        json.dump(rec, f, indent=2, default=float)
    print(f"[info] wrote {js}")
    return js


def score_external_fold(args, scaler, pca, clf, kind: str, stem: str, src: dict) -> dict:
    """This fold's probe on another dataset's embeddings (every labelled row). Refuses
    the training table itself (however its root was spelt) and drops any file that
    is also in the training table."""
    from brset_dataset import load_any
    from dataset import LABEL_REGISTRY
    from extract_retfound import load_features
    ext = load_any(args.external_root, args.external_dataset, image_ext=args.image_ext)
    if os.path.realpath(ext["csv"]) == os.path.realpath(src["csv"]):
        raise SystemExit(f"[fatal] the external table {ext['csv']} is the training table")
    df = external_table(ext["df"], args.task, args.external_dataset)
    train_files = set(src["df"]["file"].astype(str))
    clash = df["file"].astype(str).isin(train_files)
    if clash.any():
        print(f"[warn] {int(clash.sum())} external rows share a file name with the training table; dropped")
        df = df.loc[~clash].reset_index(drop=True)
    rows, y = materialise(df, ext["images_dir"], args.task)
    files, feats, _ = load_features(args.external_features)
    idx = {f: i for i, f in enumerate(files)}
    p = predict(scaler, pca, clf, feature_matrix(rows["file"].astype(str).tolist(), idx, feats,
                                                 "external"), kind)
    out = pd.DataFrame({"file": rows["file"].astype(str), "patient": rows["patient"].astype(str),
                        "label": y, "p_image": p})
    for c in ("age", "sex", "camera", "device_class"):
        if c in rows.columns:
            out[c] = rows[c].to_numpy()
    out.to_csv(stem + "_external.csv", index=False)
    res = {"dataset": args.external_dataset, "root": os.path.abspath(args.external_root),
           "label_col": LABEL_REGISTRY[args.task].source_cols[0], "n": int(len(y))}
    if kind == "classifier":
        res.update(auroc=_auroc(y, p), pos=int(np.nansum(y)))
        print(f"  external {args.external_dataset}: AUROC {res['auroc']:.4f} (n={res['n']}, "
              f"{res['pos']} positive images)")
    return res


# --------------------------------------------------------------------------- #
# Pooled summary for continuous targets (binary ones go to summarize_systemic.py)
# --------------------------------------------------------------------------- #
_RUN = re.compile(r"^(?P<cond>.+)_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.json$")


def summarize_regression(task_dir: str, cond: str, n_boot: int = 2000) -> Optional[dict]:
    """Pool a regression probe's fold test rows: R2 of covariates vs covariates +
    photo on every patient once, with a patient-cluster bootstrap CI on the gain.
    Completeness is read from the fold JSONs (written last), not from the CSVs."""
    from covariate_baseline import r2_block
    by_seed: Dict[int, Dict[int, dict]] = {}
    for p in glob.glob(os.path.join(task_dir, f"{cond}_fold*_seed*.json")):
        m = _RUN.match(os.path.basename(p))
        if m and m["cond"] == cond:
            with open(p) as f:
                rec = json.load(f)
            rec["_csv"] = os.path.splitext(p)[0] + "_test.csv"
            by_seed.setdefault(int(m["seed"]), {})[int(m["fold"])] = rec
    out = {}
    lines = [f"REGRESSION PROBE, POOLED OUT OF FOLD  --  {task_dir}  ({cond})"]
    for s, folds in sorted(by_seed.items()):
        n = folds[min(folds)].get("n_folds")
        if sorted(folds) != list(range(n)) or not all(os.path.isfile(folds[k]["_csv"]) for k in folds):
            lines.append(f"  seed {s}: folds present {sorted(folds)} of {n}; not pooled")
            continue
        rows = pd.concat([pd.read_csv(folds[k]["_csv"]) for k in range(n)], ignore_index=True)
        if not {"p_covariates", "p_combined"} <= set(rows.columns):
            lines.append(f"  seed {s}: no covariate predictions in the rows; not pooled")
            continue
        ok = rows[["label", "p_image", "p_covariates", "p_combined"]].notna().all(1).to_numpy()
        blocks = {"all": ok}
        for c in sorted(c for c in rows.columns if c.startswith("in_")):
            blocks[c[3:]] = ok & rows[c].astype(bool).to_numpy()
        res = {name: r2_block(rows["label"].to_numpy(), rows["patient"].astype(str).to_numpy(),
                              rows["p_covariates"].to_numpy(), rows["p_combined"].to_numpy(),
                              rows["p_image"].to_numpy(), msk, n_boot=n_boot, seed=s)
               for name, msk in blocks.items()}
        out[s] = res
        lines.append(f"  seed {s}: {int(rows.loc[ok, 'patient'].nunique())} patients, each tested once")
        for name, b in res.items():
            if "delta" not in b:
                lines.append(f"    [{name}] {b.get('reason', 'not computed')} (n={b['n']})")
                continue
            lo, hi = b["delta_ci"]
            lines.append(f"    [{name:<20}] R2 covariates {b['r2_covariates']:.4f}  +photo "
                         f"{b['r2_combined']:.4f}  gain {b['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                         f"photo alone r={b['r_image']:.3f}  ({b['patients']} patients)")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(task_dir, f"summary_regression_{cond}.md"), "w") as f:
        f.write("```\n" + txt + "\n```\n")
    return out


def main(argv=None) -> int:
    from brset_dataset import DATASETS
    from extract_retfound import load_features
    from train_mbrset import load_table
    p = argparse.ArgumentParser(description="Frozen-RETFound probe on the CNN runs' folds.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--root", required=True)
    p.add_argument("--camera", default=None, help='e.g. "Canon CR"; must equal the CNN runs\' --camera')
    p.add_argument("--task", required=True)
    p.add_argument("--features", required=True, help="extract_retfound.py output for this --root")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--out", required=True, help="results root -> <out>/<task>/<cond>_fold<k>_seed<s>.*")
    p.add_argument("--cond", default="probe", help="condition name (summarize_systemic.py --control)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42, help="must equal the CNN runs' FOLD_SEED")
    p.add_argument("--folds", type=int, nargs="+", default=None, help="default: every fold")
    p.add_argument("--seed", type=int, default=0, help="run label (and PCA random_state)")
    p.add_argument("--covariate-features", nargs="+", default=["age", "sex"],
                   help="must equal the CNN runs' COVARIATES for a like-for-like comparison")
    p.add_argument("--pca", type=int, default=32)
    p.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID))
    p.add_argument("--alpha-grid", type=float, nargs="+", default=list(ALPHA_GRID))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--external-dataset", default=None, choices=list(DATASETS))
    p.add_argument("--external-root", default=None)
    p.add_argument("--external-features", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if bool(args.external_root) != bool(args.external_features) or \
            bool(args.external_root) != bool(args.external_dataset):
        raise SystemExit("[fatal] --external-dataset, --external-root and --external-features go together")
    if args.external_root and (args.external_dataset == args.dataset or
                               os.path.realpath(args.external_root) == os.path.realpath(args.root)):
        raise SystemExit("[fatal] the external set must be another dataset than the training one")

    from dataset import LABEL_REGISTRY
    if args.task not in LABEL_REGISTRY:
        raise SystemExit(f"[fatal] unknown task {args.task!r}")
    src = load_table(args.root, args.dataset, args.task, camera=args.camera, image_ext=args.image_ext)
    src["root"] = args.root
    files, feats, fmeta = load_features(args.features)
    if fmeta.get("root") and os.path.realpath(fmeta["root"]) != os.path.realpath(args.root):
        print(f"[warn] features were extracted from {fmeta['root']}, this run reads {args.root}; "
              f"rows are matched by file name")
    index = {f: i for i, f in enumerate(files)}
    print(f"[info] {args.task} on {args.dataset}{f' ({args.camera})' if args.camera else ''}: "
          f"{len(src['df'])} table rows; {len(files)} embeddings from {args.features}")
    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    for k in folds:
        if not 0 <= k < args.n_folds:
            raise SystemExit(f"[fatal] fold {k} is outside 0..{args.n_folds - 1}")
        run_fold(args, src, index, feats, fmeta, k)
    if LABEL_REGISTRY[args.task].num_classes is None:
        summarize_regression(os.path.join(args.out, args.task), args.cond, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
