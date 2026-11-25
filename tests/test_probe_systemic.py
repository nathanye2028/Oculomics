"""The frozen-RETFound probe: retfound.py loads a RETFound-style checkpoint
(teacher weights, backbone. prefix, w12/w3 names, another position grid) or
refuses it; the scaler -> PCA -> linear model folds into one linear map;
probe_systemic.py runs on the CNN's folds and writes rows summarize_systemic.py
pools unchanged; a continuous target gets ridge + R2; the probe teacher refuses a
student on another partition."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from test_folds import _brset_with_flags

SMALL = "vit_small_patch14_dinov2"


def _fake_retfound(path, image_size=56, drop=None, seed=0):
    """A RETFound-DINOv2-style checkpoint of a small ViT: weights under 'teacher',
    'backbone.' prefixed, mlp.fc1/fc2 named w12/w3, plus projection-head junk."""
    import timm
    torch.manual_seed(seed)
    m = timm.create_model(SMALL, pretrained=False, img_size=image_size, num_classes=0).eval()
    sd = {}
    for k, v in m.state_dict().items():
        if drop and k.startswith(drop):
            continue
        sd["backbone." + k.replace("mlp.fc1.", "mlp.w12.").replace("mlp.fc2.", "mlp.w3.")] = v
    sd["dino_head.mlp.0.weight"] = torch.zeros(3, 3)
    torch.save({"teacher": sd, "student": {"x": torch.zeros(1)}}, path)
    return m


def test_retfound_checkpoint_loads_exactly_and_refuses_partial(tmp_path):
    from retfound import build_encoder, embed
    ref = _fake_retfound(tmp_path / "rf.pth")
    enc, info = build_encoder(str(tmp_path / "rf.pth"), timm_name=SMALL, image_size=56)
    x = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        assert torch.allclose(embed(enc, x, "cls"), ref(x), atol=1e-6)
        assert embed(enc, x, "mean").shape == (2, info["feat_dim"])
    assert not any(p.requires_grad for p in enc.parameters()) and not enc.training
    # trained on another grid: the position embedding is resampled, not dropped
    _fake_retfound(tmp_path / "rf70.pth", image_size=70)
    build_encoder(str(tmp_path / "rf70.pth"), timm_name=SMALL, image_size=56)
    # a missing transformer block is fatal: the features would be partly random
    _fake_retfound(tmp_path / "bad.pth", drop="blocks.3.")
    with pytest.raises(SystemExit, match="missing"):
        build_encoder(str(tmp_path / "bad.pth"), timm_name=SMALL, image_size=56)
    with pytest.raises(SystemExit, match="neither"):
        build_encoder(str(tmp_path / "nope.pth"), timm_name=SMALL, image_size=56)


def test_collapsed_probe_equals_the_pipeline():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from retfound import collapse_probe
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 40)) * rng.uniform(0.5, 3, 40) + rng.normal(size=40)
    y = (X[:, 0] - X[:, 3] + rng.normal(size=300) > 0).astype(int)
    sc = StandardScaler().fit(X)
    pca = PCA(8, random_state=0).fit(sc.transform(X))
    for clf in (LogisticRegression(class_weight="balanced").fit(pca.transform(sc.transform(X)), y),
                Ridge(alpha=3.0).fit(pca.transform(sc.transform(X)), X[:, 1])):
        w, b = collapse_probe(sc, pca, clf)
        Z = pca.transform(sc.transform(X))
        ref = clf.decision_function(Z) if hasattr(clf, "predict_proba") else clf.predict(Z)
        assert np.allclose(X @ w + b, ref, atol=1e-8)


def _features_for(root, out, signal=3.0, seed=0, d=24, drop=None):
    """An extract_retfound.py-format file whose first dimension carries the label."""
    lab = pd.read_csv(root / "labels_brset.csv")
    rng = np.random.default_rng(seed)
    pid_htn = {r.patient_id: r.comorbidities == "hypertension" for r in lab.itertuples()}
    files, feats = [], []
    for r in lab.itertuples():
        f = rng.normal(size=d)
        f[0] += signal * pid_htn[r.patient_id]
        files.append(f"{r.image_id}.jpg")
        feats.append(f)
    if drop:
        keep = [i for i, f in enumerate(files) if f != drop]
        files, feats = [files[i] for i in keep], [feats[i] for i in keep]
    meta = {"weights": "fake", "weights_path": "/nowhere/rf.pth", "timm_name": SMALL,
            "image_size": 56, "pool": "cls", "root": str(root)}
    np.savez(out, files=np.asarray(files), feats=np.asarray(feats, np.float32), meta=json.dumps(meta))
    return out


def test_probe_runs_on_the_cnn_folds_and_pools_unchanged(tmp_path):
    import probe_systemic
    from dataset import fold_split
    from summarize_systemic import summarize
    from train_mbrset import load_table
    root = _brset_with_flags(tmp_path, n_patients=150)
    feats = _features_for(root, tmp_path / "f.npz")
    out = tmp_path / "exp"
    argv = ["--dataset", "brset", "--root", str(root), "--camera", "Canon CR", "--task", "cm_hypertension",
            "--features", str(feats), "--out", str(out), "--n-folds", "3",
            "--covariate-features", "age", "sex", "--n-boot", "50"]
    assert probe_systemic.main(argv) == 0
    d = out / "cm_hypertension"
    table = load_table(str(root), "brset", "cm_hypertension", camera="Canon CR")["df"]
    for k in range(3):
        rec = json.loads((d / f"probe_fold{k}_seed0.json").read_text())
        pr = pd.read_csv(d / f"probe_fold{k}_seed0_test.csv")
        assert rec["fold"] == k and rec["n_folds"] == 3 and rec["fold_seed"] == 42
        # the same columns train_mbrset.py writes (minus the ablation scores)
        assert {"file", "patient", "label", "p_image", "age", "sex", "p_covariates", "p_combined",
                "task", "fold", "seed", "in_no_diabetes_dr0", "in_normal_looking"} <= set(pr.columns)
        # fold k's test patients are exactly the CNN's fold k test patients
        cnn = fold_split(table, task="cm_hypertension", fold=k, n_folds=3, seed=42)["test"]
        assert set(pr["patient"].astype(str)) == set(cnn["patient"].astype(str))
        assert rec["probe"]["pca"] <= 24 and rec["test"]["auroc"] > 0.75
        assert (d / f"probe_fold{k}_seed0_probe.npz").exists()
    rep = summarize(str(out), ["cm_hypertension"], control="probe", n_boot=50)
    pooled = rep["cm_hypertension"]["pooled_out_of_fold"]["seeds"][0]
    assert pooled["n_folds"] == 3 and "delta" in pooled["blocks"]["all"]
    assert "concordance" in pooled["matched_pairs"]["all"]["intact"]
    # a rerun skips finished folds; features from another table are refused
    assert probe_systemic.main(argv) == 0
    bad = _features_for(root, tmp_path / "g.npz", drop="img00002_1.jpg")
    with pytest.raises(SystemExit, match="no embedding"):
        probe_systemic.main(argv[:9] + [str(bad), "--out", str(tmp_path / "exp2"), "--n-folds", "3",
                                        "--covariate-features", "age", "sex"])


def test_regression_probe_reports_r2_gain(tmp_path, monkeypatch):
    import probe_systemic
    from covariate_baseline import incremental_value_regression
    rng = np.random.default_rng(1)
    n = 400
    age = rng.uniform(40, 80, n)
    latent = rng.normal(size=n)
    sbp = 90 + 0.6 * age + 8 * latent + rng.normal(scale=4, size=n)
    df = pd.DataFrame({"patient": np.arange(n).astype(str), "age": age, "sbp": sbp})
    pred = sbp.mean() + 7 * latent + rng.normal(scale=3, size=n)    # the photo sees the latent part
    v, t = df.iloc[:150], df.iloc[150:]
    inc = incremental_value_regression(v, pred[:150], t, pred[150:], "sbp", features=("age",), n_boot=100)
    b = inc["test"]
    assert b["r2_combined"] > b["r2_covariates"] + 0.2 and b["delta_ci"][0] > 0
    assert len(inc["_p_cov_rows"]) == len(t)
    assert incremental_value_regression(v, pred[:150], t, pred[150:], "htn_measured")["reason"]


def test_probe_teacher_refuses_another_partition(tmp_path, monkeypatch):
    import os
    import train_mbrset
    from retfound import ProbeTeacher, build_encoder, save_probe_head
    root = _brset_with_flags(tmp_path, n_patients=20)
    _fake_retfound(tmp_path / "rf.pth")
    enc, _ = build_encoder(str(tmp_path / "rf.pth"), timm_name=SMALL, image_size=56)
    w = np.random.default_rng(0).normal(size=enc.num_features)
    t = ProbeTeacher(enc, w, 0.3, "cls", 56)
    logits, f = t.forward_with_feat(torch.randn(2, 3, 112, 112))       # any student size
    assert logits.shape == (2, 2) and torch.all(logits[:, 0] == 0) and f.shape[1] == t.feat_dim
    meta = {"kind": "classifier", "task": "cm_hypertension", "dataset": "brset", "root": str(tmp_path / "elsewhere"),
            "label_csv": os.path.realpath(root / "labels_brset.csv"),
            "camera": "Canon CR", "fold": 1, "n_folds": 5, "fold_seed": 42, "val_auroc": 0.7,
            "features_meta": {"weights_path": str(tmp_path / "rf.pth"), "timm_name": SMALL,
                              "image_size": 56, "pool": "cls"}}
    save_probe_head(str(tmp_path / "p.npz"), w, 0.3, meta)
    args = type("A", (), dict(teacher=str(tmp_path / "p.npz"), teacher_weights=None, image_ext=".jpg",
                              task="cm_hypertension", dataset="brset", camera="Canon CR",
                              fold=1, n_folds=5, fold_seed=42, root=str(root), seed=0))()
    # same label table (the recorded root string differs: only the resolved CSV counts)
    teacher, name, t_args, _ = train_mbrset.load_probe_as_teacher(args, torch.device("cpu"), 2)
    assert name == "retfound_probe" and t_args["fold"] == 1
    args.fold = 2
    with pytest.raises(SystemExit, match="different partition"):
        train_mbrset.load_probe_as_teacher(args, torch.device("cpu"), 2)
    # weights whose sha256 differs from the ones the embeddings came from are refused
    meta["features_meta"]["weights_sha256"] = "0" * 64
    save_probe_head(str(tmp_path / "q.npz"), w, 0.3, meta)
    args.teacher, args.fold = str(tmp_path / "q.npz"), 1
    with pytest.raises(SystemExit, match="sha256"):
        train_mbrset.load_probe_as_teacher(args, torch.device("cpu"), 2)


def _mbrset_like(tmp_path, n=60, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    root = tmp_path / "mBRSET"
    (root / "images").mkdir(parents=True)
    rows = []
    for pid in range(n):
        htn = rng.random() < 0.6
        for e in (1, 2):
            f = f"{pid}.{e}.jpg"
            Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(root / "images" / f)
            rows.append({"file": f, "patient": pid, "age": float(rng.uniform(40, 80)), "sex": int(rng.integers(0, 2)),
                         "systemic_hypertension": "yes" if htn else "no", "final_icdr": 0})
    pd.DataFrame(rows).to_csv(root / "labels_mbrset.csv", index=False)
    feats = np.stack([np.r_[3.0 * (r["systemic_hypertension"] == "yes"), rng.normal(size=23)] for r in rows])
    np.savez(tmp_path / "mf.npz", files=np.asarray([r["file"] for r in rows], dtype=str),
             feats=feats.astype(np.float32), meta=json.dumps({"weights": "fake", "pool": "cls"}))
    return root, tmp_path / "mf.npz"


def test_probe_scores_an_external_set_into_a_fresh_folder(tmp_path):
    import probe_systemic
    root = _brset_with_flags(tmp_path, n_patients=150)
    feats = _features_for(root, tmp_path / "f.npz")
    mroot, mfeats = _mbrset_like(tmp_path)
    out = tmp_path / "fresh"                                            # does not exist yet
    base = ["--dataset", "brset", "--root", str(root), "--camera", "Canon CR", "--task", "cm_hypertension",
            "--features", str(feats), "--n-folds", "3", "--covariate-features", "age", "sex", "--n-boot", "30"]
    assert probe_systemic.main(base + ["--out", str(out), "--external-dataset", "mbrset", "--external-root",
                                       str(mroot), "--external-features", str(mfeats)]) == 0
    rec = json.loads((out / "cm_hypertension" / "probe_fold0_seed0.json").read_text())
    assert rec["external"]["dataset"] == "mbrset" and rec["external"]["auroc"] > 0.8
    assert (out / "cm_hypertension" / "probe_fold0_seed0_external.csv").exists()
    with pytest.raises(SystemExit, match="another dataset"):
        probe_systemic.main(base + ["--out", str(tmp_path / "x"), "--external-dataset", "brset",
                                    "--external-root", str(root), "--external-features", str(feats)])


def test_compare_probe_cnn_pairs_the_same_rows(tmp_path, capsys):
    import compare_probe_cnn
    import probe_systemic
    root = _brset_with_flags(tmp_path, n_patients=150)
    strong, weak = _features_for(root, tmp_path / "s.npz", signal=3.0), _features_for(root, tmp_path / "w.npz",
                                                                                      signal=0.0, seed=1)
    base = ["--dataset", "brset", "--root", str(root), "--camera", "Canon CR", "--task", "cm_hypertension",
            "--out", str(tmp_path / "exp"), "--n-folds", "3", "--covariate-features", "age", "sex", "--n-boot", "30"]
    probe_systemic.main(base + ["--features", str(strong)])
    probe_systemic.main(base + ["--features", str(weak), "--cond", "ctrl"])     # stands in for the CNN folds
    res = compare_probe_cnn.compare(str(tmp_path / "exp"), str(tmp_path / "exp"), ["cm_hypertension"],
                                    n_boot=30)["cm_hypertension"]
    assert res["paired"]["row_overlap"] == 1.0
    assert res["paired"]["image"]["delta"] > 0.2                         # the strong embedding wins
    assert {"probe", "cnn"} <= set(res) and "PROBE" in capsys.readouterr().out


def test_compare_refuses_incomplete_or_mismatched_arms(tmp_path, capsys):
    import compare_probe_cnn
    import probe_systemic
    root = _brset_with_flags(tmp_path, n_patients=150)
    feats = _features_for(root, tmp_path / "s.npz")
    base = ["--dataset", "brset", "--root", str(root), "--camera", "Canon CR", "--task", "cm_hypertension",
            "--features", str(feats), "--n-folds", "3", "--n-boot", "20"]
    probe_systemic.main(base + ["--out", str(tmp_path / "a"), "--covariate-features", "age", "sex"])
    probe_systemic.main(base + ["--out", str(tmp_path / "a"), "--covariate-features", "age", "sex", "--cond", "ctrl",
                                "--folds", "0", "1"])                           # fold 2 still "running"
    assert compare_probe_cnn.compare(str(tmp_path / "a"), str(tmp_path / "a"), ["cm_hypertension"], n_boot=20) == {}
    assert "incomplete: folds [0, 1] of 3" in capsys.readouterr().out
    probe_systemic.main(base + ["--out", str(tmp_path / "b"), "--covariate-features", "age", "sex"])
    probe_systemic.main(base + ["--out", str(tmp_path / "b"), "--covariate-features", "age", "--cond", "ctrl"])
    assert compare_probe_cnn.compare(str(tmp_path / "b"), str(tmp_path / "b"), ["cm_hypertension"], n_boot=20) == {}
    assert "differ in" in capsys.readouterr().out
