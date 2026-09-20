"""lesion_burden.py: the grade-0 cohort, per-image scoring with a real (random-weight)
segmenter checkpoint, resume, and the mediation check on planted data."""
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lesion_burden as lb                                       # noqa: E402
from model_seg import build_model                                # noqa: E402


def _fundus(seed, h=96, w=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    disc = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.45 * min(h, w)) ** 2
    a = np.zeros((h, w, 3), np.uint8)
    a[disc] = rng.integers(90, 200, (int(disc.sum()), 3), dtype=np.uint8)
    return a


def _pred_table(n_pat=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        dm = float(p % 3 == 0); age = float(rng.integers(35, 80))
        grade = 2 if (dm and p % 12 == 0) else 0                  # a few diabetics with DR: must be excluded
        for eye in (1, 2):
            rows.append(dict(dataset="brset", file=f"img{p}_{eye}.jpg", patient=p, split="test",   # int ids, as in BRSET
                             cohort="healthy" if not dm else "nonhealthy", age=age, sex=float(p % 2),
                             dr_grade=grade, gradable=1.0 if (p + eye) % 10 else 0.0,
                             diabetes="yes" if dm else "no", dm_time=np.nan, final_edema="no", insulin="no",
                             camera="A" if p % 2 else "B", gap_corrected=2.0 * dm + rng.normal(0, 1.5)))
    return pd.DataFrame(rows)


def test_grade0_cohort_excludes_dr_and_ungradable_images():
    df = _pred_table()
    pat, rows = lb.grade0_cohort(df)
    assert (pat["dr_grade"] == 0).all() and pat["diabetes"].notna().all()
    assert set(rows["patient"]) == set(pat["patient"]) and (rows["gradable"] == 1.0).all()
    dr_patients = set(df.loc[df["dr_grade"] == 2, "patient"])
    assert dr_patients and not (dr_patients & set(pat["patient"]))


def test_score_image_and_resume(tmp_path):
    net = build_model(num_classes=4, pretrained=False, use_gcg=True).eval()
    lesions = ["MA", "HE", "EX", "SE"]
    imgs = tmp_path / "imgs"; imgs.mkdir()
    for i in range(3):
        Image.fromarray(_fundus(i)).save(imgs / f"img{i}.jpg")
    s = lb.score_image(net, str(imgs / "img0.jpg"), lesions, torch.device("cpu"), size=64)
    assert s["fov_px"] > 1000 and all(0.0 <= s[f"ppm_{c}"] <= 1e6 for c in lesions)
    assert 0.0 <= s["ppm_any"] <= 1e6 and all(0.0 <= s[f"pmax_{c}"] <= 1.0 for c in lesions)
    assert s["ppm_any"] >= max(s[f"ppm_{c}"] for c in lesions) - 1e-6
    t = lb.score_image(net, str(imgs / "img0.jpg"), lesions, torch.device("cpu"), tiled=True, tile=64, overlap=0)
    assert set(t) == set(s)
    rows = pd.DataFrame({"file": ["img0.jpg", "img1.jpg", "missing.jpg"], "patient": ["a", "a", "b"]})
    out = str(tmp_path / "scores.csv")
    sc = lb.score_all(net, lesions, rows, str(imgs), out, torch.device("cpu"), size=64)
    assert list(sc["file"]) == ["img0.jpg", "img1.jpg"]           # the missing file is skipped, not fatal
    rows2 = pd.concat([rows, pd.DataFrame({"file": ["img2.jpg"], "patient": ["b"]})])
    sc2 = lb.score_all(net, lesions, rows2, str(imgs), out, torch.device("cpu"), size=64)
    assert list(sc2["file"]) == ["img0.jpg", "img1.jpg", "img2.jpg"]   # resumed: only the new image was scored
    pb = lb.patient_burden(sc2, rows2)
    assert set(pb["patient"]) == {"a", "b"} and "ppm_any" in pb.columns


def _patients(n=900, seed=1, lesions_drive_gap=False):
    rng = np.random.default_rng(seed)
    dm = (rng.random(n) < 0.35).astype(float)
    age = rng.integers(35, 85, n).astype(float)
    # log-burden: independent of diabetes, or strongly raised by it
    logb = rng.normal(1.0, 0.5, n) + (1.2 * dm if lesions_drive_gap else 0.0)
    ppm = 10 ** np.clip(logb, 0, None) - 1.0
    gap = (2.5 * logb if lesions_drive_gap else 2.0 * dm) + rng.normal(0, 1.5, n)
    d = pd.DataFrame({"patient": [f"p{i}" for i in range(n)], "diabetes": dm, "age": age, "gap": gap,
                      "sex": (rng.random(n) < 0.5).astype(float), "camera": np.where(rng.random(n) < 0.6, "A", "B"),
                      "amd": (rng.random(n) < 0.2).astype(float)})
    for c in ("MA", "HE", "EX", "SE"):
        d[f"ppm_{c}"] = ppm * rng.uniform(0.8, 1.2, n)
    d["ppm_any"] = d[[f"ppm_{c}" for c in ("MA", "HE", "EX", "SE")]].sum(axis=1)
    return d


def test_mediation_separates_the_two_explanations():
    cols = [f"ppm_{c}" for c in ("MA", "HE", "EX", "SE")]
    a = lb.mediation(_patients(lesions_drive_gap=False), cols)
    assert 1.6 < a["before"]["d"] < 2.4 and abs(a["attenuation"]) < 0.15     # burden explains nothing
    b = lb.mediation(_patients(lesions_drive_gap=True), cols)
    assert b["before"]["d"] > 2.0 and b["attenuation"] > 0.7                 # burden explains nearly all
    assert lb.mediation(_patients(n=12), cols) is None or True               # tiny sets do not crash
    text = "\n".join(lb.burden_report(_patients(lesions_drive_gap=True), ["MA", "HE", "EX", "SE"]))
    assert "no other ophthalmic flag" in text and "attenuation" in text and "| any |" in text


def test_end_to_end_cli(tmp_path):
    """BRSET tree + pooled predictions + a seg checkpoint -> the three-table report, resumable."""
    df = _pred_table(n_pat=60)
    df.insert(0, "condition", "medium_mix"); df["n_seeds"] = 3
    pred = tmp_path / "predictions_pooled.csv"; df.to_csv(pred, index=False)
    B = tmp_path / "BRSET"; (B / "fundus_photos").mkdir(parents=True)
    for i, f in enumerate(df["file"].unique()):
        Image.fromarray(_fundus(i)).save(B / "fundus_photos" / f)
    raw = pd.DataFrame({"image_id": [f[:-4] for f in df["file"].unique()], "patient_id": 0, "DR_ICDR": 0,
                        "amd": [int(i % 5 == 0) for i in range(df["file"].nunique())]})
    raw.to_csv(B / "labels_brset.csv", index=False)
    net = build_model(num_classes=4, pretrained=False, use_gcg=True)
    ck = tmp_path / "seg.pt"
    torch.save({"model": net.state_dict(), "arch": "gcg_unet", "lesions": ["MA", "HE", "EX", "SE"],
                "args": {"no_gcg": False, "gcg_variant": "baseline", "image_size": 64, "patch_size": 0,
                         "encoder": "mobilenetv3", "decoder": "dense"}}, ck)
    out = tmp_path / "lb"
    argv = ["--seg-checkpoint", str(ck), "--predictions", str(pred), "--condition", "medium_mix", "--root", str(B),
            "--brset-csv", str(B / "labels_brset.csv"), "--size", "64", "--out", str(out), "--device", "cpu"]
    assert lb.main(argv) == 0
    md = (out / "lesion_burden.md").read_text()
    assert "### 1. Do grade-0 diabetics carry more predicted lesion?" in md and "### 3. Does burden explain" in md
    assert "no other ophthalmic flag" in md
    imgs = pd.read_csv(out / "lesion_burden_images.csv"); pats = pd.read_csv(out / "lesion_burden_patients.csv")
    assert len(imgs) > 50 and imgs["file"].is_unique and {"ppm_MA", "ppm_any", "gap", "diabetes"} <= set(pats.columns)
    n = len(imgs)
    assert lb.main(argv) == 0 and len(pd.read_csv(out / "lesion_burden_images.csv")) == n     # resume: nothing rescored
