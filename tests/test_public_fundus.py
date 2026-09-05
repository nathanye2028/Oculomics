"""CPU-only synthetic tests for the public glaucoma / AMD adapters (AIROGS, REFUGE,
PAPILA, ODIR-5K), the load_any dispatch, nested image layouts in MBRSETDataset,
score_external.score_checkpoint and summarize_external."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brset_dataset import DATASETS, load_any, load_brset
from dataset import LABEL_REGISTRY, MBRSETDataset, stratified_split
from model import MBRSETClassifier
from public_fundus import (PUBLIC_DATASETS, _odir_eye_labels, load_airogs, load_odir,
                           load_papila, load_public, load_refuge)
from score_external import score_checkpoint
from summarize_external import summarize


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.zeros((48, 56, 3), np.uint8); a[6:42, 8:48] = 120
    Image.fromarray(a).save(path)


# ---- AIROGS ------------------------------------------------------------------- #
def test_airogs_nested_images_and_rg_labels(tmp_path):
    pd.DataFrame({"challenge_id": ["TRAIN000001", "TRAIN000002", "TRAIN000003"],
                  "class": ["RG", "NRG", "weird"]}).to_csv(tmp_path / "train_labels.csv", index=False)
    _img(tmp_path / "train" / "0" / "TRAIN000001.jpg"); _img(tmp_path / "train" / "1" / "TRAIN000002.jpg")
    src = load_airogs(str(tmp_path))
    df = src["df"]
    assert src["source"] == "airogs" and src["images_dir"].endswith("train")
    assert list(df["glaucoma"].fillna(-1)) == [1.0, 0.0, -1]           # unknown class -> NaN
    assert df["file"][0].replace("\\", "/") == "0/TRAIN000001.jpg"    # nested relpath kept
    assert df["file"][2] == "TRAIN000003.jpg"                          # not on disk: id + ext
    # the dataset keeps nested files under drop_missing_files and reads them
    ds = MBRSETDataset(csv=df, images_dir=src["images_dir"], task="glaucoma", split="val",
                       image_size=32, drop_missing_files=True)
    assert len(ds) == 2 and ds[0]["image"].shape == (3, 32, 32)


def test_airogs_light_folder_layout_reads_one_release(tmp_path):
    for rel in ("release-crop", "release-raw", "release-pad"):
        _img(tmp_path / rel / "train" / "RG" / "a.jpg"); _img(tmp_path / rel / "train" / "NRG" / "b.jpg")
        _img(tmp_path / rel / "validation" / "RG" / "c.jpg"); _img(tmp_path / rel / "test" / "NRG" / "d.jpg")
    src = load_airogs(str(tmp_path))
    df = src["df"]
    assert src["release"] == "release-crop" and len(df) == 4               # one release, not three
    assert dict(zip(df["patient"], df["glaucoma"])) == {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    assert set(df["source_split"]) == {"train", "validation", "test"}
    assert all(f.startswith(("train", "validation", "test")) for f in df["file"])
    assert load_airogs(str(tmp_path), airogs_release="release-raw")["images_dir"].endswith("release-raw")
    with pytest.raises(FileNotFoundError):
        load_airogs(str(tmp_path), airogs_release="release-nope")
    ds = MBRSETDataset(csv=df, images_dir=src["images_dir"], task="glaucoma", split="val",
                       image_size=32, drop_missing_files=True)
    assert len(ds) == 4 and ds.class_counts().tolist() == [2, 2]
    # no release dirs, plain RG/NRG folders also work
    _img(tmp_path / "flat" / "RG" / "x.jpg"); _img(tmp_path / "flat" / "NRG" / "y.jpg")
    flat = load_airogs(str(tmp_path / "flat"))
    assert flat["release"] is None and sorted(flat["df"]["glaucoma"]) == [0.0, 1.0]


def test_airogs_light_v2_metadata_csv_prefers_filename_over_numeric_id(tmp_path):
    d = tmp_path / "eyepac-light-v2-512-jpg"
    _img(d / "train" / "RG" / "EyePACS-TRAIN-RG-2580.jpg"); _img(d / "validation" / "NRG" / "EyePACS-TRAIN-NRG-11.jpg")
    pd.DataFrame({"id": [2580, 11], "file_name": ["EyePACS-TRAIN-RG-2580.jpg", "EyePACS-TRAIN-NRG-11.jpg"],
                  "label": ["RG", "NRG"], "label_binary": [1, 0], "folder": ["train", "validation"]}
                 ).to_csv(d / "metadata.csv", index=False)
    src = load_airogs(str(tmp_path))
    df = src["df"]
    assert src["csv"].endswith("metadata.csv") and len(df) == 2
    assert df["file"].str.startswith("eyepac-light-v2-512-jpg/").all()      # found via the filename, not the id
    assert list(df["glaucoma"]) == [1.0, 0.0] and list(df["source_split"]) == ["train", "validation"]
    assert list(df["patient"]) == ["EyePACS-TRAIN-RG-2580", "EyePACS-TRAIN-NRG-11"]


# ---- REFUGE ------------------------------------------------------------------- #
def test_refuge_folder_labels_table_labels_and_mask_skipping(tmp_path):
    _img(tmp_path / "Training400" / "Glaucoma" / "g0001.jpg")
    _img(tmp_path / "Training400" / "Non-Glaucoma" / "n0001.jpg")
    _img(tmp_path / "Training400" / "Non-Glaucoma" / "n0002.jpg")
    _img(tmp_path / "Validation400" / "V0001.jpg"); _img(tmp_path / "Validation400" / "V0002.jpg")
    _img(tmp_path / "Validation400" / "V0003.jpg")                     # not in the table -> NaN
    _img(tmp_path / "Annotation-Training400" / "Disc_Cup_Masks" / "g0001.png")   # must be skipped
    pd.DataFrame({"ImgName": ["V0001.jpg", "V0002.jpg"], "Glaucoma Label": [1, 0]}).to_csv(
        tmp_path / "Validation400" / "Fovea_locations.csv", index=False)
    df = load_refuge(str(tmp_path))["df"]
    lab = dict(zip(df["patient"], df["glaucoma"].fillna(-1)))
    assert lab == {"g0001": 1.0, "n0001": 0.0, "n0002": 0.0, "V0001": 1.0, "V0002": 0.0, "V0003": -1}
    assert not any("Masks" in f for f in df["file"])


# ---- PAPILA ------------------------------------------------------------------- #
def _papila(tmp_path):
    cd = tmp_path / "ClinicalData"; cd.mkdir()
    for eye in ("od", "os"):
        rows = [["Patient data (title row)", "", "", ""], ["ID", "Age", "Gender", "Diagnosis"],
                ["#001", 60, 0, 0], ["#002", 70, 1, 1], ["#003", 65, 0, 2]]
        pd.DataFrame(rows).to_csv(cd / f"patient_data_{eye}.csv", index=False, header=False)
    for i in (1, 2, 3):
        _img(tmp_path / "FundusImages" / f"RET{i:03d}OD.jpg"); _img(tmp_path / "FundusImages" / f"RET{i:03d}OS.jpg")


def test_papila_header_row_both_eyes_and_suspect_policy(tmp_path):
    _papila(tmp_path)
    df = load_papila(str(tmp_path))["df"]
    assert len(df) == 6 and df["patient"].nunique() == 3 and set(df["laterality"]) == {"R", "L"}
    assert df.loc[df["patient"] == "003", "glaucoma"].isna().all()            # suspect excluded
    assert df.loc[df["patient"] == "002", "glaucoma"].eq(1.0).all()
    assert set(df["file"]) == {f"RET{i:03d}{e}.jpg" for i in (1, 2, 3) for e in ("OD", "OS")}
    assert load_papila(str(tmp_path), papila_suspect="positive")["df"].query("patient == '003'")["glaucoma"].eq(1.0).all()
    assert load_papila(str(tmp_path), papila_suspect="negative")["df"].query("patient == '003'")["glaucoma"].eq(0.0).all()
    with pytest.raises(ValueError):
        load_papila(str(tmp_path), papila_suspect="maybe")
    # both eyes of a patient never straddle the split
    sp = stratified_split(load_papila(str(tmp_path), papila_suspect="negative")["df"], task="glaucoma",
                          val_frac=0.34, test_frac=0.33, group_col="patient", seed=0)
    assert not (set(sp["train"]["patient"]) & set(sp["test"]["patient"]))


# ---- ODIR-5K ------------------------------------------------------------------ #
def test_odir_eye_labels_from_keywords():
    assert _odir_eye_labels("normal fundus") == (0.0, 0.0)
    assert _odir_eye_labels("glaucoma") == (1.0, 0.0)
    assert _odir_eye_labels("wet age-related macular degeneration，moderate non proliferative retinopathy") == (0.0, 1.0)
    g, a = _odir_eye_labels("low image quality，glaucoma")
    assert np.isnan(g) and np.isnan(a)
    assert all(np.isnan(v) for v in _odir_eye_labels(np.nan))


def test_odir_full_df_and_per_patient_sheet(tmp_path):
    # Kaggle full_df layout: one row per eye, `filename` says the side
    full = pd.DataFrame({"ID": [1, 1, 2], "Patient Age": [60, 60, 70], "Patient Sex": ["Male", "Male", "Female"],
                         "Left-Diagnostic Keywords": ["glaucoma", "glaucoma", "normal fundus"],
                         "Right-Diagnostic Keywords": ["normal fundus", "normal fundus", "dry age-related macular degeneration"],
                         "filename": ["1_left.jpg", "1_right.jpg", "2_right.jpg"]})
    full.to_csv(tmp_path / "full_df.csv", index=False)
    (tmp_path / "preprocessed_images").mkdir()
    df = load_odir(str(tmp_path))["df"]
    assert list(df["glaucoma"]) == [1.0, 0.0, 0.0] and list(df["amd"]) == [0.0, 0.0, 1.0]
    assert list(df["laterality"]) == ["L", "R", "R"] and list(df["sex"]) == [0.0, 0.0, 1.0]
    assert df["patient"].nunique() == 2 and "amd" in LABEL_REGISTRY and "glaucoma" in LABEL_REGISTRY
    # per-patient sheet layout expands to two rows per patient
    (tmp_path / "full_df.csv").unlink()
    pp = pd.DataFrame({"ID": [5], "Patient Age": [55], "Patient Sex": ["Female"],
                       "Left-Fundus": ["5_left.jpg"], "Right-Fundus": ["5_right.jpg"],
                       "Left-Diagnostic Keywords": ["suspected glaucoma"],
                       "Right-Diagnostic Keywords": ["lens dust"]})
    pp.to_csv(tmp_path / "data.csv", index=False)
    df = load_odir(str(tmp_path))["df"]
    assert len(df) == 2 and df["glaucoma"].tolist()[0] == 1.0 and np.isnan(df["glaucoma"].tolist()[1])


# ---- dispatch ------------------------------------------------------------------ #
def test_load_any_dispatches_public_sets_and_brset_carries_amd(tmp_path):
    assert set(PUBLIC_DATASETS) <= set(DATASETS) and {"mbrset", "brset"} <= set(DATASETS)
    _papila(tmp_path)
    assert load_any(str(tmp_path), "papila")["source"] == "papila"
    assert load_any(str(tmp_path), "papila", papila_suspect="positive")["df"]["glaucoma"].notna().all()
    with pytest.raises(ValueError):
        load_public(str(tmp_path), "nope")
    with pytest.raises(ValueError):
        load_any(str(tmp_path), "nope")
    raw = pd.DataFrame({"image_id": ["a", "b"], "patient_id": ["p1", "p2"], "DR_ICDR": [0, 1], "amd": [1, 0]})
    assert list(load_brset(raw)["amd"]) == [1.0, 0.0]


# ---- score_external ----------------------------------------------------------- #
def test_score_checkpoint_zero_shot_and_adabn_and_guards(tmp_path):
    torch.manual_seed(0)
    # a tiny REFUGE-style external set: 6 glaucoma, 6 non-glaucoma
    ext = tmp_path / "refuge"
    for i in range(6):
        _img(ext / "Glaucoma" / f"g{i}.jpg"); _img(ext / "Non-Glaucoma" / f"n{i}.jpg")
    m = MBRSETClassifier(num_classes=2, pretrained=False)
    ck = tmp_path / "ctrl_seed0.pt"
    torch.save({"model": m.state_dict(), "use_gcg": m.use_gcg,
                "args": {"task": "glaucoma", "root": str(tmp_path / "airogs"), "dataset": "airogs",
                         "backbone": "mobilenetv3_small", "image_size": 32, "seed": 0}}, ck)
    r = score_checkpoint(str(ck), str(ext), "refuge", bn_adapt=True, batch_size=4, num_workers=0,
                         device=torch.device("cpu"))
    assert r["n_external"] == 12 and r["external_pos"] == 6 and r["task"] == "glaucoma"
    assert 0.0 <= r["external"]["auroc"] <= 1.0 and r["bn_layers"] > 0
    assert r["external_bnadapt"] and 0.0 <= r["external_bnadapt"]["auroc"] <= 1.0 and r["bn_adapt_transductive"]
    with pytest.raises(SystemExit):                                     # wrong task
        score_checkpoint(str(ck), str(ext), "refuge", task="amd", device=torch.device("cpu"))
    with pytest.raises(SystemExit):                                     # its own training root
        score_checkpoint(str(ck), str(tmp_path / "airogs"), "airogs", device=torch.device("cpu"))


# ---- summarize_external ------------------------------------------------------- #
def _rec(d, cond, seed, ds, zs, ad):
    (d / f"{cond}_seed{seed}_on_{ds}.json").write_text(json.dumps({
        "external": {"auroc": zs, "n": 100}, "external_bnadapt": ({"auroc": ad, "n": 100} if ad is not None else None),
        "n_external": 100, "external_pos": 30}))


def test_summarize_external_pairs_per_dataset(tmp_path):
    for s, (c, k) in enumerate([(0.80, 0.84), (0.81, 0.85), (0.79, 0.82)]):
        _rec(tmp_path, "ctrl", s, "papila", c, c + 0.01); _rec(tmp_path, "kd", s, "papila", k, k + 0.01)
        _rec(tmp_path, "ctrl", s, "odir", c, None); _rec(tmp_path, "kd", s, "odir", k, None)
    rep = summarize(str(tmp_path), "ctrl", "kd")
    pz = rep["papila"]["paired_zero_shot"]
    assert pz["n"] == 3 and pz["mean_delta"] == pytest.approx((0.04 + 0.04 + 0.03) / 3) and pz["significant"]
    assert rep["papila"]["paired_bnadapt"]["mean_delta"] == pytest.approx(pz["mean_delta"])
    assert rep["odir"]["paired_bnadapt"] is None and rep["odir"]["n_external"] == 100
    assert (tmp_path / "summary_external.md").exists()
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        summarize(str(tmp_path / "empty"), "ctrl", "kd")
