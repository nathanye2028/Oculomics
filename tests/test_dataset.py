import itertools

import numpy as np
import pandas as pd
from PIL import Image

from dataset import MBRSETDataset, stratified_split


def _make_data(tmp_path, n=48):
    img_dir = tmp_path / "images"; img_dir.mkdir()
    rng = np.random.default_rng(0); rows = []
    for i in range(n):
        a = np.zeros((64, 72, 3), np.uint8)           # black border -> FOV crop target
        a[10:54, 12:60] = rng.integers(40, 255, (44, 48, 3), dtype=np.uint8)
        Image.fromarray(a).save(img_dir / f"{i}.jpg")
        rows.append(dict(file=f"{i}.jpg", patient=f"p{i//3}", age=float(rng.integers(30, 80)),
                         laterality=rng.choice(["R", "L"]),
                         final_icdr=int(rng.integers(0, 5)), final_edema=rng.choice(["yes", "no"])))
    df = pd.DataFrame(rows)
    csv = tmp_path / "d.csv"; df.to_csv(csv, index=False)
    return df, str(img_dir), str(csv)


def test_multiclass_and_imbalance_helpers(tmp_path):
    df, images, _ = _make_data(tmp_path)
    ds = MBRSETDataset(csv=df, images_dir=images, task="dr_grade", split="val", image_size=64)
    assert ds[0]["image"].shape == (3, 64, 64)
    assert ds.num_classes == 5
    assert int(ds.class_counts().sum()) == len(ds)
    assert ds.sample_weights().shape[0] == len(ds)


def test_fov_crop_toggle(tmp_path):
    df, images, _ = _make_data(tmp_path)
    for fov in (True, False):
        ds = MBRSETDataset(csv=df, images_dir=images, task="dr_grade",
                           split="val", image_size=64, fov_crop=fov)
        assert ds[0]["image"].shape == (3, 64, 64)


def test_regression_and_label_map(tmp_path):
    df, images, _ = _make_data(tmp_path)
    da = MBRSETDataset(csv=df, images_dir=images, task="age", split="val", image_size=64)
    assert da.num_classes is None and da[0]["label"].dtype.is_floating_point
    dm = MBRSETDataset(csv=df, images_dir=images, task=None, label_col="laterality",
                       label_map={"R": 0, "L": 1}, split="val", image_size=64)
    assert dm.num_classes == 2


def test_stratified_split_no_patient_leakage(tmp_path):
    _, _, csv = _make_data(tmp_path)
    sp = stratified_split(csv, task="dr_referable", val_frac=0.2, test_frac=0.2)
    psets = {k: set(v["patient"]) for k, v in sp.items()}
    for a, b in itertools.combinations(psets, 2):
        assert not (psets[a] & psets[b])
