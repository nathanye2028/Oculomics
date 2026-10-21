"""Messidor-2 adapter (public_fundus.load_messidor2), on synthetic copies of the two
grade layouts in circulation: Krause et al.'s adjudicated messidor_data.csv next to
the release's IMAGES/ folder and its two-images-per-row examination sheet, and a
repackaging that renames the columns (id_code / diagnosis) and ships only a
preprocessed copy without the sheet."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from capability_gate import encoding_audit
from public_fundus import load_messidor2, main as inspect


def _img(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (16, 16), (120, 60, 30)).save(path)


def _krause_release(root):
    names = ["20051020_43808_0100_PP.png", "20051020_43832_0100_PP.png",
             "IM000001.JPG", "IM000002.JPG", "IM000003.JPG", "IM000004.JPG"]
    for n in names:
        _img(os.path.join(root, "IMAGES", n))
    _img(os.path.join(root, "preprocess", "IM000001.png"))     # a second copy of one stem
    pd.DataFrame({"image_id": names,
                  "adjudicated_dr_grade": [0, 2, 4, np.nan, 1, 3],
                  "adjudicated_dme": [0, 1, 0, np.nan, 0, 0],
                  "adjudicated_gradable": [1, 1, 1, 0, 1, 1]}
                 ).to_csv(os.path.join(root, "messidor_data.csv"), index=False)
    with open(os.path.join(root, "messidor-2.csv"), "w") as f:   # the release's sheet uses ';'
        f.write("left;right\n" + "\n".join(f"{names[i]};{names[i + 1]}" for i in (0, 2, 4)) + "\n")
    return names


def test_krause_grades_with_the_examination_sheet(tmp_path):
    root = str(tmp_path / "m2")
    names = _krause_release(root)
    src = load_messidor2(root)
    df = src["df"].set_index(src["df"]["file"].map(lambda f: os.path.basename(f)))
    assert len(df) == 6 and df["patient"].nunique() == 3          # three examinations of two eyes
    assert df.loc[names[0], "patient"] == df.loc[names[1], "patient"]
    assert df.loc[names[0], "laterality"] == "L" and df.loc[names[1], "laterality"] == "R"
    assert np.isnan(df.loc["IM000002.JPG", "final_icdr"])          # ungradable: no grade
    assert df.loc["IM000002.JPG", "final_quality"] == 0.0
    assert df.loc[names[1], "final_edema"] == 1.0 and df.loc[names[2], "final_icdr"] == 4.0
    # the original photo wins over the preprocessed copy of the same stem
    assert df.loc["IM000001.JPG", "file"] == os.path.join("IMAGES", "IM000001.JPG")
    assert all(os.path.isfile(os.path.join(src["images_dir"], f)) for f in src["df"]["file"])
    assert src["pairs_sheet"].endswith("messidor-2.csv") and src["missing_images"] == 0
    for t in ("dr_referable", "edema", "quality"):
        assert encoding_audit(src["df"], t)["ok"], t
    from brset_dataset import load_any
    assert len(load_any(root, "messidor2")["df"]) == 6
    assert inspect(["--root", root, "--dataset", "messidor2", "--strict"]) == 0


def test_renamed_columns_without_the_sheet(tmp_path, capsys):
    root = str(tmp_path / "pre")
    ids = ["a_PP.png", "b_PP.png", "c_PP.png", "d_PP.png"]
    for n in ids:
        _img(os.path.join(root, "messidor-2", "preprocess", n))
    pd.DataFrame({"id_code": ids, "diagnosis": [0, 2, 1, 3], "adjudicated_dme": [0, 1, 0, 0],
                  "adjudicated_gradable": [1, 1, 1, 1]}).to_csv(os.path.join(root, "messidor_data.csv"),
                                                                index=False)
    src = load_messidor2(root)
    df = src["df"]
    assert list(df["final_icdr"]) == [0.0, 2.0, 1.0, 3.0]
    assert df["patient"].nunique() == 4 and src["pairs_sheet"] is None   # every image its own patient
    assert src["preprocessed"] == 4
    assert inspect(["--root", root, "--dataset", "messidor2", "--strict"]) == 0
    out = capsys.readouterr().out
    assert "no examination sheet" in out and "preprocess" in out


def test_no_grades_table_is_a_clear_error(tmp_path):
    root = str(tmp_path / "bare")
    _img(os.path.join(root, "IMAGES", "IM000001.JPG"))
    # an original-Messidor style sheet (0-3 scale of its own) is NOT read as ICDR grades
    pd.DataFrame({"Image name": ["IM000001.JPG"], "Retinopathy grade": [3]}
                 ).to_csv(os.path.join(root, "Annotation_Base11.csv"), index=False)
    with pytest.raises(FileNotFoundError, match="no grades table"):
        load_messidor2(root)


def test_the_adjudicated_grades_win_over_other_tables(tmp_path):
    root = str(tmp_path / "both")
    names = _krause_release(root)
    pd.DataFrame({"id_code": names, "diagnosis": [4] * len(names)}
                 ).to_csv(os.path.join(root, "aaa_other.csv"), index=False)   # walked first
    src = load_messidor2(root)
    assert src["grade_column"] == "adjudicated_dr_grade"
    assert src["df"]["final_icdr"].max() == 4.0 and src["df"]["final_icdr"].min() == 0.0
