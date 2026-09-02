# legacy/

Superseded scripts kept for reference only. Nothing imports them and pytest
does not collect them (`pyproject.toml` sets `testpaths = ["tests"]`).

| file | superseded by | why it is here |
|---|---|---|
| `train.py` | `train_mbrset.py` | selects on val accuracy, no patient grouping, no AMP/EMA |
| `train_seg.py`, `seg_dataset.py` | `train_idrid.py` + `idrid_dataset.py` / `fgadr_dataset.py` | RFMiD *placeholder* masks — plumbing only, never real lesions |
| `test_loader.py`, `test_classifier.py` | `tests/` | one-batch smoke tests on a Kaggle DR dataset the project no longer uses |

Each file has a `sys.path` shim so `python legacy/<file>.py` still runs from the repo root.
