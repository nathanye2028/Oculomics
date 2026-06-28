import math

import torch

from metrics import quadratic_weighted_kappa, dice_iou_from_counts, binary_average_precision, CSVLogger


def test_kappa_perfect_agreement():
    y = [0, 1, 2, 3, 4, 0, 2]
    assert quadratic_weighted_kappa(y, y, 5) == 1.0


def test_kappa_off_by_more_is_worse():
    true = [0, 0, 4, 4]
    near = [0, 1, 4, 3]      # off by 1
    far = [4, 4, 0, 0]       # maximally wrong
    assert quadratic_weighted_kappa(near, true, 5) > quadratic_weighted_kappa(far, true, 5)


def test_dice_iou_from_counts():
    inter = torch.tensor([5.0]); psum = torch.tensor([10.0]); tsum = torch.tensor([10.0])
    dice, iou = dice_iou_from_counts(inter, psum, tsum, ["X"])
    assert abs(dice["X"] - (2 * 5) / (10 + 10)) < 1e-6      # 0.5
    assert abs(iou["X"] - 5 / (10 + 10 - 5)) < 1e-6         # 5/15


def test_auprc_handles_empty_positive():
    assert math.isnan(binary_average_precision([0.2, 0.8], [0, 0]))
    ap = binary_average_precision([0.9, 0.1, 0.8], [1, 0, 1])
    assert 0.0 <= ap <= 1.0


def test_csv_logger(tmp_path):
    p = tmp_path / "m.csv"
    log = CSVLogger(str(p))
    log.log({"epoch": 1, "loss": 0.5})
    log.log({"epoch": 2, "loss": 0.4})
    log.close()
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "epoch,loss" and len(lines) == 3
