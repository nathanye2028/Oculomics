import numpy as np
import torch
import torch.nn as nn

from fundus_utils import (fov_bbox, crop_to_fov, focal_tversky_loss, tversky_loss,
                          random_patch, make_rng, tiled_predict, seed_everything)


def test_fov_bbox_and_crop():
    img = np.zeros((100, 120, 3), np.uint8)
    img[20:80, 30:90] = 200
    assert fov_bbox(img) == (20, 80, 30, 90)
    cropped, box = crop_to_fov(img)
    assert cropped.shape[:2] == (60, 60)


def test_seg_losses_backward_and_range():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = (torch.rand(2, 3, 16, 16) > 0.9).float()
    for fn in (focal_tversky_loss, tversky_loss):
        loss = fn(logits, target)
        assert 0.0 <= loss.item() <= 1.0 + 1e-4
        loss.backward()
        assert logits.grad is not None
        logits.grad = None


def test_random_patch_shape_and_lesion_bias():
    img = np.random.randint(0, 255, (200, 240, 3), np.uint8)
    masks = np.zeros((4, 200, 240), np.uint8)
    masks[0, 100, 120] = 1
    rng = np.random.default_rng(0)
    ip, mp = random_patch(img, masks, 64, rng, fg_bias=1.0)
    assert ip.shape == (64, 64, 3) and mp.shape == (4, 64, 64)
    assert mp.sum() > 0       # fg_bias=1.0 -> patch centred on the lesion


def test_make_rng_reproducible():
    a = make_rng(42, 3).random()
    b = make_rng(42, 3).random()
    c = make_rng(42, 4).random()
    assert a == b and a != c


def test_tiled_predict_shape():
    seed_everything(0)
    model = nn.Conv2d(3, 4, 1)       # trivial [3,h,w] -> [4,h,w]
    img = torch.randn(3, 300, 360)
    out = tiled_predict(model, img, tile=128, overlap=32, device=torch.device("cpu"), tile_batch=4)
    assert out.shape == (4, 300, 360)
    assert (out >= 0).all() and (out <= 1).all()   # sigmoid probabilities
