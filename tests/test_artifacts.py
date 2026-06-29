import numpy as np

from artifacts import (focus_score, detect_glare, assess_quality, preprocess,
                       normalize_illumination, suppress_glare)


def _sharp_image(h=128, w=128):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, ::4] = 200            # high-frequency stripes -> sharp
    return img


def _blurry_version(rgb):
    from PIL import Image, ImageFilter
    from PIL import Image as I
    return np.asarray(I.fromarray(rgb).filter(ImageFilter.GaussianBlur(4)))


def test_focus_sharper_than_blurry():
    sharp = _sharp_image()
    blurry = _blurry_version(sharp)
    assert focus_score(sharp) > focus_score(blurry)


def test_glare_detection():
    img = np.full((100, 100, 3), 60, np.uint8)     # dim, no glare
    _, frac0 = detect_glare(img)
    img[40:60, 40:60] = 255                          # bright desaturated patch = glare
    mask, frac1 = detect_glare(img)
    assert frac1 > frac0
    assert mask[50, 50]


def test_assess_quality_keys_and_flag():
    img = np.zeros((120, 140, 3), np.uint8)
    img[20:100, 30:110] = np.random.randint(40, 200, (80, 80, 3))   # FOV with texture
    q = assess_quality(img)
    for k in ("focus", "glare_frac", "illumination_spread", "quality_score", "gradable", "reasons"):
        assert k in q
    assert isinstance(q["gradable"], bool)
    assert 0.0 <= q["quality_score"] <= 1.0


def test_preprocess_outputs_uint8_rgb():
    img = np.zeros((150, 150, 3), np.uint8)
    img[20:130, 20:130] = np.random.randint(30, 220, (110, 110, 3))
    out = preprocess(img)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3


def test_normalize_and_suppress_preserve_shape():
    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    assert normalize_illumination(img).shape == img.shape
    mask = np.zeros((64, 64), bool); mask[10:20, 10:20] = True
    assert suppress_glare(img, mask).shape == img.shape
