"""GCG attention-map capture + save_gcg_maps.py (CPU, synthetic, no network)."""
import os

import numpy as np
import pytest
import torch

from gcg_blocks import GCG_VARIANTS
from model import build_classifier
from model_seg import build_model, collect_gcg_gates, gcg_gate_modules, record_gcg_gates
import save_gcg_maps as sgm


def _fundus(h=96, w=96, seed=0):
    """A fake fundus: orange disc on black, so FOV logic has something to find."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    disc = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.45 * min(h, w)) ** 2
    img = np.zeros((h, w, 3), np.uint8)
    img[disc] = (rng.integers(120, 200, (disc.sum(), 3))).astype(np.uint8)
    img[disc, 0] = 220
    return img


def test_unet_records_only_inside_context():
    net = build_model(num_classes=4, pretrained=False, use_gcg=True).eval()
    x = torch.randn(1, 3, 64, 64)
    names = [n for n, _ in gcg_gate_modules(net)]
    assert names == ["decoders.0.gcg", "decoders.1.gcg", "decoders.2.gcg", "decoders.3.gcg", "up_full.gcg"]
    with torch.no_grad():
        net(x)
    assert collect_gcg_gates(net) == {}                  # nothing recorded by default
    with record_gcg_gates(net), torch.no_grad():
        net(x)
        gates = collect_gcg_gates(net)
    assert list(gates) == names
    for n, g in gates.items():
        assert g["spatial"].shape[:2] == (1, 1) and g["channel"].ndim == 2
        assert 0 <= g["spatial"].min() and g["spatial"].max() <= 1
    # strides: 16, 8, 4, 2, 1 for the MobileNetV3 taps
    assert [64 // g["spatial"].shape[-1] for g in gates.values()] == [16, 8, 4, 2, 1]
    assert all(m.record_gates is False for _, m in gcg_gate_modules(net))
    # control has no gates at all
    assert gcg_gate_modules(build_model(num_classes=4, pretrained=False, use_gcg=False)) == []


@pytest.mark.parametrize("name,keys", [("attention", {"spatial"}), ("se", {"channel"}),
                                       ("cbam", {"spatial", "channel"}), ("none", set())])
def test_variants_record_what_they_compute(name, keys):
    blk = GCG_VARIANTS[name](8, 16).eval()
    skip, guide = torch.randn(2, 8, 12, 12), torch.randn(2, 16, 6, 6)
    blk.record_gates = True
    with torch.no_grad():
        blk(skip, guide)
    got = getattr(blk, "last_gate", None) or {}
    assert set(got) == keys
    if "spatial" in keys:
        assert got["spatial"].shape == (2, 1, 12, 12)
    if "channel" in keys:
        assert got["channel"].shape == (2, 8)


def test_classifier_gate_is_captured():
    net = build_classifier(num_classes=2, pretrained=False, use_gcg=True).eval()
    with record_gcg_gates(net), torch.no_grad():
        net(torch.randn(1, 3, 64, 64))
        gates = collect_gcg_gates(net)
    assert list(gates) == ["gcg"]
    assert 64 // gates["gcg"]["spatial"].shape[-1] == 16       # the stride-16 mid feature


def test_tiled_matches_whole_when_one_tile():
    torch.manual_seed(0)
    net = build_model(num_classes=2, pretrained=False, use_gcg=True).eval()
    x = torch.randn(3, 64, 64)
    p_whole, g_whole = sgm.infer_whole(net, x, torch.device("cpu"), "seg")
    p_tiled, g_tiled = sgm.infer_tiled(net, x, tile=64, overlap=0, device=torch.device("cpu"))
    assert torch.allclose(p_whole, p_tiled, atol=1e-5)
    for a, b in zip(g_whole, g_tiled):
        assert a["name"] == b["name"] and a["stride"] == b["stride"]
        assert np.allclose(a["spatial"], b["spatial"], atol=1e-4)
        assert np.allclose(a["channel"], b["channel"], atol=1e-5)


def test_end_to_end_on_a_folder(tmp_path):
    torch.manual_seed(0)
    net = build_model(num_classes=4, pretrained=False, use_gcg=True)
    ck = tmp_path / "gcg.pt"
    torch.save({"model": net.state_dict(), "arch": "gcg_unet", "lesions": ["MA", "HE", "EX", "SE"],
                "args": {"no_gcg": False, "gcg_variant": "baseline", "image_size": 64,
                         "patch_size": 0, "encoder": "mobilenetv3", "decoder": "dense"}}, ck)
    imgs = tmp_path / "imgs"; imgs.mkdir()
    for i in range(2):
        sgm.Image.fromarray(_fundus(seed=i)).save(imgs / f"img{i}.png")
    out = tmp_path / "maps"
    rc = sgm.main(["--checkpoint", str(ck), "--images", str(imgs), "--out-dir", str(out),
                   "--image-size", "64", "--separate-pngs", "--save-pred", "--png-size", "96",
                   "--device", "cpu"])
    assert rc == 0
    for i in range(2):
        z = np.load(out / f"img{i}.npz")
        assert list(z["gate_names"]) == ["decoders.0.gcg", "decoders.1.gcg", "decoders.2.gcg",
                                         "decoders.3.gcg", "up_full.gcg"]
        assert z["spatial_0"].shape == (4, 4) and z["spatial_4"].shape == (64, 64)
        assert z["channel_0"].ndim == 1 and z["pred"].shape == (4, 64, 64)
        assert (out / f"img{i}.png").exists() and (out / f"img{i}_gate4_stride1.png").exists()
    assert (out / "gate_stats.csv").exists() and (out / "run.json").exists()
    rows = open(out / "gate_stats.csv").read().splitlines()
    assert len(rows) == 1 + 2 * 5 and "sp_mean" in rows[0]


def test_stats_on_vs_off_lesion(tmp_path):
    img = _fundus()
    masks = np.zeros((1, 96, 96), np.uint8); masks[0, 40:56, 40:56] = 1
    hot = np.zeros((24, 24), np.float32); hot[10:14, 10:14] = 1.0          # stride-4 gate lit on the lesion
    gates = [{"name": "decoders.2.gcg", "spatial": hot, "channel": np.full(8, 0.5, np.float32), "stride": 4}]
    (row,) = sgm.gate_stats_rows("x", gates, img, masks, ["MA"])
    assert row["on_MA"] > row["off_MA"] and row["ratio_any_lesion"] > 1
    sgm.write_npz(str(tmp_path / "g.npz"), gates, (96, 96), ["MA"])
    z = np.load(tmp_path / "g.npz")
    assert z["strides"].tolist() == [4] and z["spatial_0"].dtype == np.float16


def test_no_gcg_checkpoint_is_refused(tmp_path):
    net = build_model(num_classes=4, pretrained=False, use_gcg=False)
    ck = tmp_path / "nogcg.pt"
    torch.save({"model": net.state_dict(), "arch": "gcg_unet", "lesions": ["MA", "HE", "EX", "SE"],
                "args": {"no_gcg": True, "image_size": 64, "patch_size": 0}}, ck)
    imgs = tmp_path / "imgs"; imgs.mkdir()
    sgm.Image.fromarray(_fundus()).save(imgs / "a.png")
    with pytest.raises(SystemExit, match="no GCG gates"):
        sgm.main(["--checkpoint", str(ck), "--images", str(imgs), "--out-dir", str(tmp_path / "o"),
                  "--device", "cpu"])
