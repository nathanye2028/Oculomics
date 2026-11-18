"""phone_sim_effects.py: the variant table from a finished checkpoint, end to end."""
import numpy as np
import pandas as pd
import pytest

import phone_sim as ps
import phone_sim_effects as pe


def test_variants_cover_every_stage_once():
    names = set()
    for sets, want in ((["drop"], {f"drop_{s}" for s in ps.STAGES}),
                       (["only"], {"only_field", "field+bloom"} | {f"only_{s}" for s in ps.STAGES[1:] if s != "bloom"}),
                       (["strength"], {"strength_0", "strength_0.25", "strength_0.5", "strength_0.75"}),
                       (["field"], {"whole_field"})):
        class _P:                                   # variants() only hands the profile on
            pass
        made = {}
        orig = pe.PhoneSimulator
        try:
            pe.PhoneSimulator = lambda profile, **kw: made.setdefault(len(made), kw) or kw
            got = pe.variants(_P(), 64, sets)
        finally:
            pe.PhoneSimulator = orig
        assert set(got) == want | {"full"}
        names |= set(got)
        if sets == ["drop"]:
            for s in ps.STAGES:
                assert s not in got[f"drop_{s}"]["stages"] and len(got[f"drop_{s}"]["stages"]) == len(ps.STAGES) - 1
        if sets == ["field"]:
            assert got["whole_field"]["field"] == "whole"
    assert len(names) == 1 + 7 + 7 + 4 + 1


def test_effects_end_to_end(tmp_path):
    from test_retinareach import BACKBONE, _make_brset, _make_mbrset
    from train_retinareach import main as train
    rng = np.random.default_rng(0)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 30, rng, camera_of=lambda pid: "Canon CR" if pid % 4 else "NIKON NF5050")
    _make_mbrset(M, 30, rng)
    prof = tmp_path / "ck_phone_sim" / "profile.json"
    assert ps.main(["fit", "--target-root", str(M), "--source-root", str(B), "--source-camera", "Canon CR",
                    "--image-size", "64", "--validate", "0", "--num-workers", "0", "--device", "cpu",
                    "--out", str(prof)]) == 0
    ck = tmp_path / "base.pt"
    assert train(["--source-root", str(B), "--target-root", str(M), "--source-camera", "Canon CR",
                  "--probe-targets", "hypertension", "--backbone", BACKBONE, "--no-pretrained", "--image-size", "64",
                  "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--calib-batch", "4",
                  "--calib-sizes", "4", "--calib-repeats", "1", "--envelope-repeats", "2", "--prior-strengths", "0",
                  "--min-probe-pos", "2", "--n-boot", "10", "--cv-repeats", "1", "--min-test-pos-patients", "2",
                  "--demo-n", "4", "--n-shuffle", "0", "--ablation-frac", "0", "--out", str(tmp_path / "out"),
                  "--ckpt", str(ck)]) == 0

    def run(out, *extra):
        assert pe.main(["--ckpt", f"canon_only={ck}", f"again={ck}", "--profile", str(prof), "--out", str(out),
                        "--num-workers", "0", "--batch-size", "8", "--device", "cpu", "--probe-n", "20",
                        *extra]) == 0
        return pd.read_csv(out / "effects.csv")

    df = run(tmp_path / "fx")
    want = {pe.BASE, pe.REAL, "full", "whole_field", "only_field", "field+bloom", "strength_0", "strength_0.75"}
    want |= {f"drop_{s}" for s in ps.STAGES}
    assert want <= set(df.variant) and set(df.stats) == set(pe.STATS) and (df.part == "val").all()
    from style_transfer_xfer import load_device
    n_val = len(load_device(str(B), "brset", 64, 42, camera="Canon CR")["val"])
    assert (df[df.variant != pe.REAL].n <= n_val).all() and df[df.variant == "full"].n.max() > 0
    # one checkpoint under two names: the same draws, so the same numbers, row for row
    num = ["auroc", "sensitivity", "specificity", "oracle_spec", "dlogit_icdr0", "dlogit_referable", "p_handheld_icdr0"]
    a, b = (df[df.arm == arm].set_index(["variant", "stats"])[num] for arm in ("canon_only", "again"))
    pd.testing.assert_frame_equal(a, b)
    # the unsimulated row is its own reference; a render moves the logits
    base = df[(df.variant == pe.BASE) & (df.arm == "canon_only")]
    assert np.allclose(base.dlogit_icdr0.dropna(), 0) and np.allclose(base.dlogit_referable.dropna(), 0)
    full = df[(df.variant == "full") & (df.arm == "canon_only") & (df.stats == pe.BASE)].iloc[0]
    assert np.isfinite(full.dlogit_icdr0) and abs(full.dlogit_icdr0) > 0
    assert 0 <= full.p_handheld_icdr0 <= 1 and df[df.stats == "handheld"].p_handheld_icdr0.isna().all()
    # dropping a stage changes the render, same draws otherwise (on this small fixture the
    # bloom is gated off and the bank carries no grain, so those two rows equal ``full``)
    tab = df[(df.arm == "canon_only") & (df.stats == pe.BASE)].set_index("variant")
    moved = lambda s: any(abs(tab.loc[f"drop_{s}", c] - tab.loc["full", c]) > 1e-6
                          for c in ("dlogit_icdr0", "dlogit_referable"))
    assert all(moved(s) for s in ("field", "light", "colour", "rim"))
    assert abs(tab.loc["drop_colour", "p_handheld_icdr0"] - tab.loc["full", "p_handheld_icdr0"]) > 0.1
    txt = (tmp_path / "fx" / "effects_summary.txt").read_text()
    assert "IN-SAMPLE" in txt and "canon_only, tabletop statistics" in txt and "drop_colour" in txt
    assert (tmp_path / "fx" / "grid.jpg").exists() and "full" in (tmp_path / "fx" / "grid.txt").read_text()

    t = run(tmp_path / "fx_test", "--part", "test", "--sets", "field", "--max-n", "6", "--grid-n", "0")
    assert set(t.variant) == {pe.BASE, pe.REAL, "full", "whole_field"} and (t.part == "test").all()
    assert (t.n <= 6).all() and not (tmp_path / "fx_test" / "grid.jpg").exists()
    assert "IN-SAMPLE" not in (tmp_path / "fx_test" / "effects_summary.txt").read_text()
    with pytest.raises(SystemExit):
        pe.main(["--ckpt", str(ck), "--profile", str(prof), "--out", str(tmp_path / "bad")])
