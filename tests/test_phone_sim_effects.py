"""phone_sim_effects.py: the variant table from a finished checkpoint, end to end."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

import phone_sim as ps
import phone_sim_effects as pe


def test_variants_cover_every_stage_once():
    names = set()
    for sets, want in ((["drop"], {f"drop_{s}" for s in ps.STAGES}),
                       (["only"], {"only_field", "field+bloom"} | {f"only_{s}" for s in ps.STAGES[1:] if s != "bloom"}),
                       (["strength"], {"strength_0", "strength_0.25", "strength_0.5", "strength_0.75"}),
                       (["field"], {"whole_field"})):
        class _P:                                   # variants() only hands the profile on (no ranks to re-read)
            params = {"detail_by": "band", "fill": 0.18, "prefilter": "matched"}
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


def test_render_variants_flip_each_option_the_bank_can_serve():
    from test_phone_sim import _as_v2_blob, _profile
    new = _profile()
    old = ps.PhoneProfile(new.image_size, _as_v2_blob(new)["bank"], _as_v2_blob(new)["params"], format="phone_sim/2")
    assert pe.render_variants(old) == {"fill_0.18": {"fill": 0.18}, "prefilter_matched": {"prefilter": "matched"},
                                       "render_v3": {"fill": 0.18, "prefilter": "matched"}}      # no edge reading
    assert pe.render_variants(new) == {"fill_0": {"fill": 0.0}, "prefilter_gauss": {"prefilter": "gauss"},
                                       "detail_by_band": {"detail_by": "band"}, "detail_by_edge": {"detail_by": "edge"},
                                       "render_v2": {"fill": 0.0, "prefilter": "gauss", "detail_by": "band"}}
    # a fit that could not rank its renders falls back to the drawn photo's reading: still the new render
    drawn = new.with_params(detail_by="edge", edge_src=None)
    assert pe.render_variants(drawn).keys() == {"fill_0", "prefilter_gauss", "detail_by_band", "render_v2"}
    # the old render's options on a /3 bank: the whole new render is its newest criterion too
    assert pe.render_variants(new.with_params(**ps.LEGACY_PARAMS))["render_v3"] == {
        "fill": 0.18, "prefilter": "matched", "detail_by": "rank"}
    # a bank without an edge reading keeps the band criterion in every render row, and that
    # stage's blur reacts to what reaches it: each row has a twin without the detail stage,
    # next to the profile's own render without it
    v = pe.variants(old, 64, ["render"])
    rows = {"fill_0.18", "prefilter_matched", "render_v3"}
    assert set(v) == {"full", "drop_detail"} | rows | {f"{r}_no_detail" for r in rows}
    assert v["full"].p["fill"] == 0.0 and v["render_v3"].p["fill"] == 0.18 and v["render_v3"].p["detail_by"] == "band"
    no_detail = [s for s in ps.STAGES if s != "detail"]
    for r in rows:
        assert list(v[r].stages) == list(ps.STAGES) and list(v[f"{r}_no_detail"].stages) == no_detail, r
        assert v[f"{r}_no_detail"].p == v[r].p
    assert list(v["drop_detail"].stages) == no_detail and v["drop_detail"].p == v["full"].p
    assert set(pe.variants(old, 64, ["render", "drop"])) >= {"drop_detail", "drop_noise", "render_v3_no_detail"}
    assert not [n for n in pe.variants(new, 64, ["render"]) if n.endswith("_no_detail") or n == "drop_detail"]
    # one option flipped by hand: neither generation, so both flipped (the criterion stays)
    assert pe.render_variants(new.with_params(fill=0.0))["render_flipped"] == {"fill": 0.18, "prefilter": "gauss"}
    # the generation is the field and the resampling: a rank profile switched to another
    # criterion is still the new render, and its other generation the old one
    for by in ("edge", "band"):
        rv = pe.render_variants(new.with_params(detail_by=by))
        assert rv["render_v2"] == ps.LEGACY_PARAMS and "render_flipped" not in rv and "detail_by_rank" in rv
    # ``fit --render v2`` on a /3 bank: an edge reading, but no reading of the renders. The
    # whole new render is by rank only when the caller re-reads the ranks (photos at hand);
    # short of that the row says which criterion it is
    v2fit = new.with_params(**ps.LEGACY_PARAMS, edge_src=None, edge_src_whole=None)
    rv = pe.render_variants(v2fit)
    assert rv.keys() == {"fill_0.18", "prefilter_matched", "detail_by_edge", "render_v3_edge"}
    assert rv["render_v3_edge"] == {"fill": 0.18, "prefilter": "matched", "detail_by": "edge"}
    rv = pe.render_variants(v2fit, can_read=True)
    assert rv.keys() == {"fill_0.18", "prefilter_matched", "detail_by_edge", "detail_by_rank", "render_v3"}
    assert rv["render_v3"] == {"fill": 0.18, "prefilter": "matched", "detail_by": "rank"}
    assert "detail_by_rank" in pe.render_variants(drawn, can_read=True)


def test_variants_rank_against_a_reading_of_their_own_chain():
    """``detail_by: rank`` ranks a render against fit's reading of its own renders. A
    variant whose renders reach the detail stage differently (another prefilter or
    scale, a stage dropped, a lower strength) is ranked against a reading of ITS chain,
    taken on the calibration photos; one that changes nothing before the detail stage
    keeps the profile's."""
    from test_phone_sim import _fundus, _profile
    size, quiet = 64, lambda t: None
    prof = _profile(size)
    cal = _fundus(8, 96, seed=9)
    own, own_whole = prof.params["edge_src"], prof.params["edge_src_whole"]
    assert prof.params["detail_by"] == "rank" and own and own_whole and own != own_whole
    said = []
    v = pe.variants(prof, size, ["drop", "strength", "field", "render"], cal=cal, seed=3, log=said.append)
    for name in ("full", "drop_noise", "drop_detail", "detail_by_edge", "detail_by_band", "render_v2", "whole_field"):
        assert v[name].p["edge_src"] == own and v[name].p["edge_src_whole"] == own_whole, name
    read = lambda change=None, **kw: ps.calibrate_detail(prof.with_params(**(change or {})), cal, size, 3, log=quiet, **kw)
    without = lambda s: [t for t in ps.STAGES if t != s]
    fitted = read()                                                        # the fitted chain on the same photos
    for name, want in (("prefilter_gauss", read({"prefilter": "gauss"})), ("fill_0", read({"fill": 0.0})),
                       ("drop_colour", read(stages=without("colour"))), ("drop_field", read(stages=without("field"))),
                       ("drop_light", read(stages=without("light"))), ("strength_0.5", read(strength=0.5)),
                       ("strength_0", read(strength=0.0))):
        assert want is not None and v[name].p["edge_src"] == want and want != fitted, name
    # (the rim stage belongs to the chain: the light stage draws the dark rim bands only with it)
    assert v["drop_rim"].p["edge_src"] == read(stages=without("rim")) != own
    # the re-read rows' reference: the fitted chain ranked on the same photos. ``full`` keeps
    # fit's reading (other photos), so a re-read row minus ``full`` is the variant AND the
    # photos; a variant that changes nothing before the detail stage (this bank blooms no
    # disc) renders exactly as ``full_cal``, and not as ``full``
    assert v["full_cal"].p["edge_src"] == fitted != own and list(v["full_cal"].stages) == list(ps.STAGES)
    assert float(np.max(prof.bank["bloom"])) == 0.0 and v["drop_bloom"].p["edge_src"] == read(stages=without("bloom"))
    x = _fundus(4, 96, seed=10)
    draw = lambda name: v[name](x, torch.Generator().manual_seed(0))
    assert torch.equal(draw("drop_bloom"), draw("full_cal")) and not torch.equal(draw("drop_bloom"), draw("full"))
    assert len(said) == 1 and "re-read on 8 tabletop training photos" in said[0]
    listed = said[0].split("for: ")[1].split(", ")
    assert {"prefilter_gauss", "fill_0", "drop_colour", "drop_rim", "strength_0", "strength_0.75", "full_cal"} <= set(listed)
    assert not {"full", "drop_noise", "drop_detail", "whole_field", "render_v2", "detail_by_edge"} & set(listed)
    # no photos: every variant keeps the profile's reading, and there is no second reference
    v0 = pe.variants(prof, size, ["render", "field", "strength"])
    assert all(sim.p["edge_src"] == own for sim in v0.values()) and "full_cal" not in v0
    # by the drawn photo's reading nothing is ranked: no second reference either
    assert "full_cal" not in pe.variants(prof.with_params(detail_by="edge"), size, ["drop"], cal=cal, seed=3, log=quiet)
    # a strength row is the trainer's own render only without the re-reading
    half = lambda sim: sim(x, torch.Generator().manual_seed(0))
    assert torch.equal(half(v0["strength_0.5"]), half(ps.PhoneSimulator(prof, strength=0.5, out_size=size)))
    # a profile fitted before the whole-field chain had its own reading: read here
    older = prof.with_params(edge_src_whole=None)
    v = pe.variants(older, size, ["field"], cal=cal, seed=3, log=quiet)
    assert v["whole_field"].p["edge_src_whole"] == ps.calibrate_detail(older, cal, size, 3, log=quiet, field="whole")
    assert v["whole_field"].p["edge_src"] == own and v["full"].p["edge_src_whole"] is None
    # a bank with an edge reading whose fit never read its renders (fit --render v2): by rank
    # only with photos, each rank variant on its own chain
    v2fit = prof.with_params(**ps.LEGACY_PARAMS, edge_src=None, edge_src_whole=None)
    v = pe.variants(v2fit, size, ["render"], cal=cal, seed=3, log=quiet)
    assert set(v) == {"full", "fill_0.18", "prefilter_matched", "detail_by_edge", "detail_by_rank", "render_v3"}
    assert v["detail_by_rank"].p["edge_src"] == ps.calibrate_detail(v2fit.with_params(detail_by="rank"), cal, size, 3, log=quiet)
    assert v["render_v3"].p["edge_src"] == ps.calibrate_detail(
        v2fit.with_params(**ps.RENDER_V3, detail_by="rank"), cal, size, 3, log=quiet)
    assert v["render_v3"].p["edge_src"] != v["detail_by_rank"].p["edge_src"] and v["full"].p["detail_by"] == "band"
    assert set(pe.variants(v2fit, size, ["render"])) == {"full", "fill_0.18", "prefilter_matched", "detail_by_edge",
                                                         "render_v3_edge"}
    # a bank whose handheld photos have no edge reading at all: fit fell back to the drawn
    # photo's (which then asks for no blur), and a rank row has nothing to rank against --
    # left out, and said
    blank = copy.deepcopy(prof)
    blank.bank["edge"][:] = float("nan")
    ps.fit_detail(blank, cal, size, 3, log=quiet)
    assert blank.params["detail_by"] == "edge" and blank.params["edge_src"] is None
    said = []
    v = pe.variants(blank, size, ["render"], cal=cal, seed=3, log=said.append)
    assert "detail_by_rank" not in v and "full_cal" not in v and "detail_by_band" in v
    assert any("left out (no reading of the renders to rank against): detail_by_rank" in t for t in said), said


def test_a_phone_only_referee_is_told_which_tabletop_photos_it_is_scored_on(tmp_path, monkeypatch, capsys):
    """A checkpoint trained on handheld photos alone takes the tabletop set and camera of
    the arms beside it: said in the log, and refused when those arms read different
    photos (it took the first one listed, silently)."""
    from test_phone_sim import _profile
    prof = tmp_path / "profile.json"
    _profile().save(str(prof))
    rr = lambda camera: {"args": {"source_root": "/B", "target_root": "/M", "source_dataset": "brset",
                                  "target_dataset": "mbrset", "split_seed": 42, "source_camera": camera,
                                  "source_targets": [pe.PRIMARY]},
                         "config": {"image_size": 64}, "thresholds": {"calibrated": {pe.PRIMARY: 0.5}}}
    phone = {"args": {"root": "/M", "dataset": "mbrset", "split_seed": 42, "targets": [pe.PRIMARY]},
             "config": {"image_size": 64}, "thresholds": {"trained": {pe.PRIMARY: 0.5}}}
    cks = {"canon.pt": rr("Canon CR"), "nikon.pt": rr("NIKON NF5050"), "phone.pt": phone}
    groups, made = [], []
    monkeypatch.setattr(pe.torch, "load", lambda path, **kw: cks[path])
    monkeypatch.setattr(pe, "Arm", lambda arm, path, ck, dev: arm)
    monkeypatch.setattr(pe, "group_effects", lambda arms, key, a, dev: groups.append((arms, key)) or list(made))
    monkeypatch.setattr(pe, "summarise", lambda df, sens: "")
    run = lambda *arms, extra=(): pe.main(["--ckpt", *arms, "--profile", str(prof), "--out", str(tmp_path / "fx"),
                                           "--device", "cpu", *extra])
    assert run("canon=canon.pt", "phone=phone.pt") == 0
    assert len(groups) == 1 and groups[0][0] == ["canon", "phone"] and groups[0][1]["camera"] == "Canon CR"
    said = capsys.readouterr().out
    assert "phone (phone.pt) was trained on handheld photos alone" in said and "(/B, Canon CR)" in said
    txt = (tmp_path / "fx" / "effects_summary.txt").read_text()
    assert "render rows (changed from the profile's own fill 0.18, prefilter matched, detail_by rank): fill_0 = fill 0.0; " in txt
    assert "render_v2 = fill 0.0, prefilter gauss, detail_by band" in txt and "detail ranks: re-read on up to 64" in txt
    # the summary names the render rows that were made (one left out for want of a reading
    # to rank against is not in the table)
    made[:] = [{"variant": "full"}, {"variant": "fill_0"}]
    assert run("canon=canon.pt") == 0
    txt = (tmp_path / "fx" / "effects_summary.txt").read_text()
    assert "fill_0 = fill 0.0" in txt and "render_v2 =" not in txt and "prefilter_gauss =" not in txt
    made.clear(); groups.clear(); capsys.readouterr()
    # the handheld root spelled another way is the same split; alone under a root, every camera, and said
    groups.clear()
    cks["phone.pt"]["args"]["root"] = "/M/"
    assert run("phone=phone.pt", "canon=canon.pt") == 0 and groups[0][1]["camera"] == "Canon CR" and len(groups) == 1
    cks["phone.pt"]["args"]["root"] = "/M"
    groups.clear(); capsys.readouterr()
    assert run("phone=phone.pt", extra=["--source-root", "/B"]) == 0 and groups[0][1]["camera"] is None
    assert "no arm beside it shares its handheld split: scored on every camera's photos under /B" in capsys.readouterr().out
    groups.clear()
    for order in (("canon=canon.pt", "nikon=nikon.pt", "phone=phone.pt"), ("phone=phone.pt", "nikon=nikon.pt", "canon=canon.pt")):
        with pytest.raises(SystemExit):
            run(*order)
        err = capsys.readouterr().err
        assert "the arms beside it read different tabletop photos" in err and "/B, Canon CR" in err and "/B, NIKON NF5050" in err
    assert groups == []
    # named, the camera is every arm's
    assert run("canon=canon.pt", "nikon=nikon.pt", "phone=phone.pt", extra=["--source-camera", "Canon CR", "--cal-n", "0"]) == 0
    assert len(groups) == 1 and groups[0][1]["camera"] == "Canon CR" and len(groups[0][0]) == 3
    assert "every row ranks against the profile's own reading (--cal-n 0)" in (tmp_path / "fx" / "effects_summary.txt").read_text()
    # an arm trained on every camera's tabletop photos cannot be scored on one camera's: the
    # split is drawn after the camera filter, so that camera's photos split afresh are mostly
    # photos it was trained on. Without --source-camera it runs, in a group of its own
    mixed = lambda camera: {"args": {"root": "/M", "dataset": "mbrset", "split_seed": 42, "targets": [pe.PRIMARY],
                                     "mix_root": "/B", "mix_camera": camera},
                            "config": {"image_size": 64}, "thresholds": {"trained": {pe.PRIMARY: 0.5}}}
    cks.update({"every.pt": rr(None), "mixed_every.pt": mixed(None), "mixed_canon.pt": mixed("Canon CR")})
    for bad in ("every=every.pt", "mixed=mixed_every.pt"):
        groups.clear()
        with pytest.raises(SystemExit):
            run("canon=canon.pt", bad, "phone=phone.pt", extra=["--source-camera", "Canon CR"])
        err = capsys.readouterr().err
        assert "was trained on every camera's tabletop train split" in err and bad.split("=")[1] in err and groups == []
    groups.clear()
    assert run("every=every.pt", "canon=canon.pt") == 0
    assert sorted((g[0], g[1]["camera"]) for g in groups) == [(["canon"], "Canon CR"), (["every"], None)]
    groups.clear()
    assert run("mixed=mixed_canon.pt", "nikon=nikon.pt", extra=["--source-camera", "Canon CR"]) == 0 and len(groups) == 1
    # an arm trained with a simulator: is --profile the file it was trained with, and which row is its render
    import hashlib
    sha = hashlib.sha256(prof.read_bytes()).hexdigest()
    sim = lambda sha, **g: {**rr("Canon CR"), "args": {**rr("Canon CR")["args"], "phone_sim_p": 0.5, **g},
                            "phone_sim_profile": {"path": "ck/profile.json", "sha256": sha}}
    assert pe.trained_with(rr("Canon CR"), sha) is None and pe.trained_with(phone, sha) is None
    line = pe.trained_with(sim(sha), sha)
    assert "--profile is that file; its training render is the full row" in line and "p 0.5, strength 1, field phone" in line
    assert "the whole_field row" in pe.trained_with(sim(sha, phone_sim_field="whole"), sha)
    assert "the strength_0.5 under --cal-n 0 row" in pe.trained_with(sim(sha, phone_sim_strength=0.5), sha)
    assert "not among the rows" in pe.trained_with(sim(sha, phone_sim_strength=0.5, phone_sim_field="whole"), sha)
    assert "NOT --profile, so no row here is the render it was trained on" in pe.trained_with(sim("0" * 64), sha)
    mixed_sim = {**mixed("Canon CR"), "phone_sim_profile": {"path": "p.json", "sha256": sha, "p": 1.0}}
    assert "p 1.0, strength 1, field phone" in pe.trained_with(mixed_sim, sha) and "the full row" in pe.trained_with(mixed_sim, sha)
    cks.update({"sim.pt": sim(sha), "other.pt": sim("0" * 64)})
    capsys.readouterr()
    assert run("canon=canon.pt", "canon_phonesim=sim.pt", "older=other.pt") == 0
    said, txt = capsys.readouterr().out, (tmp_path / "fx" / "effects_summary.txt").read_text()
    for where in (said, txt):
        assert "canon_phonesim (sim.pt): trained with phone_sim" in where and "its training render is the full row" in where
        assert "older (other.pt): trained with phone_sim" in where and "NOT --profile" in where
        assert "canon (canon.pt)" not in where
    assert f"profile {prof} ({sha[:12]})" in txt and "full_cal is the fitted chain ranked on those same photos" in txt


def test_a_split_file_is_held_to_the_photos_it_splits(tmp_path, capsys):
    """A referee's split file against the photos scored: a photo the file puts in another
    split may be one it was trained on (fatal); a photo the file does not list is a change
    of the table when the photos are the file's own set (fatal: the split recomputed here
    is another), and another camera's photo otherwise (never a training photo: said)."""
    class _Ds:
        def __init__(self, files):
            self.df = pd.DataFrame({"file": files})
    f = tmp_path / "split.csv"
    pd.DataFrame({"file": ["a", "b", "c", "d"], "split": ["train", "val", "val", "test"]}).to_csv(f, index=False)
    pe.check_split(_Ds(["b", "c"]), str(f), "val", "mixed seed 0 (tabletop)")
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit, match="1 of the val photos scored here are in another split .*e.g. a"):
        pe.check_split(_Ds(["b", "a"]), str(f), "val", "mixed seed 0 (tabletop)")
    with pytest.raises(SystemExit, match="1 of the val photos scored here are not in .* at all .*e.g. z"):
        pe.check_split(_Ds(["b", "z"]), str(f), "val", "mixed seed 0 (tabletop)")
    pe.check_split(_Ds(["b", "z", "y"]), str(f), "val", "mixed seed 0 (tabletop)", own=False)
    assert "2 of the 3 val photos scored are not in" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="in another split"):               # another camera's photos do not excuse a train photo
        pe.check_split(_Ds(["a", "z"]), str(f), "val", "mixed seed 0 (tabletop)", own=False)
    pe.check_split(_Ds(["b"]), str(tmp_path / "gone.csv"), "val", "phone seed 0 (handheld)")
    assert "is not here; assuming the recomputed split" in capsys.readouterr().out
    pe.check_split(_Ds(["anything"]), None, "val", "phone seed 0 (handheld)")


def test_effects_with_referees_that_know_the_handheld_camera(tmp_path):
    """Phone-only and mixed checkpoints (train_phone_ceiling.py) as arms: read as trained
    at their own handheld threshold, on the Canon sweep's rows; the phone-only one takes
    its tabletop photos from the other arms; a split file that disagrees with the rows
    scored is refused."""
    from test_retinareach import BACKBONE, _make_brset, _make_mbrset
    from train_phone_ceiling import main as ceiling
    from train_retinareach import main as train
    rng = np.random.default_rng(1)
    B, M = tmp_path / "brset", tmp_path / "mbrset"
    _make_brset(B, 30, rng, camera_of=lambda pid: "Canon CR" if pid % 4 else "NIKON NF5050")
    _make_mbrset(M, 30, rng)
    prof = tmp_path / "ck_phone_sim" / "profile.json"
    assert ps.main(["fit", "--target-root", str(M), "--source-root", str(B), "--source-camera", "Canon CR",
                    "--image-size", "64", "--validate", "0", "--num-workers", "0", "--device", "cpu",
                    "--out", str(prof)]) == 0
    small = ["--backbone", BACKBONE, "--no-pretrained", "--image-size", "64", "--epochs", "1", "--batch-size", "8",
             "--num-workers", "0", "--device", "cpu", "--n-boot", "10"]
    rr, ck = tmp_path / "rr", tmp_path / "base.pt"
    assert train(["--source-root", str(B), "--target-root", str(M), "--source-camera", "Canon CR",
                  "--probe-targets", "hypertension", *small, "--calib-batch", "4", "--calib-sizes", "4",
                  "--calib-repeats", "1", "--envelope-repeats", "2", "--prior-strengths", "0", "--min-probe-pos", "2",
                  "--cv-repeats", "1", "--min-test-pos-patients", "2", "--demo-n", "4", "--n-shuffle", "0",
                  "--ablation-frac", "0", "--out", str(rr), "--ckpt", str(ck)]) == 0
    phone, mixed = tmp_path / "phone.pt", tmp_path / "mixed.pt"
    split = ["--split-file", str(rr / "split_handheld.csv")]
    assert ceiling(["--root", str(M), *split, *small, "--ema-decay", "0", "--out", str(tmp_path / "pc"),
                    "--ckpt", str(phone)]) == 0
    assert ceiling(["--root", str(M), *split, "--mix-root", str(B), "--mix-camera", "Canon CR",
                    "--mix-split-file", str(rr / "split_tabletop.csv"), *small, "--ema-decay", "0",
                    "--out", str(tmp_path / "pm"), "--ckpt", str(mixed)]) == 0

    def run(out, *arms, extra=()):
        assert pe.main(["--ckpt", *arms, "--profile", str(prof), "--out", str(out), "--num-workers", "0",
                        "--batch-size", "8", "--device", "cpu", "--probe-n", "20", "--sets", "render", "field",
                        "--grid-n", "0", *extra]) == 0
        return pd.read_csv(out / "effects.csv"), (out / "effects_summary.txt").read_text()

    # the photos the variants' detail ranks are re-read on: tabletop TRAINING photos of the
    # scored camera, at the supersampled size, in [0, 1], as many as asked for (none of
    # the photos scored, and not normalised: either would shift every re-read row unseen)
    seen, true_variants = [], pe.variants
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pe, "variants", lambda profile, size, sets, cal=None, *a, **kw:
                   seen.append(cal) or true_variants(profile, size, sets, cal, *a, **kw))
        df, txt = run(tmp_path / "fx", f"canon_only={ck}", f"phone={phone}", f"mixed={mixed}", extra=["--cal-n", "5"])
        none, _ = run(tmp_path / "fx0", f"canon_only={ck}", f"phone={phone}", f"mixed={mixed}", extra=["--cal-n", "0"])
    cal, without = seen
    sup = float(ps.PhoneProfile.load(str(prof)).params["supersample"])
    assert without is None and cal.shape == (5, 3, round(64 * sup), round(64 * sup))
    assert 0.0 <= float(cal.min()) and float(cal.max()) <= 1.0 and float(cal.max()) > 0.3
    from style_transfer_xfer import load_device
    from style_transfer import denorm
    train_split = load_device(str(B), "brset", round(64 * sup), 42, camera="Canon CR", strat=pe.PRIMARY)["train"]
    pick = np.random.default_rng(0 + 1).choice(len(train_split), 5, replace=False)      # --seed 0: a generator of their own
    want = torch.stack([denorm(train_split[int(i)]["image"][None])[0] for i in pick])
    assert torch.allclose(cal, want, atol=1e-6)
    # the rows that rank against the profile's own reading are the same with and without
    same = ["full", "whole_field", "render_v2", "detail_by_band", "detail_by_edge"]
    k = ["arm", "variant", "stats"]
    a_, b_ = df[df.variant.isin(same)].set_index(k).sort_index(), none[none.variant.isin(same)].set_index(k).sort_index()
    assert a_.index.equals(b_.index) and np.allclose(a_.auroc, b_.auroc, equal_nan=True)
    assert "full_cal" not in set(none.variant) and "full_cal" in set(df.variant)
    assert set(df.arm) == {"canon_only", "phone", "mixed"}
    assert set(df[df.arm == "canon_only"].stats) == set(pe.STATS) and not df[df.arm == "canon_only"].referee.any()
    for arm in ("phone", "mixed"):
        d = df[df.arm == arm]
        assert set(d.stats) == {pe.TRAINED} and d.referee.all()
        assert {pe.BASE, pe.REAL, "full", "full_cal", "whole_field", "render_v2", "fill_0", "prefilter_gauss",
                "detail_by_band", "detail_by_edge"} == set(d.variant)
        assert d.auroc.between(0, 1).all() and d.set_index("variant").loc["full", "p_handheld_icdr0"] >= 0
        # the threshold is the referee's own: the one its handheld validation split set
        thr = __import__("json").loads((tmp_path / ("pc" if arm == "phone" else "pm") / "results.json").read_text())
        assert np.allclose(d.threshold, thr["thresholds"][arm]["dr_referable"])
    # every arm scored the same photos, the phone-only one on the other arms' tabletop set
    assert df[df.variant == pe.BASE].n.nunique() == 1 and df[df.variant == pe.REAL].n.nunique() == 1
    assert df[df.variant == pe.BASE].prevalence.nunique() == 1
    # ... also in the documented form, the roots given beside the other arms (with the root
    # alone the phone-only referee was scored on every tabletop camera, another split)
    rooted, _ = run(tmp_path / "fx_rooted", f"phone={phone}", f"canon_only={ck}", f"mixed={mixed}",
                    extra=["--source-root", str(B), "--target-root", str(M)])
    key = ["arm", "variant", "stats"]
    one, two = df.set_index(key).sort_index(), rooted.set_index(key).sort_index()
    assert one.index.equals(two.index)
    for col in ("n", "prevalence", "auroc", "dlogit_icdr0"):
        assert np.allclose(one[col], two[col], equal_nan=True), col
    # the old render on the same bank moves the referees' logits, same draws
    tab = df[(df.arm == "mixed")].set_index("variant")
    assert abs(tab.loc["render_v2", "dlogit_icdr0"] - tab.loc["full", "dlogit_icdr0"]) > 1e-6
    assert "REFEREES' VERDICT" in txt and "minus_tabletop" in txt and "phone, as trained" in txt
    assert "IN-SAMPLE for the referees" in txt and "canon_only, tabletop statistics" in txt

    # alone, a phone-only referee names no tabletop photos
    with pytest.raises(SystemExit):
        pe.main(["--ckpt", f"phone={phone}", "--profile", str(prof), "--out", str(tmp_path / "bad"),
                 "--num-workers", "0", "--device", "cpu"])
    solo, _ = run(tmp_path / "fx_solo", f"phone={phone}", extra=["--source-root", str(B), "--source-camera", "Canon CR"])
    both = df[df.arm == "phone"].set_index("variant")
    assert np.allclose(solo.set_index("variant").loc[both.index, "auroc"], both.auroc, equal_nan=True)
    # the split it was trained on must be the split scored: a validation photo the file calls
    # a training photo is refused
    sp = pd.read_csv(rr / "split_handheld.csv")
    sp.loc[sp.index[sp.split == "val"][0], "split"] = "train"
    sp.to_csv(rr / "split_handheld.csv", index=False)
    with pytest.raises(SystemExit, match="in another split"):
        run(tmp_path / "fx_bad", f"canon_only={ck}", f"phone={phone}")


def test_the_referees_verdict_survives_an_undefined_auroc():
    """A small sample of one class has no AUROC on the tabletop rows: the summary still
    comes out (the pivot dropped the all-NaN column and the verdict looked it up)."""
    rows = []
    for arm, referee in (("phone", True), ("mixed", True), ("canon_only", False)):
        for stats in ((pe.TRAINED,) if referee else pe.STATS):
            for variant, auroc in ((pe.BASE, float("nan")), ("full", float("nan")), (pe.REAL, 0.8)):
                rows.append({"arm": arm, "seed": 0, "variant": variant, "stats": stats, "part": "val", "referee": referee,
                             "auroc": auroc, "sensitivity": 1.0, "specificity": float("nan"), "flagged": 1.0,
                             "oracle_spec": float("nan"), "dlogit_referable": 0.1, "dlogit_icdr0": float("nan"),
                             "p_handheld_icdr0": float("nan")})
    txt = pe.summarise(pd.DataFrame(rows), 0.85)
    assert "REFEREES' VERDICT" in txt and "phone, as trained" in txt
    # two seeds that differ: each re-rendered AUROC is read against the same seed's tabletop
    # one (against the arm's mean tabletop AUROC, seed 0's 0.62 would count as below it)
    au = {(0, pe.BASE): 0.60, (0, "full"): 0.62, (1, pe.BASE): 0.80, (1, "full"): 0.85}
    two = [dict(r, auroc=au.get((s, r["variant"]), r["auroc"]), seed=s) for r in rows for s in (0, 1)]
    # the mixed referee has seed 0 only, and the phone referee's third seed has no tabletop AUROC
    # (its re-rendered 0.65 is below the arm's mean tabletop AUROC: counted against that mean
    # the seeds above would be 1, seed 1 alone, and counted without a tabletop AUROC, of 3)
    two = [r for r in two if not (r["arm"] == "mixed" and r["seed"] == 1)]
    two += [dict(r, seed=2, auroc=float("nan") if r["variant"] == pe.BASE else 0.65) for r in rows if r["arm"] == "phone"]
    verdict = pe.summarise(pd.DataFrame(two), 0.85).split("REFEREES' VERDICT")[1]
    assert "of the of_seeds that have both AUROCs" in verdict.splitlines()[0]
    lines = [l.split() for l in verdict.splitlines()]
    at = next(i for i, l in enumerate(lines) if l[:2] == ["phone", "tabletop"])
    assert lines[at] == ["phone", "tabletop", "0.700"]
    #                           AUROC (3 seeds), minus tabletop, seeds above it, of the seeds with both
    assert lines[at + 1] == ["full", f"{(0.62 + 0.85 + 0.65) / 3:.3f}", "0.035", "2.000", "2.000"]
    at = next(i for i, l in enumerate(lines) if l[:2] == ["mixed", "tabletop"])
    assert lines[at + 1] == ["full", "0.620", "0.020", "1.000", "1.000"]

