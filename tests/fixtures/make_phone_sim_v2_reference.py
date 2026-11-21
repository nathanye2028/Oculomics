#!/usr/bin/env python3
"""Writes ``phone_sim_v2_reference.npz``: a phone_sim/2 profile, four synthetic photos and
what the simulator that fitted the profile rendered from them, for
``test_a_format_2_profile_renders_what_it_rendered_when_fitted``.

Run once, with the phone_sim.py of the commit whose render is to be kept (the last one
before format /3 is 45d8b37):

    git show 45d8b37:phone_sim.py > /tmp/phone_sim_v2.py
    python tests/fixtures/make_phone_sim_v2_reference.py /tmp/phone_sim_v2.py

Synthetic fundi only (``tests/test_phone_sim.py``'s fixture): no photo is involved. The
fitted bank is edited in two ways before it is saved, because the synthetic phone photos
never clip their disc and the fitted bloom would be 0 in every row: the rows' bloom runs
from 0 to 1 and their disc colour is near-white, so the stored renders pass through the
bloom (checked below)."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)]
from test_phone_sim import V2_RENDERS, _fundus, _stream_probe  # noqa: E402


def main(old_source: str) -> int:
    spec = importlib.util.spec_from_file_location("phone_sim_v2", old_source)
    old = importlib.util.module_from_spec(spec)
    sys.modules["phone_sim_v2"] = old
    spec.loader.exec_module(old)
    assert old.FORMAT == "phone_sim/2", old.FORMAT
    prof, _ = old.fit_profile([_fundus(12, 64, phone=True, seed=0)], [_fundus(12, 64, seed=1)], 64, log=lambda s: None)
    prof.bank["bloom"][:] = np.linspace(0.0, 1.0, len(prof), dtype=np.float32)
    prof.bank["hcol"][:] = np.array([0.95, 0.9, 0.85], dtype=np.float32)
    tmp = os.path.join(HERE, "_profile_v2.json")
    prof.save(tmp)
    with open(tmp) as f:
        blob = f.read()
    loaded = old.PhoneProfile.load(tmp)               # what a run reads back: the values as the file rounds them
    os.remove(tmp)
    assert json.loads(blob)["format"] == "phone_sim/2"
    x8 = (_fundus(4, 96, seed=5) * 255).round().to(torch.uint8)       # stored as a photo is, in 8 bits
    x = x8.float() / 255
    render = lambda **kw: old.PhoneSimulator(loaded, out_size=64, **kw)(x, torch.Generator().manual_seed(1))
    out = {"profile": np.array(blob), "x": x8.numpy(), "stream": _stream_probe()}
    for name, kw in V2_RENDERS.items():
        y = render(**kw)
        # the whole chain in full; the variants on every other pixel (the file stays small)
        y = y if name == "full" else y[..., ::2, ::2]
        out[name] = (y.numpy().astype(np.float64) * 65535).round().astype(np.uint16)
    # every stage has to leave its mark on the stored full render, the bloom included
    full = render()
    for stage in old.STAGES:
        moved = float((render(stages=[s for s in old.STAGES if s != stage]) - full).abs().max())
        print(f"  without {stage:7s} the render moves by up to {moved:.4f}")
        assert moved > 0.01, stage
    np.savez_compressed(os.path.join(HERE, "phone_sim_v2_reference.npz"), **out)
    size = os.path.getsize(os.path.join(HERE, "phone_sim_v2_reference.npz"))
    print("wrote phone_sim_v2_reference.npz:", {k: v.shape for k, v in out.items()}, f"{size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
