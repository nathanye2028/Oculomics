"""
export_coreml.py
================
Core ML export for on-device (iPhone/iPad) inference — the Apple Neural Engine
counterpart to :mod:`edge_optimize`.

Relationship to edge_optimize.py
--------------------------------
``edge_optimize.py`` exports ONNX + INT8 and benchmarks via ONNX Runtime, which
is a **CPU** inference path. That is the right portable baseline, but it does
not reach the ANE, and ANE-resident inference is what makes a phone-based
screening tool feel instant. This module is the iOS-specific path:

    PyTorch -> torch.jit.trace -> Core ML .mlpackage (fp16 / int8 weights)

Both are worth keeping — they answer different questions ("what does portable
CPU inference cost?" vs "what does the target device actually deliver?").

Preprocessing is baked into the graph
-------------------------------------
:class:`DeployWrapper` folds the ImageNet normalisation and the final sigmoid
into the exported model, so it takes **raw 0-255 RGB** and returns **[0,1]
per-lesion probabilities**. The app then does no arithmetic. This is not just
convenience: Core ML's ``ImageType`` exposes a single scalar ``scale``, which
cannot express the per-channel ImageNet std (0.229/0.224/0.225), so normalising
inside the graph is the only exact option.

Export is a deployment smoke test, and is worth running *before* the model is
good. With random weights the predictions are meaningless, but op coverage
(does anything fall back off the ANE?) and latency are real signal, and those
are expensive to discover late.

Run:
    python export_coreml.py --checkpoint checkpoints/gcg_seed0.pt --classes 4
    python export_coreml.py --image-size 768 --quantize int8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DeployWrapper(nn.Module):
    """Raw 0-255 RGB in, [0,1] lesion probabilities out.

    Folding normalisation and the sigmoid into the graph removes the two most
    common sources of train/deploy skew: mismatched normalisation constants,
    and forgetting to apply the sigmoid to logits on the app side.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        # Buffers so they trace as constants and travel with the state_dict.
        self.register_buffer("mean255", IMAGENET_MEAN.clone() * 255.0)
        self.register_buffer("std255", IMAGENET_STD.clone() * 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean255) / self.std255
        return torch.sigmoid(self.net(x))


def load_checkpoint(net: nn.Module, path: Optional[str]) -> bool:
    """Load weights into ``net``. Returns True if real weights were loaded.

    Accepts a bare state_dict or one wrapped under ``model`` / ``state_dict`` /
    ``model_state_dict``, and strips any ``module.`` DDP prefix.
    """
    if not path:
        return False
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise ValueError(f"could not find a state_dict inside {path}")

    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return True


def build_wrapped(args: argparse.Namespace) -> Tuple[nn.Module, bool]:
    from model_seg import build_model
    net = build_model(arch="gcg_unet", num_classes=args.classes,
                      pretrained=False, use_gcg=not args.no_gcg)
    trained = load_checkpoint(net, args.checkpoint)
    wrapper = DeployWrapper(net).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper, trained


def export(wrapper: nn.Module, args: argparse.Namespace) -> str:
    import coremltools as ct

    size = args.image_size
    example = torch.randint(0, 256, (1, 3, size, size), dtype=torch.float32)

    # jit.trace is the best-supported coremltools front end. The dynamic
    # skip.shape[-2:] sizes inside DecoderBlock bake to constants here, which
    # is exactly what a fixed-resolution mobile model wants.
    print(f"[info] tracing at {size}x{size} ...")
    traced = torch.jit.trace(wrapper, example, strict=False)
    traced.eval()

    precision = ct.precision.FLOAT32 if args.quantize == "none" else ct.precision.FLOAT16
    print(f"[info] converting to Core ML (precision={precision.value}) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, size, size),
                             color_layout=ct.colorlayout.RGB,
                             scale=1.0, bias=[0.0, 0.0, 0.0])],
        outputs=[ct.TensorType(name="lesion_prob")],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=getattr(ct.target, args.min_target),
        compute_units=ct.ComputeUnit.ALL,
    )

    if args.quantize == "int8":
        print("[info] applying int8 weight quantization ...")
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
        )
        cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", weight_threshold=512))
        mlmodel = linear_quantize_weights(mlmodel, config=cfg)

    mlmodel.short_description = (
        f"GCG-U-Net retinal lesion segmentation ({args.classes} channels). "
        "Input: 0-255 RGB. Output: per-pixel lesion probability in [0,1].")
    mlmodel.input_description["image"] = f"Fundus image, {size}x{size} RGB."
    mlmodel.output_description["lesion_prob"] = "Lesion probabilities [1,C,H,W]."

    os.makedirs(args.out_dir, exist_ok=True)
    out = args.output or os.path.join(
        args.out_dir, f"seg_coreml_{size}_{args.quantize}.mlpackage")
    mlmodel.save(out)
    print(f"[ok]   saved {out}")
    return out


def verify(path: str, wrapper: nn.Module, size: int) -> None:
    """Compare Core ML output against PyTorch on identical input."""
    import coremltools as ct
    from PIL import Image

    if sys.platform != "darwin":
        print("[skip] verification requires macOS")
        return

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)[None]).numpy()

    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
    cm_out = np.asarray(
        mlmodel.predict({"image": Image.fromarray(arr)})["lesion_prob"]).reshape(torch_out.shape)

    diff = np.abs(torch_out - cm_out)
    agree = float(((torch_out > 0.5) == (cm_out > 0.5)).mean())
    print("\n=== verification (PyTorch vs Core ML) ===")
    print(f"  max abs diff  : {diff.max():.6f}")
    print(f"  mask agreement: {agree*100:.3f}% of pixels @ thr=0.5")
    if agree < 0.99:
        print("  [warn] <99% agreement — inspect before trusting on-device output.")


def benchmark(path: str, size: int, runs: int) -> None:
    """Time the Core ML model on this Mac.

    An upper-bound sanity check, NOT an iPhone number — the Mac's ANE and
    memory bandwidth differ from a phone's. Trust it for op coverage and order
    of magnitude; measure on-device for anything quotable.
    """
    import coremltools as ct
    from PIL import Image

    if sys.platform != "darwin":
        print("[skip] benchmark requires macOS")
        return

    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    for _ in range(3):
        mlmodel.predict({"image": img})            # warm up / let the ANE compile

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        mlmodel.predict({"image": img})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    print(f"\n=== latency on this Mac ({runs} runs, ComputeUnit.ALL) ===")
    print(f"  median : {times[len(times)//2]:.1f} ms")
    print(f"  p10/p90: {times[int(len(times)*0.1)]:.1f} / {times[int(len(times)*0.9)]:.1f} ms")
    print("  (indicative only — measure on the target iPhone for real numbers)")


def main() -> int:
    p = argparse.ArgumentParser(description="Core ML export for on-device segmentation.")
    p.add_argument("--checkpoint", default=None,
                   help="Trained weights; omitted => random (deployment smoke test).")
    p.add_argument("--classes", type=int, default=4, help="Output lesion channels.")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--out-dir", default="edge_export")
    p.add_argument("--output", default=None)
    p.add_argument("--quantize", choices=("none", "fp16", "int8"), default="fp16")
    p.add_argument("--min-target", default="iOS17", help="e.g. iOS16 / iOS17 / iOS18.")
    p.add_argument("--no-gcg", action="store_true", help="Match a --no-gcg ablation checkpoint.")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()

    wrapper, trained = build_wrapped(args)
    n_params = sum(q.numel() for q in wrapper.parameters())
    print(f"[info] model    : GCG-U-Net {'WITH' if not args.no_gcg else 'NO'} GCG, "
          f"{args.classes} classes, {n_params/1e6:.2f}M params")
    print(f"[info] weights  : {args.checkpoint if trained else 'RANDOM (untrained)'}")
    if not trained:
        print("[warn] No checkpoint given — predictions are meaningless; "
              "latency and op coverage are still valid signal.")

    out = export(wrapper, args)
    if not args.skip_verify:
        verify(out, wrapper, args.image_size)
    if not args.skip_benchmark:
        benchmark(out, args.image_size, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
