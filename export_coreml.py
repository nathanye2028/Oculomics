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
    # segmentation (GCG-U-Net)
    python export_coreml.py --checkpoint checkpoints/gcg_seed0.pt --classes 4
    # classification (MBRSETClassifier; arch/backbone/size read from the ckpt)
    python export_coreml.py --checkpoint ck_kd_v4_384/kd_seed1.pt
    # architecture-only latency probe (random weights)
    python export_coreml.py --model cls --backbone timm:mobilenetv4_conv_small --image-size 384
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


class ClsDeployWrapper(nn.Module):
    """Raw 0-255 RGB in, class probabilities out (softmax folded in).

    The classification twin of :class:`DeployWrapper`: normalisation inside the
    graph (Core ML's ImageType scale is scalar, ImageNet std is per-channel) and
    softmax applied so the app reads calibrated-ish probabilities directly.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean255", IMAGENET_MEAN.clone() * 255.0)
        self.register_buffer("std255", IMAGENET_STD.clone() * 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean255) / self.std255
        return torch.softmax(self.net(x), dim=1)


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
    gcg_mismatch = [k for k in list(missing) + list(unexpected) if "gcg" in k.lower()]
    if gcg_mismatch:
        raise SystemExit(f"[fatal] GCG parameter mismatch between checkpoint and built model "
                         f"(e.g. {gcg_mismatch[:3]}). Exporting would ship randomly-initialised "
                         f"gates; refusing.")
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return True


def detect_model_kind(args) -> str:
    """'cls' vs 'seg', read from the checkpoint when --model auto.

    Classifier checkpoints (train_mbrset.py) record ``task``/``backbone`` in
    their args; segmentation checkpoints (train_idrid.py) record ``lesions``.
    """
    if args.model != "auto":
        return args.model
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ck, dict):
            if "lesions" in ck or "lesions" in ck.get("args", {}):
                return "seg"
            a = ck.get("args", {})
            if "backbone" in a or a.get("task") in (
                    "dr_referable", "dr_binary", "dr_grade", "edema", "quality", "artifacts"):
                return "cls"
    return "seg"


def build_wrapped_cls(args: argparse.Namespace) -> Tuple[nn.Module, bool, int]:
    """Classifier path: rebuild MBRSETClassifier from the checkpoint's own args."""
    from model import MBRSETClassifier
    from train_mbrset import backbone_kwargs_for
    a = {}
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        a = ck.get("args", {}) if isinstance(ck, dict) else {}
    backbone = args.backbone or a.get("backbone", "mobilenetv3_small")
    size = args.image_size or int(a.get("image_size", 224))
    n_cls = args.classes if args.classes is not None else 2
    net = MBRSETClassifier(num_classes=n_cls, pretrained=False,
                           use_gcg=not a.get("no_gcg", False),
                           gcg_variant=a.get("gcg_variant", "baseline"),
                           backbone=backbone,
                           backbone_kwargs=backbone_kwargs_for(backbone, size))
    print(f"[info] arch: {backbone}  task={a.get('task', '?')}  classes={n_cls}  size={size}")
    trained = load_checkpoint(net, args.checkpoint)
    wrapper = ClsDeployWrapper(net).eval()
    for p_ in wrapper.parameters():
        p_.requires_grad_(False)
    return wrapper, trained, size


def build_wrapped(args: argparse.Namespace) -> Tuple[nn.Module, bool]:
    from model_seg import arch_cfg_from_checkpoint, build_model
    # Peek at the checkpoint's recorded architecture BEFORE building, so a run
    # trained with --decoder separable exports as a separable-decoder model
    # instead of silently loading into the dense default.
    cfg = {"encoder": args.encoder, "decoder": args.decoder,
           "lateral_channels": None if args.lateral_channels < 0 else args.lateral_channels}
    use_gcg = not args.no_gcg
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.encoder == "auto":
            cfg = arch_cfg_from_checkpoint(ck)
        # Gating comes from the checkpoint too: exporting a --no-gcg checkpoint
        # into a GCG-enabled graph would ship randomly-initialised gates AND
        # benchmark a latency the control checkpoint doesn't have.
        ck_args = ck.get("args", {}) if isinstance(ck, dict) else {}
        if "no_gcg" in ck_args and not args.no_gcg:
            use_gcg = not ck_args["no_gcg"]
            if not use_gcg:
                print("[info] gcg: OFF (from checkpoint args)")
    elif args.encoder == "auto":
        cfg["encoder"] = "mobilenetv3"
    print(f"[info] arch: encoder={cfg['encoder']} decoder={cfg['decoder']} "
          f"lateral={cfg['lateral_channels']}")
    net = build_model(arch="gcg_unet", num_classes=args.classes,
                      pretrained=False, use_gcg=use_gcg, **cfg)
    trained = load_checkpoint(net, args.checkpoint)
    wrapper = DeployWrapper(net).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper, trained


def export(wrapper: nn.Module, args: argparse.Namespace, size: int, kind: str) -> str:
    import coremltools as ct

    out_name = "class_prob" if kind == "cls" else "lesion_prob"
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
        outputs=[ct.TensorType(name=out_name)],
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

    if kind == "cls":
        mlmodel.short_description = (
            "Referable-DR fundus classifier (MBRSETClassifier). "
            "Input: 0-255 RGB. Output: class probabilities (softmax).")
        mlmodel.output_description[out_name] = "Class probabilities [1,C]; index 1 = positive."
    else:
        mlmodel.short_description = (
            f"GCG-U-Net retinal lesion segmentation ({args.classes} channels). "
            "Input: 0-255 RGB. Output: per-pixel lesion probability in [0,1].")
        mlmodel.output_description[out_name] = "Lesion probabilities [1,C,H,W]."
    mlmodel.input_description["image"] = f"Fundus image, {size}x{size} RGB."

    os.makedirs(args.out_dir, exist_ok=True)
    out = args.output or os.path.join(
        args.out_dir, f"{kind}_coreml_{size}_{args.quantize}.mlpackage")
    mlmodel.save(out)
    print(f"[ok]   saved {out}")
    return out


def verify(path: str, wrapper: nn.Module, size: int, kind: str = "seg") -> None:
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

    out_name = "class_prob" if kind == "cls" else "lesion_prob"
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
    cm_out = np.asarray(
        mlmodel.predict({"image": Image.fromarray(arr)})[out_name]).reshape(torch_out.shape)

    diff = np.abs(torch_out - cm_out)
    print("\n=== verification (PyTorch vs Core ML) ===")
    print(f"  max abs diff  : {diff.max():.6f}")
    if kind == "cls":
        print(f"  torch probs   : {np.round(torch_out.ravel(), 4)}")
        print(f"  coreml probs  : {np.round(cm_out.ravel(), 4)}")
        if torch_out.argmax() != cm_out.argmax():
            print("  [warn] argmax disagrees — inspect before trusting on-device output.")
    else:
        agree = float(((torch_out > 0.5) == (cm_out > 0.5)).mean())
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

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    print(f"\n=== latency on this Mac ({runs} runs) ===")
    for label, cu in (("ANE/ALL", ct.ComputeUnit.ALL), ("CPU    ", ct.ComputeUnit.CPU_ONLY)):
        mlmodel = ct.models.MLModel(path, compute_units=cu)
        for _ in range(5):
            mlmodel.predict({"image": img})        # warm up / let the ANE compile
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            mlmodel.predict({"image": img})
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        print(f"  {label}: median {times[len(times)//2]:6.1f} ms   "
              f"p10/p90 {times[int(len(times)*0.1)]:.1f}/{times[int(len(times)*0.9)]:.1f} ms")
    print("  (indicative only — measure on the target iPhone for real numbers)")


def main() -> int:
    p = argparse.ArgumentParser(description="Core ML export for on-device segmentation.")
    p.add_argument("--checkpoint", default=None,
                   help="Trained weights; omitted => random (deployment smoke test).")
    p.add_argument("--model", default="auto", choices=["auto", "seg", "cls"],
                   help="'auto' reads it from the checkpoint (seg=train_idrid, cls=train_mbrset).")
    p.add_argument("--backbone", default=None,
                   help="cls only: override/for random-weight probes, e.g. "
                        "timm:mobilenetv4_conv_small (default: from checkpoint).")
    p.add_argument("--classes", type=int, default=None,
                   help="Output channels/classes (default: 4 for seg, 2 for cls).")
    p.add_argument("--image-size", type=int, default=None,
                   help="Default: from checkpoint for cls (else 224); 512 for seg.")
    p.add_argument("--out-dir", default="edge_export")
    p.add_argument("--output", default=None)
    p.add_argument("--quantize", choices=("none", "fp16", "int8"), default="fp16")
    p.add_argument("--min-target", default="iOS17", help="e.g. iOS16 / iOS17 / iOS18.")
    p.add_argument("--no-gcg", action="store_true", help="Match a --no-gcg ablation checkpoint.")
    # 'auto' reads the architecture out of the checkpoint (train_idrid.py records
    # it). Override only to export an architecture with no checkpoint, e.g. to
    # measure ANE latency for a variant before it has been trained.
    from model_seg import ENCODER_NAMES                          # noqa: E402
    p.add_argument("--encoder", default="auto", choices=["auto"] + list(ENCODER_NAMES))
    p.add_argument("--decoder", default="dense", choices=["dense", "separable"],
                   help="Ignored unless --encoder is given explicitly (else read from ckpt).")
    p.add_argument("--lateral-channels", type=int, default=-1)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()

    kind = detect_model_kind(args)
    if kind == "cls":
        wrapper, trained, size = build_wrapped_cls(args)
    else:
        if args.classes is None:
            args.classes = 4
        if args.image_size is None:
            args.image_size = 512
        size = args.image_size
        wrapper, trained = build_wrapped(args)
    n_params = sum(q.numel() for q in wrapper.parameters())
    print(f"[info] model    : {'classifier' if kind == 'cls' else 'GCG-U-Net'}, "
          f"{n_params/1e6:.2f}M params @ {size}px")
    print(f"[info] weights  : {args.checkpoint if trained else 'RANDOM (untrained)'}")
    if not trained:
        print("[warn] No checkpoint given — predictions are meaningless; "
              "latency and op coverage are still valid signal.")

    out = export(wrapper, args, size, kind)
    if not args.skip_verify:
        verify(out, wrapper, size, kind)
    if not args.skip_benchmark:
        benchmark(out, size, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
