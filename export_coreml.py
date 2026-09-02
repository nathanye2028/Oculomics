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

What is NOT baked in — and therefore what the app must reproduce — is the FOV
crop and the resize. That recipe is written into the model's
``user_defined_metadata`` (and printed) so an iOS developer can copy it rather
than guess; ``--verify-images`` checks the whole chain on real fundus photos.

Which BatchNorm statistics ship
-------------------------------
``train_mbrset.py --bn-adapt`` re-estimates BN running stats on the target
domain's images (AdaBN, label-free) and stores them alongside the source
weights under ``model_bnadapt``. ``--bn-stats adapted`` exports those. The two
are different models for deployment purposes (different domain assumptions),
so the choice is recorded in the metadata.

Latency on this Mac is an OPTIMISTIC bound
------------------------------------------
A MobileNet-class model at 384-512 px is memory-bandwidth-bound, and a Mac has
several times the bandwidth of any iPhone. Numbers from :func:`benchmark` are
therefore a lower bound on phone latency, useful for op coverage and relative
comparisons between architectures. Anything quotable as an iPhone number needs
an Xcode Core ML performance report on the target device.

Export is a deployment smoke test, and is worth running *before* the model is
good. With random weights the predictions are meaningless, but op coverage
(does anything fall back off the ANE?) and relative latency are real signal,
and those are expensive to discover late.

Run:
    # segmentation (GCG-U-Net)
    python export_coreml.py --checkpoint checkpoints/gcg_seed0.pt --classes 4
    # classification (MBRSETClassifier; arch/backbone/size read from the ckpt)
    python export_coreml.py --checkpoint ck_kd_v4_384/kd_seed1.pt
    # ...with AdaBN statistics and a real-image fidelity check
    python export_coreml.py --checkpoint ck_kd_v4_384/kd_seed1.pt --bn-stats adapted \
        --verify-images /path/to/mbrset/images --verify-n 16
    # architecture-only latency probe (random weights)
    python export_coreml.py --model cls --backbone timm:mobilenetv4_conv_small --image-size 384
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# fundus_utils.crop_to_fov's default: a pixel is "retina" when max(R,G,B) > tol.
FOV_TOL = 12

# Which state_dict inside a train_mbrset.py checkpoint each --bn-stats picks.
BN_STATS_KEY = {"source": "model", "adapted": "model_bnadapt"}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


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


# --------------------------------------------------------------------------- #
# Checkpoint handling — the file is read ONCE and the dict passed around.
# --------------------------------------------------------------------------- #
def read_checkpoint(path: Optional[str]) -> Optional[dict]:
    """torch.load the checkpoint once (None when no path was given)."""
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"could not find a state_dict inside {path}")
    return obj


def ckpt_args(ck: Optional[dict]) -> dict:
    return ck.get("args", {}) if isinstance(ck, dict) else {}


def gcg_from_args(a: dict, default: bool = True) -> bool:
    """train_mbrset.py records ``no_gcg`` in the checkpoint args; newer results
    JSON records ``use_gcg`` directly. Accept either, newest key first."""
    if "use_gcg" in a:
        return bool(a["use_gcg"])
    # Checkpoints from before 2026-09-01 record only the CLI flag. For a timm
    # backbone the old model.py silently built NO gate regardless of the flag
    # (and the new one raises on the combination), so the truthful answer for
    # every such checkpoint — including the deployed V4-Small students — is False.
    if str(a.get("backbone", "")).startswith("timm:"):
        return False
    if "no_gcg" in a:
        return not a["no_gcg"]
    return default


def extract_state(ck: dict, bn_stats: str = "source") -> dict:
    """Pick the state_dict to export from a loaded checkpoint dict.

    ``adapted`` demands the AdaBN weights and refuses to silently fall back to
    the source ones — shipping source BN stats while believing they were
    adapted is exactly the kind of skew this script exists to prevent.
    """
    if bn_stats == "adapted":
        key = BN_STATS_KEY["adapted"]
        if key not in ck:
            raise SystemExit(
                f"[fatal] --bn-stats adapted: checkpoint has no '{key}' entry — this "
                "checkpoint was trained without --bn-adapt or before the key existed. "
                "Re-run train_mbrset.py with --external-test-root ... --bn-adapt, or export "
                "with --bn-stats source.")
        return ck[key]
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ck:
            return ck[key]
    if ck and all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck                                   # bare state_dict
    raise ValueError("could not find a state_dict inside the checkpoint")


def load_checkpoint(net: nn.Module, ck: Optional[dict], bn_stats: str = "source") -> bool:
    """Load weights into ``net`` from an already-read checkpoint dict.

    Returns True if real weights were loaded. Accepts a bare state_dict or one
    wrapped under ``model`` / ``state_dict`` / ``model_state_dict`` (or
    ``model_bnadapt`` with ``bn_stats='adapted'``), and strips any ``module.``
    DDP prefix.
    """
    if ck is None:
        return False
    obj = extract_state(ck, bn_stats)
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


def detect_model_kind(args, ck: Optional[dict]) -> str:
    """'cls' vs 'seg', read from the checkpoint when --model auto.

    Classifier checkpoints (train_mbrset.py) record ``task``/``backbone`` in
    their args; segmentation checkpoints (train_idrid.py) record ``lesions``.
    """
    if args.model != "auto":
        return args.model
    if ck is not None:
        if "lesions" in ck or "lesions" in ckpt_args(ck):
            return "seg"
        a = ckpt_args(ck)
        if "backbone" in a or a.get("task") in (
                "dr_referable", "dr_binary", "dr_grade", "edema", "quality", "artifacts"):
            return "cls"
    return "seg"


def build_wrapped_cls(args: argparse.Namespace, ck: Optional[dict]) -> Tuple[nn.Module, bool, int]:
    """Classifier path: rebuild MBRSETClassifier from the checkpoint's own args."""
    from model import MBRSETClassifier
    from train_mbrset import backbone_kwargs_for
    a = ckpt_args(ck)
    backbone = args.backbone or a.get("backbone", "mobilenetv3_small")
    size = args.image_size or int(a.get("image_size", 224))
    n_cls = args.classes if args.classes is not None else 2
    use_gcg = False if args.no_gcg else gcg_from_args(a, default=True)
    net = MBRSETClassifier(num_classes=n_cls, pretrained=False,
                           use_gcg=use_gcg,
                           gcg_variant=a.get("gcg_variant", "baseline"),
                           backbone=backbone,
                           backbone_kwargs=backbone_kwargs_for(backbone, size))
    print(f"[info] arch: {backbone}  task={a.get('task', '?')}  classes={n_cls}  size={size}  "
          f"gcg={'on' if use_gcg else 'off'}")
    trained = load_checkpoint(net, ck, args.bn_stats)
    wrapper = ClsDeployWrapper(net).eval()
    for p_ in wrapper.parameters():
        p_.requires_grad_(False)
    return wrapper, trained, size


def build_wrapped(args: argparse.Namespace, ck: Optional[dict]) -> Tuple[nn.Module, bool]:
    from model_seg import arch_cfg_from_checkpoint, build_model
    # Peek at the checkpoint's recorded architecture BEFORE building, so a run
    # trained with --decoder separable exports as a separable-decoder model
    # instead of silently loading into the dense default.
    cfg = {"encoder": args.encoder, "decoder": args.decoder,
           "lateral_channels": None if args.lateral_channels < 0 else args.lateral_channels}
    use_gcg = not args.no_gcg
    if ck is not None:
        if args.encoder == "auto":
            cfg = arch_cfg_from_checkpoint(ck)
        # Gating comes from the checkpoint too: exporting a --no-gcg checkpoint
        # into a GCG-enabled graph would ship randomly-initialised gates AND
        # benchmark a latency the control checkpoint doesn't have.
        a = ckpt_args(ck)
        if ("no_gcg" in a or "use_gcg" in a) and not args.no_gcg:
            use_gcg = gcg_from_args(a)
            if not use_gcg:
                print("[info] gcg: OFF (from checkpoint args)")
    elif args.encoder == "auto":
        cfg["encoder"] = "mobilenetv3"
    print(f"[info] arch: encoder={cfg['encoder']} decoder={cfg['decoder']} "
          f"lateral={cfg['lateral_channels']}")
    net = build_model(arch="gcg_unet", num_classes=args.classes,
                      pretrained=False, use_gcg=use_gcg, **cfg)
    trained = load_checkpoint(net, ck, args.bn_stats)
    wrapper = DeployWrapper(net).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper, trained


# --------------------------------------------------------------------------- #
# Preprocessing contract — what the app must do before calling the model.
# --------------------------------------------------------------------------- #
def preprocess_spec(kind: str, size: int, bn_stats: str, checkpoint: Optional[str],
                    quantize: str) -> Dict[str, str]:
    """The exact train-time preprocessing, as strings for user_defined_metadata.

    Mirrors ``dataset.MBRSETDataset`` (cls: FOV crop -> torchvision Resize with
    antialias on a float tensor -> ImageNet Normalize) and ``idrid_dataset``
    whole-image mode (seg: FOV crop -> PIL BILINEAR resize -> ImageNet
    Normalize). Only the normalisation is inside the graph; the crop and the
    resize are the app's job, and any deviation there is silent accuracy loss.
    """
    # Rounded so the metadata reads 0.485, not float32's 0.48500001430511475.
    mean = tuple(round(v, 4) for v in IMAGENET_MEAN.flatten().tolist())
    std = tuple(round(v, 4) for v in IMAGENET_STD.flatten().tolist())
    spec = {
        "input.name": "image",
        "input.layout": f"RGB, {size}x{size}, uint8 0-255 (Core ML ImageType; no app-side scaling)",
        "input.channel_order": "RGB (not BGR)",
        "preprocess.1_fov_crop": (
            f"crop to the bounding box of pixels where max(R,G,B) > {FOV_TOL} "
            f"(fundus_utils.crop_to_fov, tol={FOV_TOL}); removes the black border"),
        "preprocess.3_normalise": (
            f"BAKED INTO THE GRAPH: (x - 255*mean)/(255*std), ImageNet mean={mean} "
            f"std={std}. The app must NOT normalise."),
        "bn_stats": bn_stats + (" (AdaBN running stats from train_mbrset.py --bn-adapt)"
                                if bn_stats == "adapted" else " (source-domain running stats)"),
        "checkpoint": checkpoint or "RANDOM (untrained; op-coverage/latency probe only)",
        "export.quantize": quantize,
        "export.torch": torch.__version__,
    }
    if kind == "cls":
        spec["preprocess.2_resize"] = (
            f"resize the FOV crop to exactly {size}x{size} (aspect ratio NOT preserved; no "
            "centre crop) with bilinear interpolation and antialiasing — torchvision "
            "Resize(antialias=True) on a float [0,1] tensor, i.e. a proper area-weighted "
            "downscale, not nearest/plain bilinear")
        spec["preprocess.0_decode"] = (
            "training decoded JPEGs with PIL draft mode (DCT-domain downscale to the "
            "smallest scale >= target size); a full-resolution decode differs by JPEG "
            "rounding only")
        spec["postprocess"] = "softmax BAKED IN; output[1] = P(positive class)"
    else:
        spec["preprocess.2_resize"] = (
            f"resize the FOV crop to exactly {size}x{size} (aspect ratio NOT preserved) with "
            "PIL Image.BILINEAR (antialiased for downscale) — idrid_dataset whole-image mode")
        spec["postprocess"] = "sigmoid BAKED IN; output is per-pixel, per-lesion probability"
    try:
        import coremltools as ct
        spec["export.coremltools"] = ct.__version__
    except Exception:  # noqa
        pass
    return spec


def print_spec(spec: Dict[str, str]) -> None:
    print("\n=== preprocessing contract (also in user_defined_metadata) ===")
    for k in sorted(spec):
        print(f"  {k:<24}: {spec[k]}")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export(wrapper: nn.Module, args: argparse.Namespace, size: int, kind: str,
           spec: Dict[str, str]) -> str:
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
    mlmodel.input_description["image"] = f"Fundus image, {size}x{size} RGB, FOV-cropped and resized."
    # The preprocessing contract rides with the artefact: Xcode shows this
    # dictionary in the model inspector, so the iOS developer never has to
    # find this script.
    for k, v in spec.items():
        mlmodel.user_defined_metadata[k] = str(v)

    os.makedirs(args.out_dir, exist_ok=True)
    out = args.output or os.path.join(
        args.out_dir, f"{kind}_coreml_{size}_{args.quantize}.mlpackage")
    mlmodel.save(out)
    print(f"[ok]   saved {out}")
    return out


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def list_images(images_dir: str, n: int) -> List[str]:
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS))
    return [os.path.join(images_dir, f) for f in files[:n]]


def preprocess_for_verify(path: str, size: int, kind: str) -> Tuple[torch.Tensor, np.ndarray]:
    """One real image -> (what the trained net saw, what the app feeds Core ML).

    Returns ``x_norm`` [1,3,H,W], the normalised float tensor produced by the
    training-time pipeline, and ``u8`` [H,W,3] uint8, the same resized image
    quantised to bytes as an iOS app would hand it to Core ML. The uint8
    rounding is a real part of deployment, so it is deliberately inside the
    comparison rather than hidden.
    """
    from PIL import Image
    from fundus_utils import crop_to_fov

    with Image.open(path) as im:
        if kind == "cls":
            im.draft("RGB", (size, size))        # MBRSETDataset(draft_decode=True)
        im = im.convert("RGB")
    arr, _ = crop_to_fov(np.asarray(im), tol=FOV_TOL)
    im = Image.fromarray(arr)

    if kind == "cls":
        from dataset import build_transforms
        # The dataset's own eval pipeline, twice: once as trained (Normalize),
        # once with an identity Normalize so the pre-normalisation [0,1] pixels
        # are exactly the ones the resize produced (Normalize is elementwise,
        # so the two share the resized tensor bit-for-bit).
        x_norm = build_transforms(split="val", image_size=size)(im)
        x01 = build_transforms(split="val", image_size=size,
                               mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))(im)
        x_norm = torch.as_tensor(x_norm).float()
        x01 = torch.as_tensor(x01).float()
    else:
        # idrid_dataset whole-image mode: PIL BILINEAR resize (outputs uint8, so
        # there is no extra rounding step for the seg path).
        res = np.asarray(im.resize((size, size), Image.BILINEAR))
        x01 = torch.from_numpy(res.astype(np.float32) / 255.0).permute(2, 0, 1)
        x_norm = (x01 - IMAGENET_MEAN) / IMAGENET_STD

    u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return x_norm[None], np.ascontiguousarray(u8)


@torch.no_grad()
def reference_output(wrapper: nn.Module, x_norm: torch.Tensor, kind: str) -> np.ndarray:
    """Run the UNWRAPPED net on already-normalised input, then the same
    activation the wrapper bakes in. This is the number the training and
    evaluation scripts would have produced for this image."""
    logits = wrapper.net(x_norm)
    out = torch.softmax(logits, dim=1) if kind == "cls" else torch.sigmoid(logits)
    return out.numpy()


def verify(path: str, wrapper: nn.Module, size: int, kind: str = "seg",
           images_dir: Optional[str] = None, n_images: int = 16, tol: float = 1e-2) -> bool:
    """Compare Core ML output against PyTorch.

    With ``images_dir``: real fundus photos through the real preprocessing,
    pass/fail against ``tol``. Without: one uniform-noise input, which proves
    the graph runs and nothing else ("smoke only", never fails).

    Pass criterion (cls): max |p_torch - p_coreml| <= tol on every image AND
    argmax agrees on every image. For seg the 99th-percentile pixel difference
    is used instead of the max: at fp16 a handful of boundary pixels on a
    512x512x4 map routinely diverge by more than any sane tolerance without
    the mask changing, and a hard max would fail every export for nothing.
    Mask agreement at thr=0.5 must also be >= 99%.
    """
    import coremltools as ct
    from PIL import Image

    if sys.platform != "darwin":
        print("[skip] verification requires macOS")
        return True

    out_name = "class_prob" if kind == "cls" else "lesion_prob"
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)

    def run_coreml(u8: np.ndarray, shape) -> np.ndarray:
        return np.asarray(mlmodel.predict({"image": Image.fromarray(u8)})[out_name],
                          dtype=np.float32).reshape(shape)

    # ---- smoke only: noise input, no pass/fail --------------------------- #
    if not images_dir:
        rng = np.random.default_rng(0)
        u8 = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        with torch.no_grad():
            torch_out = wrapper(torch.from_numpy(u8.astype(np.float32)).permute(2, 0, 1)[None]).numpy()
        cm_out = run_coreml(u8, torch_out.shape)
        diff = np.abs(torch_out - cm_out)
        print("\n=== verification: SMOKE ONLY (uniform noise, no pass/fail) ===")
        print(f"  max abs diff  : {diff.max():.6f}")
        if kind == "cls":
            print(f"  torch probs   : {np.round(torch_out.ravel(), 4)}")
            print(f"  coreml probs  : {np.round(cm_out.ravel(), 4)}")
        else:
            agree = float(((torch_out > 0.5) == (cm_out > 0.5)).mean())
            print(f"  mask agreement: {agree*100:.3f}% of pixels @ thr=0.5")
        print("  Noise says nothing about real-image fidelity; pass --verify-images DIR "
              "for a pass/fail check.")
        return True

    # ---- real images: pass/fail ------------------------------------------ #
    if not os.path.isdir(images_dir):
        raise SystemExit(f"[fatal] --verify-images: not a directory: {images_dir}")
    files = list_images(images_dir, n_images)
    if not files:
        raise SystemExit(f"[fatal] --verify-images: no {IMAGE_EXTS} files in {images_dir}")

    print(f"\n=== verification: {len(files)} real images from {images_dir} (tol={tol:g}) ===")
    worst_diff, worst_u8, disagree, agree_px = 0.0, 0.0, 0, []
    for f in files:
        x_norm, u8 = preprocess_for_verify(f, size, kind)
        ref = reference_output(wrapper, x_norm, kind)              # training pipeline
        with torch.no_grad():                                       # same bytes, fp32 torch
            ref_u8 = wrapper(torch.from_numpy(u8.astype(np.float32)).permute(2, 0, 1)[None]).numpy()
        cm = run_coreml(u8, ref.shape)
        d = np.abs(ref - cm)
        d_u8 = np.abs(ref_u8 - cm)
        if kind == "cls":
            stat, stat_u8 = float(d.max()), float(d_u8.max())
            ok_arg = int(ref.argmax()) == int(cm.argmax())
            disagree += int(not ok_arg)
            print(f"  {os.path.basename(f):<28} p_torch={ref.ravel()[1]:.4f} "
                  f"p_coreml={cm.ravel()[1]:.4f} maxdiff={stat:.5f} "
                  f"{'' if ok_arg else '  <-- ARGMAX DISAGREES'}")
        else:
            stat, stat_u8 = float(np.percentile(d, 99)), float(np.percentile(d_u8, 99))
            a = float(((ref > 0.5) == (cm > 0.5)).mean())
            agree_px.append(a)
            print(f"  {os.path.basename(f):<28} p99diff={stat:.5f} mask-agree={a*100:.3f}%")
        worst_diff, worst_u8 = max(worst_diff, stat), max(worst_u8, stat_u8)

    label = "max abs diff" if kind == "cls" else "p99 abs diff"
    print(f"  worst {label:<13}: {worst_diff:.6f}   (vs training-pipeline float input)")
    print(f"  worst {label:<13}: {worst_u8:.6f}   (vs the same uint8 bytes in fp32 torch — "
          "isolates fp16/ANE error from uint8 rounding)")
    if kind == "cls":
        print(f"  argmax agreement : {len(files) - disagree}/{len(files)}")
        ok = worst_diff <= tol and disagree == 0
    else:
        min_agree = min(agree_px)
        print(f"  min mask agreement: {min_agree*100:.3f}% @ thr=0.5")
        ok = worst_diff <= tol and min_agree >= 0.99
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  [fail] Core ML output deviates from PyTorch beyond --tol. Do not ship this "
              "artefact; try --quantize none to see whether fp16 is the cause.")
    return ok


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def benchmark(path: str, size: int, runs: int, warmup: int) -> Dict[str, float]:
    """Time the Core ML model on this Mac, per compute-unit setting.

    This is an OPTIMISTIC bound, not an iPhone number. The model is
    memory-bandwidth-bound at these resolutions and a Mac has several times a
    phone's bandwidth, so every median here is a floor on device latency.
    Read it for op coverage (does ALL match CPU_AND_NE? if not, something is
    falling off the ANE) and for relative comparisons between exports. The
    quotable number comes from an Xcode Core ML performance report on the
    target iPhone.

    Compute units are benchmarked SEPARATELY and labelled by what they are:
    ``ALL`` lets the framework choose and is NOT a guarantee of ANE residency.
    """
    import coremltools as ct
    from PIL import Image

    if sys.platform != "darwin":
        print("[skip] benchmark requires macOS")
        return {}

    units = [("ALL (framework picks)", ct.ComputeUnit.ALL),
             ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]
    if hasattr(ct.ComputeUnit, "CPU_AND_GPU"):
        units.append(("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU))
    units.append(("CPU_ONLY", ct.ComputeUnit.CPU_ONLY))

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    print(f"\n=== latency on this Mac ({warmup} warm-up + {runs} timed runs per unit) ===")
    print(f"  {'compute unit':<22}{'p10':>9}{'median':>9}{'p90':>9}")
    medians: Dict[str, float] = {}
    for label, cu in units:
        try:
            mlmodel = ct.models.MLModel(path, compute_units=cu)
            for _ in range(warmup):
                mlmodel.predict({"image": img})    # let the ANE compile / caches fill
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                mlmodel.predict({"image": img})
                times.append((time.perf_counter() - t0) * 1000.0)
        except Exception as e:  # noqa
            print(f"  {label:<22}  unavailable ({str(e)[:50]})")
            continue
        p10, med, p90 = (float(np.percentile(times, q)) for q in (10, 50, 90))
        medians[label] = med
        print(f"  {label:<22}{p10:>9.2f}{med:>9.2f}{p90:>9.2f}  ms")

    all_med = medians.get("ALL (framework picks)")
    ne_med = medians.get("CPU_AND_NE")
    if all_med is not None and ne_med is not None and ne_med > 0:
        rel = abs(all_med - ne_med) / ne_med
        if rel > 0.20:
            print(f"  [warn] median(ALL) differs from median(CPU_AND_NE) by {rel*100:.0f}% — "
                  "the framework is probably NOT keeping this graph on the ANE. Check op "
                  "coverage in an Xcode performance report before quoting any latency.")
    print("  Optimistic bound on a Mac (higher memory bandwidth than any iPhone). An iPhone "
          "number needs an Xcode Core ML performance report on the device.")
    return medians


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Core ML export for on-device inference.")
    p.add_argument("--checkpoint", default=None,
                   help="Trained weights; omitted => random (deployment smoke test).")
    p.add_argument("--bn-stats", default="source", choices=("source", "adapted"),
                   help="Which BatchNorm statistics to export: 'source' = the checkpoint's "
                        "'model' weights; 'adapted' = the AdaBN weights train_mbrset.py "
                        "--bn-adapt stores under 'model_bnadapt' (fails if absent).")
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
    # Verification
    p.add_argument("--verify-images", default=None, metavar="DIR",
                   help="Directory of real fundus JPEG/PNGs: run the training preprocessing "
                        "and compare Core ML vs PyTorch with pass/fail (exit 1 on failure). "
                        "Without it only a noise smoke test runs.")
    p.add_argument("--verify-n", type=int, default=16, help="Max images from --verify-images.")
    p.add_argument("--tol", type=float, default=1e-2,
                   help="Max |p_torch - p_coreml| allowed (1e-2 is appropriate for fp16).")
    p.add_argument("--skip-verify", action="store_true")
    # Benchmark — 10 warm-up + 60 timed matches the README's stated protocol.
    p.add_argument("--warmup", type=int, default=10, help="Untimed runs per compute unit.")
    p.add_argument("--runs", type=int, default=60, help="Timed runs per compute unit.")
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()

    ck = read_checkpoint(args.checkpoint)          # once; every consumer gets the dict
    kind = detect_model_kind(args, ck)
    if kind == "cls":
        wrapper, trained, size = build_wrapped_cls(args, ck)
    else:
        if args.classes is None:
            args.classes = 4
        if args.image_size is None:
            args.image_size = 512
        size = args.image_size
        wrapper, trained = build_wrapped(args, ck)
    n_params = sum(q.numel() for q in wrapper.parameters())
    print(f"[info] model    : {'classifier' if kind == 'cls' else 'GCG-U-Net'}, "
          f"{n_params/1e6:.2f}M params @ {size}px")
    print(f"[info] weights  : {args.checkpoint if trained else 'RANDOM (untrained)'}")
    print(f"[info] bn stats : {args.bn_stats}"
          + ("  (AdaBN from train_mbrset.py --bn-adapt)" if args.bn_stats == "adapted" else ""))
    if not trained:
        print("[warn] No checkpoint given — predictions are meaningless; "
              "latency and op coverage are still valid signal.")

    spec = preprocess_spec(kind, size, args.bn_stats, args.checkpoint if trained else None,
                           args.quantize)
    out = export(wrapper, args, size, kind, spec)
    print_spec(spec)

    ok = True
    if not args.skip_verify:
        ok = verify(out, wrapper, size, kind, images_dir=args.verify_images,
                    n_images=args.verify_n, tol=args.tol)
    if not args.skip_benchmark:
        benchmark(out, size, args.runs, args.warmup)

    print(f"\n[summary] {out}  kind={kind} size={size} quantize={args.quantize} "
          f"bn_stats={args.bn_stats} weights={'trained' if trained else 'RANDOM'} "
          f"verify={'PASS' if ok else 'FAIL'}"
          + ("" if args.verify_images else " (smoke only)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
