"""
check_env.py
============
Remote-server sanity check. Run this FIRST after setting up the environment on
a new machine (local or remote GPU box) to confirm everything the project needs
actually works on that hardware:

  * Python version inside the supported range (>=3.9, <3.13 — torch wheels)
  * Python / Torch / Torchvision versions
  * GPU visibility (CUDA device count + names, or MPS, else CPU)
  * a real forward+backward pass of the segmentation model on the chosen device
  * mixed precision works with the dtype the trainers actually use
    (bf16 on MPS, fp16 on CUDA — see train_idrid.py)
  * distributed (DDP) availability + NCCL (for multi-GPU torchrun)
  * ``--deploy``: the optional export / evaluation stack (coremltools, onnx,
    onnxruntime, timm, scikit-learn, scipy) — missing ones WARN, not fail,
    because training does not need them.

Usage:
    python check_env.py            # training readiness
    python check_env.py --deploy   # + export/eval libraries
Exit code 0 = good to train. Non-zero = something to fix before training.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The supported interpreter range. The lower bound is the syntax/typing this
# repo uses; the upper bound is where torch / coremltools wheels stop existing
# (on this Mac the default python3 is 3.14 with no torch wheel at all, which
# fails with an unhelpful pip resolver error rather than "wrong Python").
PY_MIN = (3, 9)
PY_MAX_EXCLUSIVE = (3, 14)   # 3.13 = the GPU cluster; 3.14 has no torch/coremltools wheels yet


def python_supported(version_info) -> bool:
    """True when ``version_info`` (any (major, minor, ...) sequence) is inside
    [PY_MIN, PY_MAX_EXCLUSIVE). Pure so it can be unit-tested with fake tuples."""
    major, minor = int(version_info[0]), int(version_info[1])
    return PY_MIN <= (major, minor) < PY_MAX_EXCLUSIVE


def python_gate() -> bool:
    """Print the interpreter and the supported range; False if outside it."""
    v = sys.version_info
    lo = ".".join(map(str, PY_MIN))
    hi = ".".join(map(str, PY_MAX_EXCLUSIVE))
    print(f"python      : {platform.python_version()}  ({sys.executable})")
    print(f"supported   : >={lo},<{hi}")
    if python_supported(v):
        return True
    print(f"!! Python {v.major}.{v.minor} is outside the supported range >={lo},<{hi}.")
    print("   torch / coremltools publish no wheels for it, so `pip install -r "
          "requirements.txt` will fail or install a CPU-only stub.")
    print("   Fixes:")
    print("     macOS : the Xcode/CLT interpreter is fine ->  /usr/bin/python3 -m venv .venv")
    print("     any   : pyenv install 3.11 && pyenv local 3.11  (then recreate .venv)")
    return False


def _probe(name: str, import_name: str = None, darwin_only: bool = False) -> bool:
    """Import an optional library and print its version; WARN (not fail) if absent."""
    import importlib
    if darwin_only and sys.platform != "darwin":
        print(f"{name:<12}: n/a on {sys.platform} (macOS only)")
        return True
    try:
        mod = importlib.import_module(import_name or name)
        print(f"{name:<12}: {getattr(mod, '__version__', '?')}")
        return True
    except Exception as e:  # noqa
        print(f"{name:<12}: WARN missing ({type(e).__name__}) -> pip install {name}")
        return False


def deploy_section() -> None:
    print("\n" + "=" * 60)
    print("DEPLOY / EVAL STACK (optional — WARN only)")
    print("=" * 60)
    # coremltools is the ANE path (export_coreml.py) and only runs on macOS;
    # onnx + onnxruntime are the portable CPU path (edge_optimize.py,
    # evaluate_deploy.py); timm supplies the teacher/alt backbones; sklearn and
    # scipy supply AUROC / ROC curves and the statistical tests in the report.
    missing = [n for n, imp, mac in (("coremltools", None, True), ("onnx", None, False),
                                     ("onnxruntime", None, False), ("timm", None, False),
                                     ("scikit-learn", "sklearn", False), ("scipy", None, False))
               if not _probe(n, imp, mac)]
    if missing:
        print(f"-> {len(missing)} optional package(s) missing: {', '.join(missing)}. "
              "Training is unaffected; export/evaluation scripts that need them will not run.")
    else:
        print("-> deploy stack complete.")


def main() -> int:
    p = argparse.ArgumentParser(description="Environment sanity check (training readiness).")
    p.add_argument("--deploy", action="store_true",
                   help="Also probe the export/evaluation libraries (coremltools, onnx, "
                        "onnxruntime, timm, scikit-learn, scipy). Missing ones warn, not fail.")
    args = p.parse_args()

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    if not python_gate():
        return 1
    try:
        import torch
        import torchvision
    except Exception as e:  # noqa
        print(f"!! torch/torchvision import failed: {e}")
        print("   -> run setup_remote.sh / pip install -r requirements.txt")
        return 1
    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")

    print("\n" + "=" * 60)
    print("ACCELERATOR")
    print("=" * 60)
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if cuda:
        n = torch.cuda.device_count()
        print(f"CUDA        : YES  ({n} device{'s' if n != 1 else ''})")
        for i in range(n):
            pr = torch.cuda.get_device_properties(i)
            print(f"  gpu {i}    : {pr.name}  {pr.total_memory/1e9:.1f} GB  sm_{pr.major}{pr.minor}")
        device = torch.device("cuda:0")
    elif mps:
        print("CUDA        : no  |  MPS (Apple GPU): YES")
        device = torch.device("mps")
    else:
        print("CUDA        : no  |  MPS: no  ->  CPU only")
        device = torch.device("cpu")
    print(f"using device: {device}")

    print("\n" + "=" * 60)
    print("MODEL FORWARD/BACKWARD")
    print("=" * 60)
    try:
        from model_seg import build_model
        # pretrained=False on purpose: this is a fast hardware check, not a
        # training run, and it must not block on downloading ImageNet weights.
        m = build_model(arch="gcg_unet", num_classes=4, pretrained=False, use_gcg=True).to(device)
        x = torch.randn(2, 3, 256, 256, device=device)
        # Mirror the trainers' policy (train_idrid.py): autocast on any
        # accelerator, bf16 on MPS (fp16 autocast there is both slower and the
        # MA-collapse hazard precision_check.py hunts) and fp16 on CUDA.
        use_amp = device.type in ("cuda", "mps")
        amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            y = m(x)
        loss = y.float().mean()
        loss.backward()
        amp_desc = f"on:{str(amp_dtype).replace('torch.', '')}" if use_amp else "off (cpu)"
        print(f"GCG-U-Net   : forward {tuple(y.shape)} + backward OK  "
              f"(amp={amp_desc}, output dtype={str(y.dtype).replace('torch.', '')})")
    except Exception as e:  # noqa
        print(f"!! model forward/backward failed on {device}: {e}")
        return 2

    print("\n" + "=" * 60)
    print("DISTRIBUTED (multi-GPU DDP)")
    print("=" * 60)
    avail = torch.distributed.is_available()
    nccl = torch.distributed.is_nccl_available() if avail else False
    print(f"dist avail  : {avail}")
    print(f"nccl avail  : {nccl}  ({'multi-GPU torchrun ready' if nccl else 'single-GPU only / use gloo'})")
    try:
        import importlib_metadata  # noqa: F401
        print("torchrun dep: importlib_metadata present")
    except Exception:
        print("torchrun dep: !! importlib_metadata MISSING -> pip install importlib_metadata")

    if args.deploy:
        deploy_section()

    print("\nOK: environment ready to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
