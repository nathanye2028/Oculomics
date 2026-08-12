"""
check_env.py
============
Remote-server sanity check. Run this FIRST after setting up the environment on
a new machine (local or remote GPU box) to confirm everything the project needs
actually works on that hardware:

  * Python / Torch / Torchvision versions
  * GPU visibility (CUDA device count + names, or MPS, else CPU)
  * a real forward+backward pass of the segmentation model on the chosen device
  * AMP (fp16 autocast) works
  * distributed (DDP) availability + NCCL (for multi-GPU torchrun)

Usage:
    python check_env.py
Exit code 0 = good to train. Non-zero = something to fix before training.
"""
from __future__ import annotations

import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"python      : {platform.python_version()}  ({sys.executable})")
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
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu {i}    : {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
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
        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(use_amp or device.type == "mps")):
            y = m(x)
        loss = y.float().mean()
        loss.backward()
        print(f"GCG-U-Net   : forward {tuple(y.shape)} + backward OK  (amp={'on' if use_amp else 'off/mps'})")
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

    print("\nOK: environment ready to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
