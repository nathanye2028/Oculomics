"""
retfound.py
===========
The frozen RETFound-DINOv2 encoder (``YukunZhou/RETFound_dinov2_meh``): a ViT-L/14
at 224 px whose 1,024-number CLS embedding is the input to the systemic probes
(``extract_retfound.py`` writes it once per image; ``probe_systemic.py`` fits a
small classifier on it).

Loading follows the RETFound repository's ``main_finetune.py``: the checkpoint's
``teacher`` weights, ``backbone.`` stripped, ``mlp.w12/w3`` renamed to
``mlp.fc1/fc2``, the position embedding resampled if its grid differs. Unlike
that script, loading here is CHECKED: a missing transformer weight is fatal,
because a half-loaded ViT still returns 1,024 numbers per image and nothing
downstream would notice that they are random.

Weights are gated on HuggingFace (CC BY-NC 4.0). Request access on the model
page, then on the box ``python -c "from huggingface_hub import login; login()"`` once; after that
``--weights YukunZhou/RETFound_dinov2_meh`` downloads to the HF cache. A local
``.pth`` path works too.

``ProbeTeacher`` turns a fitted probe (``probe_systemic.py``'s ``*_probe.npz``)
back into a torch module with the ``forward_with_feat`` interface of a
train_mbrset.py teacher, so ``train_mbrset.py --teacher <probe.npz>`` distils the
probe into the phone model.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = "YukunZhou/RETFound_dinov2_meh"
TIMM_NAME = "vit_large_patch14_dinov2"
IMAGE_SIZE = 224
POOLS = ("cls", "mean")


def resolve_weights(weights: str) -> str:
    """A local checkpoint path, or a HuggingFace repo id whose file is
    ``<repo name>.pth`` (the RETFound naming, as in its main_finetune.py)."""
    if os.path.isfile(weights):
        return weights
    parts = weights.strip("/").split("/")
    if len(parts) != 2 or weights.endswith((".pth", ".pt", ".bin")):
        raise SystemExit(f"[fatal] --weights {weights!r} is neither a file nor a HuggingFace "
                         f"repo id (owner/name)")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("[fatal] huggingface_hub is not installed: .venv/bin/pip install huggingface_hub")
    name = weights.rstrip("/").split("/")[-1]
    try:
        return hf_hub_download(repo_id=weights, filename=f"{name}.pth")
    except Exception as e:                                     # gated, offline, no token
        raise SystemExit(
            f"[fatal] could not download {weights} ({type(e).__name__}: {e}).\n"
            f"  The RETFound weights are gated: request access on huggingface.co/{weights},\n"
            f"  then log in once on this machine:\n"
            f"  python -c 'from huggingface_hub import login; login()'")


def file_sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def remap_state_dict(ck: dict) -> Dict[str, torch.Tensor]:
    """The encoder weights of a RETFound / DINOv2 checkpoint, with timm key names."""
    sd = ck
    for key in ("teacher", "model", "state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            sd = ck[key]
            break
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for pre in ("module.", "backbone."):
            if k.startswith(pre):
                k = k[len(pre):]
        if k.startswith(("dino_head.", "ibot_head.", "head.")):
            continue                                           # projection heads, not the encoder
        k = k.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2.")
        out[k] = v
    return out


def build_encoder(weights: str, timm_name: str = TIMM_NAME, image_size: int = IMAGE_SIZE,
                  device: Optional[torch.device] = None) -> Tuple[nn.Module, dict]:
    """The frozen encoder in eval mode, and a record of what was loaded."""
    import timm
    path = resolve_weights(weights)
    model = timm.create_model(timm_name, pretrained=False, img_size=image_size, num_classes=0)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = remap_state_dict(ck)
    if "pos_embed" in sd and sd["pos_embed"].shape != model.pos_embed.shape:
        from timm.layers import resample_abs_pos_embed
        npt = 0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
        sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=model.patch_embed.grid_size,
                                                 num_prefix_tokens=npt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("head")]
    if missing:
        raise SystemExit(f"[fatal] {path}: {len(missing)} encoder weights missing from the checkpoint "
                         f"(e.g. {missing[:5]}); the features would be partly random. Is this a "
                         f"{timm_name} checkpoint?")
    structural = [k for k in unexpected if k.startswith(("blocks.", "patch_embed.", "norm."))]
    if structural:
        raise SystemExit(f"[fatal] {path}: checkpoint weights the model has no place for "
                         f"(e.g. {structural[:5]}): the architecture does not match {timm_name}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is not None:
        model.to(device)
    info = {"weights": weights, "weights_path": os.path.abspath(path), "timm_name": timm_name,
            "image_size": int(image_size), "feat_dim": int(model.num_features),
            "ignored_keys": sorted(unexpected)[:20], "n_ignored_keys": len(unexpected)}
    return model, info


def embed(model: nn.Module, x: torch.Tensor, pool: str = "cls") -> torch.Tensor:
    """[B, D] embedding after the final norm: the CLS token (timm's pooled output,
    what RETFound fine-tunes on) or the mean of the patch tokens."""
    if pool == "cls":
        return model(x)
    if pool == "mean":
        tokens = model.forward_features(x)
        return tokens[:, model.num_prefix_tokens:].mean(1)
    raise ValueError(f"pool must be one of {POOLS}, not {pool!r}")


def collapse_probe(scaler, pca, clf) -> Tuple[np.ndarray, float]:
    """StandardScaler -> PCA -> linear model, folded into one linear map on the raw
    embedding: ``z = f @ w + b`` (the logit for a classifier, the value for a
    regressor). Exact: every stage is affine."""
    coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    b0 = float(np.asarray(clf.intercept_, dtype=np.float64).reshape(-1)[0])
    comp = np.asarray(pca.components_, dtype=np.float64)                # [k, D]
    if getattr(pca, "whiten", False):
        comp = comp / np.sqrt(pca.explained_variance_)[:, None]
    u = comp.T @ coef                                                    # [D]
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mu = np.asarray(scaler.mean_, dtype=np.float64)
    w = u / scale
    b = b0 - float(mu @ w) - float(np.asarray(pca.mean_, dtype=np.float64) @ u)
    return w, b


def save_probe_head(path: str, w: np.ndarray, b: float, meta: dict) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, w=np.asarray(w, dtype=np.float64), b=np.float64(b), meta=json.dumps(meta))
    os.replace(tmp, path)


def load_probe_head(path: str) -> Tuple[np.ndarray, float, dict]:
    z = np.load(path, allow_pickle=False)
    return z["w"], float(z["b"]), json.loads(str(z["meta"]))


class ProbeTeacher(nn.Module):
    """RETFound encoder + the fitted probe as a frozen 2-class teacher. Takes the
    student's batch (ImageNet-normalised, any size), resizes it to the encoder's
    224 px, and returns ``(logits [B, 2], embedding [B, D])`` with logits
    ``[0, z]`` so ``softmax`` gives the probe's probability."""

    def __init__(self, encoder: nn.Module, w: np.ndarray, b: float, pool: str, image_size: int):
        super().__init__()
        self.encoder, self.pool, self.image_size = encoder, pool, int(image_size)
        self.head = nn.Linear(len(w), 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(w, dtype=torch.float32)[None])
            self.head.bias.fill_(float(b))
        self.feat_dim = int(len(w))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward_with_feat(self, x: torch.Tensor):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x.float(), size=(self.image_size, self.image_size), mode="bilinear",
                              align_corners=False, antialias=True)
        f = embed(self.encoder, x, self.pool)
        z = self.head(f.float()).squeeze(1)
        return torch.stack([torch.zeros_like(z), z], dim=1), f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feat(x)[0]


def load_probe_teacher(path: str, device: torch.device, weights: Optional[str] = None):
    """``(teacher, meta)`` from a probe_systemic.py ``*_probe.npz``. The encoder
    comes from the weights the features were extracted with, unless overridden."""
    w, b, meta = load_probe_head(path)
    if meta.get("kind") != "classifier":
        raise SystemExit(f"[fatal] {path} is a {meta.get('kind')!r} probe; only a classifier "
                         f"can be distilled into the 2-class student")
    fm = meta.get("features_meta") or {}
    cached = fm.get("weights_path") if fm.get("weights_path") and os.path.isfile(fm["weights_path"]) else None
    enc, info = build_encoder(weights or cached or fm.get("weights") or DEFAULT_WEIGHTS,
                              timm_name=fm.get("timm_name", TIMM_NAME),
                              image_size=int(fm.get("image_size", IMAGE_SIZE)))
    if fm.get("weights_sha256") and file_sha256(info["weights_path"]) != fm["weights_sha256"]:
        raise SystemExit(f"[fatal] {info['weights_path']} is not the checkpoint the probe's embeddings came "
                         f"from (sha256 differs): the probe's linear map would be applied to other features")
    t = ProbeTeacher(enc, w, b, fm.get("pool", "cls"), int(fm.get("image_size", IMAGE_SIZE)))
    return t.to(device), meta
