"""
extract_retfound.py
===================
One frozen RETFound-DINOv2 embedding per image, computed once and saved, so the
systemic probes (``probe_systemic.py``) fit in seconds instead of training a CNN.

Preprocessing is the CNN runs' own evaluation path (``dataset.MBRSETDataset`` with
``split="val"``): the black border cropped off (``fundus_utils.crop_to_fov``), a
full-frame resize to 224 px, ImageNet normalisation. Every image of the table that
exists on disk is embedded, whatever its labels, so one file serves every task.

    python extract_retfound.py --dataset brset  --root <BRSET>  --out features/brset_retfound.npz
    python extract_retfound.py --dataset mbrset --root <mBRSET> --out features/mbrset_retfound.npz

Output ``.npz``: ``files`` (the table's file names), ``feats`` [N, 1024] float32,
``meta`` (JSON: weights and their sha256, model, image size, pooling, dataset,
root, camera filter, counts). Written atomically; an existing file is refused
unless ``--overwrite``, because the probes record which features they were fit on.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retfound import DEFAULT_WEIGHTS, IMAGE_SIZE, POOLS, TIMM_NAME, build_encoder, embed, file_sha256  # noqa: E402


def load_features(path: str):
    """``(files, feats, meta)`` from an extract_retfound.py file."""
    z = np.load(path, allow_pickle=False)
    return z["files"].astype(str), z["feats"].astype(np.float32), json.loads(str(z["meta"]))


def main(argv=None) -> int:
    from brset_dataset import DATASETS, filter_camera, load_any
    from dataset import MBRSETDataset
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--root", required=True)
    p.add_argument("--camera", default=None, help="Keep one camera's rows (default: every image).")
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Local .pth or HuggingFace repo id (gated: log in with huggingface_hub.login() first).")
    p.add_argument("--timm-name", default=TIMM_NAME)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--pool", default="cls", choices=list(POOLS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="First N images only (smoke test).")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if os.path.exists(args.out) and not args.overwrite:
        print(f"[skip] {args.out} exists (--overwrite to recompute)")
        return 0
    src = load_any(args.root, args.dataset, image_ext=args.image_ext)
    df = filter_camera(src["df"], args.camera) if args.camera else src["df"]
    df = df.assign(file=df["file"].astype(str)).drop_duplicates("file").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    ds = MBRSETDataset(csv=df.assign(_all=0), images_dir=src["images_dir"], label_col="_all",
                       split="val", image_size=args.image_size, drop_missing_files=True, fov_crop=True)
    n_missing = len(df) - len(ds)
    print(f"[info] {args.dataset} @ {src['images_dir']}: {len(ds)} images on disk "
          f"({n_missing} table rows without a file)")
    if len(ds) == 0:
        raise SystemExit("[fatal] no images found; check --root / --image-ext")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    enc, info = build_encoder(args.weights, timm_name=args.timm_name, image_size=args.image_size,
                              device=device)
    print(f"[info] encoder: {info['timm_name']} from {info['weights_path']} on {device} "
          f"(ignored checkpoint keys: {info['n_ignored_keys']})")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"))
    feats, files, t0 = [], [], time.time()
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i, batch in enumerate(dl):
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                f = embed(enc, x, args.pool)
            feats.append(f.float().cpu().numpy())
            files += list(batch["file"])
            if i % 20 == 0:
                done = len(files)
                print(f"  {done}/{len(ds)}  {done / max(time.time() - t0, 1e-6):.0f} img/s", flush=True)
    feats = np.concatenate(feats).astype(np.float32)
    if not np.isfinite(feats).all():
        raise SystemExit("[fatal] non-finite embeddings (fp16 overflow?); rerun on CPU or without AMP")
    meta = {**info, "weights_sha256": file_sha256(info["weights_path"]), "pool": args.pool,
            "dataset": args.dataset, "root": os.path.abspath(args.root), "camera": args.camera,
            "n_images": int(len(files)), "n_table_rows_without_file": int(n_missing),
            "preprocessing": "crop_to_fov + full-frame resize + ImageNet normalisation (MBRSETDataset val)",
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, files=np.asarray(files, dtype=str), feats=feats, meta=json.dumps(meta))
    os.replace(tmp, args.out)
    print(f"[info] wrote {args.out}: {feats.shape[0]} x {feats.shape[1]} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
