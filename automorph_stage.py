"""
automorph_stage.py
==================
Lay out one dataset's photos the way AutoMorph (github.com/rmaphoh/AutoMorph)
reads them, without copying 10,000 images:

    <out>/images/<name>.<ext>          symlinks to the dataset's files
    <out>/resolution_information.csv   fundus,res  (mm per pixel, per image)
    <out>/name_map.csv                 automorph_name -> file, patient, camera
    <out>/Results/                     empty; AutoMorph writes here

AutoMorph renames every image to ``name.split('.')[0] + '.png'``, so a file name
with a second dot would collide; the staged names are ``<stem with dots replaced>``
and ``name_map.csv`` maps them back (``vessel_associations.py`` reads it).

``res`` is approximate and only matters for absolute micrometre units: the field
of view's width in pixels (``fundus_utils.fov_bbox``) is taken to span
``--fov-mm`` of retina (45 degrees ~ 12.4 mm). The association analysis
standardises every measurement within a camera, so a constant scale error cancels.

    python automorph_stage.py --dataset brset --root <BRSET> --camera "Canon CR" --out automorph/brset_canon
    python automorph_stage.py --dataset mbrset --root <mBRSET> --out automorph/mbrset

It prints the Docker command with the dataset folder mounted read-only at the
same path, so the symlinks resolve inside the container.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def staged_name(file: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(file))
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem) + ext.lower()


def fov_width_px(path: str) -> float:
    from PIL import Image
    from fundus_utils import fov_bbox
    with Image.open(path) as im:
        full_w = im.size[0]
        im.draft("RGB", (max(full_w // 4, 1), max(im.size[1] // 4, 1)))   # fast JPEG decode
        small = im.convert("RGB")
    scale = full_w / small.size[0] if small.size[0] else 1.0
    r0, r1, c0, c1 = fov_bbox(np.asarray(small))
    return float(c1 - c0) * scale


def main(argv=None) -> int:
    from brset_dataset import DATASETS, filter_camera, load_any
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--root", required=True)
    p.add_argument("--camera", default=None)
    p.add_argument("--image-ext", default=".jpg")
    p.add_argument("--fov-mm", type=float, default=12.4,
                   help="retinal width spanned by the field of view (45 deg ~ 12.4 mm)")
    p.add_argument("--out", required=True)
    p.add_argument("--image", default="yukunzhou/image_automorph",
                   help="AutoMorph Docker image (DOCKER.md)")
    a = p.parse_args(argv)

    src = load_any(a.root, a.dataset, image_ext=a.image_ext)
    df = filter_camera(src["df"], a.camera) if a.camera else src["df"]
    df = df.assign(file=df["file"].astype(str)).drop_duplicates("file")
    img_dir = os.path.realpath(src["images_dir"])
    on_disk = set(os.listdir(img_dir))
    df = df[df["file"].isin(on_disk)].reset_index(drop=True)
    df["automorph_name"] = df["file"].map(staged_name)
    dup = df["automorph_name"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(f"[fatal] staged names collide: {df.loc[dup, 'file'].head().tolist()}")
    if df["automorph_name"].str.split(".").str[0].duplicated().any():
        raise SystemExit("[fatal] two files share a stem; AutoMorph's .png renaming would merge them")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "Results"), exist_ok=True)
    res = []
    for i, r in enumerate(df.itertuples()):
        dst = os.path.join(out, "images", r.automorph_name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(img_dir, r.file), dst)
        w = fov_width_px(os.path.join(img_dir, r.file))
        res.append(a.fov_mm / w if w > 0 else np.nan)
        if i % 1000 == 0:
            print(f"  {i}/{len(df)}", flush=True)
    df["res_mm_per_px"] = res
    keep = [c for c in ("automorph_name", "file", "patient", "camera", "laterality", "res_mm_per_px")
            if c in df.columns]
    df[keep].to_csv(os.path.join(out, "name_map.csv"), index=False)
    pd.DataFrame({"fundus": df["automorph_name"], "res": df["res_mm_per_px"]}).to_csv(
        os.path.join(out, "resolution_information.csv"), index=False)
    data_root = os.path.dirname(img_dir)
    print(f"[info] staged {len(df)} images of {a.dataset}{f' ({a.camera})' if a.camera else ''} in {out}")
    print(f"[info] median res {np.nanmedian(res):.5f} mm/px (fov {a.fov_mm} mm over the FOV width)")
    print("\nRun AutoMorph (GPU; a rerun wipes Results/). Docker, if the box allows it:\n")
    print(f"  docker pull {a.image}")
    print(f"  docker run --rm --shm-size=2g --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${{GPU:-0}} \\\n"
          f"    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\\n"
          f"    -v {out}/images:/app/AutoMorph/images \\\n"
          f"    -v {out}/Results:/app/AutoMorph/Results \\\n"
          f"    -v {out}/resolution_information.csv:/app/AutoMorph/resolution_information.csv \\\n"
          f"    -v {data_root}:{data_root}:ro \\\n"
          f"    {a.image}")
    print("\nWithout Docker (no root on the box): clone AutoMorph, follow its LOCAL.md conda setup, then\n"
          f"  export AUTOMORPH_DATA={out} && cd <AutoMorph clone> && sh run.sh")
    print(f"\nThen: python vessel_associations.py --automorph {out} --dataset {a.dataset} --root {a.root}"
          + (f' --camera "{a.camera}"' if a.camera else "") + " --prereg vessel_prereg.json --out ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
