# Oculomics — Retinal Fundus Lesion Segmentation & Clinical-Label Classification

PyTorch pipeline for diabetic-retinopathy work on fundus images, built around two tracks:

1. **Lesion segmentation** (the research focus) — a 2D U-Net with a MobileNetV3
   encoder and **Guided Context Gating (GCG)** blocks on the decoder skips, trained
   on real per-pixel lesion masks (IDRiD). The headline question: *does GCG beat a
   no-gating control?*
2. **Clinical-label classification** — a MobileNetV3-Small classifier predicting
   DR grade / macular-edema risk from the image alone (no metadata fusion).

> Image-only by design — no systemic/demographic metadata fusion in either track.

## Repository layout

| File | Role |
|------|------|
| `dataset.py` | `MBRSETDataset` — classification labels, transforms, FOV crop, class-weight helpers |
| `model.py` | `MBRSETClassifier` — MobileNetV3-Small + custom head (multiclass/multilabel/regression) |
| `train.py` | Classification trainer — CrossEntropy + Adam, val tracking (acc/F1/**kappa**), best-checkpoint, file+CSV logging |
| `test_loader.py`, `test_classifier.py` | One-batch smoke tests for the classification path |
| `model_seg.py` | `GCGUNet` — MobileNetV3-Large U-Net with GCG-gated skips (`gcg_factory` injectable) |
| `unet_baseline.py` | Vanilla U-Net — *standard-architecture reference* (NOT the GCG control) |
| `gcg_blocks.py` | GCG variants (`attention`, `cbam`, `se`, `none`) + registry — drop in your custom block here |
| `idrid_dataset.py` | `IDRiDSegDataset` — real IDRiD lesion masks → multi-label `[C,H,W]`, FOV crop, native-res patches |
| `seg_dataset.py` | RFMiD placeholder-mask loader (plumbing only; no real masks) |
| `train_idrid.py` | Segmentation trainer/benchmark — train/val/test, focal-Tversky, AMP, gradient accumulation, **DDP**, tiled eval, Dice+IoU |
| `run_experiment.py` | Orchestrator: sweeps `{GCG, control} × seeds`, prints mean±std comparison table |
| `fundus_utils.py` | Shared: seeding, worker RNG, FOV crop, losses, tiled inference |
| `metrics.py` | Quadratic-weighted kappa, Dice/IoU, AUPRC, CSV/TensorBoard logging |
| `check_env.py` | Verify torch / GPU / AMP / NCCL on a new machine |
| `setup_remote.sh`, `Dockerfile` | Reproducible environment for a remote GPU server |

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python check_env.py        # confirms device + a real forward/backward
```
On a remote GPU box: `bash setup_remote.sh` (or build the `Dockerfile`).

## The headline experiment (GCG vs control on IDRiD)

The control for "does gating help?" is the **same backbone with gating off** —
`--no-gcg` — NOT the vanilla U-Net (which differs in backbone/depth/pretraining).
The IDRiD test set is only 27 images, so always run **multiple seeds** and read
mean±std, not a single run.

```bash
# full run (single GPU): produces experiments/summary.md
python run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 --eval-tiled --amp

# benchmark a specific GCG block from gcg_blocks.py:
python run_experiment.py --seeds 0 1 2 --gcg-variant attention --epochs 200 --patch-size 512 --eval-tiled --amp

# multi-GPU per run:
torchrun --nproc_per_node=4 train_idrid.py --arch gcg_unet --amp \
    --patch-size 512 --epochs 200 --eval-tiled --batch-size 16
```

### Plugging in a custom GCG block
Add your block to `gcg_blocks.py` implementing `forward(skip, guide) -> skip-shaped`,
register it in `GCG_VARIANTS`, then benchmark with the *same* harness:
`run_experiment.py --gcg-variant <name>`.

## Classification training

```bash
python train.py --task dr_grade --epochs 50 --batch-size 32
tail -f ~/oculomics_project/logs/train_metrics.log    # metrics persisted to disk
```
Best weights → `~/oculomics_project/weights/best_model.pt`.

## Tests

```bash
.venv/bin/python -m pytest -q          # CPU-only, synthetic, no network
```

## Data

- **IDRiD segmentation** (`aaryapatel98/indian-diabetic-retinopathy-image-dataset`): real
  per-lesion masks (MA/HE/EX/SE/OD), 54 train / 27 test. Auto-downloaded via `kagglehub`.
- **RFMiD / DR classification**: image-level labels only (no masks) — used for the
  classification track and placeholder-mask plumbing.
- **mBRSET**: the classification pipeline is schema-compatible but mBRSET (PhysioNet,
  credentialed) has not yet been wired in.

## Known caveats
- First real CUDA/DDP run: watch for a DDP `find_unused_parameters` error and fp16
  loss-scaling on the focal-Tversky loss (both one-line fixes).
- Microaneurysms (MA) are ~0.05% of pixels — expect low Dice unless trained at high
  resolution via `--patch-size`.
- The GCG block currently benchmarked is a baseline; the custom block is the eventual subject.
