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
| `model_seg.py` | `GCGUNet` — mobile-encoder U-Net with GCG-gated skips (`gcg_factory` injectable); `--encoder` / `--decoder` / `--lateral-channels` select the backbone and decoder style |
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

## Mobile cost: what is actually true (measured, 2026-08-11)

The deployment claim is the point of this project, so the numbers matter more
than the intuitions. Profiling `model_seg.GCGUNet` at 512×512 gives **14.09
GMAC**, split **encoder 7.9% / decoder 92.1%** — MobileNetV3-Large was chosen
for mobile and then paired with a decoder costing 12× the backbone.

Optimising that away **does not work**. Core ML fp16 at 512×512 on Apple
silicon, median of 60 runs after 10 warmups:

| config | GMAC | ANE median | CPU median | params |
|---|---|---|---|---|
| `mobilenetv3 / dense` (default) | 14.09 | **9.0 ms** | 46.5 ms | 6.84M |
| `mobilenetv3 / separable` | 3.26 | 9.2 ms | 46.6 ms | 3.53M |
| `mobilenetv3 / dense --lateral-channels 256` | 12.41 | **8.4 ms** | **40.3 ms** | 5.38M |
| `mobilenetv4_m / separable` | 6.86 | 9.3 ms | 83.5 ms | 7.84M |
| `efficientvit_b1 / separable` | 4.61 | 12.7 ms | 61.2 ms | 5.05M |

**A 4.33× MAC reduction bought 0% latency.** The network is memory-bandwidth-
bound, not compute-bound: activations dominate at this resolution, and
depthwise-separable convs trade arithmetic for an extra intermediate activation.
Three things follow:

1. **Do not justify an architecture change here with FLOPs/MACs.** Export and
   time it — `export_coreml.py` is the ground truth. The 135 ms in
   `edge_export/seg_edge_metrics.json` is ONNX-INT8 on **CPU** and is ~15× the
   real Core ML figure; it must not be quoted as the deployment latency.
2. **There is large latency headroom, so spend it on resolution** — the lever
   that actually matters for ~1–2 px microaneurysms. Same config, ANE median:
   512 → 7.8 ms, 768 → 21.6 ms, 1024 → 34.3 ms, 1280 → 48.3 ms.
3. `--lateral-channels 256` is the one cost knob that won on both paths
   (1.07× ANE, 1.15× CPU, −1.46M params), because it cuts activation *traffic*
   (960 → 256 channels at stride 32), not just arithmetic.

GCG itself costs **3.9% of MACs** — worth reporting alongside its Dice delta.

```bash
# recommended cost config (also the only one that measured faster on both paths)
python train_idrid.py --decoder dense --lateral-channels 256 --patch-size 512 ...

# modern encoders (need timm; pick for ACCURACY — neither is faster)
python train_idrid.py --encoder mobilenetv4_m ...
python train_idrid.py --encoder efficientvit_b1 ...

# re-measure any variant end to end
python export_coreml.py --image-size 512 --quantize fp16 --decoder separable
```

Defaults are unchanged (`mobilenetv3` / `dense` / no projection), so every
checkpoint in `checkpoints/` and `exp_*/` still loads and all prior results stay
reproducible — `tests/test_models.py` guards this.

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
