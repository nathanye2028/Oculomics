# Oculomics — on-device diabetic-retinopathy screening from fundus photographs

PyTorch research code for an ISEF project whose premise is **mobile viability**: a
model that stays accurate on unconstrained smartphone fundus captures at
on-device latency. Two tracks share one codebase:

1. **Classification (the headline experiment)** — referable-DR classification
   trained on **BRSET** (tabletop camera, 11,405 images) and tested zero-shot on
   all of **mBRSET** (smartphone, 4,884 images, patient-disjoint). The levers
   that closed the domain gap: a MobileNetV4-Small student at 384 px, knowledge
   distillation from a ConvNeXt-S teacher, and label-free test-time BatchNorm
   adaptation (AdaBN).
2. **Segmentation** — a mobile-encoder U-Net with **Guided Context Gating
   (GCG)** on the decoder skips, trained on pixel-level lesion masks (IDRiD,
   FGADR). The GCG-vs-control comparison was null; the track is kept for the
   lesion-level evidence and the latency measurements.

**Status document:** [REPORT.md](REPORT.md) (numbers, statistics, what is
claimable). **Changes:** [CHANGELOG.md](CHANGELOG.md). Image-only by design — no
demographic/systemic metadata fusion in either track.

| mBRSET AUROC (zero-shot unless noted) | |
|---|---|
| old baseline (V3-Small, 224 px) | 0.813 |
| V4-Small, new recipe, 384 px (control, 5 seeds) | 0.882 |
| + AdaBN (transductive: BN stats from the evaluation images) | 0.896 |
| + distillation + AdaBN | 0.909 |
| teacher, 49.7 M params | 0.936 |

Deployed network: 2.82 M params, fp16 Core ML, **0.7 ms** median on a Mac's
Apple Neural Engine at 384 px — an *optimistic* proxy for an iPhone, see
[Mobile cost](#mobile-cost-what-is-actually-true).

## Quick start

Supported Python: **3.9 – 3.13** (`pyproject.toml`; `check_env.py` refuses
anything else). On macOS the default `python3` may be 3.14, which has no torch
wheels yet — use the Xcode CLT interpreter:

```bash
/usr/bin/python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: make setup PY=/usr/bin/python3
.venv/bin/python check_env.py --deploy   # Python gate, device, forward/backward, deploy toolchain
.venv/bin/python -m pytest               # 67 CPU-only synthetic tests, ~30 s
make smoke                               # 1-epoch GCG-vs-control run into scratch dirs
```

`make help` lists every target. On a Linux GPU box: `bash setup_remote.sh`
(or build the `Dockerfile`; both use the `cu126` index — torch 2.8.0 is not on
`cu121`).

## Reproduce the headline result

BRSET and mBRSET are PhysioNet-credentialed and are never downloaded by this
repo. With both on disk:

```bash
B=<BRSET root> M=<mBRSET root> bash reproduce.sh            # train 5 seeds -> stats -> deploy eval -> Core ML
B=... M=... STAGE=stats bash reproduce.sh                    # re-summarise an existing sweep
make reproduce B=... M=... SEEDS="0 1 2"                     # same via make
```

`reproduce.sh` runs `run_kd_xfer.sh` (ctrl / teacher / kd per seed, all with
`--bn-adapt`), `summarize_xfer.py` (paired statistics), `evaluate_deploy.py`
(val-calibrated operating point on mBRSET, INT8 cost) and, on a Mac,
`export_coreml.py` with real-image verification. Outputs land in
`exp_kd_v4_384/` and `ck_kd_v4_384/` (gitignored).

## Repository layout

| File | Role |
|------|------|
| **Classification** | |
| `dataset.py` | `MBRSETDataset` — labels, transforms, FOV crop, patient-grouped `stratified_split` (raises if the `patient` column is missing), GPU-side `DeviceAug` |
| `brset_dataset.py` | BRSET → mBRSET schema adapter (`load_brset` / `load_any`; NaN-preserving re-encoders) + `--inspect` CSV auditor |
| `model.py` | `MBRSETClassifier` — `mobilenetv3_small` or any `timm:<name>` backbone; GCG only on the V3 path (a timm backbone with GCG requested is an error) |
| `train_mbrset.py` | Trainer: AUROC selection on in-domain val, AMP (bf16 on MPS), EMA, warmup→cosine, `--external-test-root` transfer, `--teacher` distillation, `--bn-adapt` AdaBN (adapted weights saved as `model_bnadapt`), `.done` completion marker |
| `run_kd_xfer.sh` | The paired ctrl / teacher / kd design per seed; `B=`/`M=` required, `--help` lists every knob |
| `summarize_xfer.py` | Paired treatment-vs-control statistics + the AdaBN table over `<condition>_seed<n>.json` files |
| `run_mbrset.py` | Small in-domain GCG-vs-control sweep on mBRSET |
| **Ophthalmic labels beyond DR (BRSET)** | |
| `run_ophthalmic.sh` | Per seed: one multi-label model (`--task ophthalmic`) and one binary model per label on the same patient split; `B=` required, `--help` lists every knob |
| `summarize_ophthalmic.py` | Paired multi-minus-single AUROC per label over `multi_seed<n>.json` / `single_<label>_seed<n>.json` |
| **Segmentation** | |
| `model_seg.py` | `GCGUNet` — `--encoder` / `--decoder` / `--lateral-channels`; gate init is RNG-isolated so GCG and control share every non-gate weight at a seed |
| `gcg_blocks.py` | GCG variants (`attention`, `cbam`, `se`, `none`) + registry — drop a custom block in here |
| `unet_baseline.py` | Vanilla U-Net — a *standard-architecture reference*, NOT the GCG control |
| `train_idrid.py` | Segmentation trainer: patches, focal-Tversky / `lesion_seg`, AMP, accumulation, DDP, tiled eval, `--eval-tiled-val`; results JSON records the full eval config + git commit |
| `run_experiment.py` | `{GCG, control} × seeds` harness with paired CI; `--eval-tiled-val` on by default with tiled eval; `--quick` writes to `*_quick/` scratch dirs; non-default configs get their own `experiments/<slug>/` |
| `run_arch_sweep.py` | Encoder / decoder sweep on the same harness |
| `eval_fgadr.py` | Score a checkpoint on FGADR (gating and GCG variant read from the checkpoint) |
| `idrid_dataset.py`, `fgadr_dataset.py`, `multi_seg_dataset.py`, `retlesion_dataset.py`, `vessel_dataset.py`, `ddr_dataset.py`, `rfmid_dataset.py` | Lesion / vessel / classification sources; every seg source implements `load_full(idx) -> (img, masks)` + a per-channel `valid` vector |
| `pretrain_encoder.py`, `pretrain_vessel.py`, `pretrain_retlesion.py` | In-domain pretraining options (`--init-encoder`, `--init-weights`) |
| `losses.py` | `lesion_seg` loss (focal Tversky + focal BCE) via `--loss lesion_seg` |
| `overfit_test.py`, `precision_check.py` | Can-it-learn-at-all and fp16/TF32 sanity checks |
| **Shared** | |
| `fundus_utils.py` | Seeding (`make_rng` is safe with persistent workers), FOV crop, losses, tiled inference, `pick_device` |
| `metrics.py` | Kappa, Dice/IoU (NaN for absent lesions), AUPRC, CSV/TensorBoard logging |
| **Deployment** | |
| `export_coreml.py` | Core ML export (seg or cls, read from the checkpoint); `--bn-stats {source,adapted}`; per-compute-unit ANE/GPU/CPU benchmark; `--verify-images DIR` real-image pass/fail; preprocessing spec written into the model metadata |
| `evaluate_deploy.py` | FP32 vs INT8 accuracy, val-calibrated operating point (also on `--external-root`), ONNX-CPU proxy latency (key `latency_ms_cpu_onnx` — not a device number) |
| `edge_optimize.py` | ONNX export + static INT8 quantisation helpers |
| `artifacts.py`, `validate_artifacts.py` | Artifact-reduction preprocessing and its validation |
| **Environment** | |
| `check_env.py` | Python gate, device, forward/backward, `--deploy` toolchain probe |
| `requirements.txt`, `pyproject.toml` | Pins (per-Python markers), supported range, pytest config |
| `setup_remote.sh`, `Dockerfile`, `Job*.sh` | GPU box bootstrap, container, SLURM jobs |
| `reproduce.sh`, `Makefile` | One-command reproduction and the common targets |
| `tests/` | CPU-only synthetic tests, run by GitHub Actions on every push |
| `legacy/` | Superseded scripts kept for reference (`train.py`, placeholder-mask `train_seg.py`, old smoke tests) |

## Classification: BRSET → mBRSET transfer

```bash
# in-domain: train and test on mBRSET (patient-grouped 70/10/20)
python train_mbrset.py --root <mBRSET> --task dr_referable --seed 0

# the domain-shift experiment, one arm by hand (run_kd_xfer.sh does the paired design):
python train_mbrset.py --dataset brset --root <BRSET> \
    --external-test-root <mBRSET> --external-test-dataset mbrset \
    --task dr_referable --image-size 384 --seed 0 \
    --backbone timm:mobilenetv4_conv_small.e2400_r224_in1k --no-gcg \
    --bn-adapt --results-json exp_xfer/ctrl_seed0.json
# ... and the control for a GCG test on the V3 student: --no-gcg --results-json exp_xfer/nogcg_seed0.json

# after >=3 seeds of two conditions named <cond>_seed<n>.json:
python summarize_xfer.py --dir exp_xfer --treatment kd --control ctrl
```

Design points that the code enforces:

- **Splits** are patient-grouped and stratified; the same `--seed` gives the
  same split, so conditions within a seed are paired. The external set must be a
  different directory from `--root`, and a `--teacher` must have been trained on
  the same dataset (both are fatal otherwise).
- **Selection** is on in-domain val AUROC only (falls back to accuracy if AUROC
  is undefined, recorded as `selection_metric`); test and mBRSET are scored
  once with the selected (EMA) weights.
- **AdaBN** (`--bn-adapt`) re-estimates BatchNorm statistics on the external
  *images* (never labels) and reports `external_bnadapt` beside the zero-shot
  `external`. It is **transductive** — the statistics come from the same
  images that are scored — and is reported that way. The adapted weights are
  saved into the checkpoint as `model_bnadapt` so they can be exported
  (`export_coreml.py --bn-stats adapted`). Backbones without BatchNorm (the
  ConvNeXt teacher) get no adapted number.
- **Distillation** (`--teacher <ckpt>`): loss = (1−α)·CE + α·T²·KL(teacher ‖
  student), α = 0.7, T = 4, optional cosine feature matching. ctrl and kd are
  identical apart from that term (the teacher is built under a forked RNG).
- **Completion**: a run is finished when `<ckpt-dir>/<run>.done` exists;
  `run_kd_xfer.sh` retrains a teacher whose checkpoint exists without it.
- **Imbalance** is corrected once (`--imbalance sampler`); `--amp` is bf16 on
  MPS, fp16 + GradScaler on CUDA, identically for every arm.

## Ophthalmic labels beyond DR: multi-label head on BRSET

BRSET grades every image for more than DR: **AMD, drusen, increased cup-to-disc
ratio (glaucoma suspect), hypertensive retinopathy, vascular occlusion,
hemorrhage, myopic fundus, retinal detachment, scar, nevus**. The BRSET adapter
now carries those columns through (`brset_dataset.OPHTHALMIC_MAP`), each is a
binary task (`--task amd`), and `--task ophthalmic` trains **one model with a
sigmoid logit per label** (`dataset.OPHTHALMIC_LABELS`). mBRSET has none of
these labels, so this is **in-domain on BRSET only** — no smartphone transfer
claim is available for it.

```bash
python brset_dataset.py --csv <BRSET>/labels_brset.csv --inspect    # OPHTHALMIC LABELS block: raw encodings + prevalence
B=<BRSET> bash run_ophthalmic.sh 0 1 2 3 4                          # multi + one-per-label, paired by seed
python summarize_ophthalmic.py --dir exp_ophthalmic
```

What changes for the multi-label head, and only there:

- **Loss / imbalance**: `BCEWithLogitsLoss`; `--imbalance loss` uses per-label
  `neg/pos` as `pos_weight`, `--imbalance sampler` (default) draws each image by
  the inverse frequency of its *rarest* positive label.
- **Metrics / selection**: per-label AUROC (`per_label_auroc` in the JSON, NaN
  where a label has one class in the split) with the macro mean over scored
  labels as the selection metric; `kappa` is NaN.
- **Splits**: stratified on each row's rarest positive label so the rare labels
  reach every partition; a row with any missing label is dropped.
- **Distillation**: `--teacher` works with a per-logit binary KD term.
- **The paired question** (`summarize_ophthalmic.py`): does sharing one trunk
  across labels help or hurt each label versus a dedicated model, same seed,
  same split? Rare labels have a handful of test positives — quote the CI.
- **Not yet adapted**: `export_coreml.py` / `evaluate_deploy.py` assume a
  softmax head; exporting the multi-label model needs a sigmoid output path.

## Segmentation: GCG vs control

The control for "does gating help?" is the **same backbone with gating off**
(`--no-gcg`), not the vanilla U-Net. IDRiD's test set is 27 images, so always
run several seeds and read the paired CI.

```bash
# headline harness (single GPU): tiled test AND tiled checkpoint selection, paired CI
python run_experiment.py --seeds 0 1 2 --epochs 200 --patch-size 512 --eval-tiled --amp

# a different block, encoder or data mix gets its own experiments/<slug>/ automatically
python run_experiment.py --seeds 0 1 2 --gcg-variant attention --epochs 200 --patch-size 512 --eval-tiled --amp
python run_experiment.py --seeds 0 1 2 --datasets idrid fgadr ...

# multi-GPU per run
torchrun --nproc_per_node=4 train_idrid.py --arch gcg_unet --amp \
    --patch-size 512 --epochs 200 --eval-tiled --eval-tiled-val --batch-size 16

# score any checkpoint on FGADR's fixed test split (split_seed=42)
python eval_fgadr.py --checkpoint checkpoints/gcg_seed0.pt --tiled
```

Plug in a custom GCG block: implement `forward(skip, guide) -> skip-shaped` in
`gcg_blocks.py`, register it in `GCG_VARIANTS`, benchmark with
`run_experiment.py --gcg-variant <name>`.

Recorded in every results JSON and checkpoint: git commit, FGADR
`split_seed`/fractions, `eval_tiled`, `eval_tiled_val`, `tile_overlap`,
`accum_steps`, `fg_bias`, worker settings — so runs from before and after a
fix are distinguishable. Whole-image training with `--eval-tiled` is refused
(the tiles would be 8× larger than anything the model saw).

## Mobile cost: what is actually true

Measured 2026-08-11 with `export_coreml.py`, Core ML fp16 at 512×512, median
after warmup:

| config | GMAC | ANE median | CPU median | params |
|---|---|---|---|---|
| `mobilenetv3 / dense` (default) | 14.09 | **9.0 ms** | 46.5 ms | 6.84M |
| `mobilenetv3 / separable` | 3.26 | 9.2 ms | 46.6 ms | 3.53M |
| `mobilenetv3 / dense --lateral-channels 256` | 12.41 | **8.4 ms** | **40.3 ms** | 5.38M |
| `mobilenetv4_m / separable` | 6.86 | 9.3 ms | 83.5 ms | 7.84M |
| `efficientvit_b1 / separable` | 4.61 | 12.7 ms | 61.2 ms | 5.05M |

**A 4.33× MAC reduction bought 0% latency.** The network is memory-bandwidth-
bound, not compute-bound. What follows:

1. **Never justify an architecture change with FLOPs/MACs.** Export and time
   it. `export_coreml.py` now benchmarks `CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`
   and `ALL` separately and warns when `ALL` is not running on the ANE.
2. **The per-pass number is not the per-image number for segmentation.** Dice
   is scored `--eval-tiled` at native resolution: 9 tiles on FGADR (1280²),
   ~36 on IDRiD (4288×2848), so ~80–320 ms per image at 9 ms per tile. Quote
   tiles × per-tile, or the whole-image row (512 → 7.8 ms, 768 → 21.6 ms,
   1024 → 34.3 ms, 1280 → 48.3 ms).
3. **The Mac ANE is an optimistic bound**, not an upper bound: a
   bandwidth-bound model runs slower on an iPhone's narrower memory system.
   The device number comes from an Xcode Core ML performance report.
4. The ONNX-INT8 figure in `evaluate_deploy.py` (`latency_ms_cpu_onnx`) is a
   CPU proxy and must never be quoted as deployment latency.
5. `--lateral-channels 256` is the one cost knob that won on both paths
   (−1.46M params) because it cuts activation *traffic*, not arithmetic. GCG
   itself costs 3.9% of MACs.

## Deployment

```bash
# Core ML for the deployed classifier, with a real-image pass/fail check (exit 1 on failure)
python export_coreml.py --checkpoint ck_kd_v4_384/kd_seed1.pt --verify-images <mBRSET>/images
python export_coreml.py --checkpoint ck_kd_v4_384/kd_seed1.pt --bn-stats adapted   # the AdaBN weights

# operating point (threshold calibrated on in-domain VAL, applied once to test and to mBRSET) + INT8 cost
python evaluate_deploy.py --root <BRSET> --ckpt ck_kd_v4_384/kd_seed1.pt \
    --external-root <mBRSET> --target-sens 0.90 --calib sweep
```

The exported model bakes in normalisation and softmax. The FOV crop
(`tol=12`) and the antialiased square resize are **app-side**; their exact
spec is written into the `.mlpackage`'s `user_defined_metadata` and printed at
export so the iOS side can reproduce it.

## Tests and CI

```bash
.venv/bin/python -m pytest          # tests/ only; CPU, synthetic, no network
```

`.github/workflows/tests.yml` runs the same suite on every push with CPU torch.
Covered: model round-trips for every encoder/decoder, GCG/control weight
sharing, worker RNG diversity under persistent workers, tiled inference, Dice
with absent lesions, BRSET re-encoders, patient-grouped splits, AdaBN, the KD
loss, the deploy wrapper vs `Normalize`, the operating-point rule, and the
Python gate.

## Data

- **IDRiD** (`aaryapatel98/indian-diabetic-retinopathy-image-dataset`, CC BY
  4.0): pixel masks for MA/HE/EX/SE/OD, 54 train / 27 test. Auto-downloaded via
  `kagglehub`; the 27-image test set is the segmentation benchmark.
- **FGADR Seg-set**: 1,842 images with MA/HE/EX/SE masks, under an IIAI
  research-use agreement — non-commercial, **non-redistributable**, cite Zhou
  et al. (arXiv:2008.09772). Lives under `data/` (gitignored, dockerignored).
  Partition fixed at `split_seed=42`: 1290 / 185 / 367.
- **BRSET** and **mBRSET** (PhysioNet, credentialed): the classification
  transfer pair. `brset_dataset.py --csv <BRSET>/labels.csv --inspect` audits a
  CSV before training (BRSET encodes `patient_sex` 1/2, `quality`
  Adequate/Inadequate, `artifacts` 1/2 with 2 = present).
- **RFMiD / DDR / e-ophtha / RetLesion / vessel sets**: optional extra sources
  for the segmentation and pretraining scripts.

## Known caveats

- **Segmentation numbers before 2026-09-01 predate three fixes** (persistent-
  worker augmentation collapse, GCG/control init sharing, tiled val
  selection in the harness) and should be re-run before being compared with
  new ones. FGADR numbers before 2026-08-20 used per-seed partitions and are
  not comparable at all.
- Microaneurysms are ~0.05% of pixels: train with `--patch-size` and keep
  `--eval-tiled-val` (the harness default) so selection sees them.
- `--cache-images` (train_idrid) keeps decoded IDRiD in RAM — a big win on the
  GPU box, skip it on a laptop.
- coremltools 9.0 warns that torch 2.8 is untested; export and verification
  have been checked to work. Do not add `executorch` to this environment on
  Python 3.9 (it downgrades torch).
- Any trainer or harness writes into its default output dirs; `--quick` and
  `make smoke` use `*_quick/` so they never overwrite real checkpoints. Pass
  `--out-dir`/`--ckpt-dir` for anything else you want kept apart.

## Licence

Code: MIT (see `LICENSE`). Datasets are not included and carry their own terms.
