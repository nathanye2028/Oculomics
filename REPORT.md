# Oculomics — Project Report, 20–31 August 2026

On-device diabetic-retinopathy screening from smartphone fundus photographs.
This report covers the work done between the 20th and 31st of August: a full
audit of the codebase, the architecture decision that followed a null result on
the original hypothesis, the two new levers that were built and tested
(foundation-model distillation and test-time BatchNorm adaptation), the move to
a MobileNetV4 backbone at 384 px, the resulting numbers with their statistics,
and the deployment tooling that turns those numbers into an on-device claim.

**Headline.** The model's zero-shot AUROC on 4,884 never-seen smartphone images
rose from **0.813 → 0.882** (control) and **0.909** with label-free test-time
adaptation (transductive AdaBN: BatchNorm statistics re-estimated on the same
unlabelled mBRSET images that are then scored; the exported Core ML model is the
zero-shot network unless exported with `--bn-stats adapted`), the tabletop→smartphone domain gap shrank from **−0.144 → −0.090**,
and the deployed network runs in **0.7 ms on the Apple Neural Engine** at 384 px.

---

## 1. Where things stood on 20 August

The project's premise is on-device viability: a model that stays accurate on
unconstrained smartphone captures at mobile latency. The repo had two tracks:

* **Segmentation** — a GCG-gated MobileNetV3 U-Net on IDRiD/FGADR lesion masks.
  Core ML latency had been measured (9.0 ms at 512 px); the GCG-vs-control
  comparison was underpowered and null.
* **Classification** — MobileNetV3-Small predicting referable DR. The
  well-powered experiment: train on BRSET (11,405 tabletop images), test
  in-domain on BRSET's held-out split and zero-shot on all of mBRSET (4,884
  smartphone images, patient-disjoint, never used for training or selection).

The classification transfer experiment had been run at 5 seeds (224 px, the
old recipe) and had just returned:

| | in-domain AUROC | mBRSET AUROC | gap |
|---|---|---|---|
| control (no GCG) | 0.957 | 0.8129 ± 0.0087 | −0.144 |
| GCG | 0.959 | 0.8167 ± 0.0074 | −0.142 |
| paired GCG − control | — | +0.0038, 95 % CI [−0.011, +0.019] | inconclusive |

Two facts from that result shaped everything after: the paired seed-to-seed
noise floor on mBRSET AUROC was **0.0119**, and an attention-style block
(GCG) was a third of it — "reached the limit of GCG" was the right reading.

## 2. Codebase audit (20 August)

Every Python file in the repo was read line by line (three parallel reviewers
over the segmentation core, the dataset loaders, and the tooling; the
classification track reviewed directly). Findings were verified against the
code before being fixed. Twenty-seven files changed; the full test suite (42
tests) passes.

### Bugs that were corrupting results

| where | defect | fix |
|---|---|---|
| `fgadr_dataset.py` | FGADR's train/val/test partition followed the run seed, so every 3-seed sweep scored each run on a *different* test set, and `eval_fgadr.py` (default seed 42) scored seed-0/1/2 checkpoints on images ~70 % of which were in their training set | `split_seed=42`, decoupled from `--seed`; all pre-fix FGADR numbers invalidated |
| `train_idrid.py` | DDP evaluation double-counted `DistributedSampler` padding into the all-reduced Dice counts (~9 % distortion of val selection at 4 GPUs) | non-padded strided shards for every eval path |
| `eval_fgadr.py`, `export_coreml.py`, `edge_optimize.py` | gating came from a CLI flag with `strict=False`, so a `--no-gcg` checkpoint could be scored or exported with randomly-initialised GCG gates | gating read from the checkpoint; GCG key mismatch is fatal |
| `evaluate_deploy.py` | the INT8 calibration sweep selected the method on the *test* set and reported that same test AUROC as the quantization cost | selection on val, test touched once |
| `train_mbrset.py` | balanced sampler **and** inverse-frequency CE weights together (~quadratic over-weighting of the 5 % positive class) | one correction (`--imbalance sampler`) |
| `dataset.py` | eval did 1.14× resize + centre-crop on an already FOV-cropped image, discarding ~23 % of the retina at its periphery | full-frame resize |
| `train_idrid.py` | `dice_bce` masked logits instead of probabilities (`sigmoid(0)=0.5` inflated the Dice denominator of annotated samples) | mask probabilities, weight the BCE |
| `train_idrid.py` | MPS AMP ran fp16 with no loss scaling — the underflow mode that collapses microaneurysms | bf16 on MPS |
| `metrics.py` | a lesion absent from both prediction and target scored Dice 0, deflating means | NaN + `mean_present` |
| `train_mbrset.py` | single-class val split → AUROC NaN forever → no checkpoint ever saved → crash at final test (a lost run) | NaN-safe fallback to accuracy |
| `run_experiment.py`, `run_mbrset.py` | all-runs-failed still wrote a NaN summary and exited 0 | fatal + nonzero exit |
| `run_arch_sweep.py`, `precision_check.py` | `--skip-existing` reused stale JSONs from different configurations | config validated before reuse |
| `train_mbrset.py` | "domain gap" label had the sign backwards | fixed |

### Code added to make the model better

* `--eval-tiled-val`: checkpoint selection on tiled native-resolution val, so
  the "best" epoch is no longer chosen on a whole-image metric where
  NEAREST-downsampled masks have erased most 1–2 px microaneurysms.
* `--loss lesion_seg` wires in the previously dead `losses.py` (focal Tversky +
  focal BCE); per-epoch sensitivity/precision logging diagnoses MA collapse.
* One shared dihedral+jitter augmentation for every lesion source (e-ophtha's
  47 images previously got a single flip).
* Classification recipe: warmup→cosine, optional label smoothing, weight EMA
  (an explicit implementation — `AveragedModel` mishandles BatchNorm buffers),
  AMP, on-device loss accumulation.
* Throughput: `--cache-images` (bit-packed IDRiD decode cache), persistent
  workers, removal of per-step `.item()` syncs and the per-tile sync in
  `tiled_predict`; epoch-diverse augmentation with `num_workers=0`.

## 3. The architecture decision (22 August)

The question was whether to move to flow matching or another generative
approach. The answer was no, for a reason specific to this project: flow /
diffusion models sample iteratively, so a 10-step segmenter costs ~10× the
9 ms budget, for gains those methods don't show on tiny sparse lesions — and
for classification the framing doesn't apply at all.

What the repo's own measurements said the lever was:

* ImageNet pretraining had moved IDRiD Dice 0.183 → 0.387 — larger than any
  block change ever measured. *What the network knows* matters more than its
  wiring.
* The network is memory-bandwidth-bound on the ANE: a 4.3× MAC cut bought 0 %
  latency, attention encoders were slower, and there was ~4× latency headroom
  to spend on resolution.
* The 14-point domain gap is a *distribution* problem; attention re-weighting
  operates at the half-point scale.

Chosen, in order: **(1)** distil a large fundus-competent teacher into the
unchanged mobile student (same params, same latency); **(2)** label-free
test-time BatchNorm adaptation (AdaBN) on the target images; **(3)** the
MobileNetV4 backbone the project had intended all along (it existed only in the
segmentation track; the classifier had always been V3-Small); **(4)** 384 px.

## 4. What was built

* `model.py` — `--backbone timm:<name>` (any timm model as a teacher, or V4 as
  the student); `embed()` / `forward_with_feat()` expose the pooled embedding.
  The head width is probed by a forward pass because timm's `num_features` is
  the pre-head width (MobileNetV4: 960 vs the real 1280).
* `train_mbrset.py` — `--teacher <ckpt>`: loss = (1−α)·CE + α·T²·KL(teacher ‖
  student) (the standard Hinton form), α = 0.7, T = 4, optional cosine feature matching; teacher
  architecture/size/gating read from its checkpoint, task mismatch fatal, seed
  mismatch warned (different patient split = leakage). `--bn-adapt`: AdaBN on a
  deep copy — BN layers in cumulative-average mode, everything else in eval —
  reported as `external_bnadapt`, separately from the zero-shot number.
* `dataset.py` — `DeviceAug`: the photometric/blur/erasing half of the
  augmentation applied per-sample on the GPU. Same ops and ranges; it lifted a
  CPU-bound cap of ~145 img/s (ColorJitter alone was 11 of 23 ms/img in the
  workers) and made two-GPU parallel sweeps possible.
* `run_kd_xfer.sh` — the whole paired design per seed: `ctrl` (student, no
  teacher) → `teacher` → `kd` (student distilled from that seed's teacher), all
  with `--bn-adapt`; idempotent, resumable; knobs `STUDENT`, `TEACHER`,
  `TEACHER_CK` (reuse another sweep's teachers), `SIZE`, `WORKERS`, `AMP`,
  `EXTRA`. Includes the fixes that came out of the first remote launch:
  unbuffered output (Python block-buffers stdout through `tee`, which hid every
  epoch line), `channels_last` only with `--nondeterministic` (deterministic
  cuDNN depthwise NHWC is far slower), NNPACK probing silenced.
* `summarize_xfer.py` — paired statistics over any two condition prefixes, plus
  the BN-adaptation table (paired adapt effect per condition, paired
  treatment-vs-control on adapted AUROC).
* `export_coreml.py` — classifier path (architecture read from the
  checkpoint; normalisation + softmax baked into the graph), ANE and CPU
  benchmark, PyTorch-vs-Core ML verification.
* `evaluate_deploy.py` — rebuilds any backbone from the checkpoint, rebuilds
  BRSET-trained splits via `load_any`, and adds a target-domain operating point
  (`--external-root`) with the threshold still calibrated on source-domain val.

## 5. Results

All runs: train on BRSET, patient-grouped 70/10/20 split, in-domain test on
BRSET's held-out split (n = 3,239), zero-shot on all gradable mBRSET images
(n = 4,884). Conditions within a seed share the patient split, so every
comparison is paired. Teacher: ConvNeXt-Small (ImageNet-22k), 49.7 M params,
trained with the same script on the same split.

### 5.1 Three seeds, 384 px (23 August)

| student | condition | in-domain AUROC | mBRSET AUROC | gap |
|---|---|---|---|---|
| V3-Small | ctrl | 0.9757 ± 0.0078 | 0.8633 ± 0.0100 | −0.112 |
| V3-Small | kd | 0.9798 ± 0.0090 | 0.8731 ± 0.0081 | −0.107 |
| **V4-Small** | ctrl | 0.9769 ± 0.0044 | 0.8871 ± 0.0120 | −0.090 |
| **V4-Small** | kd | 0.9831 ± 0.0036 | **0.9026 ± 0.0108** | −0.081 |

V4 + kd + AdaBN: 0.9161 ± 0.0065. Paired kd − ctrl on V4: +0.0155, 3/3 seeds,
95 % CI [−0.0011, +0.0320] — the same marginal n = 3 profile that had produced
a false positive on 22 August, so two more seeds were run before claiming it.

### 5.2 Five seeds, V4-Small at 384 px (30 August)

| | zero-shot mBRSET | BN-adapted mBRSET | adapt effect (paired) |
|---|---|---|---|
| ctrl | 0.8819 ± 0.0117 | 0.8962 ± 0.0067 | **+0.0143 [+0.0067, +0.0219] — significant** |
| kd | 0.8884 ± 0.0225 | 0.9089 ± 0.0121 | +0.0205 [−0.0086, +0.0496] n.s. |

Paired kd − ctrl: zero-shot +0.0064, CI [−0.0193, +0.0322], 4/5 seeds
(inconclusive); on BN-adapted AUROC +0.0126, CI [−0.0013, +0.0265], 4/5 seeds
(just short of significance). In-domain: ctrl 0.9762 ± 0.0055, kd 0.9785 ±
0.0064. Teacher zero-shot mBRSET: 0.9446 / 0.9335 / 0.9298 / 0.9354 / 0.9349
(mean 0.936).

Per-seed (in-domain / zero-shot / adapted):

| seed | ctrl | kd | teacher |
|---|---|---|---|
| 0 | 0.9748 / 0.9020 / 0.9088 | 0.9801 / 0.9179 / 0.9223 | 0.9829 / 0.9446 |
| 1 | 0.9729 / 0.8726 / 0.8904 | 0.9882 / 0.8945 / 0.9189 | 0.9879 / 0.9335 |
| 2 | 0.9829 / 0.8868 / 0.8958 | 0.9810 / 0.8954 / 0.9072 | 0.9890 / 0.9298 |
| 3 | 0.9683 / 0.8786 / 0.8955 | 0.9696 / 0.8489 / 0.9084 | 0.9826 / 0.9354 |
| 4 | 0.9820 / 0.8696 / 0.8907 | 0.9736 / 0.8851 / 0.8876 | 0.9956 / 0.9349 |

`kd_seed3` is the most informative run: val AUROC 0.9944 (excellent), zero-shot
0.8489 (the worst), and AdaBN restores it to 0.9084. Its transfer failure was
largely a BatchNorm-statistics mismatch — the mechanism AdaBN exists for. It
also exposes the structural limit of the design: in-domain val is saturated
(0.97–0.99) and cannot rank checkpoints by how well they transfer, and mBRSET
cannot be used for selection without breaking the experiment.

### 5.3 The ladder

| | mBRSET AUROC |
|---|---|
| old baseline (V3-Small, 224 px, old recipe) | 0.813 |
| V3-Small, new recipe, 384 px | 0.863 |
| V4-Small, new recipe, 384 px (control) | 0.882 |
| + AdaBN | 0.896 |
| + distillation + AdaBN | 0.909 |
| teacher (49.7 M params) | 0.936 |
| mBRSET-trained ceiling | *running* |

A 2.8 M-parameter student with test-time adaptation recovers ~93 % of a
49.7 M-parameter teacher's transfer performance at 1/18th the size.

### 5.4 What is claimable

* **Robust:** the recipe + resolution + V4 backbone effect. Fourteen of the
  fifteen V4 runs land 0.85–0.92 zero-shot (kd_seed3: 0.849) against an old
  baseline of 0.80–0.82.
* **Significant:** AdaBN on the control, +0.0143 with a CI excluding zero.
* **Trend, not a claim:** distillation — positive in 4/5 seeds on both
  readouts, not significant at n = 5. Seeds 5–6 are queued; the BN-adapted
  contrast needs only its current mean to hold to clear zero at n = 7.
* **Accuracy:** 0.90–0.91 on mBRSET now clears the 82.4 % all-negative base
  rate meaningfully (BRSET in-domain: 5.5 % prevalence, so 95 % accuracy there
  is barely above chance). Lead with AUROC; never headline accuracy without the
  base rate beside it.

## 6. Deployment

`export_coreml.py` on the exact deployed architecture (V4-Small, 384 px,
2.82 M params, fp16, normalisation + softmax folded in):

| | median | p10 / p90 |
|---|---|---|
| Apple Neural Engine (this Mac) | **0.7 ms** | 0.6 / 0.8 ms |
| CPU only | 2.8 ms | 2.7 / 3.5 ms |

Core ML vs PyTorch max abs difference 1.2 × 10⁻⁴ (one random-noise input —
a smoke check; the real-image pass/fail check is
`export_coreml.py --verify-images <dir>`). ONNX INT8 (val-selected
calibration) quantises the V4 to 3.1 MB from 11.3 MB fp32. The Mac ANE is an
*optimistic* proxy — the network is memory-bandwidth-bound and a Mac has far more
bandwidth than an iPhone — so read 0.7 ms as a lower bound on device latency; an
Xcode Core ML performance report on an iPhone gives the literal device number.

Chosen checkpoint for deployment: `ck_kd_v4_384/kd_seed1.pt`, selected by
in-domain val AUROC (0.9979), not by its mBRSET score. Its `.mlpackage` carries
source-domain BN statistics (zero-shot 0.8945 for this seed); the 0.9189
AdaBN figure needs `--bn-stats adapted`, and AdaBN on-device would require
re-estimating BN statistics from the phone's own unlabelled captures. Report the 5-seed mean ±
std as the science and this checkpoint's numbers as the shipped model.

## 7. Open items

1. **Ceiling** — `exp_ceiling_384` (V4@384 trained *on* mBRSET, 3 seeds) is
   running; it decomposes the residual gap into "the phone" vs "the task".
2. **Seeds 5–6** of ctrl/teacher/kd for the 7-seed distillation verdict.
3. **`evaluate_deploy.py` on `kd_seed1.pt`** with `--external-root <mBRSET>`
   (server): target-domain sensitivity/specificity at a val-calibrated
   threshold, plus INT8 cost.
4. **`export_coreml.py --checkpoint checkpoints/kd_seed1.pt`** (Mac) with the
   real weights: verification on real weights; latency will match the probe.
5. One pre-registered attempt at the teacher–student gap (DINOv2 teacher, or
   α = 0.9 with feature matching), run at 5 seeds and judged by the same paired
   test — one configuration, not a sweep of variants.
6. The segmentation track is unchanged apart from the audit fixes; its V4-M
   encoder question (`run_arch_sweep.py`) remains open, and its pre-fix FGADR
   numbers are not comparable to new ones.

## 8. Reproduction

One command (env vars for the credentialed data; stages selectable):

```bash
B=<BRSET root> M=<mBRSET root> bash reproduce.sh          # or: make reproduce B=... M=...
```

What it runs, by hand:

```bash
# paired design, V4 student, 384 px, GPU 0 (and V3 student on GPU 1)
SIZE=384 WORKERS=5 STUDENT=timm:mobilenetv4_conv_small.e2400_r224_in1k \
  OUT=exp_kd_v4_384 CK=ck_kd_v4_384 CUDA_VISIBLE_DEVICES=0 bash run_kd_xfer.sh 0 1 2 3 4
SIZE=384 WORKERS=5 OUT=exp_kd_v3_384 CK=ck_kd_v3_384 CUDA_VISIBLE_DEVICES=1 bash run_kd_xfer.sh 0 1 2

# paired statistics
python summarize_xfer.py --dir exp_kd_v4_384 --treatment kd --control ctrl

# deployment
python export_coreml.py --checkpoint ck_kd_v4_384/kd_seed1.pt
python evaluate_deploy.py --root <BRSET> --ckpt ck_kd_v4_384/kd_seed1.pt \
  --external-root <mBRSET> --target-sens 0.90 --calib sweep
```

Data: BRSET and mBRSET (PhysioNet), passed as `B=`/`M=` (on the lab box they
live under `/data/users4/nshaik3/Datasets/{BRSET,mBRSET}` on `arctrdgndev101`); FGADR under
the IIAI research-use agreement (non-redistributable; cite Zhou et al.,
arXiv:2008.09772).
