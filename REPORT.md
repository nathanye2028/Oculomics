# Oculomics — Project Report, 20 August – 5 September 2026

On-device diabetic-retinopathy screening from smartphone fundus photographs.
This report covers the work done from the 20th of August to the 5th of
September: a full audit of the codebase, the architecture decision that followed
a null result on the original hypothesis, the two new levers that were built and
tested (foundation-model distillation and test-time BatchNorm adaptation), the
move to a MobileNetV4 backbone at 384 px, the resulting numbers with their
statistics, a full replication of the headline sweep under the final code
(which surfaced a reproducibility finding and demoted one result), the
deployment evaluation — including an operating-point shift that is the most
practically important finding of the project — and the extension of the harness
to glaucoma / AMD and systemic targets.

**Headline.** The model's zero-shot AUROC on 4,884 never-seen smartphone images
rose from **0.813 → 0.882–0.886** (control, two independent 7-seed sweeps) and
**0.904–0.907** with label-free test-time
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

### 5.2 Seven seeds, V4-Small at 384 px (31 August, first sweep)

Five seeds (30 August) reproduced the 22-August lesson exactly: the 3-seed
zero-shot distillation delta (+0.0155, 3/3) fell to +0.0064 with a CI spanning
zero. Two more seeds were added for a 7-seed verdict:

| | zero-shot mBRSET | BN-adapted mBRSET | adapt effect (paired) |
|---|---|---|---|
| ctrl | 0.8855 ± 0.0115 | 0.8959 ± 0.0073 | **+0.0104 [+0.0024, +0.0185] — significant** |
| kd | 0.8877 ± 0.0193 | **0.9068 ± 0.0113** | **+0.0191 [+0.0013, +0.0369] — significant** |

Paired kd − ctrl: **zero-shot +0.0022, CI [−0.0149, +0.0193], 4/7 seeds —
null.** On BN-adapted AUROC **+0.0109, CI [+0.0020, +0.0198], 6/7 seeds —
significant.** In-domain: ctrl 0.9756 ± 0.0067, kd 0.9780 ± 0.0071; gap
−0.090 for both. Teacher zero-shot mBRSET over the 7 seeds: 0.9446 / 0.9335 /
0.9298 / 0.9354 / 0.9349 / 0.9349 / 0.9396 (mean 0.936).

The reading: distillation on its own does **not** improve zero-shot transfer,
but the distilled student benefits about twice as much from test-time BN
adaptation as the control does, and the combination is significantly better
than AdaBN alone. The two levers interact: the teacher's representation is
more transferable, but that only shows once the student's BatchNorm statistics
match the target domain.

**Ceiling.** The same V4-Small at 384 px trained *on* mBRSET and tested
in-domain (3 seeds, n = 964 per test split): **0.9331 ± 0.0029** (0.9297 /
0.9368 / 0.9328). So the 49.7 M-parameter teacher's zero-shot 0.936 already
sits *at* the in-domain ceiling, and the 2.8 M-parameter student with AdaBN
(0.907) is within 0.026 of it — 97 % of the ceiling without ever training on a
smartphone image.

Per-seed, first five seeds (in-domain / zero-shot / adapted):

| seed | ctrl | kd | teacher |
|---|---|---|---|
| 0 | 0.9748 / 0.9020 / 0.9088 | 0.9801 / 0.9179 / 0.9223 | 0.9829 / 0.9446 |
| 1 | 0.9729 / 0.8726 / 0.8904 | 0.9882 / 0.8945 / 0.9189 | 0.9879 / 0.9335 |
| 2 | 0.9829 / 0.8868 / 0.8958 | 0.9810 / 0.8954 / 0.9072 | 0.9890 / 0.9298 |
| 3 | 0.9683 / 0.8786 / 0.8955 | 0.9696 / 0.8489 / 0.9084 | 0.9826 / 0.9354 |
| 4 | 0.9820 / 0.8696 / 0.8907 | 0.9736 / 0.8851 / 0.8876 | 0.9956 / 0.9349 |

`kd_seed3` looked like the most informative run: val AUROC 0.9944 (excellent),
zero-shot 0.8489 (the worst), and AdaBN restores it to 0.9084 — a
BatchNorm-statistics mismatch, the mechanism AdaBN exists for. **The replication
in §5.3 changed that reading:** re-run under the final code with the same seed,
`kd_seed3` scored 0.8861 zero-shot. The collapse was an unlucky non-deterministic
roll, not a property of the model, and individual runs should not be
over-interpreted. What does survive is the structural point: in-domain val is
saturated (0.97–0.99) and cannot rank checkpoints by how well they transfer,
and mBRSET cannot be used for selection without breaking the experiment.

### 5.3 Replication under the final code (3–5 September)

The 1 September hardening commit touched the training script (teacher
construction wrapped in `fork_rng`, so the kd arm's random stream is no longer
offset from the control's; AdaBN loader shuffled; `.done` completion markers;
GCG on a timm backbone is now an error). None of it changes the model, data,
loss or selection on CUDA, but the RNG alignment does change the kd arm, so
both student arms were re-run for all seven seeds with the seven teachers
reused (`exp_kd_v4_384_v2`).

**Fixed-seed runs are not bitwise reproducible.** The control's code path was
unchanged, yet only one of the first four control seeds reproduced:

| run | sweep 1 in / zero-shot / adapted | v2 in / zero-shot / adapted | Δ zero-shot | Δ adapted |
|---|---|---|---|---|
| ctrl_seed0 | 0.9748 / 0.9020 / 0.9088 | 0.9631 / 0.9016 / 0.8995 | −0.0004 | −0.0092 |
| ctrl_seed1 | 0.9729 / 0.8726 / 0.8904 | 0.9796 / 0.8815 / 0.8965 | +0.0089 | +0.0061 |
| ctrl_seed2 | 0.9829 / 0.8868 / 0.8958 | 0.9776 / 0.8552 / 0.8743 | −0.0316 | −0.0215 |
| ctrl_seed3 | 0.9683 / 0.8786 / 0.8955 | 0.9683 / 0.8786 / 0.8959 | 0.0000 | +0.0004 |
| kd_seed0 | 0.9801 / 0.9179 / 0.9223 | 0.9809 / 0.9151 / 0.9147 | −0.0029 | −0.0075 |
| kd_seed1 | 0.9882 / 0.8945 / 0.9189 | 0.9855 / 0.8927 / 0.9040 | −0.0018 | −0.0149 |
| kd_seed2 | 0.9810 / 0.8954 / 0.9072 | 0.9895 / 0.9075 / 0.9142 | +0.0121 | +0.0070 |
| kd_seed3 | 0.9696 / 0.8489 / 0.9084 | 0.9693 / 0.8861 / 0.9181 | +0.0371 | +0.0097 |

The run-to-run SD at a *fixed* seed (≈0.02) is as large as the seed-to-seed SD
(0.012). A probe with `torch.use_deterministic_algorithms(True)` set to raise
rather than warn ran a full epoch without flagging any op, so the residual
nondeterminism is not an op lacking a deterministic kernel. The most likely
remaining source is cuDNN algorithm *selection*, which under
`cudnn.deterministic=True` guarantees each algorithm is deterministic but not
that the same algorithm is chosen when GPU memory state differs — and the v2
runs shared their GPU with other jobs while the first sweep ran alone. The
cheap test (two identical runs back to back on an idle GPU, diff the loss
trace) is queued. Three consequences hold regardless:

1. **AdaBN results are immune** — paired *within* a run (same weights, adapted
   vs not), so training nondeterminism cannot reach them.
2. **The kd − ctrl contrast is legitimate but weaker than "paired by seed"
   implies**: pairing buys little power, and the two sweeps are replicates, not
   extra seeds.
3. **Single runs are not evidence** (`kd_seed3`, above).

**v2, seven seeds — the number of record:**

| | zero-shot mBRSET | BN-adapted mBRSET | adapt effect (paired) |
|---|---|---|---|
| ctrl | 0.8822 ± 0.0151 | 0.8924 ± 0.0088 | **+0.0102 [+0.0007, +0.0197] — significant** |
| kd | 0.8832 ± 0.0297 | **0.9043 ± 0.0183** | +0.0210 [−0.0024, +0.0445] n.s. |

Paired kd − ctrl: zero-shot +0.0011, CI [−0.0347, +0.0368], 5/7 — null (as
before). On BN-adapted AUROC **+0.0119, CI [−0.0094, +0.0332], 5/7 —
inconclusive**. In-domain ctrl 0.9741 ± 0.0077, kd 0.9772 ± 0.0079; gap −0.092
/ −0.094.

**What replicated and what did not.**

* *AdaBN on the control* — replicated: +0.0104 (sweep 1) and +0.0102 (v2), both
  CIs excluding zero. This is the project's one lever that is significant in
  two independent sweeps.
* *The distillation × adaptation interaction* — replicated in **direction and
  size**, not in **significance**: +0.0109 [+0.0020, +0.0198], 6/7, in sweep 1;
  +0.0119 [−0.0094, +0.0332], 5/7, in v2. The point estimate is stable; the v2
  CI is twice as wide because the v2 kd arm had one high-variance seed
  (zero-shot SD 0.030 vs 0.019). Over both sweeps, 11 of 14 paired deltas are
  positive with a mean of about +0.011 — but those 14 pairs share 7 patient
  splits, so they are not 14 independent observations and no pooled CI is
  quoted. **This result is demoted from "significant" to "suggestive and
  replicated in direction."** Its honest sentence: distillation does not improve
  zero-shot transfer in either sweep; combined with adaptation it is
  consistently a little better than adaptation alone, by roughly one seed-SD.
* *The ceiling framing* — unchanged: the adapted student sits at 0.904–0.907
  against an in-domain ceiling of 0.933 (§5.2).

### 5.4 The ladder

| | mBRSET AUROC |
|---|---|
| old baseline (V3-Small, 224 px, old recipe) | 0.813 |
| V3-Small, new recipe, 384 px | 0.863 |
| V4-Small, new recipe, 384 px (control, n = 7; sweep 1 / v2) | 0.886 / 0.882 |
| + AdaBN | 0.896 / 0.892 |
| + distillation + AdaBN | 0.907 / 0.904 |
| teacher, zero-shot (49.7 M params) | 0.936 |
| mBRSET-trained ceiling (V4-Small, in-domain) | 0.933 |

A 2.8 M-parameter student with test-time adaptation reaches ~97 % of the
in-domain ceiling and ~97 % of a 49.7 M-parameter teacher's transfer
performance, at 1/18th the teacher's size — in both sweeps.

### 5.5 What is claimable

* **Robust:** the recipe + resolution + V4 backbone effect. Fourteen of the
  fifteen V4 runs land 0.85–0.92 zero-shot (kd_seed3: 0.849) against an old
  baseline of 0.80–0.82.
* **Significant in two independent 7-seed sweeps:** AdaBN on the control
  (+0.0104 and +0.0102, both CIs excluding zero).
* **Suggestive, replicated in direction, not in significance:** distillation
  combined with AdaBN over AdaBN alone (+0.0109 sig. / +0.0119 n.s.; 11 of 14
  paired deltas positive). Say "consistently about one seed-SD better"; do not
  say "significant".
* **Null in both sweeps:** distillation on zero-shot transfer (+0.0022, +0.0011).
* **Noise floor, restated:** seed SD 0.012 *and* fixed-seed run-to-run SD
  ≈0.02. Any effect under ~0.02 needs far more than 7 runs; the AdaBN result
  cleared significance twice *including* this noise, which is why it is
  believable.
* **The operating-point shift (§6) is robust and matters more than any AUROC
  delta:** at a threshold calibrated on tabletop images, sensitivity on
  smartphone images halves (0.88 → 0.54) while AUROC drops only 0.09. A
  screening tool ships a threshold, not an AUROC.
* **Near-parity:** the adapted student (0.907) is within 0.026 of the
  mBRSET-trained ceiling (0.933); the residual gap after adaptation is small
  compared with the 0.12 that the recipe, resolution and backbone removed.
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

**Clinical operating point (`evaluate_deploy.py`, `kd_seed1.pt`, sweep 1).**
Threshold chosen on the in-domain (BRSET) validation split for sensitivity
≥ 0.90: 0.9182 (val sens 0.905, spec 0.996).

| scored on | AUROC | sensitivity | specificity | PPV | NPV | n |
|---|---|---|---|---|---|---|
| BRSET test (in-domain) | 0.9882 | 0.880 | 0.991 | 0.867 | — | 3,230 |
| **mBRSET (smartphone), same threshold** | 0.8945 | **0.535** | 0.984 | 0.876 | 0.908 | 4,884 |

This is the most practically important number in the project. The AUROC gap
is 0.09, but the *operating point does not transfer*: at the tabletop-calibrated
threshold the model misses nearly half of referable DR on smartphone images
while remaining highly specific — its output probabilities are systematically
lower on phone captures (the same BatchNorm-statistics mismatch AdaBN
corrects). Consequences for deployment: (a) a threshold must be calibrated on
the target device, which needs a small labelled handheld set, or (b) the
network must be adapted to the device (AdaBN from unlabelled captures) before a
tabletop threshold can be trusted. Quoting AUROC alone would have hidden this.

**Quantisation (RQ3).** INT8 via ONNX Runtime (val-selected MinMax calibration):
AUROC 0.9867 vs 0.9882 fp32 (−0.0015), 3.13 MB vs 11.26 MB (3.6× smaller),
sensitivity 0.870 / specificity 0.990 at an INT8-recalibrated threshold (the
fp32 threshold does not transfer across quantisation either: 0.9182 → 0.7848).
The 56 ms / 50 ms figures in that report are ONNX-CPU proxies on the lab box,
not device latency; the device number is the Core ML one above.

Chosen checkpoint for deployment: `ck_kd_v4_384/kd_seed1.pt`, selected by
in-domain val AUROC (0.9979), not by its mBRSET score. Its `.mlpackage` carries
source-domain BN statistics (zero-shot 0.8945 for this seed); the 0.9189
AdaBN figure needs `--bn-stats adapted`, and AdaBN on-device would require
re-estimating BN statistics from the phone's own unlabelled captures. Report the two 7-seed sweeps as the science and this checkpoint's numbers as
the shipped model; once a v2 checkpoint is chosen (it carries `model_bnadapt`),
re-run this evaluation on it.

## 7. Open items

1. ~~Ceiling~~ — done (0.9331 ± 0.0029, §5.2).
2. ~~Seeds 5–6~~ — done (§5.2). ~~v2 replication~~ — done (§5.3).
3. ~~`evaluate_deploy.py` on `kd_seed1.pt`~~ — done (§6); the operating-point
   shift is now the lead deployment finding.
4. **Operating point across all v2 checkpoints** (`score_external.py` on the
   fourteen v2 ctrl/kd runs): is the sensitivity collapse at a tabletop
   threshold universal, and does AdaBN restore it? One command, no training.
5. **Nondeterminism source**: two identical 1-epoch runs back to back on an
   idle GPU; if the loss traces match, the divergence is co-scheduling (cuDNN
   algorithm choice under memory pressure) and the fix is one job per GPU.
6. **`export_coreml.py`** (Mac) with real weights — the checkpoint is not on the
   Mac yet (`scp` it); latency will match the 0.7 ms probe.
7. One pre-registered attempt at the teacher–student gap (DINOv2 teacher, or
   α = 0.9 with feature matching), run at 5 seeds and judged by the same paired
   test — one configuration, not a sweep of variants.
8. The segmentation track: the 1 September fix to `make_rng` (persistent
   workers replayed the identical augmentation every epoch) means **every
   `train_idrid.py` result with `--num-workers > 0` predates a result-changing
   fix**, on top of the FGADR split change; its V4-M encoder question
   (`run_arch_sweep.py`) remains open. Treat all prior segmentation numbers as
   superseded until re-run.

## 8. Since 1 September: hardening, deployment tooling, and beyond DR

Recorded in `CHANGELOG.md`; summarised here because some of it changes how
earlier numbers should be read.

**Result-changing fixes (1 September).** `fundus_utils.make_rng` — persistent
DataLoader workers replayed the identical augmentation every epoch; every
segmentation result trained with workers predates this. `model_seg.DecoderBlock`
— GCG and control arms now share bit-identical non-gate initialisation at the
same seed. The orchestrators pass `--eval-tiled-val` through by default with
tiled evaluation. None of these touch the classification sweeps (the
classifier's augmentation uses torch's own RNG, not `make_rng`).

**Deployment tooling.** `export_coreml.py` gained `--bn-stats {source,adapted}`
(AdaBN weights are saved in the checkpoint as `model_bnadapt` from v2 onward), a
per-compute-unit benchmark, and real-image pass/fail verification
(`--verify-images`). `evaluate_deploy.py` reports median latency, qualifies the
CPU-proxy number, and accepts BRSET-trained checkpoints with an external
target-domain operating point.

**Repository.** Supported Python 3.9–3.13 stated and enforced; CUDA index
`cu126`; GitHub Actions running the CPU suite on every push; MIT licence (code
only); `reproduce.sh` / `Makefile`; `train.py`, `train_seg.py`, `seg_dataset.py`
moved to `legacy/`.

**Beyond DR (3–4 September; branches off `main`).** Three disease branches share
the DR harness verbatim — same trainer, paired-seed runner, AdaBN, summariser:

* `disease/systemic` — ten systemic targets from mBRSET's metadata
  (hypertension, nephropathy, neuropathy, myocardial infarction, …), in-domain,
  each run paired with an age+sex logistic baseline on the same split
  (`covariate_baseline.py`) and an optional arm warm-started from the DR student
  (`--init-from`). Only hypertension (71 % prevalence) is well powered.
* `disease/ophthalmic-multilabel` — BRSET's ten ophthalmic labels as one
  multi-label head vs one model per label, in-domain on tabletop images.
* `disease/glaucoma-amd` — adapters for AIROGS, REFUGE, PAPILA and ODIR-5K
  (`public_fundus.py`), `run_public_xfer.sh`, `score_external.py`. No public
  smartphone glaucoma/AMD set exists, so those are tabletop → tabletop transfers.

Novelty assessment, honestly: each disease alone re-treads published ground
(oculomics from fundus, the BRSET multi-label benchmark, cross-dataset
glaucoma). Their value is as **replication arms** for the methodological claims
of §5 — does the adaptation effect, and the suggestive distillation ×
adaptation interaction, hold on a second disease and shift? — and, combined,
as a single self-calibrating multi-head trunk on the phone. Dataset encodings on
every branch are unverified until each inspector runs on the real files; no
results from these branches exist yet.

## 9. Reproduction

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
