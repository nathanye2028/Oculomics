# Changelog

All notable changes to this project. Dates are ISO; results referenced are in REPORT.md.

## [Unreleased] — 2026-09-24 RetinaReach: self-calibrating, capability-declaring model  (branch `retinareach`, off `disease/metadata-baseline`)

- `retinareach.py`: `RetinaReachNet` — one standard timm trunk (default MobileNetV4-Conv-Small) with one linear head per target; image only, metadata never an input. Source heads (`dr_referable`, `edema`) trained on the tabletop set; systemic targets are linear probes on the frozen, self-calibrated trunk, so the trunk never sees a handheld label. `self_calibrate` = `train_mbrset.adapt_bn` plus an optional Schneider et al. (2020) source prior. Deployable calibration as two graphs: `BatchStatCollector` (batch → every BN layer's batch mean / unbiased variance; its average over batches reproduces AdaBN exactly) and `StatInputNet` (image + statistics vector → per-target probabilities; equals the eval-mode model with those statistics). Label-free camera signature (`camera_distance`: per-channel symmetric KL of the BN Gaussians), `DeviceProfile` with a same-camera envelope, and `device_capability` — the report the phone prints: nearest validated profile's capability, or NOT_VALIDATED for every target when the camera is outside all envelopes.
- `capability_gate.py`: one patient-grouped split per dataset over all rows (`multitask_split`, split seed 42 fixed across training seeds); per-target label-encoding audit (a 1/2-coded binary column is refused, not trained); the metadata comparator refit on that split (strongest of `metadata_model.py`'s logreg / GBM over the intake-form sets `age_sex`, `clinical`, chosen by grouped CV on train+val only; the `full` chart reported, not gating); paired patient-cluster bootstrap of image − metadata AUROC; statuses SUPPORTED / THRESHOLD_DRIFT / NO_IMAGE_EVIDENCE / UNDERPOWERED / NOT_VALIDATED. Standalone CLI prints the bar each target must clear from the label CSV alone.
- `train_retinareach.py`: the whole build for one seed — stage-1 trunk on BRSET, inductive AdaBN profiles from each device's train+val pool, stage-2 probes on calibrated mBRSET embeddings, thresholds fixed on home val, every target × device × protocol (`trained` vs `calibrated`) scored and gated, calibration-size sweep (N captures × repeats × prior); envelopes from pool draws, with device recognition and false acceptance scored out of sample on test-patient captures; writes `capability.csv`, `calibration_sweep.csv`, `profiles.json`, predictions, and the checkpoint.
- `export_retinareach.py`: Core ML `RetinaReach.mlpackage` (image + `bn_stats` input, fp16), `RetinaReachCalibrator.mlpackage` (fp32 by default), optional BN-folded baseline for a fixed profile (`--fused --fused-profile`), `retinareach_profiles.json`; `--verify-images` checks Core ML calibrator+model against PyTorch AdaBN on the same uint8 captures; per-compute-unit latency.
- `run_retinareach.sh` (5 seeds by default), `summarize_retinareach.py` (paired calibrated − trained threshold transfer, multi-seed capability table with a seed-SD noise floor and worst-case status, calibration-size curve).
- `tests/test_retinareach.py` (CPU, synthetic; the Core ML fidelity test runs on macOS only).

## [Unreleased] — 2026-09-23 metadata-only baseline  (branch `disease/metadata-baseline`)

- `metadata_model.py`: every ocular + systemic target predicted from clinical metadata alone (no pixels) — the capability-gate comparator. Nested feature sets `age_sex` ⊂ `clinical` ⊂ `full` (target column and its proxies always excluded; image-derived columns and camera never used), logistic regression + shallow gradient-boosted trees. Scored on the image trainer's own patient split (`--split-seed 42`, so metadata and image AUROCs pair on identical test patients) with patient-cluster bootstrap CIs, sens/spec/PPV/flagged fraction at a threshold fixed on training OOF predictions, repeated grouped CV (noise floor), a 20-permutation shuffled-label control, underpowered flags, and optional BRSET→mBRSET external scoring on shared features.
- `.venv` untracked: `b295732` had committed it as a self-referencing symlink, and checking out `disease/systemic` deletes a real `.venv` directory.
- `tests/test_metadata_model.py` (CPU-only, synthetic).

## [Unreleased] — 2026-09-03 systemic (oculomics) targets  (branch `disease/systemic`)

- `dataset.py`: `SYSTEMIC_TASKS` — `hypertension`, `nephropathy`, `neuropathy`, `myocardial_infarction`, `vascular_disease`, `diabetic_foot`, `obesity`, `smoking`, `alcohol`, `insulin` from mBRSET's metadata columns, via a strict `_binary_flag` (unknown tokens → NaN → dropped, never a confident 0); metadata columns extended.
- `covariate_baseline.py` + `train_mbrset.py --covariate-baseline [--covariate-features]`: age+sex logistic regression on the run's own split, recorded as `covariate_baseline` / `image_minus_covariate_auroc`.
- `train_mbrset.py --init-from <ckpt>`: warm-start every non-head tensor from another checkpoint (zero-shot weights only); refuses an external-root checkpoint, warns on overlapping splits; recorded as `init_from`.
- `inspect_mbrset.py` (pre-flight, `--strict`), `run_systemic.sh` (per-task `ctrl` vs optional `drinit` sweep), `summarize_systemic.py` (paired image-minus-covariate and treatment-minus-control per task), `make systemic` / `make inspect`.
- `run_mbrset.py --task` accepts every classification task in the registry.
- `tests/test_systemic.py` (CPU-only). Branch layout documented in README: `main` shared code, `disease/<target>` per disease.

## [Unreleased] — 2026-09-01 audit fixes

Bugs that changed results (re-run affected sweeps):
- `fundus_utils.make_rng`: persistent DataLoader workers replayed the identical augmentation every epoch (≈`num_workers` variants per image for the whole run). Every `train_idrid.py` result with `--num-workers > 0` predates this fix.
- `model_seg.DecoderBlock`: GCG and `--no-gcg` arms now share bit-identical non-gate initialisation at the same seed (the gate used to consume RNG before the fuse convs).
- `run_experiment.py` / `run_arch_sweep.py`: `--eval-tiled-val` is passed through (default on with tiled eval + patches), so checkpoint selection sees native-resolution microaneurysms.
- `run_kd_xfer.sh`: a partially trained teacher is no longer "reused"; completion is tracked by a `.done` marker.
- `train_mbrset.py`: AdaBN-adapted weights are saved (`model_bnadapt`); zero-BN backbones no longer report a fake adapted number; batch-size-1 crash in adaptation fixed; results JSON written before adaptation; GCG requested with a timm backbone is an error instead of a silent no-op; `--amp` on MPS uses bf16.

Deployment claims:
- `export_coreml.py`: `--bn-stats {source,adapted}`, per-compute-unit benchmark (CPU_AND_NE vs ALL vs CPU), `--warmup`/`--runs` matching the documented protocol, real-image verification with pass/fail (`--verify-images`), preprocessing spec in model metadata.
- `evaluate_deploy.py`: median latency, CPU-proxy latency key qualified, `image_ext` honoured.

Environment / repo:
- Python range stated and enforced (`pyproject.toml`, `check_env.py`): 3.9–3.13.
- `requirements.txt`: per-version numpy markers; coremltools macOS-only.
- Dockerfile / `setup_remote.sh`: CUDA index `cu126` (cu121 has no torch 2.8.0).
- `.dockerignore` mirrors `.gitignore` (no more 8 GB `data/` in the build context).
- GitHub Actions: CPU test suite on every push.
- `LICENSE` (MIT, code only), `reproduce.sh`, `Makefile`, this changelog.
- Untracked `.DS_Store`, `exp_s34.out`, `artifact_samples/`; removed empty `summary.md`; moved `train.py`, `train_seg.py`, `seg_dataset.py`, root `test_*.py` to `legacy/`.

## 2026-08-31 — REPORT.md
- 5-seed V4-Small@384 distillation + AdaBN results; Core ML export and clinical operating point (see REPORT.md).

## 2026-08-20 — codebase audit
- FGADR partition fixed at `split_seed=42`; DDP eval padding; gating read from checkpoints; INT8 calibration selected on val; single imbalance correction; full-frame eval resize; bf16 on MPS; NaN Dice for absent lesions. Pre-fix FGADR numbers are not comparable.
