# Changelog

All notable changes to this project. Dates are ISO; results referenced are in REPORT.md.

## [Unreleased] — 2026-09-06 GCG attention maps

- `save_gcg_maps.py`: saves the attention maps every GCG gate produces (spatial map + channel vector per gated skip) as `.npz`, labelled overlay panels, per-gate PNGs and an on-lesion vs off-lesion `gate_stats.csv`; whole-image or tiled (stitched like the prediction); segmentation or classifier checkpoints; refuses `--no-gcg` checkpoints.
- `model_seg.record_gcg_gates` / `collect_gcg_gates` / `gcg_gate_modules`: opt-in capture, off by default. Every block in `gcg_blocks.py` records what it computes; custom blocks follow the same one-flag contract.
- Tests: `tests/test_gcg_maps.py` (capture, variants, classifier gate, tiled == whole, end-to-end on a folder, stats, refusal).

## [Unreleased] — 2026-09-06 retinal age track (branch `disease/retinal-age`)

- `train_retinal_age.py`: age regression on BRSET's healthy cohort (`--healthy nodm|dr0|gradable|all`, `--exclude-pathology`), patient-level cohort rule, age-stratified patient-grouped split drawn over all patients (diseased patients scored, never trained on), healthy-val-MAE selection, L1/Huber/MSE on standardised age, optional age-bin-balanced sampling, MAE by age bin, patient-level MAE, mBRSET zero-shot + AdaBN with DR-grade breakdown, Beheshti-style age-gap bias correction fit on healthy val, per-image predictions CSV. `--inspect` prints the cohort without touching images.
- `run_retinal_age.sh` (student per seed, optional teacher reference, cohort pre-flight), `summarize_retinal_age.py` (mean ± SD, per-bin table, paired contrast, pooled predictions), `launch_disease_runs.sh retinalage`.
- 2026-09-07, after the first sweep (BRSET healthy MAE ≈ 5.1 y; mBRSET zero-shot ≈ 14 y, AdaBN ≈ 9.7 y — compressed age scale on phone images): device-side recalibration of the external set (`external_recal`, 2-fold patient-grouped linear fit on DR-0 patients, out-of-fold, own within-set bias correction; `--no-recalibrate`, `--recal-group`), also applied to the AdaBN predictions; `CEILING=1` in `run_retinal_age.sh` trains the in-domain mBRSET DR-0 ceiling with BRSET as the reverse external set; summariser reports calibrated sets, the within-external referable-minus-DR-0 corrected gap, the device scale factor, and pairs only conditions with the same training set.
- Levers (2026-09-07, after the calibrated sweep put mBRSET at r = 0.43): `train_retinal_age.py` now wraps the trunk in `RetinalAgeModel` (linear or label-distribution head `--head ldl`, years from `forward`, `--tta` flip averaging), regression distillation (`--teacher`, `--kd-alpha`, `--distill-feat-weight`), `PhoneAug` (`--phone-aug`), and mixed-domain training (`--extra-train-root`, own cohort rule / split / bias correction; external numbers on held-out rows when the roots coincide). `run_retinal_age.sh`: HEAD, TTA, PHONE_AUG, MIX, EXTRA_WEIGHT, KD, KD_ALPHA, FEAT_W, TAG knobs; `kd_seed<n>` after each teacher. `summarize_retinal_age.py`: mixed-in sets, recipe line, seed-ensemble MAE / r.
- 2026-09-11: the lab box's GPUs turned out to be 11.6 GiB (512 px runs OOMed). `train_retinal_age.py --auto-batch` (CUDA default) probes one training step and halves the batch with `--grad-accum` doubling until it fits; a mid-run CUDA OOM skips the batch (bounded by `--max-oom-skips`); `batch_size_used` / `grad_accum_used` / `oom_skips` / `peak_gpu_gib` recorded. `run_retinal_age.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and gained `TEACHER_EXTRA`.
- 2026-09-13: `--gcg {baseline,attention,cbam,se}` on the age clock (MobileNetV3-Small trunk only, refused elsewhere; recorded in the checkpoint head spec), `load_age_model` shared by the teacher loader and the new `explain_retinal_age.py` (Grad-CAM on the deepest trunk activation via forward hooks, GCG gate maps via `record_gcg_gates`, PIL-only grid + central-mass CSV). `run_retinal_age.sh`: `MEDIUM=1` (MobileNetV4-Medium student arm) and `GCG=<variant>` (paired ctrl/gcg arms on `GCG_STUDENT`).
- `analyze_age_gap.py` (step 2): patient-level gap-vs-disease associations per dataset — OLS-adjusted Δ with CI / p / BH q, Cohen's d, prevalence by gap quintile and OR per +5 y (Newton logistic), DR-grade and diabetes-duration trends, image-quality artefact check; joins BRSET's ophthalmic flags from the raw CSV; external rows use the device-calibrated within-set gap. Run by `run_retinal_age.sh` after the summary. Tests in `tests/test_age_gap.py`.
- `brset_dataset.py`: BRSET `diabetes` mapped through the adapter (optional; mBRSET has none).
- Tests: `tests/test_retinal_age.py` (cohort rule, split, metrics, bias correction, 1-epoch end-to-end on synthetic trees).

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
