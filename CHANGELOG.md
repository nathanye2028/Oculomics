# Changelog

All notable changes to this project. Dates are ISO; results referenced are in REPORT.md.

## [Unreleased] — 2026-09-06 GCG attention maps

- `save_gcg_maps.py`: saves the attention maps every GCG gate produces (spatial map + channel vector per gated skip) as `.npz`, labelled overlay panels, per-gate PNGs and an on-lesion vs off-lesion `gate_stats.csv`; whole-image or tiled (stitched like the prediction); segmentation or classifier checkpoints; refuses `--no-gcg` checkpoints.
- `model_seg.record_gcg_gates` / `collect_gcg_gates` / `gcg_gate_modules`: opt-in capture, off by default. Every block in `gcg_blocks.py` records what it computes; custom blocks follow the same one-flag contract.
- Tests: `tests/test_gcg_maps.py` (capture, variants, classifier gate, tiled == whole, end-to-end on a folder, stats, refusal).

## [Unreleased] — 2026-09-06 retinal age track (branch `disease/retinal-age`)

- `train_retinal_age.py`: age regression on BRSET's healthy cohort (`--healthy nodm|dr0|gradable|all`, `--exclude-pathology`), patient-level cohort rule, age-stratified patient-grouped split drawn over all patients (diseased patients scored, never trained on), healthy-val-MAE selection, L1/Huber/MSE on standardised age, optional age-bin-balanced sampling, MAE by age bin, patient-level MAE, mBRSET zero-shot + AdaBN with DR-grade breakdown, Beheshti-style age-gap bias correction fit on healthy val, per-image predictions CSV. `--inspect` prints the cohort without touching images.
- `run_retinal_age.sh` (student per seed, optional teacher reference, cohort pre-flight), `summarize_retinal_age.py` (mean ± SD, per-bin table, paired contrast, pooled predictions), `launch_disease_runs.sh retinalage`.
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
