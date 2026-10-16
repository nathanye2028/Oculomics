# Changelog

All notable changes to this project. Dates are ISO; results referenced are in REPORT.md.

## [Unreleased] — 2026-10-16 Systemic oculomics: Canon-only image runs and "does the photo add to age+sex?"  (branch `disease/systemic-oculomics`)

- `covariate_baseline.incremental_value`: covariates vs covariates + the image model's logit, both logistic regressions fit on the validation rows and scored on the test rows; AUROC of each, the image alone, the paired delta with a patient-cluster bootstrap CI and `p_no_gain`; re-scored inside named subsets.
- `train_mbrset.py`: `--camera` (filters `--root` before the split); BRSET runs carry `diabetes`/`nationality` and, for a `cm_*` task, the parsed comorbidity labels (`brset_dataset.add_brset_extras`); `cm_hypertension` is scored on mBRSET's hypertension column as the external test; with `--covariate-baseline` on a binary task the run adds `incremental` (overall and `no_diabetes_dr0`: non-diabetic patients whose worst eye is grade 0). `evaluate(..., return_probs=True)`, `rows_for`, `nodm_dr0_mask`.
- `brset_dataset.filter_camera` / `add_brset_extras` / `BRSET_EXTRA_METADATA` (moved from metadata_model.py, which re-exports them). `brset_comorbidities.MBRSET_EQUIVALENTS` keeps hypertension only: the metadata run scored BRSET-fit kidney models 0.38–0.43 and MI 0.44–0.56 on mBRSET.
- `summarize_systemic.py`: per condition, "+image vs covariates" paired by seed for every `incremental` block; title and footnote no longer assume mBRSET.
- `run_brset_cm.sh`: BRSET `cm_*` targets on one camera (default `cm_hypertension`, `Canon CR`, MobileNetV4-Medium @ 512, 30 epochs, batch 16, EMA 0.999); pre-flight counts positive patients per task on the filtered table and refuses < 50; `PY=` for running from a worktree; `GPU=` per invocation.
- Measured on the lab box (2026-10-16, `metadata_model.py --camera "Canon CR"`): metadata bars for the cm targets — hypertension 0.742 (age+sex 0.73), heart disease 0.726, cardiovascular event 0.697, thyroid 0.684, dyslipidemia 0.658; the rest have ≤ 7 test-positive patients.
- `tests/test_incremental.py` (7): a planted image signal is found and a useless one is not, subsets and refusals, row alignment and the no-DM/DR-0 mask, a one-epoch Canon-only `cm_hypertension` run end to end with the mBRSET external test, the summariser's pairing.

## [Unreleased] — 2026-10-16 Fix: sex coded 1 = male in every dataset  (branch `disease/systemic-oculomics`)

- `brset_dataset._normalise_sex`: BRSET 1/2 -> 1/0 (1 = male), was `- 1` (1 = female). Verified on the real CSV: all 9 "prostatic hyperplasia" patients are BRSET 1 and all 6 "breast cancer" patients 2; mBRSET's code 1 drinks 2.3x and has 1.7x the MI rate, so mBRSET 1 = male. The old rule reversed sex between BRSET and mBRSET.
- `public_fundus.load_odir`: Male -> 1, Female -> 0 (was the reverse). PAPILA's Gender is still passed through raw and is marked unverified.
- `brset_comorbidities._sex01` follows the same rule. Test `test_sex_is_one_for_male_in_every_adapter` pins it.
- Affected: only models with a sex input fit on one dataset and scored on another (`metadata_model.py` external rows, `train_mbrset.py --covariate-baseline` external line). None of the reported runs did that — the systemic, RetinaReach and age-gap analyses fit and score sex within one dataset. Other branches still carry the old rule.

## [Unreleased] — 2026-10-16 Systemic oculomics: metadata baselines for the BRSET comorbidity targets  (branch `disease/systemic-oculomics`)

- `dataset.py`: `COMORBIDITY_TASKS` — every `brset_comorbidities.GROUPS` target registered as `cm_<target>` (0/1, NaN when no history was recorded), so the metadata model and the image trainer read the same labels.
- `brset_comorbidities.attach_flags` joins the patient-level flags onto the image-level BRSET table (patient id as text; refuses a join matching < 99% of rows); `MBRSET_EQUIVALENTS` / `mbrset_equivalents` map `cm_hypertension`, `cm_myocardial_infarction`, `cm_kidney` to mBRSET's structured columns.
- `metadata_model.py`: BRSET tables carry the `cm_*` targets (default target list, `cm_diabetes` excluded); `--camera` keeps one camera's rows of the primary table (the audit found histories for 75% of Canon CR vs 3% of Nikon NF5050 patients); `cm_diabetes` bans the diabetes / treatment columns as proxies; the `full` set of a comorbidity target adds the other `cm_*` flags except any group sharing a condition with it; comorbidity targets with an mBRSET equivalent are scored on mBRSET.
- `tests/test_metadata_comorbidities.py` (7): registration, the join, feature rules, the camera filter, and a Canon-only end-to-end run scored on an mBRSET-like table.

## [Unreleased] — 2026-10-15 Systemic oculomics: BRSET comorbidities  (branch `disease/systemic-oculomics`)

- `brset_comorbidities.py`: BRSET's free-text `comorbidities` column parsed into per-patient flags — `atom_<condition>` (54 conditions; vocabulary written against the real CSV's strings and the first lab-box audit's unmapped list, misspellings and Portuguese spellings included) and `cm_<target>` groups (hypertension, diabetes, dyslipidemia, heart disease, MI, stroke, cardiovascular event, kidney, neurodegenerative, Alzheimer's, brain_any, autoimmune, antimalarial use, thyroid). `na` is missing (NaN), never negative; `0` is recorded-none; unmapped items are listed, not dropped. Audit: coverage, who has a recorded history (by camera, age band, sex, diabetes, worst DR grade) with an adjusted camera odds ratio, free text vs BRSET's own `diabetes` column, and per target the size, age gap, prevalence by camera and an age/sex/diabetes-adjusted camera odds ratio (flagged when its CI excludes 1). numpy-only logistic regression (no statsmodels). Outputs `brset_comorbidity_flags.csv`, `comorbidity_audit.md`, `comorbidity_audit.json`.
- First lab-box run (8,524 patients): history recorded for 49%, 75% of Canon CR vs 3% of Nikon NF5050 patients (adjusted OR 0.13) and 100% of diabetics vs 40% of non-diabetics; the free text and the `diabetes` column agree perfectly (so that check is not independent and the report now says so). Hypertension 1,748 / 2,448, heart disease 111, stroke 25, neurodegenerative 20, Alzheimer's 8.
- `tests/test_brset_comorbidities.py` (57): vocabulary on the real strings, missing vs none, patient consolidation, BOM header, odds-ratio recovery, a planted camera confound flagged and an age-only signal not flagged, a one-camera target marked unstable without overflow, cross-check orientation and the perfect-agreement wording, CLI outputs.

## [Unreleased] — 2026-10-04 RetinaReach: DR-proxy check  (branch `retinareach`)

- `dr_proxy_check.py`: on a finished sweep's saved predictions (no GPU), per systemic probe on the handheld test split — the probe's AUROC, the AUROC of the model's own referable-DR output and of the grader's ICDR grade as predictors of the same condition, the probe's AUROC among ICDR-0 eyes, and its within-grade AUROC (pairs of eyes with the same ICDR grade; patient-bootstrap CI). `dr_proxy.csv`. Answers whether a probe reads the systemic condition or DR as a stand-in for it.

## [Unreleased] — 2026-10-03 RetinaReach: redesign round 1  (branch `retinareach`)

- REPORT.md §10.8: first redesign results on real data — prevalence drives the threshold shift (shown on the tabletop camera alone); most of the earlier AUROC gain comes from re-estimating statistics on clean source images, not from the target camera; shallow-only recalibration keeps the AUROC and drops the sensitivity cost; EM works only on unadapted scores.
- `redesign_calibration.py`: B3 and C add a shallow-only base (`--shallow-frac`, default half the BN layers), so the candidate design — shallow recalibration for the camera, label-free threshold for the prevalence — is measured end to end.

## [Unreleased] — 2026-10-02 RetinaReach: label-free threshold  (branch `retinareach`)

- `retinareach.label_free_threshold` (+ `ThresholdReference`, `fit_threshold_reference`; `em_prevalence` and `expected_sensitivity_threshold` moved here from `redesign_calibration.py`): the device re-sets each source threshold from the unlabelled captures it calibrated on — `em` (Saerens EM prevalence on Platt-calibrated scores, expected-sensitivity threshold) or `anchor` (shipped threshold moved in logit space by the shift of the captures' lower score quartile, `--anchor-quantile`). Pure numpy, scalar references, for a direct port to the phone.
- `train_retinareach.py`: references fit on tabletop validation and shipped in `profiles.json`, the checkpoint and `results.json`; `operating_point.csv` (each device's full pool × method: sensitivity with a patient-bootstrap CI, specificity, flagged fraction, EM prevalence vs the pool's); the calibration-size sweep adds `sens_<t>__<method>`, `flagged_<t>__<method>`, `prev_<t>__em` from the same N captures; `--calib-sizes` default now starts at 4.
- `redesign_calibration.py`: B3 runs both methods through the shared function; new experiment C — N captures (`--small-n`, `--small-n-repeats`) give both the statistics and the threshold, every method on tabletop statistics and on AdaBN; the summary reports how often a single draw reaches the sensitivity floor. `--anchor-q`.
- `summarize_retinareach.py`: label-free sweep columns and an operating-point table. `plot_retinareach.py`: fig3 draws the shipped, EM and anchored thresholds from the same captures with the floor marked (older runs: prior lines as before); title no longer collides with the legend on a one-target figure.
- REPORT.md §10.6 (flag rate below prevalence on the phone), §10.7 (the two methods in simulation: EM fails under a score shift, the anchor holds; capture count for the threshold).

## [Unreleased] — 2026-09-30 RetinaReach: calibration redesign  (branch `retinareach`)

- First lab-box results (seeds 0 and 3) in REPORT.md §10.6: AdaBN raises phone/ODIR AUROC but lowers sensitivity at the shipped threshold.
- `retinareach.calibrate_blended`: calibration with the source prior applied IN the forward pass (Schneider et al. 2020) and/or only the first K BN layers recalibrated (shallow-only; the first K entries equal full AdaBN's, so the existing collector serves it on device). `self_calibrate(prior_strength > 0)` uses it (blend weight from the dataset size). Fixes the post-hoc per-layer blend, which on real data pushed sensitivity below both endpoints; `train_retinareach.py`'s sweep recalibrates the same captures per prior (and no longer depends on the order of `--prior-strengths`); results record `prior_mode`, and the summarizer and plots drop pre-fix prior rows.
- `redesign_calibration.py`: on finished checkpoints, no retraining — (A) prevalence-controlled calibration pools on every camera, including the tabletop one (mechanism); (B1) shallow-only depth sweep; (B2) in-forward prior; (B3) EM prevalence estimate (Saerens et al. 2002) on Platt-calibrated outputs + expected-sensitivity threshold. `redesign.csv`, `redesign_summary.txt`.

## [Unreleased] — 2026-09-28 RetinaReach: preflight, vessel ablation, shared-vs-per-target, figures  (branch `retinareach`)

- `train_retinareach.py --preflight` (run once by `run_retinareach.sh` before any seed, `PREFLIGHT=1`): data roots, encodings, images on disk, positive patients per target and device (UNDERPOWERED flagged up front), unfamiliar cameras, one training step on the real device (loss must be plausible), one calibration (finite statistics), pretrained weights, disk, GPU memory, a wall-clock estimate; the sweep stops if it fails.
- `vessel_ablation.py`: training-free vessel segmentation (multi-scale line detector, Nguyen et al. 2013; window scaled from DRIVE's 15 px at 565 px), fixed-area mask, inpainting by normalised Gaussian convolution, control = the same mask rotated 90° inside the FOV. `train_retinareach.py` scores every validated camera's test split intact / vessels removed / control removed (`anatomy_ablation.csv`, paired patient-bootstrap `vessels_minus_control`); `--ablation-frac` (0 disables).
- `train_mbrset.py --split-file` (fixed partition from a `file,split` CSV; a patient in two splits is refused): per-target fine-tuned models on RetinaReach's own handheld split. `PERTARGET="hypertension insulin"` in the run script; the summary pairs them with the shared-trunk probes by seed.
- `plot_retinareach.py`: five poster figures + a CSV of what each plots (threshold transfer, capability matrix, calibration size, mechanism, unfamiliar camera); needs only pandas + matplotlib (the Mac's system python3); drawn at the end of the run script when available.
- `capability_gate.py` CLI accepts every dataset `load_any` knows (e.g. `--dataset odir`).

## [Unreleased] — 2026-09-26 RetinaReach: controls, mechanism checks, third camera, INT8, Swift reference  (branch `retinareach`)

- Negative controls (plan 5.8): every probe refit on permuted labels (`--n-shuffle`, must sit at 0.5; warns otherwise); `--shuffle-source-labels` turns a whole run into a control (`SHUFFLE=1` in `run_retinareach.sh` → `shuffle_seed<s>/`); a quality-flags-only comparator (`quality_only_auroc`) on every gate row.
- Mechanism checks (plan 5.7): camera decodability — a patient-grouped linear read-out of camera identity from the trunk embedding, as-trained vs calibrated; `quality_strata.csv` — AUROC / sensitivity on gradable vs ungradable images. Calibration metrics on every row (Brier, ECE, mean predicted risk).
- Unfamiliar cameras: `--unfamiliar ROOT DATASET [NAME]` (`U=` in the run script, `kaggle:` ids resolved) — a labelled camera the model is not validated on builds nothing; its full gate is compared with what the phone's label-free lookup shows after N captures (`unfamiliar_lookup.csv`: shown, unsafe and missed rates). `public_fundus.py` ported verbatim from `disease/retinal-age` and `brset_dataset.load_any` dispatch aligned with that branch, so ODIR-5K (DR grade from keywords, age, sex) loads as a third camera.
- `summarize_retinareach.py`: controls, shuffle runs, decodability (paired by seed), quality strata, unfamiliar-camera capability and lookup sections.
- `export_retinareach.py`: `--weights int8` (convolutions int8, head float), `--bundle` (one multifunction package, iOS 18+), package sizes in the report, NaN-free `retinareach_profiles.json`, `--swift-cli` parity check against the Swift reference.
- `app/RetinaReachKit`: Swift package (library + `retinareach-cli`) — training-identical preprocessing, on-device calibration, camera lookup, screening; `README.md`. Opt-in parity test `RR_SWIFT=1`.
- `.gitignore`: Swift build products.
- Mac (MPS) fixes, both silent before: a `non_blocking` copy of the strided label slice to MPS delivered garbage labels (loss 1e15, then negative) — copies are non-blocking only on CUDA now (also in `train_mbrset.adapt_bn`); with spawn-started DataLoader workers a numpy view of the dataset labels taken before the first loader ran pointed at freed memory (validation AUROC undefined every epoch) — `MultiTargetDataset.label_array()` returns a copy. Regression test runs on MPS with workers.

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
