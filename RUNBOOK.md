# Lab box run order: the November poster

Every command, in order, from committing on the Mac to the one Messidor-2 look.
Box paths assume the GSU datasets folder (`/data/users4/nshaik3/Datasets`); if
your roots differ, change `B`, `M` and `--root`. Steps marked **Paste** produce
output worth checking before you go on.

Conventions:
- One job per GPU (two ~11.6 GiB cards).
- Long jobs run inside `tmux`: detach with `Ctrl-b d`, reattach with `tmux attach -t <name>`.
- `~/Oculomics` runs the `retinareach` branch. `~/oculomics-systemic` runs
  `disease/systemic-oculomics`. Both use `~/Oculomics/.venv`.
- Both sweep scripts skip runs that already finished. After a crash or a
  dropped connection, run the same command again.

## 0. On the Mac

- [ ] Commit the `retinareach` work. In GitHub Desktop: File → Add Local Repository →
      `~/Documents/GitHub/Oculomics-retinareach`. Commit everything on `retinareach`,
      then Push origin.
- [ ] Commit and push `disease/systemic-oculomics` in `~/Documents/GitHub/Oculomics`
      (this file included).
- [ ] For section 7: commit and push `disease/retinal-age`. In GitHub Desktop: File → Add Local Repository →
      `~/Documents/GitHub/Oculomics-retinal-age` (train_retinal_age.py, run_retinal_age.sh,
      pool_age_folds.py, tests/test_age_folds.py, CHANGELOG.md), then Push origin.
- [ ] Write the dated one-page analysis plan and commit it **before step 4**. It names:
  - the targets: sensitivity > 85 % and specificity > 82.5 %, per patient;
  - the Messidor-2 primary result as `PROTOCOL:METHOD:RULE` (e.g. `calibrated:em2:dr_referable`;
    methods: `shipped`, `em`, `em2`, `rate`, `anchor`) and the calibration (`full` or `shallow`),
    chosen from the ladder, not after the look;
  - the systemic primary: hypertension against age + sex + diabetes, in all eyes and
    in normal-looking eyes;
  - what counts as success.

## 1. Connect and update the code on the box

GSU VPN on, then log in (password):

```bash
ssh dongye@arctrdgndev101.rs.gsu.edu
```

- [ ] See which branch each folder has, and that nothing is half-edited:

```bash
cd ~/Oculomics && git status -sb && git worktree list
```

- [ ] RetinaReach code:

```bash
cd ~/Oculomics && git fetch && git checkout retinareach && git pull
```

- [ ] Systemic code:

```bash
cd ~/oculomics-systemic && git fetch origin && git checkout --detach origin/disease/systemic-oculomics
```

  (This folder is a detached worktree, so `git pull` would refuse: "not currently on a branch".)

- [ ] Both folders show the commits you pushed in step 0:

```bash
cd ~/Oculomics && git log --oneline -1 && cd ~/oculomics-systemic && git log --oneline -1
```

## 2. Checks that spend nothing (minutes) **Paste all three**

- [ ] Messidor-2 grades, images and pairing sheet:

```bash
cd ~/Oculomics && .venv/bin/python public_fundus.py --root /data/users4/nshaik3/Datasets/Messidor2 --dataset messidor2 --inspect --strict
```

  Look for three things:
  - the grade column (`adjudicated_dr_grade` or `diagnosis`);
  - "grade ids with no image on disk: 0";
  - whether an examination sheet was found. Without one, every image counts as its own patient.

- [ ] RetinaReach Canon-only preflight. It checks the data, the camera names and
      the power per target, runs one GPU training step and prints a time estimate:

```bash
cd ~/Oculomics && CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_retinareach.py --source-root /data/users4/nshaik3/Datasets/BRSET --target-root /data/users4/nshaik3/Datasets/mBRSET --source-camera "Canon CR" --holdout-camera "NIKON NF5050" --out exp_rr_canon/preflight --ckpt ck_rr_canon/preflight.pt --preflight
```

  If a camera name is wrong, the error lists the real ones.

- [ ] Both GPUs free:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

  No rows below the header means both cards are free; a row is someone's job on that card.

## 3. Wait for the dated plan

- [ ] The plan from step 0 is committed and pushed. Nothing below runs before it.

## 4. The two sweeps (one per GPU, at the same time)

GPU 0 runs RetinaReach on Canon only: 5 seeds, then the shuffled-label control.
Open a tmux session first:

```bash
tmux new -s rr_canon
```

```bash
cd ~/Oculomics && B=/data/users4/nshaik3/Datasets/BRSET M=/data/users4/nshaik3/Datasets/mBRSET SOURCE_CAMERA="Canon CR" HOLDOUT_CAMERA="NIKON NF5050" SHUFFLE=1 OUT=exp_rr_canon CK=ck_rr_canon CUDA_VISIBLE_DEVICES=0 bash run_retinareach.sh 0 1 2 3 4 2>&1 | tee -a exp_rr_canon_sweep.log
```

Detach (`Ctrl-b d`). GPU 1 runs the systemic column: 4 targets × 5 folds, with vessel ablation on.

```bash
tmux new -s cm_folds
```

```bash
cd ~/oculomics-systemic && PY=~/Oculomics/.venv/bin/python GPU=1 FOLDS=5 COVARIATES="age sex diabetes" TASKS="cm_hypertension cm_heart_disease cm_cardiovascular_event cm_dyslipidemia" B=/data/users4/nshaik3/Datasets/BRSET M=/data/users4/nshaik3/Datasets/mBRSET OUT=exp_cm_folds CK=ck_cm_folds bash run_brset_cm.sh 0 2>&1 | tee -a exp_cm_folds.log
```

Its first lines are the positive and negative patient counts per target. It stops
if a target has fewer than 50 positive patients. **Paste** those lines.

- [ ] Progress, at any time:

```bash
tail -n 20 ~/Oculomics/exp_rr_canon_sweep.log ~/oculomics-systemic/exp_cm_folds.log
```

## 5. Summaries (after the sweeps)

- [ ] RetinaReach multi-seed table. The sweep prints it when it ends; this re-runs it:

```bash
cd ~/Oculomics && .venv/bin/python summarize_retinareach.py --dir exp_rr_canon
```

- [ ] Per-1,000 referral outcomes on every camera: refer, retake, missed and
      unnecessary referrals, plus the IDx-DR targets:

```bash
cd ~/Oculomics && .venv/bin/python referral_outcomes.py --dir exp_rr_canon
```

- [ ] Are the phone's systemic probes reading DR as a stand-in?

```bash
cd ~/Oculomics && .venv/bin/python dr_proxy_check.py --dir exp_rr_canon
```

- [ ] Camera ladder at matched prevalence, shallow recalibration, the label-free
      prevalence-matched pool (experiment D) and every threshold method (`em`,
      `em2`, `rate`, `anchor`) on every base. GPU, tens of minutes per checkpoint
      (experiment C draws 60 calibration sets per camera), so run it inside tmux:

```bash
cd ~/Oculomics && CUDA_VISIBLE_DEVICES=0 .venv/bin/python redesign_calibration.py --ckpt ck_rr_canon/seed*.pt --out exp_rr_canon/redesign
```

  The ladder at its end prints sensitivity AND specificity per camera and
  setting: the phone row needs both at 0.85 / 0.825 to meet the IDx-DR targets.

- [ ] Cross-fitted systemic probes: every handheld patient tested once, so the
      four UNDERPOWERED targets get 46-98 positive patients. Minutes per
      checkpoint on the GPU, one call per seed (the loop below does all five):

```bash
cd ~/Oculomics && for s in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python probe_folds.py --ckpt ck_rr_canon/seed$s.pt --target-root /data/users4/nshaik3/Datasets/mBRSET --folds 5 --out exp_rr_canon/seed$s; done
```

  Then re-run the multi-seed summary; the new rows carry protocol `calibrated_oof5`:

```bash
cd ~/Oculomics && .venv/bin/python summarize_retinareach.py --dir exp_rr_canon
```

- [ ] Systemic, pooled out of fold: the photo's gain over age + sex + diabetes,
      matched pairs and vessel ablation, per subset:

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python summarize_systemic.py --dir exp_cm_folds --control ctrl
```

  **Paste** these four files:
  - `exp_rr_canon/summary.md`;
  - the referral table;
  - `exp_rr_canon/redesign/redesign_summary.txt` (the ladder is at its end);
  - the `calibrated_oof5` rows of the re-run `exp_rr_canon/summary.md`;
  - `exp_cm_folds/summary.md`.

- [ ] Copy the results to the Mac. Run these on the Mac, not on the box:

```bash
scp -r dongye@arctrdgndev101.rs.gsu.edu:Oculomics/exp_rr_canon ~/Documents/GitHub/Oculomics-retinareach/
```

```bash
scp -r dongye@arctrdgndev101.rs.gsu.edu:oculomics-systemic/exp_cm_folds ~/Documents/GitHub/Oculomics/
```

- [ ] Poster figures. They need matplotlib, and the box venv may not have it.
      If `exp_rr_canon/figures/` is missing, run this on the Mac:

```bash
cd ~/Documents/GitHub/Oculomics-retinareach && python3 plot_retinareach.py --dir exp_rr_canon
```

## 6. Nov 8: freeze, then the Messidor-2 look

- [ ] The analysis is frozen: no more changes to thresholds, calibration choice or the primary.
- [ ] Preflight. It scores nothing, writes nothing and reports the ledger's state:

```bash
cd ~/Oculomics && CUDA_VISIBLE_DEVICES=0 .venv/bin/python score_once.py --dataset messidor2 --root /data/users4/nshaik3/Datasets/Messidor2 --checkpoints "ck_rr_canon/seed*.pt" --preflight
```

- [ ] The look, exactly once. Replace the primary, the calibration and the plan path
      with your plan's:

```bash
cd ~/Oculomics && CUDA_VISIBLE_DEVICES=0 .venv/bin/python score_once.py --dataset messidor2 --root /data/users4/nshaik3/Datasets/Messidor2 --checkpoints "ck_rr_canon/seed*.pt" --primary calibrated:em:dr_or_edema --calibration full --plan <path to your plan on the box> --out exp_messidor2_once 2>&1 | tee messidor2_once.log
```

  The ledger, `~/Oculomics/external_test_ledger.json`, now refuses any second look at
  Messidor-2. **Paste** the PRIMARY line and the tables.

- [ ] Bring the record home and commit it right away. On the Mac:

```bash
scp dongye@arctrdgndev101.rs.gsu.edu:Oculomics/external_test_ledger.json ~/Documents/GitHub/Oculomics-retinareach/
```

```bash
mkdir -p ~/Documents/GitHub/Oculomics-retinareach/external_test/messidor2 && scp "dongye@arctrdgndev101.rs.gsu.edu:Oculomics/exp_messidor2_once/*.json" "dongye@arctrdgndev101.rs.gsu.edu:Oculomics/exp_messidor2_once/*.csv" ~/Documents/GitHub/Oculomics-retinareach/external_test/messidor2/
```

  Then commit both on `retinareach` in GitHub Desktop. The `exp_*` folders are
  gitignored, which is why the outputs go to `external_test/`. The commit date is
  part of the record.

## 7. Reach 2, across the body: probe, vessels, measured labels

Three tracks, in this order of cost. Box folders: `~/oculomics-systemic` (this branch) and
`~/oculomics-age` (`disease/retinal-age`, for the cross-fitted clock). Paste the lines marked **Paste**.

### 7.0 One-time setup

- [ ] On huggingface.co, sign in and request access to `YukunZhou/RETFound_dinov2_meh` (gated, CC BY-NC 4.0).
- [ ] On the box, the two new packages and the HuggingFace login (it asks for a token from your HF settings):

```bash
~/Oculomics/.venv/bin/pip install "pydicom>=2.4" huggingface_hub
```

```bash
~/Oculomics/.venv/bin/python -c "from huggingface_hub import login; login()"
```

- [ ] A worktree for the age clock, detached like the systemic one:

```bash
cd ~/Oculomics && git fetch origin && git worktree add --detach ~/oculomics-age origin/disease/retinal-age
```

### 7.1 Frozen RETFound + small classifier (Recommendation 1)

- [ ] The mBRSET CNNs on the SAME 5 folds (GPU, hours; skip a task you do not need):

```bash
tmux new -s sys_folds
```

```bash
cd ~/oculomics-systemic && PY=~/Oculomics/.venv/bin/python GPU=1 FOLDS=5 M=/data/users4/nshaik3/Datasets/mBRSET OUT=exp_systemic_folds CK=ck_systemic_folds bash run_systemic.sh 0 2>&1 | tee -a exp_systemic_folds.log
```

- [ ] The probes (embeddings take minutes on the GPU, the probes seconds on the CPU). It also compares
      against `exp_systemic_folds` and the step-4 `exp_cm_folds` once they exist; rerun it then:

```bash
cd ~/oculomics-systemic && PY=~/Oculomics/.venv/bin/python GPU=0 B=/data/users4/nshaik3/Datasets/BRSET M=/data/users4/nshaik3/Datasets/mBRSET bash run_probe_systemic.sh 2>&1 | tee -a exp_probe.log
```

  **Paste** `exp_probe/mbrset/summary.md`, `exp_probe/brset_canon/summary.md` and the
  "PROBE ... vs CNN" blocks. Read the matched-pair concordance and the incremental delta.

- [ ] Only if a task's probe PASSES and its CNN does not: distil fold K's probe into fold K's
      phone student (one run per fold; same folds, same covariates). Example, mBRSET nephropathy fold 0:

```bash
cd ~/oculomics-systemic && CUDA_VISIBLE_DEVICES=1 ~/Oculomics/.venv/bin/python train_mbrset.py --dataset mbrset --root /data/users4/nshaik3/Datasets/mBRSET --task nephropathy --fold 0 --n-folds 5 --fold-seed 42 --backbone timm:mobilenetv4_conv_small.e2400_r224_in1k --no-gcg --image-size 384 --epochs 25 --covariate-baseline --covariate-features age sex --teacher exp_probe/mbrset/nephropathy/probe_fold0_seed0_probe.npz --ckpt-dir ck_distil/nephropathy --run-name kd_fold0_seed0 --results-json exp_distil/nephropathy/kd_fold0_seed0.json
```

### 7.2 Vessel measurements (Recommendation 2)

- [ ] **Before any AutoMorph output or cross-fitted prediction is opened:** fill in `vessel_prereg.json`
      (verify every suggested source, set `status` to `final` and `dated`), commit and push it.
      `vessel_associations.py` refuses a draft or uncommitted file. Then bring it to the box:

```bash
cd ~/oculomics-systemic && git fetch origin && git checkout --detach origin/disease/systemic-oculomics && git log --oneline -1 -- vessel_prereg.json
```
- [ ] Stage the photos (symlinks, no copies) and print the AutoMorph command:

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python automorph_stage.py --dataset brset --root /data/users4/nshaik3/Datasets/BRSET --camera "Canon CR" --out ~/automorph/brset_canon
```

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python automorph_stage.py --dataset mbrset --root /data/users4/nshaik3/Datasets/mBRSET --out ~/automorph/mbrset
```

- [ ] Run AutoMorph with the printed command (Docker if the box allows it; otherwise its conda setup,
      LOCAL.md in github.com/rmaphoh/AutoMorph). A rerun wipes `Results/`.
- [ ] The cross-fitted, bias-corrected age clock (the v5 recipe, 5 folds, one seed; two GPUs):

```bash
cd ~/oculomics-age && PY=~/Oculomics/.venv/bin/python GPU=0 FOLDS=5 FOLD_LIST="0 1 2" B=/data/users4/nshaik3/Datasets/BRSET M=/data/users4/nshaik3/Datasets/mBRSET MIX=1 MEDIUM=1 SKIP_STUDENT=1 SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 EXTRA="--ema-decay 0.999" OUT=exp_retinal_age_cf CK=ck_retinal_age_cf bash run_retinal_age.sh 0 2>&1 | tee -a exp_retinal_age_cf_gpu0.log
```

```bash
cd ~/oculomics-age && PY=~/Oculomics/.venv/bin/python GPU=1 FOLDS=5 FOLD_LIST="3 4" B=/data/users4/nshaik3/Datasets/BRSET M=/data/users4/nshaik3/Datasets/mBRSET MIX=1 MEDIUM=1 SKIP_STUDENT=1 SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 EXTRA="--ema-decay 0.999" OUT=exp_retinal_age_cf CK=ck_retinal_age_cf bash run_retinal_age.sh 0 2>&1 | tee -a exp_retinal_age_cf_gpu1.log
```

  Whichever finishes last pools the folds (`pool_age_folds.py`). **Paste** its "bias check" lines:
  the healthy slope of gap on age should be near 0 after correction.

- [ ] The pre-registered tests and the forest plot (BRSET; then mBRSET without `--age-gap`):

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python vessel_associations.py --automorph ~/automorph/brset_canon --dataset brset --root /data/users4/nshaik3/Datasets/BRSET --prereg vessel_prereg.json --age-gap ~/oculomics-age/exp_retinal_age_cf/predictions_oof_medium_mix.csv --out exp_vessels/brset
```

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python vessel_associations.py --automorph ~/automorph/mbrset --dataset mbrset --root /data/users4/nshaik3/Datasets/mBRSET --prereg vessel_prereg.json --out exp_vessels/mbrset
```

  **Paste** both `summary.md`. Redraw the forest on the Mac: `scp` the `forest.csv`, then
  `python3 vessel_associations.py --plot-only forest.csv`.

### 7.3 Measured labels from AI-READI (Recommendation 3)

- [ ] Register on fairhub.io and request the dataset (verified-ID login, type 2 diabetes research
      only, licence acceptance; your name, email and stated purpose become public on the project site).
- [ ] Download to the box (Azure SAS link from FAIRhub), then look before trusting any name:

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python aireadi_dataset.py inspect --root <AI-READI download>
```

  **Paste** it. If `targets` later says a name is ambiguous, pin it with `--albumin-source "<name>"` etc.

- [ ] Convert the colour photographs (Aurora = handheld, the rest tabletop) and build the targets:

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python aireadi_dataset.py convert --root <AI-READI download> --out ~/aireadi --workers 8
```

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python aireadi_dataset.py targets --root <AI-READI download> --out ~/aireadi
```

- [ ] Embeddings, then the same probe on the measured targets (continuous: R2 gain; flags: AUROC delta),
      comparator age + diabetes stage as three indicators (no sex in the public release):

```bash
cd ~/oculomics-systemic && CUDA_VISIBLE_DEVICES=0 ~/Oculomics/.venv/bin/python extract_retfound.py --dataset aireadi --root ~/aireadi --out features/aireadi_retfound.npz
```

```bash
cd ~/oculomics-systemic && for t in sbp log_uacr moca htn_measured albuminuria moca_impaired; do ~/Oculomics/.venv/bin/python probe_systemic.py --dataset aireadi --root ~/aireadi --task $t --features features/aireadi_retfound.npz --covariate-features age dm_pre dm_oral dm_insulin --out exp_probe/aireadi; done
```

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python summarize_systemic.py --dir exp_probe/aireadi --tasks htn_measured albuminuria moca_impaired --control probe
```

  **Paste** `exp_probe/aireadi/summary.md` and the three `exp_probe/aireadi/*/summary_regression_probe.md`.

- [ ] Vessels on the measured targets, one device per run (`convert` printed the camera strings and fields).
      Set the `aireadi` camera in `vessel_prereg.json` (a dated amendment if it is already final), commit, push,
      refresh `~/oculomics-systemic` as in 7.2, then per device (a tabletop one, then the Optomed Aurora):

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python automorph_stage.py --dataset aireadi --root ~/aireadi --camera "<device string>" --out ~/automorph/aireadi_<device>
```

  Run AutoMorph with the printed command, then:

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python vessel_associations.py --automorph ~/automorph/aireadi_<device> --dataset aireadi --root ~/aireadi --camera "<device string>" --prereg vessel_prereg.json --out exp_vessels/aireadi_<device>
```

  **Paste** each `summary.md`.

- [ ] The outside check: the BRSET hypertension models against AI-READI's measured blood pressure,
      handheld and tabletop separately:

```bash
cd ~/oculomics-systemic && ~/Oculomics/.venv/bin/python score_external.py --task cm_hypertension --dataset aireadi --root ~/aireadi --probes "exp_probe/brset_canon/cm_hypertension/probe_fold*_seed0_probe.npz" --features features/aireadi_retfound.npz --checkpoints "ck_cm_folds/cm_hypertension/ctrl_fold*_seed0.pt" --out exp_external/aireadi_htn
```

  **Paste** `exp_external/aireadi_htn/summary.md`. Keep every AI-READI file on the box (licence).

## Not scripted yet

- DR heatmaps of the same lesion type on the tabletop and the phone (mechanism panel).
- Sensitivity by sex, age, education and insurance, with confidence intervals.
- Vessel ablation through AutoMorph's vessel masks instead of the line detector.
