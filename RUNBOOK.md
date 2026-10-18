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
- [ ] Write the dated one-page analysis plan and commit it **before step 4**. It names:
  - the targets: sensitivity > 85 % and specificity > 82.5 %, per patient;
  - the Messidor-2 primary result as `PROTOCOL:METHOD:RULE` (e.g. `calibrated:em:dr_or_edema`)
    and the calibration (`full` or `shallow`);
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
cd ~/oculomics-systemic && git pull
```

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

- [ ] Camera ladder at matched prevalence, plus shallow recalibration. This runs on
      the GPU and takes minutes per checkpoint, so run it inside tmux:

```bash
cd ~/Oculomics && CUDA_VISIBLE_DEVICES=0 .venv/bin/python redesign_calibration.py --ckpt ck_rr_canon/seed*.pt --out exp_rr_canon/redesign
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

## Not scripted yet

- DR heatmaps of the same lesion type on the tabletop and the phone (mechanism panel).
- Sensitivity by sex, age, education and insurance, with confidence intervals.
- The cross-fitted retinal age clock (December, `disease/retinal-age`).
