#!/usr/bin/env bash
# run_kd_xfer.sh — foundation-model distillation + test-time BN adaptation on the
# BRSET -> mBRSET transfer experiment. Run it on the GPU box inside tmux:
#
#   tmux new -s kd
#   cd ~/Oculomics && B=<BRSET root> M=<mBRSET root> bash run_kd_xfer.sh 0 1 2 2>&1 | tee -a exp_kd/sweep.log
#   (Ctrl-b, d to detach;  tmux attach -t kd  to come back)
#
# Per seed it trains THREE models on the same patient split (same --seed), all
# scored zero-shot on mBRSET and again after label-free AdaBN (--bn-adapt):
#   ctrl_seed<s>     mobile student ($STUDENT; MobileNetV3-Small default), no teacher
#   teacher_seed<s>  large timm backbone (ImageNet-22k ConvNeXt-S by default)
#   kd_seed<s>       same student architecture, distilled from teacher_seed<s>
# ctrl and kd are identical in every way except the distillation term, so
# summarize_xfer.py's paired kd-minus-ctrl is attributable to distillation; the
# BN-adapt effect is paired within each run. The deployed model (kd) has the
# SAME architecture, params and Core ML latency as ctrl — nothing mobile changes.
#
# Idempotent: a run whose JSON exists is skipped, so re-running resumes. A
# teacher is only REUSED from $TEACHER_CK when its `.done` marker exists: the
# trainer saves the .pt on every val improvement, so a bare .pt is just as
# likely to be a killed run as a finished one.
# Wall-clock (A100, 224px): ~40 min ctrl, ~60-90 min teacher, ~50 min kd per seed.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> M=<mBRSET root> [KNOB=value ...] bash run_kd_xfer.sh [seed ...]

Trains ctrl / teacher / kd per seed on BRSET, scores each on mBRSET zero-shot
and after label-free BN adaptation, then prints paired kd-minus-ctrl statistics.
Seeds default to 0 1 2. Every knob is an environment variable:

  required
    B              BRSET root      (dir holding fundus_photos/ + labels_brset.csv)
    M              mBRSET root     (dir holding images/ + labels_mbrset.csv)
  outputs
    OUT            results JSONs + summary        (default exp_kd)
    CK             checkpoints for ctrl/kd         (default ck_kd)
    TEACHER_CK     checkpoints for teachers        (default $CK). Point a 2nd sweep
                   (different STUDENT, same seeds) at the 1st sweep's dir to reuse
                   its finished teachers instead of retraining them.
  models
    STUDENT        deployable backbone for ctrl AND kd (default mobilenetv3_small;
                   e.g. timm:mobilenetv4_conv_small.e2400_r224_in1k)
    TEACHER        timm backbone (default timm:convnext_small.fb_in22k_ft_in1k;
                   or timm:vit_base_patch14_dinov2.lvd142m)
    TEACHER_LR     teacher learning rate           (default 1e-4)
  training
    SIZE           image size                      (default 224)
    EPOCHS         epochs per run                  (default 25)
    WORKERS        DataLoader workers              (default 8)
    KD_ALPHA       weight on the KD term           (default 0.7)
    KD_TEMP        distillation temperature        (default 4.0)
    FEAT_W         cosine feature-matching weight  (default 0.0 = off)
    AMP            "" = trainer default, "--amp" / "--no-amp" to force
    EXTRA          extra flags for BOTH student arms, e.g. "--ema-decay 0.999"
    TEACHER_EXTRA  extra flags for the teacher only
    TRAIN_DATASET  schema of $B (default brset; mbrset only for local smoke tests)
    EXT_DATASET    schema of $M (default mbrset; any of brset_dataset.DATASETS, e.g. refuge)
    TASK           task for every arm (default dr_referable; e.g. glaucoma, amd)

example
    B=/data/BRSET/1.0.1 M=/data/mBRSET/1.0 SIZE=384 EPOCHS=30 bash run_kd_xfer.sh 0 1 2
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1    # epoch lines must reach `tee` live, not at process exit

# The dataset roots are machine-specific; a default pointing into someone
# else's home directory only produces a confusing "not found" on every other box.
: "${B:?set B=<BRSET root> (dir holding fundus_photos/ + labels_brset.csv); see --help}"
: "${M:?set M=<mBRSET root> (dir holding images/ + labels_mbrset.csv); see --help}"
OUT=${OUT:-exp_kd}
CK=${CK:-ck_kd}
TEACHER_CK=${TEACHER_CK:-$CK}    # where teachers live. Point a 2nd sweep (different
                                 # STUDENT, same seeds) at the 1st sweep's dir to reuse them.
STUDENT=${STUDENT:-mobilenetv3_small}   # the deployable model (ctrl AND kd). e.g.
                                         # timm:mobilenetv4_conv_small.e2400_r224_in1k
TEACHER=${TEACHER:-timm:convnext_small.fb_in22k_ft_in1k}   # or timm:vit_base_patch14_dinov2.lvd142m
TEACHER_LR=${TEACHER_LR:-1e-4}
SIZE=${SIZE:-224}
EPOCHS=${EPOCHS:-25}
WORKERS=${WORKERS:-8}
KD_ALPHA=${KD_ALPHA:-0.7}
KD_TEMP=${KD_TEMP:-4.0}
FEAT_W=${FEAT_W:-0.0}            # >0 adds cosine feature matching on top of logit KD
AMP=${AMP:-}                     # "" = trainer default; "--no-amp" / "--amp" to force
EXTRA=${EXTRA:-}                 # extra flags for BOTH student arms, e.g. "--ema-decay 0.999"
TEACHER_EXTRA=${TEACHER_EXTRA:-} # extra flags for the teacher only
TRAIN_DATASET=${TRAIN_DATASET:-brset}   # schema of $B (brset); mbrset only for local smoke tests
EXT_DATASET=${EXT_DATASET:-mbrset}      # schema of $M; any of brset_dataset.DATASETS
TASK=${TASK:-dr_referable}              # every arm trains and is scored on this task

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
export OUT    # the teacher-ceiling printout below reads it from the environment
mkdir -p "$OUT" "$CK"
[ -d "$B" ] || { echo "[fatal] BRSET root not found: $B"; exit 1; }
[ -d "$M" ] || { echo "[fatal] mBRSET root not found: $M"; exit 1; }
.venv/bin/python -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: .venv/bin/pip install -r requirements.txt"; exit 1; }

# GCG is a MobileNetV3-specific gate: model.py now REFUSES use_gcg on a timm:
# backbone instead of silently dropping it, so timm students and the teacher
# must be told --no-gcg explicitly. (For the MobileNetV3 student the ctrl/kd
# contrast is about distillation, and both arms keep the same GCG setting.)
STUDENT_GCG=()
case "$STUDENT" in timm:*) STUDENT_GCG=(--no-gcg);; esac

COMMON=(--dataset "$TRAIN_DATASET" --root "$B" --external-test-root "$M" --external-test-dataset "$EXT_DATASET"
        --task "$TASK" --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS"
        --bn-adapt --ckpt-dir "$CK" $AMP)

run() {  # run <name> <flags...>
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "[skip] $name (exists)"; return 0; fi
  echo; echo "=== $name   $(date) ==="
  .venv/bin/python -u train_mbrset.py "${COMMON[@]}" --run-name "$name" \
      --results-json "$OUT/$name.json" "$@"
}

for s in "${SEEDS[@]}"; do
  # 1) control: the deployable student, no teacher
  run "ctrl_seed$s"    --seed "$s" --backbone "$STUDENT" ${STUDENT_GCG[@]+"${STUDENT_GCG[@]}"} $EXTRA
  # 2) teacher: large backbone, same split, --no-gcg (timm backbone). Reused
  #    (not retrained) only when TEACHER_CK holds a FINISHED teacher from another
  #    sweep — the trainer writes <name>.done after its results JSON, whereas the
  #    .pt appears at the first val improvement — legitimate because the same
  #    seed gives the same patient split.
  tpt="$TEACHER_CK/teacher_seed$s.pt"; tdone="$TEACHER_CK/teacher_seed$s.done"
  if [ -f "$tdone" ] && { [ "$TEACHER_CK" != "$CK" ] || [ -f "$OUT/teacher_seed$s.json" ]; }; then
    echo "[reuse] teacher $tpt (completed: $(tr -d '\n' < "$tdone"))"
  else
    if [ -f "$tpt" ]; then
      echo "[warn] $tpt exists without a completion marker ($tdone) or results JSON;"
      echo "       treating it as a killed partial run: deleting it and retraining the teacher."
      rm -f "$tpt" "$tdone"
    fi
    mkdir -p "$TEACHER_CK"
    run "teacher_seed$s" --seed "$s" --backbone "$TEACHER" --no-gcg --lr "$TEACHER_LR" \
                         --ckpt-dir "$TEACHER_CK" $TEACHER_EXTRA
  fi
  # 3) student distilled from that teacher (same seed => same patient split)
  run "kd_seed$s"      --seed "$s" --backbone "$STUDENT" ${STUDENT_GCG[@]+"${STUDENT_GCG[@]}"} \
                       --teacher "$tpt" \
                       --kd-alpha "$KD_ALPHA" --kd-temp "$KD_TEMP" \
                       --distill-feat-weight "$FEAT_W" $EXTRA
done

echo; echo "=== paired statistics   $(date) ==="
.venv/bin/python summarize_xfer.py --dir "$OUT" --treatment kd --control ctrl
echo
echo "teacher ceiling (what the student is being pulled toward), per seed:"
.venv/bin/python - <<'PY'
import glob, json, os
for p in sorted(glob.glob(os.path.join(os.environ.get("OUT", "exp_kd"), "teacher_seed*.json"))):
    r = json.load(open(p)); e = r["external"]; a = r.get("external_bnadapt") or {}
    ext = r.get("external_dataset") or "ext"
    print(f"  {os.path.basename(p)[:-5]:<16} in={r['test']['auroc']:.4f}  {ext}={e['auroc']:.4f}"
          + (f"  {ext}+BNadapt={a['auroc']:.4f}" if a else "") + f"  params={r.get('params_m')}M")
PY
echo "done=$(date)  ->  $OUT/summary.md"
