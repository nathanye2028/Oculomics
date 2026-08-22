#!/usr/bin/env bash
# run_kd_xfer.sh — foundation-model distillation + test-time BN adaptation on the
# BRSET -> mBRSET transfer experiment. Run it on the GPU box inside tmux:
#
#   tmux new -s kd
#   cd ~/Oculomics && bash run_kd_xfer.sh 0 1 2 2>&1 | tee -a exp_kd/sweep.log
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
# Idempotent: a run whose JSON exists is skipped, so re-running resumes.
# Wall-clock (A100, 224px): ~40 min ctrl, ~60-90 min teacher, ~50 min kd per seed.
#
# Override any of these via the environment, e.g. SIZE=384 EPOCHS=30 bash run_kd_xfer.sh 0
set -euo pipefail
cd "$(dirname "$0")"

B=${B:-/data/users4/nshaik3/Datasets/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1}
M=${M:-/data/users4/nshaik3/Datasets/mBRSET/physionet.org/files/mbrset/1.0}
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
EXTRA=${EXTRA:-}                 # extra flags for BOTH student arms, e.g. "--ema-decay 0.999"
TEACHER_EXTRA=${TEACHER_EXTRA:-} # extra flags for the teacher only
TRAIN_DATASET=${TRAIN_DATASET:-brset}   # schema of $B (brset); mbrset only for local smoke tests

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
mkdir -p "$OUT" "$CK"
[ -d "$B" ] || { echo "[fatal] BRSET root not found: $B"; exit 1; }
[ -d "$M" ] || { echo "[fatal] mBRSET root not found: $M"; exit 1; }
.venv/bin/python -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: .venv/bin/pip install -r requirements.txt"; exit 1; }

COMMON=(--dataset "$TRAIN_DATASET" --root "$B" --external-test-root "$M" --external-test-dataset mbrset
        --task dr_referable --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS"
        --bn-adapt --ckpt-dir "$CK")

run() {  # run <name> <flags...>
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "[skip] $name (exists)"; return 0; fi
  echo; echo "=== $name   $(date) ==="
  .venv/bin/python train_mbrset.py "${COMMON[@]}" --run-name "$name" \
      --results-json "$OUT/$name.json" "$@"
}

for s in "${SEEDS[@]}"; do
  # 1) control: the deployable student, no teacher
  run "ctrl_seed$s"    --seed "$s" --backbone "$STUDENT" $EXTRA
  # 2) teacher: large backbone, same split. GCG flags are ignored for timm backbones.
  #    Reused (not retrained) if TEACHER_CK already holds it from another sweep —
  #    legitimate because the same seed gives the same patient split.
  if [ -f "$TEACHER_CK/teacher_seed$s.pt" ] && [ ! -f "$OUT/teacher_seed$s.json" ]; then
    echo "[reuse] teacher $TEACHER_CK/teacher_seed$s.pt (trained by another sweep)"
  else
    mkdir -p "$TEACHER_CK"
    run "teacher_seed$s" --seed "$s" --backbone "$TEACHER" --lr "$TEACHER_LR" \
                         --ckpt-dir "$TEACHER_CK" $TEACHER_EXTRA
  fi
  # 3) student distilled from that teacher (same seed => same patient split)
  run "kd_seed$s"      --seed "$s" --backbone "$STUDENT" --teacher "$TEACHER_CK/teacher_seed$s.pt" \
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
    print(f"  {os.path.basename(p)[:-5]:<16} in={r['test']['auroc']:.4f}  mBRSET={e['auroc']:.4f}"
          + (f"  mBRSET+BNadapt={a['auroc']:.4f}" if a else "") + f"  params={r.get('params_m')}M")
PY
echo "done=$(date)  ->  $OUT/summary.md"
