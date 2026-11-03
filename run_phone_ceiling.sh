#!/usr/bin/env bash
# run_phone_ceiling.sh — the camera ladder's in-domain references, per seed
# (train_phone_ceiling.py), on the Canon sweep's own splits, with that seed's Canon
# checkpoint scored on the same test rows:
#   phone  trained on phone photos only                      -> $OUT/seed<s>
#   mixed  trained on the Canon train split + phone photos   -> $MIX_OUT/seed<s>   (needs B)
#
#   tmux new -s ceiling
#   cd ~/Oculomics && mkdir -p exp_phone_ceiling
#   B=<BRSET root> M=<mBRSET root> bash run_phone_ceiling.sh 0 2>&1 | tee -a exp_phone_ceiling/run.log
#   B=<BRSET root> M=<mBRSET root> bash run_phone_ceiling.sh 1 2 3 4   # the rest, once seed 0 looks sane
#
# Knobs (environment): M (required), B (BRSET root; without it only the phone run),
# MODES="phone mixed" (default: both when B is set), CAM="Canon CR" (the mixed run's
# tabletop camera = the sweep's source camera), SHARE=0.5 (phone share of the mixed
# run's draws), RR=exp_rr_canon and CKRR=ck_rr_canon (the sweep whose
# split_handheld.csv, split_tabletop.csv and seed<s>.pt are reused; missing files are
# skipped with a note), OUT=exp_phone_ceiling, CK=ck_phone_ceiling,
# MIX_OUT=exp_phone_mixed, MIX_CK=ck_phone_mixed, STUDENT, SIZE=384, EPOCHS=30,
# WORKERS=5, EXTRA (extra flags). Idempotent: a run whose results.json exists is skipped.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
: "${M:?set M=<mBRSET root>}"
B=${B:-}
OUT=${OUT:-exp_phone_ceiling}
CK=${CK:-ck_phone_ceiling}
MIX_OUT=${MIX_OUT:-exp_phone_mixed}
MIX_CK=${MIX_CK:-ck_phone_mixed}
RR=${RR:-exp_rr_canon}
CKRR=${CKRR:-ck_rr_canon}
CAM=${CAM:-Canon CR}
SHARE=${SHARE:-0.5}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
SIZE=${SIZE:-384}
EPOCHS=${EPOCHS:-30}
WORKERS=${WORKERS:-5}
EXTRA=${EXTRA:-}
if [ -n "$B" ]; then MODES=${MODES:-phone mixed}; else MODES=${MODES:-phone}; fi
PY=.venv/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4)
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
[ -d "$M" ] || { echo "[fatal] mBRSET root not found: $M"; exit 1; }
case " $MODES " in *" mixed "*)
  [ -n "$B" ] || { echo "[fatal] MODES has mixed but B (BRSET root) is not set"; exit 1; }
  [ -d "$B" ] || { echo "[fatal] BRSET root not found: $B"; exit 1; } ;;
esac
mkdir -p "$OUT" "$CK"

for s in "${SEEDS[@]}"; do
  SPLIT=(); CMP=()
  if [ -f "$RR/seed$s/split_handheld.csv" ]; then SPLIT=(--split-file "$RR/seed$s/split_handheld.csv")
  else echo "[note] no $RR/seed$s/split_handheld.csv: recomputing the phone split (seed 42)"; fi
  if [ -f "$CKRR/seed$s.pt" ]; then CMP=(--compare "$CKRR/seed$s.pt")
  else echo "[note] no $CKRR/seed$s.pt: no transfer gap for seed $s"; fi
  for mode in $MODES; do
    MIX=()
    if [ "$mode" = phone ]; then o=$OUT; c=$CK
    elif [ "$mode" = mixed ]; then
      o=$MIX_OUT; c=$MIX_CK
      # the camera name holds a space, so it travels as an array element, never via EXTRA
      MIX=(--mix-root "$B" --mix-camera "$CAM" --phone-share "$SHARE")
      if [ -f "$RR/seed$s/split_tabletop.csv" ]; then MIX+=(--mix-split-file "$RR/seed$s/split_tabletop.csv")
      else echo "[note] no $RR/seed$s/split_tabletop.csv: recomputing the $CAM split (seed 42)"; fi
    else echo "[fatal] unknown mode $mode (phone | mixed)"; exit 1; fi
    if [ -f "$o/seed$s/results.json" ]; then echo "[skip] $mode seed$s (results.json exists)"; continue; fi
    mkdir -p "$o" "$c"
    echo; echo "=== $mode seed$s   $(date) ==="
    # shellcheck disable=SC2086
    $PY -u train_phone_ceiling.py --root "$M" --backbone "$STUDENT" --image-size "$SIZE" \
        --epochs "$EPOCHS" --num-workers "$WORKERS" --seed "$s" \
        --out "$o/seed$s" --ckpt "$c/seed$s.pt" \
        ${SPLIT[@]+"${SPLIT[@]}"} ${CMP[@]+"${CMP[@]}"} ${MIX[@]+"${MIX[@]}"} $EXTRA
  done
done

for mode in $MODES; do
  if [ "$mode" = phone ]; then o=$OUT; else o=$MIX_OUT; fi
  echo; echo "=== $mode: summary over seeds   $(date) ==="
  $PY train_phone_ceiling.py --summarize "$o"
done
