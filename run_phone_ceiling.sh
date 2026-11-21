#!/usr/bin/env bash
# run_phone_ceiling.sh — the camera ladder's in-domain references, per seed
# (train_phone_ceiling.py), on the Canon sweep's own splits, with that seed's Canon
# checkpoint scored on the same test rows:
#   phone  trained on phone photos only                      -> $OUT/seed<s>
#   mixed  trained on the Canon train split + phone photos   -> $MIX_OUT/seed<s>   (needs B)
#   mixed_sim  mixed, its Canon photos re-rendered as phone photos by phone_sim.py before
#          the model sees them (opt-in: MODES=mixed_sim)     -> $SIM_OUT/seed<s>   (needs B, PROFILE)
#          Phone-only does worse than mixed, so the Canon photos add something the phone
#          photos alone do not teach: does a simulated phone photo add as much? Its summary
#          ends with mixed_sim minus mixed and minus phone, paired by seed.
#
#   tmux new -s ceiling
#   cd ~/Oculomics && mkdir -p exp_phone_ceiling
#   B=<BRSET root> M=<mBRSET root> bash run_phone_ceiling.sh 0 2>&1 | tee -a exp_phone_ceiling/run.log
#   B=<BRSET root> M=<mBRSET root> bash run_phone_ceiling.sh 1 2 3 4   # the rest, once seed 0 looks sane
#   B=<BRSET root> M=<mBRSET root> MODES=mixed_sim PROFILE=ck_phone_sim_v3/profile.json bash run_phone_ceiling.sh 0
#
# Knobs (environment): M (required), B (BRSET root; without it only the phone run),
# MODES="phone mixed" (default: both when B is set), CAM="Canon CR" (the mixed run's
# tabletop camera = the sweep's source camera), SHARE=0.5 (phone share of the mixed
# run's draws), RR=exp_rr_canon and CKRR=ck_rr_canon (the sweep whose
# split_handheld.csv, split_tabletop.csv and seed<s>.pt are reused; missing files are
# skipped with a note), OUT=exp_phone_ceiling, CK=ck_phone_ceiling,
# MIX_OUT=exp_phone_mixed, MIX_CK=ck_phone_mixed, STUDENT, SIZE=384, EPOCHS=30,
# WORKERS=5, EXTRA (extra flags). mixed_sim: PROFILE=ck_phone_sim/profile.json (phone_sim.py
# fit's; its bank must be the sweep's handheld train split), SIM_P=1.0 (share of Canon draws
# re-rendered), SIM_OUT=exp_phone_mixed_sim, SIM_CK=ck_phone_mixed_sim (a fresh pair for
# another PROFILE or SIM_P: a finished seed is skipped whatever it was trained with).
# Idempotent: a run whose results.json exists is skipped.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
: "${M:?set M=<mBRSET root>}"
B=${B:-}
OUT=${OUT:-exp_phone_ceiling}
CK=${CK:-ck_phone_ceiling}
MIX_OUT=${MIX_OUT:-exp_phone_mixed}
MIX_CK=${MIX_CK:-ck_phone_mixed}
SIM_OUT=${SIM_OUT:-exp_phone_mixed_sim}
SIM_CK=${SIM_CK:-ck_phone_mixed_sim}
PROFILE=${PROFILE:-ck_phone_sim/profile.json}
SIM_P=${SIM_P:-1.0}
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
case " $MODES " in *" mixed "*|*" mixed_sim "*)
  [ -n "$B" ] || { echo "[fatal] MODES has mixed / mixed_sim but B (BRSET root) is not set"; exit 1; }
  [ -d "$B" ] || { echo "[fatal] BRSET root not found: $B"; exit 1; } ;;
esac
case " $MODES " in *" mixed_sim "*)
  [ -f "$PROFILE" ] || { echo "[fatal] MODES has mixed_sim but there is no profile at $PROFILE (phone_sim.py fit)"; exit 1; } ;;
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
    elif [ "$mode" = mixed ] || [ "$mode" = mixed_sim ]; then
      if [ "$mode" = mixed ]; then o=$MIX_OUT; c=$MIX_CK; else o=$SIM_OUT; c=$SIM_CK; fi
      # the camera name holds a space, so it travels as an array element, never via EXTRA
      MIX=(--mix-root "$B" --mix-camera "$CAM" --phone-share "$SHARE")
      if [ -f "$RR/seed$s/split_tabletop.csv" ]; then MIX+=(--mix-split-file "$RR/seed$s/split_tabletop.csv")
      else echo "[note] no $RR/seed$s/split_tabletop.csv: recomputing the $CAM split (seed 42)"; fi
      if [ "$mode" = mixed_sim ]; then MIX+=(--mix-phone-sim "$PROFILE" --mix-sim-p "$SIM_P"); fi
    else echo "[fatal] unknown mode $mode (phone | mixed | mixed_sim)"; exit 1; fi
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
  VS=()
  if [ "$mode" = phone ]; then o=$OUT
  elif [ "$mode" = mixed ]; then o=$MIX_OUT
  else
    o=$SIM_OUT                              # against the sweeps it is a variant of, where they exist
    for d in "$MIX_OUT" "$OUT"; do if compgen -G "$d/seed*/ceiling.csv" > /dev/null; then VS+=("$d"); fi; done
  fi
  echo; echo "=== $mode: summary over seeds   $(date) ==="
  $PY train_phone_ceiling.py --summarize "$o" ${VS[@]+--versus "${VS[@]}"}
done
