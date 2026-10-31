#!/usr/bin/env bash
# run_phone_sim.sh — does training on simulated phone photos close the tabletop -> phone gap?
#
#   tmux new -s phonesim
#   cd ~/Oculomics && mkdir -p exp_phone_sim
#   B=<BRSET root> M=<mBRSET root> bash run_phone_sim.sh 0 2>&1 | tee -a exp_phone_sim/run.log
#   B=<BRSET root> M=<mBRSET root> STAGES="train score" bash run_phone_sim.sh 1 2 3 4 2>&1 | tee -a exp_phone_sim/run.log
#
# Stages (each idempotent; STAGES picks a subset):
#   fit    phone_sim.py fit: the handheld bank from the mBRSET TRAIN split (no label enters
#          it; the split is the trainer's, stratified on referable DR) and the tabletop
#          statistics from the Canon train split -> $CK/profile.json, fit_report.txt
#          (cross-fitted realism per statistic, the field the renders get) and fit_grid.jpg.
#          An existing profile is reused only if it has this format, size and camera.
#   check  style_transfer_xfer.py check on the Canon sweep's checkpoint: realism among
#          ICDR-0 eyes (camera distance, camera probe) and label preservation (judge,
#          logit shifts, vessel_r) for phone_sim next to colour / fda_sharp ->
#          $CK/check_seed<s>/ (check_seed<s>_strength<x>/ at another STRENGTH;
#          check_summary.txt; grid.jpg is dataset photos, so ck_*).
#   train  the Canon sweep again, identical except --phone-sim (run_retinareach.sh with
#          SOURCE_CAMERA / HOLDOUT_CAMERA, so the Nikon rung and every output exist too).
#          Its preflight prints "phone_sim: N ms per batch" and the loader's own img/s.
#          Refuses an ARM_OUT whose seeds were trained with another P, STRENGTH or profile.
#   score  style_transfer_xfer.py score, Canon-only vs Canon + simulated phone photos, on
#          the same real phone and Canon rows (PART: test by default), for EVERY seed with
#          both checkpoints (a seed missing on one side is left out of both): AUROC,
#          sensitivity / specificity at the shipped threshold, oracle specificity at 0.85
#          sensitivity, the per-seed paired difference, and each arm's settings ->
#          $OUT/score/ for the pre-specified arm on test, $OUT/score_<ARM_CK>_<PART>/ else.
#
# Ablation arms (P, STRENGTH or the profile changed) are chosen on the handheld VALIDATION
# split and need a fresh ARM_OUT / ARM_CK each (and a fresh CK for another profile):
#   P=0.25 ARM_OUT=exp_rr_p025 ARM_CK=ck_rr_p025 PART=val STAGES="train score" bash run_phone_sim.sh 0 1 2 3 4
#   STRENGTH=0.5 ARM_OUT=exp_rr_s05 ARM_CK=ck_rr_s05 PART=val STAGES="check train score" bash run_phone_sim.sh 0 1 2 3 4
#   CK=ck_phone_sim_mk FIT_EXTRA="--colour-mode mk" ARM_OUT=exp_rr_mk ARM_CK=ck_rr_mk PART=val bash run_phone_sim.sh 0 1 2 3 4
#   CK=ck_phone_sim_icdr0 FIT_EXTRA="--select icdr0" ARM_OUT=exp_rr_icdr0 ARM_CK=ck_rr_icdr0 PART=val bash run_phone_sim.sh 0 1 2 3 4
# On val the tabletop rows are in-sample (that split chose the epoch and the threshold), so
# the Canon side of a choice is only a guard there. The chosen arm is then read on test once,
# with its own knobs plus FINAL=1 STAGES=score. Scoring an ablation arm on test without
# FINAL=1 is refused. The trainer prints and writes every arm's handheld test numbers too:
# do not read them while choosing.
#
# Knobs (environment): B, M (required); STAGES (default "fit check train score");
# CAM="Canon CR", HOLD="NIKON NF5050"; P=0.5 (share of Canon training photos simulated);
# STRENGTH=1.0 (0-1); CK=ck_phone_sim (profile); FIT_EXTRA (more fit flags: --colour-mode,
# --select); PART=test|val (score); FINAL (1: an ablation arm on test); OUT=exp_phone_sim;
# ARM_OUT=exp_rr_phonesim, ARM_CK=ck_rr_phonesim (the retrained sweep); BASE_CK=ck_rr_canon
# (the Canon-only sweep the arm is compared with); SIZE=384; DEVICE=auto; WORKERS=5 (the arm
# decodes larger views: raise it when the preflight says the loader sets the pace and no
# other job shares the CPUs); GPUS (CUDA_VISIBLE_DEVICES); EXTRA (more trainer flags,
# default "--ema-decay 0.999", the Canon sweep's).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
: "${B:?set B=<BRSET root>}"
: "${M:?set M=<mBRSET root>}"
STAGES=${STAGES:-fit check train score}
CAM=${CAM:-Canon CR}
HOLD=${HOLD:-NIKON NF5050}
P=${P:-0.5}
STRENGTH=${STRENGTH:-1.0}
CK=${CK:-ck_phone_sim}
FIT_EXTRA=${FIT_EXTRA:-}
PART=${PART:-test}
FINAL=${FINAL:-0}
OUT=${OUT:-exp_phone_sim}
ARM_OUT=${ARM_OUT:-exp_rr_phonesim}
ARM_CK=${ARM_CK:-ck_rr_phonesim}
BASE_CK=${BASE_CK:-ck_rr_canon}
SIZE=${SIZE:-384}
DEVICE=${DEVICE:-auto}
WORKERS=${WORKERS:-5}
EXTRA=${EXTRA:---ema-decay 0.999}
PY=.venv/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0)
for st in $STAGES; do
  case $st in fit|check|train|score) ;; *) echo "[fatal] unknown stage '$st' (fit check train score)"; exit 1;; esac
done
case $PART in test|val) ;; *) echo "[fatal] PART must be test or val, not '$PART'"; exit 1;; esac
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
for v in B M; do [ -d "${!v}" ] || { echo "[fatal] $v root not found: ${!v}"; exit 1; }; done
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$GPUS
PROFILE="$CK/profile.json"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
ST=$(printf %g "$STRENGTH")
# the pre-specified arm: p 0.5, strength 1, the default profile; anything else is an ablation
PRIMARY=0
if [ "$(printf %g "$P")" = 0.5 ] && [ "$ST" = 1 ] && [ "$CK" = ck_phone_sim ] && [ -z "$FIT_EXTRA" ]; then PRIMARY=1; fi
if has score && [ "$PART" = test ] && [ $PRIMARY = 0 ] && [ "$FINAL" != 1 ]; then
  echo "[fatal] an ablation arm (P $P, STRENGTH $STRENGTH, CK $CK${FIT_EXTRA:+, FIT_EXTRA $FIT_EXTRA}) is chosen on" \
       "the validation split: PART=val. FINAL=1 reads the chosen arm on test, once."; exit 1
fi
ARM_TAG=${ARM_CK%/}; ARM_TAG=${ARM_TAG##*/}
if [ "$PART" = test ] && [ $PRIMARY = 1 ]; then SCORE_OUT="$OUT/score"; else SCORE_OUT="$OUT/score_${ARM_TAG}_$PART"; fi
mkdir -p "$OUT" "$CK"

if has fit; then
  if [ -f "$PROFILE" ]; then
    $PY - "$PROFILE" "$SIZE" "$CAM" <<'EOF' || { echo "[fatal] $PROFILE is another format, size or camera: move it aside and rerun the fit stage"; exit 1; }
import json, sys
import phone_sim
b = json.load(open(sys.argv[1]))
s = b.get("source", {})
sys.exit(0 if b.get("format") == phone_sim.FORMAT and s.get("image_size") == int(sys.argv[2])
         and (s.get("source_camera") or "") == sys.argv[3] else 1)
EOF
    echo "[fit] $PROFILE exists (same format, size and camera), skipping"
  else
    # shellcheck disable=SC2086  # FIT_EXTRA: separate flags
    $PY phone_sim.py fit --device "$DEVICE" --target-root "$M" --source-root "$B" --source-camera "$CAM" \
      --image-size "$SIZE" --num-workers "$WORKERS" --out "$PROFILE" $FIT_EXTRA
  fi
fi
if has check || has train; then
  [ -f "$PROFILE" ] || { echo "[fatal] no profile at $PROFILE (run the fit stage)"; exit 1; }
fi

if has check; then
  for s in "${SEEDS[@]}"; do
    base="$BASE_CK/seed$s.pt"
    if [ ! -f "$base" ]; then echo "[check] no $base, skipping seed $s"; continue; fi
    d="$CK/check_seed$s"                   # its grid.jpg holds dataset photos: a git-ignored ck_* folder
    [ "$ST" = 1 ] || d="${d}_strength$ST"
    if [ -f "$d/check_summary.txt" ]; then echo "[check] $d exists, skipping"; continue; fi
    $PY style_transfer_xfer.py check --device "$DEVICE" --ckpt "$base" --source-root "$B" --target-root "$M" \
      --methods colour fda_sharp --phone-profile "$PROFILE" --phone-strength "$STRENGTH" \
      --num-workers "$WORKERS" --out "$d"
  done
fi

if has train; then
  # a seed already trained in ARM_OUT is skipped by run_retinareach.sh: it must be this arm's
  $PY - "$ARM_OUT" "$PROFILE" "$P" "$STRENGTH" "${SEEDS[@]}" <<'EOF' || exit 1
import hashlib, json, os, sys
arm_out, profile, p, strength, seeds = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5:]
sha = hashlib.sha256(open(profile, "rb").read()).hexdigest()
for s in seeds:
    path = os.path.join(arm_out, f"seed{s}", "results.json")
    if not os.path.isfile(path):
        continue
    r = json.load(open(path))
    a, prof = r.get("args", {}), r.get("phone_sim_profile") or {}
    got = (a.get("phone_sim_p"), a.get("phone_sim_strength"), prof.get("sha256"))
    if not a.get("phone_sim") or abs(got[0] - p) > 1e-9 or abs(got[1] - strength) > 1e-9 or got[2] != sha:
        sys.exit(f"[fatal] {path} was trained with phone_sim p {got[0]}, strength {got[1]}, profile "
                 f"{(got[2] or 'none')[:12]}, not this arm's (p {p}, strength {strength}, {sha[:12]}): "
                 f"give the arm a fresh ARM_OUT and ARM_CK")
EOF
  # the Canon sweep's own pipeline, one flag different
  B="$B" M="$M" SOURCE_CAMERA="$CAM" HOLDOUT_CAMERA="$HOLD" OUT="$ARM_OUT" CK="$ARM_CK" SIZE="$SIZE" \
    WORKERS="$WORKERS" EXTRA="$EXTRA --phone-sim $PROFILE --phone-sim-p $P --phone-sim-strength $STRENGTH" \
    bash run_retinareach.sh "${SEEDS[@]}"
fi

if has score; then
  # every seed trained on both sides, whatever seeds this call was given: the decision
  # reads a mean over seeds paired with the Canon sweep
  cks=()
  for f in "$BASE_CK"/seed*.pt; do
    [ -f "$f" ] || continue
    s=${f##*/seed}; s=${s%.pt}
    if [ -f "$ARM_CK/seed$s.pt" ]; then cks+=("canon_only=$f" "canon_phonesim=$ARM_CK/seed$s.pt")
    else echo "[score] seed $s: no $ARM_CK/seed$s.pt, left out of both arms"; fi
  done
  for f in "$ARM_CK"/seed*.pt; do
    [ -f "$f" ] || continue
    s=${f##*/seed}; s=${s%.pt}
    if [ ! -f "$BASE_CK/seed$s.pt" ]; then echo "[score] seed $s: no $BASE_CK/seed$s.pt, left out of both arms"; fi
  done
  [ ${#cks[@]} -gt 0 ] || { echo "[score] no seed has a checkpoint in both $BASE_CK and $ARM_CK"; exit 1; }
  $PY style_transfer_xfer.py score --device "$DEVICE" --source-root "$B" --target-root "$M" --ckpt "${cks[@]}" \
    --part "$PART" --num-workers "$WORKERS" --out "$SCORE_OUT"
fi
wrote=()
for f in "$CK/fit_report.txt" "$CK/fit_grid.jpg" "$CK"/check_seed*/check_summary.txt "$SCORE_OUT/score_summary.txt"; do
  if [ -f "$f" ]; then wrote+=("$f"); fi
done
echo "[done] ${wrote[*]:-no output files yet}"
