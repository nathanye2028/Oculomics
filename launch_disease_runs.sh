#!/usr/bin/env bash
# launch_disease_runs.sh — start the disease-branch sweeps on the GPU box, each in
# its OWN git worktree and its OWN tmux session, logging to a timestamped file.
#
#   ssh arctrdgndev101
#   cd ~/Oculomics && git pull && bash launch_disease_runs.sh systemic glaucoma   # any subset
#   bash launch_disease_runs.sh systemic        # mBRSET systemic sweep            (GPU 0)
#   bash launch_disease_runs.sh ophthalmic      # BRSET ophthalmic multi-label sweep (GPU 1)
#   bash launch_disease_runs.sh glaucoma        # AIROGS-light v2 -> ODIR-5K glaucoma transfer (GPU 1);
#                                               # data is fetched on the box with kagglehub (anonymous)
#   bash launch_disease_runs.sh all             # all three (ophthalmic and glaucoma share GPU 1 unless
#                                               # GPU_OPHTHALMIC / GPU_GLAUCOMA say otherwise)
#   bash launch_disease_runs.sh status          # sessions, GPU load, finished runs, log tails
#   bash launch_disease_runs.sh stop systemic   # kill that session (finished runs are kept)
#
# Data roots: B= / M= may point at a PARENT of the real directory (PhysioNet drops a
# version dir, e.g. mBRSET/1.0/); the pre-flight resolves them to the directory that
# holds the label CSV and fails with the CSVs it did find otherwise.
#
# Why worktrees: each run lives on its own branch (disease/systemic,
# disease/ophthalmic-multilabel) and they run at the same time, so each gets a
# checkout of its branch next to this repo (~/Oculomics-systemic, ...) with the
# shared .venv symlinked in. Checkpoints (ck_*/) and results (exp_*/) land inside
# the worktree and are gitignored. Both sweeps are idempotent: re-running this
# script after a crash or reboot resumes where they stopped.
#
# Labels: tmux session = oculomics-<run>-<dataset>, window = the worktree name,
# log = exp_<run>/sweep-<YYYYmmdd-HHMM>.log, run names inside the sweeps are
# <condition>_seed<n> (ctrl / drinit; multi / single_<label>).
set -euo pipefail

REPO=${REPO:-$HOME/Oculomics}
DATA=${DATA:-/data/users4/nshaik3/Datasets}
B=${B:-$DATA/BRSET}
M=${M:-$DATA/mBRSET}
DR_CK=${DR_CK:-$REPO/ck_kd_v4_384/kd_seed1.pt}     # the deployed DR student (REPORT.md §6)
WORKERS=${WORKERS:-5}
GPU_SYSTEMIC=${GPU_SYSTEMIC:-0}
GPU_OPHTHALMIC=${GPU_OPHTHALMIC:-1}
GPU_GLAUCOMA=${GPU_GLAUCOMA:-1}
# glaucoma: AIROGS-light v2 (9,540 imgs, kagglehub) -> ODIR-5K (6,392 eyes, kagglehub) with the ctrl /
# teacher / kd design: roughly 1-1.5 h per student run and 2-3 h per teacher @384px -> ~15 h for 3 seeds.
GLC_SEEDS=${GLC_SEEDS:-0 1 2}
GLC_SRC=${GLC_SRC:-kaggle:deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2}
GLC_EXT=${GLC_EXT:-kaggle:andrewmvd/ocular-disease-recognition-odir5k}
REFUGE_ROOT=${REFUGE_ROOT:-}                       # optional extra external sets, scored after training
PAPILA_ROOT=${PAPILA_ROOT:-}
KAGGLEHUB_CACHE=${KAGGLEHUB_CACHE:-$HOME/.cache/kagglehub}   # where kagglehub puts the public sets (~2.2 GB)
# systemic: 4 tasks x 2 arms x seeds on mBRSET (~5k images): roughly 25-30 min per run on an A100 @384px
SYS_SEEDS=${SYS_SEEDS:-0 1 2 3 4}
SYS_TASKS=${SYS_TASKS:-hypertension nephropathy neuropathy myocardial_infarction}
# ophthalmic: (1 multi + one model per label) x seeds on BRSET (~11k images): roughly 1.5-2 h per run @384px,
# so the default is the five labels with clinical weight and 3 seeds (18 runs). Widen with OPH_LABELS / OPH_SEEDS.
OPH_SEEDS=${OPH_SEEDS:-0 1 2}
OPH_LABELS=${OPH_LABELS:-amd drusens increased_cup_disc hypertensive_retinopathy vascular_occlusion}
DRY_RUN=${DRY_RUN:-0}                              # 1 = print what would be launched, launch nothing
STAMP=$(date +%Y%m%d-%H%M)

WT_SYSTEMIC=${WT_SYSTEMIC:-$HOME/Oculomics-systemic}
WT_OPHTHALMIC=${WT_OPHTHALMIC:-$HOME/Oculomics-ophthalmic}
WT_GLAUCOMA=${WT_GLAUCOMA:-$HOME/Oculomics-glaucoma}

say() { printf '%s\n' "$*"; }

worktree() {  # worktree <branch> <dir> -- create/refresh a checkout of <branch> at <dir>
  local branch=$1 dir=$2
  if [ "$DRY_RUN" = 1 ]; then say "[dry] worktree $branch -> $dir"; return; fi
  git -C "$REPO" fetch -q origin
  if [ ! -d "$dir/.git" ] && [ ! -f "$dir/.git" ]; then
    git -C "$REPO" worktree add -q "$dir" "$branch" 2>/dev/null \
      || git -C "$REPO" worktree add -q --track -b "$branch" "$dir" "origin/$branch"
  fi
  git -C "$dir" checkout -q "$branch"
  git -C "$dir" pull -q --ff-only origin "$branch"
  [ -e "$dir/.venv" ] || ln -s "$REPO/.venv" "$dir/.venv"
  say "[worktree] $dir @ $(git -C "$dir" log --oneline -1)"
}

launch() {  # launch <session> <dir> <gpu> <log> <command>
  local session=$1 dir=$2 gpu=$3 log=$4 cmd=$5
  if [ "$DRY_RUN" = 1 ]; then
    say "[dry] tmux new-session -s $session -c $dir  CUDA_VISIBLE_DEVICES=$gpu"
    say "[dry]   $cmd"
    say "[dry]   log -> $log"; return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    say "[skip] tmux session '$session' is already running  (tmux attach -t $session)"; return
  fi
  mkdir -p "$(dirname "$log")"
  local runner="$(dirname "$log")/launch-$STAMP.sh"
  cat > "$runner" <<RUN
#!/usr/bin/env bash
set -o pipefail
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 KAGGLEHUB_CACHE="$KAGGLEHUB_CACHE"
cd "$dir"
echo "[launch] session=$session host=\$(hostname) gpu=\$CUDA_VISIBLE_DEVICES branch=\$(git branch --show-current) commit=\$(git rev-parse --short HEAD) start=\$(date)" | tee -a "$log"
$cmd 2>&1 | tee -a "$log"
echo "[done] session=$session exit=\${PIPESTATUS[0]} end=\$(date)" | tee -a "$log"
RUN
  tmux new-session -d -s "$session" -n "$(basename "$dir")" -c "$dir" "bash '$runner'; exec bash"
  say "[started] $session   GPU $gpu   log: $log"
  say "          watch: tmux attach -t $session   (Ctrl-b d to detach)   or   tail -f $log"
}

run_systemic() {
  worktree disease/systemic "$WT_SYSTEMIC"
  local dr=""
  if [ -f "$DR_CK" ]; then dr="DR_CK='$DR_CK'"; else say "[warn] DR student $DR_CK not found: systemic sweep runs the ctrl arm only (no drinit)"; fi
  launch oculomics-systemic-mbrset "$WT_SYSTEMIC" "$GPU_SYSTEMIC" "$WT_SYSTEMIC/exp_systemic/sweep-$STAMP.log" \
    "M='$M' $dr TASKS='$SYS_TASKS' WORKERS=$WORKERS bash run_systemic.sh $SYS_SEEDS"
}

run_ophthalmic() {
  worktree disease/ophthalmic-multilabel "$WT_OPHTHALMIC"
  launch oculomics-ophthalmic-brset "$WT_OPHTHALMIC" "$GPU_OPHTHALMIC" "$WT_OPHTHALMIC/exp_ophthalmic/sweep-$STAMP.log" \
    "B='$B' LABELS='$OPH_LABELS' WORKERS=$WORKERS bash run_ophthalmic.sh $OPH_SEEDS"
}

run_glaucoma() {
  worktree disease/glaucoma-amd "$WT_GLAUCOMA"
  local more=""
  [ -n "$REFUGE_ROOT" ] && { [ -d "$REFUGE_ROOT" ] && more="$more refuge=$REFUGE_ROOT" || say "[warn] REFUGE_ROOT=$REFUGE_ROOT not found; skipped"; }
  [ -n "$PAPILA_ROOT" ] && { [ -d "$PAPILA_ROOT" ] && more="$more papila=$PAPILA_ROOT" || say "[warn] PAPILA_ROOT=$PAPILA_ROOT not found; skipped"; }
  launch oculomics-glaucoma-airogs2odir "$WT_GLAUCOMA" "$GPU_GLAUCOMA" "$WT_GLAUCOMA/exp_glaucoma/sweep-$STAMP.log" \
    "SRC='$GLC_SRC' SRC_DATASET=airogs TASK=glaucoma EXT='$GLC_EXT' EXT_DATASET=odir MORE='${more# }' WORKERS=$WORKERS bash run_public_xfer.sh $GLC_SEEDS"
}

resolve() {  # resolve <VAR> <dataset> -- replace $VAR with the directory that holds the label CSV, or fail loudly
  local var=$1 ds=$2 val out
  val=${!var}
  if out=$("$REPO/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$REPO')
from brset_dataset import resolve_root
try:
    print(resolve_root(sys.argv[1], sys.argv[2]))
except FileNotFoundError as e:
    sys.exit(str(e))" "$val" "$ds" 2>&1); then
    [ "$out" != "$val" ] && say "[preflight] $var: $val -> $out"
    printf -v "$var" '%s' "$out"
  else
    say "[fatal] $var=$val: $out"
    say "        contents: $(ls "$val" 2>/dev/null | head -8 | tr '\n' ' ')"
    exit 1
  fi
}

status() {
  say "=== tmux sessions ==="; tmux ls 2>/dev/null || say "(none)"
  say; say "=== GPUs ==="; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || say "(nvidia-smi unavailable)"
  for pair in "systemic:$WT_SYSTEMIC/exp_systemic" "ophthalmic:$WT_OPHTHALMIC/exp_ophthalmic" "glaucoma:$WT_GLAUCOMA/exp_glaucoma"; do
    local name=${pair%%:*} dir=${pair#*:}
    say; say "=== $name  ($dir) ==="
    [ -d "$dir" ] || { say "(not started)"; continue; }
    say "finished runs: $(find "$dir" -name '*_seed*.json' | wc -l | tr -d ' ')   summaries: $(ls "$dir"/*/summary.md "$dir"/summary*.md 2>/dev/null | wc -l | tr -d ' ')"
    local log; log=$(ls -t "$dir"/sweep-*.log 2>/dev/null | head -1)
    [ -n "$log" ] && { say "last log: $log"; grep -E "^=== |^\[(done|skip|fatal|warn)\]|AUROC=" "$log" | tail -6; }
  done
}

stop() {  # stop <systemic|ophthalmic>
  case "${1:-}" in
    systemic)   tmux kill-session -t oculomics-systemic-mbrset && say "[stopped] oculomics-systemic-mbrset";;
    ophthalmic) tmux kill-session -t oculomics-ophthalmic-brset && say "[stopped] oculomics-ophthalmic-brset";;
    glaucoma)   tmux kill-session -t oculomics-glaucoma-airogs2odir && say "[stopped] oculomics-glaucoma-airogs2odir";;
    *) say "usage: bash launch_disease_runs.sh stop systemic|ophthalmic|glaucoma"; exit 2;;
  esac
}

preflight() {  # preflight <targets...>
  [ -d "$REPO/.git" ] || { say "[fatal] $REPO is not the repo"; exit 1; }
  [ -x "$REPO/.venv/bin/python" ] || { say "[fatal] $REPO/.venv missing: bash setup_remote.sh first"; exit 1; }
  command -v tmux >/dev/null || { say "[fatal] tmux not installed"; exit 1; }
  for t in "$@"; do
    case "$t" in
      systemic)   resolve M mbrset;;
      ophthalmic) resolve B brset;;
      glaucoma)   "$REPO/.venv/bin/python" -c "import kagglehub" 2>/dev/null || { say "[fatal] kagglehub missing in $REPO/.venv (pip install -r requirements.txt)"; exit 1; };;
    esac
  done
  say "[preflight] repo=$REPO  DR_CK=$DR_CK $([ -f "$DR_CK" ] && echo '(found)' || echo '(MISSING -> systemic ctrl arm only)')"
  say "[preflight] GPUs: systemic->$GPU_SYSTEMIC  ophthalmic->$GPU_OPHTHALMIC  glaucoma->$GPU_GLAUCOMA"
  case " $* " in *" ophthalmic "*" glaucoma "*|*" glaucoma "*" ophthalmic "*)
    [ "$GPU_OPHTHALMIC" = "$GPU_GLAUCOMA" ] && say "[warn] ophthalmic and glaucoma both on GPU $GPU_GLAUCOMA; set GPU_GLAUCOMA= to separate them";; esac
}

case "${1:-all}" in
  status) status; exit 0;;
  stop)   stop "${2:-}"; exit 0;;
  -h|--help) sed -n '2,24p' "$0"; exit 0;;
esac
TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=(all)
[ "${TARGETS[*]}" = "all" ] && TARGETS=(systemic ophthalmic glaucoma)
for t in "${TARGETS[@]}"; do
  case "$t" in systemic|ophthalmic|glaucoma) ;; *) say "unknown target '$t' (systemic | ophthalmic | glaucoma | all | status | stop <run>)"; exit 2;; esac
done
[ "$DRY_RUN" = 1 ] || preflight "${TARGETS[@]}"
for t in "${TARGETS[@]}"; do "run_$t"; done
say; say "started: ${TARGETS[*]}.  'bash launch_disease_runs.sh status' to check;  'tmux ls' lists sessions."
