#!/usr/bin/env bash
# launch_disease_runs.sh — start the disease-branch sweeps on the GPU box, each in
# its OWN git worktree and its OWN tmux session, logging to a timestamped file.
#
#   ssh arctrdgndev101
#   cd ~/Oculomics && git pull && bash launch_disease_runs.sh            # both runs
#   bash launch_disease_runs.sh systemic        # only the mBRSET systemic sweep (GPU 0)
#   bash launch_disease_runs.sh ophthalmic      # only the BRSET ophthalmic sweep (GPU 1)
#   bash launch_disease_runs.sh status          # sessions, GPU load, finished runs, log tails
#   bash launch_disease_runs.sh stop systemic   # kill that session (finished runs are kept)
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
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
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

status() {
  say "=== tmux sessions ==="; tmux ls 2>/dev/null || say "(none)"
  say; say "=== GPUs ==="; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || say "(nvidia-smi unavailable)"
  for pair in "systemic:$WT_SYSTEMIC/exp_systemic" "ophthalmic:$WT_OPHTHALMIC/exp_ophthalmic"; do
    local name=${pair%%:*} dir=${pair#*:}
    say; say "=== $name  ($dir) ==="
    [ -d "$dir" ] || { say "(not started)"; continue; }
    say "finished runs: $(find "$dir" -name '*_seed*.json' | wc -l | tr -d ' ')   summaries: $(ls "$dir"/*/summary.md "$dir"/summary.md 2>/dev/null | wc -l | tr -d ' ')"
    local log; log=$(ls -t "$dir"/sweep-*.log 2>/dev/null | head -1)
    [ -n "$log" ] && { say "last log: $log"; grep -E "^=== |^\[(done|skip|fatal|warn)\]|AUROC=" "$log" | tail -6; }
  done
}

stop() {  # stop <systemic|ophthalmic>
  case "${1:-}" in
    systemic)   tmux kill-session -t oculomics-systemic-mbrset && say "[stopped] oculomics-systemic-mbrset";;
    ophthalmic) tmux kill-session -t oculomics-ophthalmic-brset && say "[stopped] oculomics-ophthalmic-brset";;
    *) say "usage: bash launch_disease_runs.sh stop systemic|ophthalmic"; exit 2;;
  esac
}

preflight() {
  [ -d "$REPO/.git" ] || { say "[fatal] $REPO is not the repo"; exit 1; }
  [ -x "$REPO/.venv/bin/python" ] || { say "[fatal] $REPO/.venv missing: bash setup_remote.sh first"; exit 1; }
  command -v tmux >/dev/null || { say "[fatal] tmux not installed"; exit 1; }
  [ -d "$B" ] || { say "[fatal] BRSET root not found: $B  (set B=)"; exit 1; }
  [ -d "$M" ] || { say "[fatal] mBRSET root not found: $M  (set M=)"; exit 1; }
  say "[preflight] repo=$REPO  BRSET=$B  mBRSET=$M  DR_CK=$DR_CK $([ -f "$DR_CK" ] && echo '(found)' || echo '(MISSING -> ctrl only)')"
  say "[preflight] GPUs: systemic->$GPU_SYSTEMIC  ophthalmic->$GPU_OPHTHALMIC   seeds: systemic [$SYS_SEEDS]  ophthalmic [$OPH_SEEDS]"
}

case "${1:-all}" in
  status) status;;
  stop)   stop "${2:-}";;
  systemic)   [ "$DRY_RUN" = 1 ] || preflight; run_systemic;;
  ophthalmic) [ "$DRY_RUN" = 1 ] || preflight; run_ophthalmic;;
  all)        [ "$DRY_RUN" = 1 ] || preflight; run_systemic; run_ophthalmic
              say; say "both started. 'bash launch_disease_runs.sh status' to check; 'tmux ls' lists sessions.";;
  -h|--help)  sed -n '2,20p' "$0";;
  *) say "unknown command '$1' (systemic | ophthalmic | all | status | stop <run>)"; exit 2;;
esac
