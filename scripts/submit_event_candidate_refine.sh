#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_event_candidate_refine.sh --mode train [--dry-run]
  scripts/submit_event_candidate_refine.sh --mode eval --run-manifest refine-logs/event_candidate_refine_train_jobs_YYYYmmdd_HHMMSS.tsv [--dry-run]

Submit short event-candidate local-refinement jobs through Slurm.

Options:
  --mode train|eval       Submit training resumes or eval renders.
  --run-manifest PATH     Required for --mode eval; produced by --mode train.
  --dry-run               Print sbatch commands without submitting.
  -h, --help              Show this help.

Environment overrides:
  SCENES="cut_roasted_beef flame_steak sear_steak"
  SOURCE_MANIFEST=refine-logs/hide_reveal_real_windows.json
  CONFIG=configs/n3v/event_candidate_local_refine_6200.yaml
  RUN_LABEL=event_candidate_local_refine_6200
  START_CKPT_ITER=6000
  REFINE_ITER=6200
  TIME=00:45:00
  EVAL_TIME=00:45:00
  CPUS_PER_TASK=8
  WANDB_MODE=offline
EOF
}

MODE="train"
RUN_MANIFEST=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --run-manifest)
      RUN_MANIFEST="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
  echo "ERROR: --mode must be train or eval." >&2
  exit 2
fi
if [[ "$MODE" == "eval" && -z "$RUN_MANIFEST" ]]; then
  echo "ERROR: --run-manifest is required for --mode eval." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${ADAGS_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$ADAGS_PROJECT_ROOT"
elif [[ -n "${WORK:-}" ]]; then
  PROJECT_ROOT="$WORK/proj_adags"
else
  PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
fi
if [[ -z "${WORK:-}" ]]; then
  export WORK="$(cd "$PROJECT_ROOT/.." && pwd)"
fi

ENV_SCRIPT="${ADAGS_ENV_SCRIPT:-$PROJECT_ROOT/exp_index/leonardo_env.sh}"
if [[ -f "$ENV_SCRIPT" ]]; then
  # Needed on login nodes for the lightweight manifest parsing below.
  # shellcheck source=/dev/null
  source "$ENV_SCRIPT" >/dev/null 2>&1
fi

SCENES="${SCENES:-cut_roasted_beef flame_steak sear_steak}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-refine-logs/hide_reveal_real_windows.json}"
CONFIG="${CONFIG:-configs/n3v/event_candidate_local_refine_6200.yaml}"
RUN_LABEL="${RUN_LABEL:-event_candidate_local_refine_6200}"
START_CKPT_ITER="${START_CKPT_ITER:-6000}"
REFINE_ITER="${REFINE_ITER:-6200}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-n3v}"
WANDB_MODE="${WANDB_MODE:-offline}"
PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"
TIME="${TIME:-00:45:00}"
EVAL_TIME="${EVAL_TIME:-00:45:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

timestamp="$(date +%Y%m%d_%H%M%S)"
submit_manifest="$REPO_ROOT/refine-logs/event_candidate_refine_${MODE}_jobs_${timestamp}.tsv"
log_dir="$REPO_ROOT/logs"
mkdir -p "$log_dir"

manifest_scene_source() {
  local scene="$1"
  python - "$SOURCE_MANIFEST" "$scene" "$START_CKPT_ITER" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
scene = sys.argv[2]
iteration = int(sys.argv[3])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
source = payload["scene_sources"][scene]
eval_dir = Path(source["route0_eval_dir"])
run_dir = eval_dir.parent.parent
ckpt = run_dir / f"chkpnt{iteration}.pth"
run_root = run_dir.parent.parent
print(f"{run_dir}\t{ckpt}\t{eval_dir}\t{run_root}")
PY
}

submit_train() {
  if [[ "$DRY_RUN" == "0" ]]; then
    printf "job_id\tmode\tscene\tsource_run_dir\tsource_ckpt\trun_dir\tconfig\tlog_stdout\tlog_stderr\n" > "$submit_manifest"
  fi

  for scene in $SCENES; do
    IFS=$'\t' read -r source_run_dir source_ckpt route0_eval_dir run_root < <(manifest_scene_source "$scene")
    if [[ ! -f "$source_ckpt" ]]; then
      echo "ERROR: missing checkpoint for $scene: $source_ckpt" >&2
      exit 3
    fi
    run_id="${timestamp}_${scene}_${RUN_LABEL}"
    run_dir="$run_root/$RUN_LABEL/$run_id"
    stdout="$log_dir/event_candidate_refine_train_${scene}_%j.out"
    stderr="$log_dir/event_candidate_refine_train_${scene}_%j.err"
    wandb_extra_tags="experiment:event_candidate_refine method:M1_candidate_local_refine source_ckpt:${START_CKPT_ITER} refine_iter:${REFINE_ITER}"
    cmd=(
      sbatch --parsable
      -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
      -t "$TIME"
      -o "$stdout"
      -e "$stderr"
      --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",ADAGS_MAX_ITERATIONS="$REFINE_ITER",SCENE="$scene",RUN_TAG="$RUN_LABEL",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CKPT_PATH="$source_ckpt",CONFIG="$CONFIG",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="event_candidate_refine",METHOD_FAMILY="M1_candidate_local_refine",BUDGET_LABEL="$RUN_LABEL"
      "$REPO_ROOT/scripts/run_leonardo.sh" train
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN train %s: ' "$scene"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      job_id="$("${cmd[@]}")"
      printf "%s\ttrain\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job_id" "$scene" "$source_run_dir" "$source_ckpt" "$run_dir" "$CONFIG" \
        "${stdout//%j/$job_id}" "${stderr//%j/$job_id}" >> "$submit_manifest"
      echo "Submitted event-candidate refine train $scene: $job_id"
    fi
  done
}

submit_eval() {
  if [[ "$DRY_RUN" == "0" ]]; then
    printf "job_id\tmode\tscene\trun_dir\tckpt\tconfig\tlog_stdout\tlog_stderr\n" > "$submit_manifest"
  fi

  tail -n +2 "$RUN_MANIFEST" | while IFS=$'\t' read -r _train_job _mode scene _source_run _source_ckpt run_dir config _stdout _stderr; do
    ckpt="$run_dir/chkpnt${REFINE_ITER}.pth"
    if [[ ! -f "$ckpt" ]]; then
      echo "ERROR: missing refined checkpoint for $scene: $ckpt" >&2
      exit 4
    fi
    run_id="$(basename "$run_dir")_eval${REFINE_ITER}"
    stdout="$log_dir/event_candidate_refine_eval_${scene}_%j.out"
    stderr="$log_dir/event_candidate_refine_eval_${scene}_%j.err"
    wandb_extra_tags="experiment:event_candidate_refine_eval method:M1_candidate_local_refine eval_ckpt:${REFINE_ITER}"
    cmd=(
      sbatch --parsable
      -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
      -t "$EVAL_TIME"
      -o "$stdout"
      -e "$stderr"
      --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="${RUN_LABEL}_eval",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$REFINE_ITER",CKPT_PATH="$ckpt",CONFIG="$config",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="event_candidate_refine_eval",METHOD_FAMILY="M1_candidate_local_refine",BUDGET_LABEL="$RUN_LABEL"
      "$REPO_ROOT/scripts/run_leonardo.sh" eval
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN eval %s: ' "$scene"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      job_id="$("${cmd[@]}")"
      printf "%s\teval\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job_id" "$scene" "$run_dir" "$ckpt" "$config" \
        "${stdout//%j/$job_id}" "${stderr//%j/$job_id}" >> "$submit_manifest"
      echo "Submitted event-candidate refine eval $scene: $job_id"
    fi
  done
}

if [[ "$MODE" == "train" ]]; then
  submit_train
else
  submit_eval
fi

if [[ "$DRY_RUN" == "0" ]]; then
  echo "Manifest: $submit_manifest"
fi
