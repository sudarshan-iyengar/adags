#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_visibility_event_pilot.sh --variant smooth|event --mode train [--dry-run]
  scripts/submit_visibility_event_pilot.sh --variant smooth|event --mode eval --run-manifest refine-logs/visibility_event_*_train_jobs_YYYYmmdd_HHMMSS.tsv [--dry-run]

Submit full matched visibility-event pilot jobs through Slurm.

Environment overrides:
  SCENES="cut_roasted_beef flame_steak sear_steak"
  CONFIG_SMOOTH=configs/n3v/visibility_event_smooth_control_6000.yaml
  CONFIG_EVENT=configs/n3v/visibility_event_train_6000.yaml
  ITER=6000
  TIME=02:30:00
  EVAL_TIME=00:45:00
EOF
}

MODE="train"
VARIANT=""
RUN_MANIFEST=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
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
if [[ "$VARIANT" != "smooth" && "$VARIANT" != "event" ]]; then
  echo "ERROR: --variant must be smooth or event." >&2
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
  source "$ENV_SCRIPT" >/dev/null 2>&1
fi

SCENES="${SCENES:-cut_roasted_beef flame_steak sear_steak}"
CONFIG_SMOOTH="${CONFIG_SMOOTH:-configs/n3v/visibility_event_smooth_control_6000.yaml}"
CONFIG_EVENT="${CONFIG_EVENT:-configs/n3v/visibility_event_train_6000.yaml}"
if [[ "$VARIANT" == "smooth" ]]; then
  CONFIG="${CONFIG:-$CONFIG_SMOOTH}"
  RUN_LABEL="${RUN_LABEL:-visibility_event_smooth_control_6000}"
  JOB_PREFIX="${JOB_PREFIX:-visibility_event_smooth}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-visibility_event_smooth_control}"
  METHOD_FAMILY="${METHOD_FAMILY:-M3_visibility_event_smooth_control}"
  METHOD_TAG="${METHOD_TAG:-method:M3_visibility_event_smooth_control}"
else
  CONFIG="${CONFIG:-$CONFIG_EVENT}"
  RUN_LABEL="${RUN_LABEL:-visibility_event_train_6000}"
  JOB_PREFIX="${JOB_PREFIX:-visibility_event_train}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-visibility_event_train}"
  METHOD_FAMILY="${METHOD_FAMILY:-M3_visibility_event_gate}"
  METHOD_TAG="${METHOD_TAG:-method:M3_visibility_event_gate}"
fi

ITER="${ITER:-6000}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-n3v}"
WANDB_MODE="${WANDB_MODE:-offline}"
PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d36_068}"
QOS="${QOS:-boost_qos_lprod}"
TIME="${TIME:-02:30:00}"
EVAL_TIME="${EVAL_TIME:-00:45:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

timestamp="$(date +%Y%m%d_%H%M%S)"
submit_manifest="$REPO_ROOT/refine-logs/${JOB_PREFIX}_${MODE}_jobs_${timestamp}.tsv"
log_dir="$REPO_ROOT/logs"
mkdir -p "$log_dir"

submit_train() {
  if [[ "$DRY_RUN" == "0" ]]; then
    printf "job_id\tmode\tvariant\tscene\trun_dir\tconfig\tlog_stdout\tlog_stderr\n" > "$submit_manifest"
  fi
  for scene in $SCENES; do
    run_id="${timestamp}_${scene}_${RUN_LABEL}"
    run_dir="$PROJECT_ROOT/runs/$RUN_LABEL/$run_id"
    stdout="$log_dir/${JOB_PREFIX}_train_${scene}_%j.out"
    stderr="$log_dir/${JOB_PREFIX}_train_${scene}_%j.err"
    wandb_extra_tags="experiment:${EXPERIMENT_NAME} $METHOD_TAG iter:${ITER}"
    cmd=(
      sbatch --parsable
      -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
      -t "$TIME"
      -o "$stdout"
      -e "$stderr"
      --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",ADAGS_MAX_ITERATIONS="$ITER",SCENE="$scene",RUN_TAG="$RUN_LABEL",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CONFIG="$CONFIG",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="$EXPERIMENT_NAME",METHOD_FAMILY="$METHOD_FAMILY",BUDGET_LABEL="$RUN_LABEL"
      "$REPO_ROOT/scripts/run_leonardo.sh" train
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN train %s %s: ' "$VARIANT" "$scene"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      job_id="$("${cmd[@]}")"
      printf "%s\ttrain\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job_id" "$VARIANT" "$scene" "$run_dir" "$CONFIG" \
        "${stdout//%j/$job_id}" "${stderr//%j/$job_id}" >> "$submit_manifest"
      echo "Submitted $JOB_PREFIX train $scene: $job_id"
    fi
  done
}

submit_eval() {
  if [[ "$DRY_RUN" == "0" ]]; then
    printf "job_id\tmode\tvariant\tscene\trun_dir\tckpt\tconfig\tlog_stdout\tlog_stderr\n" > "$submit_manifest"
  fi
  tail -n +2 "$RUN_MANIFEST" | while IFS=$'\t' read -r _train_job _mode _variant scene run_dir config _stdout _stderr; do
    ckpt="$run_dir/chkpnt${ITER}.pth"
    if [[ ! -f "$ckpt" ]]; then
      echo "ERROR: missing checkpoint for $scene: $ckpt" >&2
      exit 4
    fi
    run_id="$(basename "$run_dir")_eval${ITER}"
    stdout="$log_dir/${JOB_PREFIX}_eval_${scene}_%j.out"
    stderr="$log_dir/${JOB_PREFIX}_eval_${scene}_%j.err"
    wandb_extra_tags="experiment:${EXPERIMENT_NAME}_eval $METHOD_TAG eval_iter:${ITER}"
    cmd=(
      sbatch --parsable
      -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
      -t "$EVAL_TIME"
      -o "$stdout"
      -e "$stderr"
      --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="${RUN_LABEL}_eval",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$ITER",CKPT_PATH="$ckpt",CONFIG="$config",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="${EXPERIMENT_NAME}_eval",METHOD_FAMILY="$METHOD_FAMILY",BUDGET_LABEL="$RUN_LABEL"
      "$REPO_ROOT/scripts/run_leonardo.sh" eval
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN eval %s %s: ' "$VARIANT" "$scene"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      job_id="$("${cmd[@]}")"
      printf "%s\teval\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$job_id" "$VARIANT" "$scene" "$run_dir" "$ckpt" "$config" \
        "${stdout//%j/$job_id}" "${stderr//%j/$job_id}" >> "$submit_manifest"
      echo "Submitted $JOB_PREFIX eval $scene: $job_id"
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
