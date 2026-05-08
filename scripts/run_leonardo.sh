#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   sbatch ... scripts/run_leonardo.sh train
#   sbatch ... --export=ALL,RUN_ID=20260226_154023_cut_roasted_beef_baseline scripts/run_leonardo.sh eval
#   sbatch ... --export=ALL,RUN_DIR=$WORK/proj_adags/runs/20260226_154023_cut_roasted_beef_baseline scripts/run_leonardo.sh eval
#   sbatch ... --export=ALL,SCENE=cut_roasted_beef,RUN_TAG=baseline scripts/run_leonardo.sh train

MODE="${1:-train}"                           # train | eval

# Config / dataset
CONFIG="${CONFIG:-configs/n3v/default.yaml}"
DATASET_ROOT="${DATASET_ROOT:-$WORK/proj_adags/data/n3v}"
SCENE="${SCENE:-cut_roasted_beef}"           # override per job via --export
RUN_TAG="${RUN_TAG:-baseline}"               # free text
RUN_LABEL="${RUN_LABEL:-}"                   # optional subdirectory under runs/

# W&B
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-n3v}"
WANDB_MODE="${WANDB_MODE:-offline}"         # offline is recommended on compute nodes

# For eval checkpoint selection
CKPT_ITER="${CKPT_ITER:-6000}"
CKPT_PATH="${CKPT_PATH:-}"                   # optional explicit path to .pth

# Reuse existing run:
# - Prefer RUN_DIR if set
# - Else if RUN_ID set, use $WORK/proj_adags/runs/$RUN_ID
RUN_DIR="${RUN_DIR:-}"
RUN_ID="${RUN_ID:-}"

# ---- derived paths ----
DATASET_PATH="${DATASET_ROOT}/${SCENE}"

# If no run was provided:
if [[ -z "$RUN_DIR" ]]; then
  if [[ -n "$RUN_ID" ]]; then
    if [[ -n "$RUN_LABEL" ]]; then
      RUN_DIR="$WORK/proj_adags/runs/$RUN_LABEL/$RUN_ID"
    else
      RUN_DIR="$WORK/proj_adags/runs/$RUN_ID"
    fi
  else
    # only create a new run for training; for eval, require an existing run
    if [[ "$MODE" == "train" ]]; then
      TS="$(date +%Y%m%d_%H%M%S)"
      RUN_ID="${TS}_${SCENE}_${RUN_TAG}"
      if [[ -n "$RUN_LABEL" ]]; then
        RUN_DIR="$WORK/proj_adags/runs/$RUN_LABEL/${RUN_ID}"
      else
        RUN_DIR="$WORK/proj_adags/runs/${RUN_ID}"
      fi
    else
      echo "ERROR: For eval you must set RUN_DIR or RUN_ID to an existing run." >&2
      echo "Example: --export=ALL,RUN_ID=20260226_154023_cut_roasted_beef_baseline" >&2
      exit 2
    fi
  fi
fi

META_DIR="$RUN_DIR/meta"
mkdir -p "$META_DIR"

# ---- Environment ----
source "$WORK/proj_adags/exp_index/leonardo_env.sh"
cd "$WORK/proj_adags/repo/adags"

# ---- Minimal logging (append per run, don’t overwrite) ----
{
  echo "timestamp: $(date -Iseconds)"
  echo "mode: ${MODE}"
  echo "scene: ${SCENE}"
  echo "run_tag: ${RUN_TAG}"
  echo "run_label: ${RUN_LABEL:-none}"
  echo "run_dir: ${RUN_DIR}"
  echo "host: $(hostname)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "python: $(which python)"
  python -V
  echo "config: ${CONFIG}"
  echo "dataset_path: ${DATASET_PATH}"
  echo "ckpt_iter: ${CKPT_ITER}"
  [[ -n "$CKPT_PATH" ]] && echo "ckpt_path: ${CKPT_PATH}"
  echo "wandb_project: ${WANDB_PROJECT}"
  echo "wandb_entity: ${WANDB_ENTITY:-none}"
  echo "wandb_group: ${WANDB_GROUP}"
  echo "wandb_mode: ${WANDB_MODE}"

  echo "---"
} | tee -a "$META_DIR/run_info.txt"

# ---- Command ----
if [[ "$MODE" == "train" ]]; then
    CMD=(
      python main.py
      --config "$CONFIG"
      --model_path "$RUN_DIR"
      --source_path "$DATASET_PATH"
      --use_wandb
      --wandb_project "$WANDB_PROJECT"
      --wandb_mode "$WANDB_MODE"
      --wandb_run_name "$RUN_ID"
      --wandb_group "$WANDB_GROUP"
      --wandb_resume "$RUN_ID"
      --wandb_tags "$SCENE" "$RUN_TAG" train
    )

    if [[ -n "$WANDB_ENTITY" ]]; then
      CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi
  if [[ -n "$CKPT_PATH" ]]; then
        CMD+=(--start_checkpoint "$CKPT_PATH")
  fi
  if [[ -n "${TEACHER_CKPT:-}" ]]; then
        CMD+=("--teacher_ckpt" "$TEACHER_CKPT")
  fi
elif [[ "$MODE" == "eval" ]]; then
  if [[ -z "$CKPT_PATH" ]]; then
    CKPT_PATH="${RUN_DIR}/chkpnt${CKPT_ITER}.pth"
  fi
  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
    exit 3
  fi

  CMD=(
    python main.py
    --config "$CONFIG"
    --model_path "$RUN_DIR"
    --source_path "$DATASET_PATH"
    --start_checkpoint "$CKPT_PATH"
    --val
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_mode "$WANDB_MODE"
    --wandb_run_name "$RUN_ID"
    --wandb_group "$WANDB_GROUP"
    --wandb_resume "$RUN_ID"
    --wandb_tags "$SCENE" "$RUN_TAG" eval
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
  fi


else
  echo "ERROR: MODE must be 'train' or 'eval'. Got: $MODE" >&2
  exit 2
fi

CMD_STAMP="$(date +%Y%m%d_%H%M%S)"
CMD_FILE="$META_DIR/command_${MODE}_${CMD_STAMP}.sh"
printf "%q " "${CMD[@]}" | tee "$CMD_FILE"
echo >> "$CMD_FILE"

echo "Launching..."
"${CMD[@]}" 2>&1 | tee -a "$META_DIR/${MODE}.log"
echo "Done."
