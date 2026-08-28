#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   sbatch ... scripts/run_leonardo.sh train
#   sbatch ... --export=ALL,RUN_ID=20260226_154023_cut_roasted_beef_baseline scripts/run_leonardo.sh eval
#   sbatch ... --export=ALL,RUN_DIR=$WORK/proj_adags/runs/20260226_154023_cut_roasted_beef_baseline scripts/run_leonardo.sh eval
#   sbatch ... --export=ALL,SCENE=cut_roasted_beef,RUN_TAG=baseline scripts/run_leonardo.sh train

MODE="${1:-train}"                           # train | eval
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="${ADAGS_REPO_DIR:-$DEFAULT_REPO_DIR}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"
PROJECT_ROOT="${ADAGS_PROJECT_ROOT:-${WORK:?WORK must be set}/proj_adags}"

# Config / dataset
CONFIG="${CONFIG:-configs/n3v/default.yaml}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
SCENE="${SCENE:-cut_roasted_beef}"           # override per job via --export
# main.py's own default is 6666, so leaving SEED unset reproduces every run
# submitted before this variable existed. Until now the seed could not be
# varied from here at all, which silently turned any multi-seed design on
# Leonardo into repeated runs at one seed.
SEED="${SEED:-6666}"
RUN_TAG="${RUN_TAG:-baseline}"               # free text
RUN_LABEL="${RUN_LABEL:-}"                   # optional subdirectory under runs/

# W&B
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-n3v}"
WANDB_MODE="${WANDB_MODE:-offline}"         # offline is recommended on compute nodes
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
METHOD_FAMILY="${METHOD_FAMILY:-}"
BUDGET_LABEL="${BUDGET_LABEL:-}"

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
      RUN_DIR="$PROJECT_ROOT/runs/$RUN_LABEL/$RUN_ID"
    else
      RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
    fi
  else
    # only create a new run for training; for eval, require an existing run
    if [[ "$MODE" == "train" ]]; then
      TS="$(date +%Y%m%d_%H%M%S)"
      RUN_ID="${TS}_${SCENE}_${RUN_TAG}"
      if [[ -n "$RUN_LABEL" ]]; then
        RUN_DIR="$PROJECT_ROOT/runs/$RUN_LABEL/${RUN_ID}"
      else
        RUN_DIR="$PROJECT_ROOT/runs/${RUN_ID}"
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
source "$PROJECT_ROOT/exp_index/leonardo_env.sh"
cd "$REPO_DIR"

# Keep repo-local CUDA extension packages importable on Leonardo.
export PYTHONPATH="$REPO_DIR/simple-knn:${PYTHONPATH:-}"

if [[ -n "${ADAGS_TORCH_EXTENSIONS_DIR:-}" ]]; then
  export TORCH_EXTENSIONS_DIR="$ADAGS_TORCH_EXTENSIONS_DIR"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/build/torch_extensions_jobs/$SLURM_JOB_ID"
else
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/build/torch_extensions_manual}"
fi
mkdir -p "$TORCH_EXTENSIONS_DIR"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-${CPUS_PER_TASK:-4}}"

# ---- Minimal logging (append per run, don’t overwrite) ----
{
  echo "timestamp: $(date -Iseconds)"
  echo "mode: ${MODE}"
  echo "scene: ${SCENE}"
  echo "run_tag: ${RUN_TAG}"
  echo "run_label: ${RUN_LABEL:-none}"
  echo "run_dir: ${RUN_DIR}"
  echo "repo_dir: ${REPO_DIR}"
  echo "project_root: ${PROJECT_ROOT}"
  echo "host: $(hostname)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "python: $(which python)"
  python -V
  echo "torch_extensions_dir: ${TORCH_EXTENSIONS_DIR}"
  echo "torch_cuda_arch_list: ${TORCH_CUDA_ARCH_LIST}"
  echo "max_jobs: ${MAX_JOBS}"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "git_dirty: true"
    git status --short | sed 's/^/git_status: /'
  else
    echo "git_dirty: false"
  fi
  echo "config: ${CONFIG}"
  echo "dataset_path: ${DATASET_PATH}"
  echo "seed: ${SEED}"
  echo "ckpt_iter: ${CKPT_ITER}"
  [[ -n "$CKPT_PATH" ]] && echo "ckpt_path: ${CKPT_PATH}"
  echo "wandb_project: ${WANDB_PROJECT}"
  echo "wandb_entity: ${WANDB_ENTITY:-none}"
  echo "wandb_group: ${WANDB_GROUP}"
  echo "wandb_mode: ${WANDB_MODE}"
  echo "wandb_extra_tags: ${WANDB_EXTRA_TAGS:-none}"
  echo "experiment_name: ${EXPERIMENT_NAME:-none}"
  echo "method_family: ${METHOD_FAMILY:-none}"
  echo "budget_label: ${BUDGET_LABEL:-none}"

  echo "---"
} | tee -a "$META_DIR/run_info.txt"

WANDB_TAGS=("$SCENE" "$RUN_TAG")
if [[ -n "$WANDB_EXTRA_TAGS" ]]; then
  read -r -a WANDB_EXTRA_TAG_ARRAY <<< "$WANDB_EXTRA_TAGS"
  WANDB_TAGS+=("${WANDB_EXTRA_TAG_ARRAY[@]}")
fi

# ---- Command ----
if [[ "$MODE" == "train" ]]; then
  CMD=(
    python main.py
    --config "$CONFIG"
    --model_path "$RUN_DIR"
    --source_path "$DATASET_PATH"
    --seed "$SEED"
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_mode "$WANDB_MODE"
    --wandb_run_name "$RUN_ID"
    --wandb_group "$WANDB_GROUP"
    --wandb_resume "$RUN_ID"
    --wandb_tags "${WANDB_TAGS[@]}" train
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$EXPERIMENT_NAME" ]]; then
    CMD+=(--experiment_name "$EXPERIMENT_NAME")
  fi
  if [[ -n "$METHOD_FAMILY" ]]; then
    CMD+=(--method_family "$METHOD_FAMILY")
  fi
  if [[ -n "$BUDGET_LABEL" ]]; then
    CMD+=(--budget_label "$BUDGET_LABEL")
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
    --seed "$SEED"
    --start_checkpoint "$CKPT_PATH"
    --val
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_mode "$WANDB_MODE"
    --wandb_run_name "$RUN_ID"
    --wandb_group "$WANDB_GROUP"
    --wandb_resume "$RUN_ID"
    --wandb_tags "${WANDB_TAGS[@]}" eval
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$EXPERIMENT_NAME" ]]; then
    CMD+=(--experiment_name "$EXPERIMENT_NAME")
  fi
  if [[ -n "$METHOD_FAMILY" ]]; then
    CMD+=(--method_family "$METHOD_FAMILY")
  fi
  if [[ -n "$BUDGET_LABEL" ]]; then
    CMD+=(--budget_label "$BUDGET_LABEL")
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
