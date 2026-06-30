#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./runit.sh
#   DATASET=panopticsports ./runit.sh
#   DATASET=panopticsports SCENE=basketball SMOKE=1 ./runit.sh
#
# Optional overrides:
#   SCENE=<single_scene>
#   CONFIG=<path/to/config.yaml> CFG_NAME=<label> CKPT_ITER=<iter>
#   EXPERIMENT=single|fixed_budget|mechanism_screen
#   FIXED_BUDGET_SCENES="cut_roasted_beef flame_steak"
#   FIXED_BUDGETS="400k 600k 800k"
#   FIXED_BUDGET_METHODS="lora_route0 scaffold_lora_route0_noreg scaffold_lora_route0_dyn"
#   MECHANISM_SCREEN_SCENES="cut_roasted_beef flame_steak sear_steak"
#   MECHANISM_SCREEN_METHODS="lora_route0 lora_route0_dyn rendergate_flow prior_admission prior_admission_no_flow prior_admission_no_static_anchor prior_admission_no_hard_floor"
#   MECHANISM_SCREEN_BUDGETS="600k"
#   TRAIN_TIME=HH:MM:SS EVAL_AFTER=0|1 WANDB_MODE=offline|online|disabled

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="${ADAGS_PROJECT_ROOT:-${WORK:?WORK must be set}/proj_adags}"
SUBMIT_SCRIPT="$REPO_DIR/scripts/run_leonardo.sh"
DRY_RUN="${DRY_RUN:-0}"

submit_sbatch() {
  if [[ "$DRY_RUN" == "1" ]]; then
    {
      printf 'DRY_RUN sbatch --parsable'
      printf ' %q' "$@"
      printf '\n'
    } >&2
    echo "dryrun-${RANDOM}"
  else
    sbatch --parsable "$@"
  fi
}

DATASET="${DATASET:-n3v}"
SMOKE="${SMOKE:-0}"
CONFIG_OVERRIDE="${CONFIG:-}"
CFG_NAME_OVERRIDE="${CFG_NAME:-}"
DEFAULT_FIXED_BUDGET_SCENES=()
DEFAULT_MECHANISM_SCREEN_SCENES=()

case "$DATASET" in
  n3v)
    DEFAULT_SCENES=(
      coffee_martini
      cook_spinach
      cut_roasted_beef
      flame_salmon_1
      sear_steak
      flame_steak
    )
    DEFAULT_DATASET_ROOT="$PROJECT_ROOT/data/n3v"
    DEFAULT_WANDB_GROUP="n3v"
    DEFAULT_RUN_TAG_PREFIX="Scaffold_Motion_Priors"
    DEFAULT_RUN_LABEL_PREFIX=""
    DEFAULT_TRAIN_TIME="15:00:00"
    DEFAULT_EVAL_AFTER="1"
    DEFAULT_CKPT_ITER="6000"
    DEFAULT_EXPERIMENT="mechanism_screen"
    DEFAULT_FIXED_BUDGET_SCENES=(
      cut_roasted_beef
      flame_steak
    )
    DEFAULT_MECHANISM_SCREEN_SCENES=(
      cut_roasted_beef
      flame_steak
      sear_steak
    )
    DEFAULT_CONFIG_NAME="scaffold_lora_route0_dyn_densify_ptbudget"
    DEFAULT_CONFIG_PATH="$REPO_DIR/configs/n3v/scaffold_lora_route0_dyn_densify_ptbudget.yaml"
    ;;

  panopticsports|panoptic|panoptic_sports)
    DATASET="panopticsports"
    DEFAULT_SCENES=(
      basketball
      boxes
      football
      juggle
      softball
      tennis
    )
    DEFAULT_DATASET_ROOT="$PROJECT_ROOT/data/panopticsports"
    DEFAULT_WANDB_GROUP="panopticsports"
    DEFAULT_RUN_TAG_PREFIX="panoptic"
    DEFAULT_RUN_LABEL_PREFIX="panopticsports_"
    if [[ "$SMOKE" == "1" ]]; then
      DEFAULT_TRAIN_TIME="00:20:00"
      DEFAULT_EVAL_AFTER="0"
      DEFAULT_CKPT_ITER="20"
      DEFAULT_EXPERIMENT="single"
      DEFAULT_CONFIG_NAME="smoke"
      DEFAULT_CONFIG_PATH="$REPO_DIR/configs/panopticsports/smoke.yaml"
    else
      DEFAULT_TRAIN_TIME="15:00:00"
      DEFAULT_EVAL_AFTER="1"
      DEFAULT_CKPT_ITER="6000"
      DEFAULT_EXPERIMENT="single"
      DEFAULT_CONFIG_NAME="scaffold_lora_route0_dyn_densify_ptbudget"
      DEFAULT_CONFIG_PATH="$REPO_DIR/configs/panopticsports/scaffold_lora_route0_dyn_densify_ptbudget.yaml"
    fi
    ;;

  *)
    echo "ERROR: DATASET must be 'n3v' or 'panopticsports'. Got: $DATASET" >&2
    exit 2
    ;;
esac

DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
TRAIN_TIME="${TRAIN_TIME:-$DEFAULT_TRAIN_TIME}"
EVAL_AFTER="${EVAL_AFTER:-$DEFAULT_EVAL_AFTER}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
EXPERIMENT="${EXPERIMENT:-$DEFAULT_EXPERIMENT}"

if [[ -n "${SCENE:-}" ]]; then
  SCENES=("$SCENE")
elif [[ "$EXPERIMENT" == "fixed_budget" && -n "${FIXED_BUDGET_SCENES:-}" ]]; then
  read -r -a SCENES <<< "$FIXED_BUDGET_SCENES"
elif [[ "$EXPERIMENT" == "fixed_budget" && "${#DEFAULT_FIXED_BUDGET_SCENES[@]}" -gt 0 ]]; then
  SCENES=("${DEFAULT_FIXED_BUDGET_SCENES[@]}")
elif [[ "$EXPERIMENT" == "mechanism_screen" && -n "${MECHANISM_SCREEN_SCENES:-}" ]]; then
  read -r -a SCENES <<< "$MECHANISM_SCREEN_SCENES"
elif [[ "$EXPERIMENT" == "mechanism_screen" && "${#DEFAULT_MECHANISM_SCREEN_SCENES[@]}" -gt 0 ]]; then
  SCENES=("${DEFAULT_MECHANISM_SCREEN_SCENES[@]}")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi

CKPT_ITER="${CKPT_ITER:-$DEFAULT_CKPT_ITER}"

# W&B settings
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-$DEFAULT_WANDB_GROUP}"
if [[ "$SMOKE" == "1" ]]; then
  WANDB_MODE="${WANDB_MODE:-offline}"
else
  WANDB_MODE="${WANDB_MODE:-offline}"   # compute nodes should stay offline by default
fi

if [[ -n "$CONFIG_OVERRIDE" || -n "$CFG_NAME_OVERRIDE" ]]; then
  EXPERIMENT="single"
fi

declare -a CFG_NAMES
declare -a CFG_PATHS
declare -a CFG_METHODS
declare -a CFG_BUDGETS

if [[ "$EXPERIMENT" == "fixed_budget" || "$EXPERIMENT" == "mechanism_screen" ]]; then
  if [[ "$DATASET" != "n3v" ]]; then
    echo "ERROR: EXPERIMENT=$EXPERIMENT is only defined for DATASET=n3v." >&2
    exit 2
  fi

  if [[ "$EXPERIMENT" == "mechanism_screen" ]]; then
    FIXED_BUDGET_METHODS="${MECHANISM_SCREEN_METHODS:-${FIXED_BUDGET_METHODS:-lora_route0 lora_route0_dyn rendergate_flow prior_admission prior_admission_no_flow prior_admission_no_static_anchor prior_admission_no_hard_floor}}"
    FIXED_BUDGETS="${MECHANISM_SCREEN_BUDGETS:-${FIXED_BUDGETS:-600k}}"
  else
    FIXED_BUDGET_METHODS="${FIXED_BUDGET_METHODS:-lora_route0 scaffold_lora_route0_noreg scaffold_lora_route0_dyn}"
    FIXED_BUDGETS="${FIXED_BUDGETS:-400k 600k 800k}"
  fi

  for METHOD in $FIXED_BUDGET_METHODS; do
    for BUDGET in $FIXED_BUDGETS; do
      CFG_NAME="fixed_budget_${METHOD}_${BUDGET}"
      CFG_PATH="$REPO_DIR/configs/n3v/${CFG_NAME}.yaml"
      if [[ ! -f "$CFG_PATH" ]]; then
        echo "ERROR: missing fixed-budget config: $CFG_PATH" >&2
        exit 3
      fi
      CFG_NAMES+=("$CFG_NAME")
      CFG_PATHS+=("$CFG_PATH")
      CFG_METHODS+=("$METHOD")
      CFG_BUDGETS+=("$BUDGET")
    done
  done
elif [[ "$EXPERIMENT" == "single" ]]; then
  if [[ -n "$CFG_NAME_OVERRIDE" ]]; then
    CFG_NAME="$CFG_NAME_OVERRIDE"
  elif [[ -n "$CONFIG_OVERRIDE" ]]; then
    CFG_NAME="$(basename "$CONFIG_OVERRIDE")"
    CFG_NAME="${CFG_NAME%.yaml}"
  else
    CFG_NAME="$DEFAULT_CONFIG_NAME"
  fi
  CFG_PATH="${CONFIG_OVERRIDE:-$DEFAULT_CONFIG_PATH}"
  if [[ ! -f "$CFG_PATH" ]]; then
    echo "ERROR: missing config: $CFG_PATH" >&2
    exit 3
  fi
  CFG_NAMES=("$CFG_NAME")
  CFG_PATHS=("$CFG_PATH")
  CFG_METHODS=("$CFG_NAME")
  CFG_BUDGETS=("")
else
  echo "ERROR: EXPERIMENT must be 'single', 'fixed_budget', or 'mechanism_screen'. Got: $EXPERIMENT" >&2
  exit 2
fi

echo "dataset: $DATASET"
echo "experiment: $EXPERIMENT"
echo "repo_dir: $REPO_DIR"
echo "submit_script: $SUBMIT_SCRIPT"
echo "dataset_root: $DATASET_ROOT"
echo "wandb_group: $WANDB_GROUP"
echo "wandb_mode: $WANDB_MODE"
echo "eval_after: $EVAL_AFTER"
echo "scenes: ${SCENES[*]}"
echo "configs:"
for IDX in "${!CFG_NAMES[@]}"; do
  echo "  - ${CFG_NAMES[$IDX]}: ${CFG_PATHS[$IDX]}"
done
echo "---"

for IDX in "${!CFG_NAMES[@]}"; do
  CFG_NAME="${CFG_NAMES[$IDX]}"
  CFG_PATH="${CFG_PATHS[$IDX]}"
  METHOD_FAMILY="${CFG_METHODS[$IDX]}"
  BUDGET_LABEL="${CFG_BUDGETS[$IDX]}"
  if [[ "$EXPERIMENT" == "fixed_budget" || "$EXPERIMENT" == "mechanism_screen" ]]; then
    RUN_LABEL_FOR_JOB="${RUN_LABEL:-${DEFAULT_RUN_LABEL_PREFIX}${CFG_NAME}}"
    RUN_TAG_FOR_JOB="${RUN_TAG:-$EXPERIMENT}"
    WANDB_EXTRA_TAGS_FOR_JOB="${WANDB_EXTRA_TAGS:-experiment:${EXPERIMENT} method:${METHOD_FAMILY} budget:${BUDGET_LABEL}}"
    EXPERIMENT_NAME_FOR_JOB="$EXPERIMENT"
  else
    RUN_LABEL_FOR_JOB="${RUN_LABEL:-${DEFAULT_RUN_LABEL_PREFIX}${CFG_NAME}}"
    RUN_TAG_FOR_JOB="${RUN_TAG:-${DEFAULT_RUN_TAG_PREFIX}_${CFG_NAME}}"
    WANDB_EXTRA_TAGS_FOR_JOB="${WANDB_EXTRA_TAGS:-}"
    EXPERIMENT_NAME_FOR_JOB="${EXPERIMENT_NAME:-}"
  fi

  for SCENE in "${SCENES[@]}"; do
    TS="$(date +%Y%m%d_%H%M%S)"
    RUN_ID="${TS}_${SCENE}_${CFG_NAME}"
    LOG_PREFIX="${DATASET}_${SCENE}_${CFG_NAME}"

    TRAIN_JOBID=$(
      submit_sbatch \
        -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
        -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1 \
        -t "$TRAIN_TIME" \
        -o "$PROJECT_ROOT/exp_index/${LOG_PREFIX}_%j.out" \
        -e "$PROJECT_ROOT/exp_index/${LOG_PREFIX}_%j.err" \
        --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG_FOR_JOB",RUN_ID="$RUN_ID",RUN_LABEL="$RUN_LABEL_FOR_JOB",DATASET_ROOT="$DATASET_ROOT",CONFIG="$CFG_PATH",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$WANDB_EXTRA_TAGS_FOR_JOB",EXPERIMENT_NAME="$EXPERIMENT_NAME_FOR_JOB",METHOD_FAMILY="$METHOD_FAMILY",BUDGET_LABEL="$BUDGET_LABEL",ADAGS_REPO_DIR="$REPO_DIR",ADAGS_PROJECT_ROOT="$PROJECT_ROOT" \
        "$SUBMIT_SCRIPT" train
    )

    echo "Submitted train for $DATASET/$SCENE [$CFG_NAME] as job $TRAIN_JOBID"
    echo "RUN_ID=$RUN_ID"

    if [[ "$EVAL_AFTER" == "1" ]]; then
      EVAL_JOBID=$(
        submit_sbatch \
          --dependency=afterok:${TRAIN_JOBID} \
          -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
          -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
          -t 00:50:00 \
          -o "$PROJECT_ROOT/exp_index/${LOG_PREFIX}_eval_%j.out" \
          -e "$PROJECT_ROOT/exp_index/${LOG_PREFIX}_eval_%j.err" \
          --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG_FOR_JOB",RUN_ID="$RUN_ID",RUN_LABEL="$RUN_LABEL_FOR_JOB",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$CFG_PATH",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$WANDB_EXTRA_TAGS_FOR_JOB",EXPERIMENT_NAME="$EXPERIMENT_NAME_FOR_JOB",METHOD_FAMILY="$METHOD_FAMILY",BUDGET_LABEL="$BUDGET_LABEL",ADAGS_REPO_DIR="$REPO_DIR",ADAGS_PROJECT_ROOT="$PROJECT_ROOT" \
          "$SUBMIT_SCRIPT" eval
      )
      echo "Submitted eval  for $DATASET/$SCENE [$CFG_NAME] as job $EVAL_JOBID (depends on $TRAIN_JOBID)"
    fi
    echo "---"
  done
done

echo "Done."
