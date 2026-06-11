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
#   TRAIN_TIME=HH:MM:SS EVAL_AFTER=0|1 WANDB_MODE=offline|online|disabled

DATASET="${DATASET:-n3v}"
SMOKE="${SMOKE:-0}"

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
    DEFAULT_DATASET_ROOT="$WORK/proj_adags/data/n3v"
    DEFAULT_WANDB_GROUP="n3v"
    DEFAULT_RUN_TAG_PREFIX="Scaffold_Motion_Priors"
    DEFAULT_RUN_LABEL_PREFIX=""
    DEFAULT_TRAIN_TIME="15:00:00"
    DEFAULT_EVAL_AFTER="1"
    DEFAULT_CKPT_ITER="9000"
    DEFAULT_CONFIG_NAME="scaffold_lora_route0_dyn_densify_ptbudget"
    DEFAULT_CONFIG_PATH="$WORK/proj_adags/repo/adags/configs/n3v/scaffold_lora_route0_dyn_densify_ptbudget.yaml"
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
    DEFAULT_DATASET_ROOT="$WORK/proj_adags/data/panopticsports"
    DEFAULT_WANDB_GROUP="panopticsports"
    DEFAULT_RUN_TAG_PREFIX="panoptic"
    DEFAULT_RUN_LABEL_PREFIX="panopticsports_"
    if [[ "$SMOKE" == "1" ]]; then
      DEFAULT_TRAIN_TIME="00:20:00"
      DEFAULT_EVAL_AFTER="0"
      DEFAULT_CKPT_ITER="20"
      DEFAULT_CONFIG_NAME="smoke"
      DEFAULT_CONFIG_PATH="$WORK/proj_adags/repo/adags/configs/panopticsports/smoke.yaml"
    else
      DEFAULT_TRAIN_TIME="15:00:00"
      DEFAULT_EVAL_AFTER="1"
      DEFAULT_CKPT_ITER="6000"
      DEFAULT_CONFIG_NAME="scaffold_lora_route0_dyn_densify_ptbudget"
      DEFAULT_CONFIG_PATH="$WORK/proj_adags/repo/adags/configs/panopticsports/scaffold_lora_route0_dyn_densify_ptbudget.yaml"
    fi
    ;;

  *)
    echo "ERROR: DATASET must be 'n3v' or 'panopticsports'. Got: $DATASET" >&2
    exit 2
    ;;
esac

if [[ -n "${SCENE:-}" ]]; then
  SCENES=("$SCENE")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi

DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
TRAIN_TIME="${TRAIN_TIME:-$DEFAULT_TRAIN_TIME}"
EVAL_AFTER="${EVAL_AFTER:-$DEFAULT_EVAL_AFTER}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

CFG_NAME="${CFG_NAME:-$DEFAULT_CONFIG_NAME}"
CFG_PATH="${CONFIG:-$DEFAULT_CONFIG_PATH}"
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

RUN_LABEL="${RUN_LABEL:-${DEFAULT_RUN_LABEL_PREFIX}${CFG_NAME}}"
RUN_TAG="${RUN_TAG:-${DEFAULT_RUN_TAG_PREFIX}_${CFG_NAME}}"

echo "dataset: $DATASET"
echo "dataset_root: $DATASET_ROOT"
echo "config: $CFG_PATH"
echo "run_label: $RUN_LABEL"
echo "run_tag: $RUN_TAG"
echo "wandb_group: $WANDB_GROUP"
echo "wandb_mode: $WANDB_MODE"
echo "eval_after: $EVAL_AFTER"
echo "scenes: ${SCENES[*]}"
echo "---"

for SCENE in "${SCENES[@]}"; do
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_ID="${TS}_${SCENE}_${CFG_NAME}"
  LOG_PREFIX="${DATASET}_${SCENE}_${CFG_NAME}"

  TRAIN_JOBID=$(
    sbatch --parsable \
      -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1 \
      -t "$TRAIN_TIME" \
      -o "$WORK/proj_adags/exp_index/${LOG_PREFIX}_%j.out" \
      -e "$WORK/proj_adags/exp_index/${LOG_PREFIX}_%j.err" \
      --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG",RUN_ID="$RUN_ID",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CONFIG="$CFG_PATH",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE" \
      "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" train
  )

  echo "Submitted train for $DATASET/$SCENE [$CFG_NAME] as job $TRAIN_JOBID"
  echo "RUN_ID=$RUN_ID"

  if [[ "$EVAL_AFTER" == "1" ]]; then
    EVAL_JOBID=$(
      sbatch --parsable \
        --dependency=afterok:${TRAIN_JOBID} \
        -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
        -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
        -t 00:50:00 \
        -o "$WORK/proj_adags/exp_index/${LOG_PREFIX}_eval_%j.out" \
        -e "$WORK/proj_adags/exp_index/${LOG_PREFIX}_eval_%j.err" \
        --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG",RUN_ID="$RUN_ID",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$CFG_PATH",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE" \
        "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" eval
    )
    echo "Submitted eval  for $DATASET/$SCENE [$CFG_NAME] as job $EVAL_JOBID (depends on $TRAIN_JOBID)"
  fi
  echo "---"
done

echo "Done."
