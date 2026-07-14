#!/usr/bin/env bash
set -euo pipefail

SCENES=(
  basketball
  boxes
  football
  juggle
  softball
  tennis
)

if [[ -n "${SCENE:-}" ]]; then
  SCENES=("$SCENE")
fi

SMOKE="${SMOKE:-0}"
DATASET_ROOT="${DATASET_ROOT:-$WORK/proj_adags/data/panopticsports}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-panopticsports}"
REPO_ROOT="${WORK}/proj_adags/repo/adags"
LOG_ROOT="${REPO_ROOT}/logs"

mkdir -p "$LOG_ROOT"

if [[ "$SMOKE" == "1" ]]; then
  CONFIG="${CONFIG:-$REPO_ROOT/configs/panopticsports/smoke.yaml}"
  RUN_LABEL="${RUN_LABEL:-panopticsports_smoke}"
  RUN_TAG="${RUN_TAG:-panoptic_smoke}"
  TRAIN_TIME="${TRAIN_TIME:-00:20:00}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
  CKPT_ITER="${CKPT_ITER:-20}"
  WANDB_MODE="${WANDB_MODE:-offline}"
  EVAL_AFTER="${EVAL_AFTER:-0}"
else
  if [[ -z "${CONFIG:-}" ]]; then
    echo "CONFIG must be set explicitly when SMOKE is not 1." >&2
    exit 2
  fi
  CONFIG_BASENAME="$(basename -- "$CONFIG")"
  CONFIG_STEM="${CONFIG_BASENAME%.*}"
  RUN_LABEL="${RUN_LABEL:-panopticsports_${CONFIG_STEM}}"
  RUN_TAG="${RUN_TAG:-panoptic_${CONFIG_STEM}}"
  TRAIN_TIME="${TRAIN_TIME:-15:00:00}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
  CKPT_ITER="${CKPT_ITER:-6000}"
  WANDB_MODE="${WANDB_MODE:-offline}"
  EVAL_AFTER="${EVAL_AFTER:-1}"
fi

for SCENE in "${SCENES[@]}"; do
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_ID="${TS}_${SCENE}_${RUN_TAG}"

  TRAIN_JOBID=$(
    sbatch --parsable \
      -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1 \
      -t "$TRAIN_TIME" \
      -o "$LOG_ROOT/panopticsports_${SCENE}_%j.out" \
      -e "$LOG_ROOT/panopticsports_${SCENE}_%j.err" \
      --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG",RUN_ID="$RUN_ID",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CONFIG="$CONFIG",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE" \
      "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" train
  )

  echo "Submitted PanopticSports train for $SCENE as job $TRAIN_JOBID"
  echo "RUN_ID=$RUN_ID"

  if [[ "$EVAL_AFTER" == "1" ]]; then
    EVAL_JOBID=$(
      sbatch --parsable \
        --dependency=afterok:${TRAIN_JOBID} \
        -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
        -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
        -t 00:50:00 \
        -o "$LOG_ROOT/panopticsports_${SCENE}_eval_%j.out" \
        -e "$LOG_ROOT/panopticsports_${SCENE}_eval_%j.err" \
        --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG",RUN_ID="$RUN_ID",RUN_LABEL="$RUN_LABEL",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$CONFIG",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE" \
        "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" eval
    )
    echo "Submitted PanopticSports eval for $SCENE as job $EVAL_JOBID (depends on $TRAIN_JOBID)"
  fi
  echo "---"
done

echo "Done."
