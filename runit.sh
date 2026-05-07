#!/usr/bin/env bash
set -euo pipefail

SCENES=(
  coffee_martini
  cook_spinach
  cut_roasted_beef
  flame_salmon_1
  sear_steak
  flame_steak
)

# label|config_path|ckpt_iter
CONFIGS=(
  "part_basis_k8_r8_a32|$WORK/proj_adags/repo/adags/configs/n3v/part_basis_k8_r8_a32.yaml|15000"
  "part_basis_k16_r8_a32|$WORK/proj_adags/repo/adags/configs/n3v/part_basis_k16_r8_a32.yaml|15000"
  "part_basis_k8_r16_a32|$WORK/proj_adags/repo/adags/configs/n3v/part_basis_k8_r16_a32.yaml|15000"
)


# W&B settings
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-n3v}"
WANDB_MODE="${WANDB_MODE:-offline}"   # compute nodes should stay offline


for SCENE in "${SCENES[@]}"; do
  for ENTRY in "${CONFIGS[@]}"; do
    IFS='|' read -r CFG_NAME CFG_PATH CKPT_ITER <<< "$ENTRY"

    TS="$(date +%Y%m%d_%H%M%S)"
    RUN_TAG="Part_Basis_Motion_${CFG_NAME}"
    RUN_ID="${TS}_${SCENE}_${CFG_NAME}"

    TRAIN_JOBID=$(
      sbatch --parsable \
        -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
        -N 1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 \
        -t 09:00:00 \
        -o "$WORK/proj_adags/exp_index/${CFG_NAME}_%j.out" \
        -e "$WORK/proj_adags/exp_index/${CFG_NAME}_%j.err" \
        --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG",RUN_LABEL="$CFG_NAME",RUN_ID="$RUN_ID",DATASET_ROOT="$WORK/proj_adags/data/n3v",CONFIG="$CFG_PATH",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE" \
        "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" train
    )

    EVAL_JOBID=$(
      sbatch --parsable \
        --dependency=afterok:${TRAIN_JOBID} \
        -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
        -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
        -t 00:50:00 \
        -o "$WORK/proj_adags/exp_index/${CFG_NAME}_eval_%j.out" \
        -e "$WORK/proj_adags/exp_index/${CFG_NAME}_eval_%j.err" \
        --export=ALL,SCENE="$SCENE",RUN_TAG="$RUN_TAG",RUN_LABEL="$CFG_NAME",RUN_ID="$RUN_ID",DATASET_ROOT="$WORK/proj_adags/data/n3v",CKPT_ITER="$CKPT_ITER",CONFIG="$CFG_PATH",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE" \
        "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" eval
    )

    echo "Submitted train for $SCENE [$CFG_NAME] as job $TRAIN_JOBID"
    echo "Submitted eval  for $SCENE [$CFG_NAME] as job $EVAL_JOBID (depends on $TRAIN_JOBID)"
    echo "RUN_ID=$RUN_ID"
    echo "---"
  done
done

echo "Done."
