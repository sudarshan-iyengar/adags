#!/usr/bin/env bash
# Batch eval script for all opticalblur runs
# This script will submit eval jobs for all runs listed below

set -euo pipefail

declare -A RUNS_CONFIG

# opticalblur_default (6 runs)
RUNS_CONFIG[opticalblur_default]="cut_roasted_beef sear_steak flame_steak coffee_martini cook_spinach flame_salmon_1"

# opticalblur_runB (6 runs)
RUNS_CONFIG[opticalblur_runB]="coffee_martini cut_roasted_beef sear_steak flame_steak cook_spinach flame_salmon_1"

# opticalblur_runC (5 runs)
RUNS_CONFIG[opticalblur_runC]="coffee_martini sear_steak flame_steak cook_spinach flame_salmon_1"

# opticalblur_runD (6 runs)
RUNS_CONFIG[opticalblur_runD]="coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 sear_steak flame_steak"

SUBMITTED_JOBS=()
FAILED_RUNS=()
TOTAL_SUBMITTED=0

echo "========================================="
echo "Submitting evals for all opticalblur runs"
echo "========================================="
echo ""

for RUNS_SUBDIR in opticalblur_default opticalblur_runB opticalblur_runC opticalblur_runD; do
  if [[ ! -v RUNS_CONFIG[$RUNS_SUBDIR] ]]; then
    continue
  fi

  IFS=' ' read -ra SCENES <<< "${RUNS_CONFIG[$RUNS_SUBDIR]}"

  echo "📁 Processing: $RUNS_SUBDIR (${#SCENES[@]} scenes)"

  for SCENE in "${SCENES[@]}"; do
    # Find the actual run directory
    RUN_PATTERN="$WORK/proj_adags/runs/${RUNS_SUBDIR}/*${SCENE}*"
    matching_runs=(${RUN_PATTERN})

    if [[ ${#matching_runs[@]} -eq 0 ]]; then
      echo "   ❌ $SCENE - NOT FOUND"
      FAILED_RUNS+=("$RUNS_SUBDIR/$SCENE")
      continue
    fi

    if [[ ${#matching_runs[@]} -gt 1 ]]; then
      echo "   ❌ $SCENE - MULTIPLE MATCHES"
      FAILED_RUNS+=("$RUNS_SUBDIR/$SCENE (multiple)")
      continue
    fi

    RUN_DIR="${matching_runs[0]}"
    RUN_ID="$(basename "$RUN_DIR")"

    # Verify checkpoint exists
    CKPT_PATH="${RUN_DIR}/chkpnt15000.pth"
    if [[ ! -f "$CKPT_PATH" ]]; then
      echo "   ⚠️  $SCENE - NO CHECKPOINT AT 15000"
      # Try to find any checkpoint
      CKPT=$(find "$RUN_DIR" -name "chkpnt*.pth" | head -1)
      if [[ -z "$CKPT" ]]; then
        echo "      (no checkpoints found)"
        FAILED_RUNS+=("$RUNS_SUBDIR/$SCENE (no checkpoint)")
        continue
      fi
      CKPT_PATH="$CKPT"
      CKPT_ITER=$(basename "$CKPT" | grep -oE '[0-9]+' | head -1)
      echo "      Using: chkpnt${CKPT_ITER}.pth"
    else
      CKPT_ITER=15000
    fi

    # Submit the job
    JOB_ID=$(sbatch -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
         -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -t 00:30:00 \
         -o "$WORK/proj_adags/exp_index/eval_%j.out" \
         -e "$WORK/proj_adags/exp_index/eval_%j.err" \
         --export="ALL,RUN_DIR=${RUN_DIR},SCENE=${SCENE},DATASET_ROOT=$WORK/proj_adags/data/n3v,CKPT_ITER=${CKPT_ITER}" \
         "$WORK/proj_adags/repo/adags/scripts/run_leonardo.sh" eval \
         2>&1 | awk '{print $NF}')

    SUBMITTED_JOBS+=("$JOB_ID")
    ((TOTAL_SUBMITTED++))
    echo "   ✓ $SCENE - Job $JOB_ID (ckpt: $CKPT_ITER)"
  done

  echo ""
done

echo "========================================="
echo "✅ SUBMISSION SUMMARY"
echo "========================================="
echo "Total submitted: $TOTAL_SUBMITTED"
echo "Job IDs: ${SUBMITTED_JOBS[@]}"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo ""
  echo "⚠️  FAILED TO SUBMIT (${#FAILED_RUNS[@]}):"
  printf '   - %s\n' "${FAILED_RUNS[@]}"
fi

echo ""
echo "========================================="
echo "Monitor with:"
if [[ ${#SUBMITTED_JOBS[@]} -gt 0 ]]; then
  echo "  squeue -u \$USER | grep eval"
  echo "  tail -f \$WORK/proj_adags/exp_index/eval_${SUBMITTED_JOBS[0]}.out"
fi