#!/usr/bin/env bash
# Submit the preregistered Phase 0 primitive-centric evidence census.
# Usage: bash scripts/submit_phase0_census.sh
# One scientific job only; capture the printed job ID immediately and check
# squeue/sacct before any resubmission (AGENTS.md).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${ADAGS_PROJECT_ROOT:-${WORK:?WORK must be set}/proj_adags}"
CONFIG="${PHASE0_CENSUS_CONFIG:-$REPO_ROOT/configs/depth_visibility/phase0_census_v1.json}"
LOG_DIR="$REPO_ROOT/logs"
# EUHPC_D21_034's allocation ended 2026-07-30; jobs now charge
# EUHPC_D36_068. Overridable, which the hardcoded value below was not.
ACCOUNT="${ACCOUNT:-euhpc_d36_068}"
mkdir -p "$LOG_DIR"

sbatch \
  --job-name=phase0-census-v1 \
  --account="$ACCOUNT" \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  --time=02:00:00 \
  --output="$LOG_DIR/phase0-census-v1_%j.out" \
  --error="$LOG_DIR/phase0-census-v1_%j.err" \
  --wrap="set -euo pipefail
source '$PROJECT_ROOT/exp_index/leonardo_env.sh'
export PYTHONPATH='$REPO_ROOT/simple-knn:$REPO_ROOT'
cd '$REPO_ROOT'
'$PROJECT_ROOT/envs/adags/bin/python' scripts/run_phase0_primitive_census.py --config '$CONFIG'"
