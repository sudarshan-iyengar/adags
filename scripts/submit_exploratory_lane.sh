#!/usr/bin/env bash
# Submit one CSVL-VPL v2 exploratory lane: full training then --val evaluation.
# Usage: LANE=lane_l3_full ROUND=r1 bash scripts/submit_exploratory_lane.sh
# Contract: research-wiki/operations/csvl-vpl-v2-exploratory-contract.md
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${ADAGS_PROJECT_ROOT:-${WORK:?WORK must be set}/proj_adags}"
LANE="${LANE:?LANE must be set, e.g. lane_l0_route0}"
ROUND="${ROUND:-r1}"
SEED="${SEED:-0}"
SCENE="${SCENE:-cut_roasted_beef}"
case "$SCENE" in
  cut_roasted_beef|cook_spinach) ;;
  *) echo "SCENE=$SCENE is not an approved development scene" >&2; exit 2 ;;
esac
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PROJECT_ROOT/runs/csvl_vpl_v2_exploratory/${STAMP}_${SCENE}_${LANE}_${ROUND}_seed${SEED}"
CONFIG="$REPO_ROOT/configs/n3v/${LANE}.yaml"
[[ -f "$CONFIG" ]] || { echo "missing config $CONFIG" >&2; exit 2; }
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$RUN_DIR"

sbatch \
  --job-name="${LANE}-${ROUND}" \
  --account=euhpc_d21_034 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 \
  --time=06:00:00 \
  --output="$LOG_DIR/${LANE}_${ROUND}_%j.out" \
  --error="$LOG_DIR/${LANE}_${ROUND}_%j.err" \
  --wrap="set -euo pipefail
source '$PROJECT_ROOT/exp_index/leonardo_env.sh'
export PYTHONPATH='$REPO_ROOT/simple-knn:$REPO_ROOT'
cd '$REPO_ROOT'
PY='$PROJECT_ROOT/envs/adags/bin/python'
\$PY main.py --config '$CONFIG' \
  --source_path '$PROJECT_ROOT/data/n3v/$SCENE' \
  --model_path '$RUN_DIR' \
  --seed $SEED \
  --save_iterations 1000 3000 6000 \
  --test_iterations 1000 2000 3000 4500 6000
echo '=== training done; running --val evaluation ==='
\$PY main.py --config '$CONFIG' \
  --source_path '$PROJECT_ROOT/data/n3v/$SCENE' \
  --model_path '$RUN_DIR' \
  --seed $SEED \
  --val --start_checkpoint '$RUN_DIR/chkpnt6000.pth'
echo '=== lane complete ==='"
echo "RUN_DIR=$RUN_DIR"
