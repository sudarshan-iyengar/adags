#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"

RUNS_ROOT="${RUNS_ROOT:-$WORK/proj_adags/runs}"
REPO_ROOT="${REPO_ROOT:-$WORK/proj_adags/repo/adags}"
PY_SCRIPT="${PY_SCRIPT:-$REPO_ROOT/scripts/extract_results.py}"
ENV_SCRIPT="${ENV_SCRIPT:-$WORK/proj_adags/exp_index/leonardo_env.sh}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK/proj_adags/results_tables}"
LOG_ROOT="${LOG_ROOT:-$WORK/proj_adags/exp_index}"

METHODS=""
METHOD_CONTAINS=""
METHOD_REGEX=""
EXCLUDE_METHODS=""
EXCLUDE_METHOD_CONTAINS=""
SCENES=""
RUN_CONTAINS=""
RUN_REGEX=""
METRICS="psnr ssim lpips num_GS static"
OUTPUT_NAME=""
DEPENDENCY=""

DETAILED_CAPTION="Detailed results across selected methods and scenes. Best values per scene are bolded."
DETAILED_LABEL="tab:detailed_results"
AVG_CAPTION="Aggregate mean metrics across selected methods."
AVG_LABEL="tab:mean_results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) METHODS="$2"; shift 2 ;;
    --method-contains) METHOD_CONTAINS="$2"; shift 2 ;;
    --method-regex) METHOD_REGEX="$2"; shift 2 ;;
    --exclude-methods) EXCLUDE_METHODS="$2"; shift 2 ;;
    --exclude-method-contains) EXCLUDE_METHOD_CONTAINS="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --run-contains) RUN_CONTAINS="$2"; shift 2 ;;
    --run-regex) RUN_REGEX="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --detailed-caption) DETAILED_CAPTION="$2"; shift 2 ;;
    --detailed-label) DETAILED_LABEL="$2"; shift 2 ;;
    --avg-caption) AVG_CAPTION="$2"; shift 2 ;;
    --avg-label) AVG_LABEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

CMD=(
  python "$PY_SCRIPT"
  --base-path "$RUNS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --print-console
  --detailed-caption "$DETAILED_CAPTION"
  --detailed-label "$DETAILED_LABEL"
  --avg-caption "$AVG_CAPTION"
  --avg-label "$AVG_LABEL"
)

append_multi_arg() {
  local flag="$1"
  local values="$2"
  if [[ -n "$values" ]]; then
    # shellcheck disable=SC2206
    local arr=($values)
    CMD+=("$flag" "${arr[@]}")
  fi
}

append_single_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    CMD+=("$flag" "$value")
  fi
}

append_multi_arg --methods "$METHODS"
append_multi_arg --method-contains "$METHOD_CONTAINS"
append_single_arg --method-regex "$METHOD_REGEX"
append_multi_arg --exclude-methods "$EXCLUDE_METHODS"
append_multi_arg --exclude-method-contains "$EXCLUDE_METHOD_CONTAINS"
append_multi_arg --scenes "$SCENES"
append_multi_arg --run-contains "$RUN_CONTAINS"
append_single_arg --run-regex "$RUN_REGEX"
append_multi_arg --metrics "$METRICS"
append_single_arg --output-name "$OUTPUT_NAME"

CMD_STR="$(printf '%q ' "${CMD[@]}")"

WRAP_CMD=$(cat <<EOF
set -euo pipefail
source "$ENV_SCRIPT"
cd "$REPO_ROOT"
$CMD_STR
EOF
)

SBATCH_ARGS=(
  --parsable
  -p "$PARTITION"
  -A "$ACCOUNT"
  --qos="$QOS"
  -N 1
  --ntasks=1
  --cpus-per-task=4
  -t "$TIME_LIMIT"
  -o "$LOG_ROOT/results_tables_%j.out"
  -e "$LOG_ROOT/results_tables_%j.err"
)

if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--dependency="afterok:${DEPENDENCY}")
fi

JOBID=$(
  sbatch "${SBATCH_ARGS[@]}" --wrap "$WRAP_CMD"
)

echo "Submitted results-table job: $JOBID"
echo "Output root: $OUTPUT_ROOT"
[[ -n "$OUTPUT_NAME" ]] && echo "Output subfolder: $OUTPUT_NAME"