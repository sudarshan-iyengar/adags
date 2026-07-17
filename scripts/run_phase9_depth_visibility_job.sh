#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ADAGS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PROJECT_ROOT="${ADAGS_PROJECT_ROOT:-${WORK:?WORK must be set}/proj_adags}"
ENV_SCRIPT="${ADAGS_ENV_SCRIPT:-$PROJECT_ROOT/exp_index/leonardo_env.sh}"
RUN_ID="${PHASE9_RUN_ID:?PHASE9_RUN_ID must be set}"
ACTION="${PHASE9_ACTION:?PHASE9_ACTION must be set}"
EXECUTION_MANIFEST="${PHASE9_EXECUTION_MANIFEST:?PHASE9_EXECUTION_MANIFEST must be set}"
if [[ -n "${PHASE9_PYTHON:-}" ]]; then
  PYTHON_BIN="$PHASE9_PYTHON"
elif [[ "$ACTION" == "da3-conformance" || "$ACTION" == "produce-da3" || "$ACTION" == "fast-visibility-pilot" || "$ACTION" == "anchor-abstention-pilot" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/envs/da3/bin/python"
else
  PYTHON_BIN="$PROJECT_ROOT/envs/adags/bin/python"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Phase 9 job wrapper must run under Slurm." >&2
  exit 2
fi
if [[ ! -f "$ENV_SCRIPT" ]]; then
  echo "ERROR: environment setup is missing: $ENV_SCRIPT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: project Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_SCRIPT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/build/torch_extensions_jobs/$SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

cd "$REPO_ROOT"
RESOLVED_ARGV=()
mapfile -d '' -t RESOLVED_ARGV < <(
  "$PYTHON_BIN" - "$EXECUTION_MANIFEST" "$RUN_ID" "$ACTION" <<'PY'
import json
import sys

from depth_visibility.evaluator import resolved_python_argv

manifest_path, run_id, action = sys.argv[1:]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
argv = resolved_python_argv(
    manifest,
    run_id=run_id,
    action=action,
    launcher_path="scripts/run_phase9_depth_visibility.py",
    execution_manifest_path=manifest_path,
)
for value in argv:
    sys.stdout.buffer.write(value.encode("utf-8") + b"\0")
PY
)
if (( ${#RESOLVED_ARGV[@]} < 3 )); then
  echo "ERROR: failed to load exact resolved argv from $EXECUTION_MANIFEST" >&2
  exit 2
fi
exec "$PYTHON_BIN" "${RESOLVED_ARGV[@]:1}"
