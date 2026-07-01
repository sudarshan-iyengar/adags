#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ADAGS_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
STAGE="${HIDE_REVEAL_STAGE:-synthetic}"
OUT_DIR="${HIDE_REVEAL_OUT_DIR:-$REPO_ROOT/refine-logs/hide_reveal_poc/$STAGE}"
MANIFEST="${HIDE_REVEAL_MANIFEST:-}"
SEEDS="${HIDE_REVEAL_SEEDS:-0 1 2}"
CLIPS_PER_TYPE="${HIDE_REVEAL_CLIPS_PER_TYPE:-8}"
COMPUTE_LPIPS="${HIDE_REVEAL_COMPUTE_LPIPS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$PYTHON_BIN" == "python" && -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"
fi

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

{
  echo "timestamp: $(date -Iseconds)"
  echo "stage: $STAGE"
  echo "out_dir: $OUT_DIR"
  echo "repo_root: $REPO_ROOT"
  echo "host: $(hostname)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "python_bin: $PYTHON_BIN"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "git_dirty: true"
    git status --short | sed 's/^/git_status: /'
  else
    echo "git_dirty: false"
  fi
  echo "---"
} | tee "$OUT_DIR/job_metadata.txt"

if [[ "$STAGE" == "synthetic" ]]; then
  read -r -a SEED_ARGS <<< "$SEEDS"
  "$PYTHON_BIN" scripts/run_hide_reveal_poc.py synthetic \
    --out-dir "$OUT_DIR" \
    --seeds "${SEED_ARGS[@]}" \
    --clips-per-type "$CLIPS_PER_TYPE"
elif [[ "$STAGE" == "real" ]]; then
  if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: HIDE_REVEAL_MANIFEST is required for real stage." >&2
    exit 2
  fi
  CMD=("$PYTHON_BIN" scripts/run_hide_reveal_poc.py real-eval --manifest "$MANIFEST" --out-dir "$OUT_DIR")
  if [[ "$COMPUTE_LPIPS" == "1" ]]; then
    CMD+=(--compute-lpips)
  fi
  "${CMD[@]}"
else
  echo "ERROR: HIDE_REVEAL_STAGE must be synthetic or real. Got: $STAGE" >&2
  exit 2
fi
