#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ADAGS_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
if [[ -n "${ADAGS_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$ADAGS_PROJECT_ROOT"
elif [[ -n "${WORK:-}" ]]; then
  PROJECT_ROOT="$WORK/proj_adags"
else
  PROJECT_ROOT="$REPO_ROOT"
fi
STAGE="${HIDE_REVEAL_STAGE:-synthetic}"
OUT_DIR="${HIDE_REVEAL_OUT_DIR:-$REPO_ROOT/refine-logs/hide_reveal_poc/$STAGE}"
MANIFEST="${HIDE_REVEAL_MANIFEST:-}"
ROUTE0_EVAL="${HIDE_REVEAL_ROUTE0_EVAL:-}"
RESIDUAL_MANIFEST="${HIDE_REVEAL_RESIDUAL_MANIFEST:-}"
MATCHED_MANIFEST="${HIDE_REVEAL_MATCHED_MANIFEST:-}"
EVAL_OUT_DIR="${HIDE_REVEAL_EVAL_OUT_DIR:-$REPO_ROOT/refine-logs/hide_reveal_poc/real}"
OVERWRITE_DERIVED="${HIDE_REVEAL_OVERWRITE:-0}"
SEEDS="${HIDE_REVEAL_SEEDS:-0 1 2}"
CLIPS_PER_TYPE="${HIDE_REVEAL_CLIPS_PER_TYPE:-8}"
COMPUTE_LPIPS="${HIDE_REVEAL_COMPUTE_LPIPS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SCRIPT="${ADAGS_ENV_SCRIPT:-$PROJECT_ROOT/exp_index/leonardo_env.sh}"

if [[ -f "$ENV_SCRIPT" ]]; then
  source "$ENV_SCRIPT"
fi

# Leonardo boost nodes are A100-backed; set a default so PyTorch's CUDA
# extension JIT does not need to infer architectures before CUDA is visible.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/build/torch_extensions_jobs/${SLURM_JOB_ID:-manual}}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

{
  echo "timestamp: $(date -Iseconds)"
  echo "stage: $STAGE"
  echo "out_dir: $OUT_DIR"
  echo "manifest: ${MANIFEST:-none}"
  echo "route0_eval: ${ROUTE0_EVAL:-none}"
  echo "residual_manifest: ${RESIDUAL_MANIFEST:-none}"
  echo "matched_manifest: ${MATCHED_MANIFEST:-none}"
  echo "eval_out_dir: $EVAL_OUT_DIR"
  echo "repo_root: $REPO_ROOT"
  echo "project_root: $PROJECT_ROOT"
  echo "env_script: $ENV_SCRIPT"
  echo "env_loaded: $([[ -f "$ENV_SCRIPT" ]] && echo true || echo false)"
  echo "host: $(hostname)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "slurm_job_gpus: ${SLURM_JOB_GPUS:-unset}"
  echo "slurm_gpus: ${SLURM_GPUS:-unset}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "torch_cuda_arch_list: ${TORCH_CUDA_ARCH_LIST:-unset}"
  echo "torch_extensions_dir: ${TORCH_EXTENSIONS_DIR:-unset}"
  echo "max_jobs: ${MAX_JOBS:-unset}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | sed 's/^/nvidia_smi: /' || true
  else
    echo "nvidia_smi: unavailable"
  fi
  echo "python: $(command -v "$PYTHON_BIN" 2>/dev/null || echo "$PYTHON_BIN")"
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
elif [[ "$STAGE" == "derive-real-renders" || "$STAGE" == "derive" ]]; then
  if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: HIDE_REVEAL_MANIFEST is required for derive-real-renders stage." >&2
    exit 2
  fi
  CMD=(
    "$PYTHON_BIN" scripts/run_hide_reveal_poc.py derive-real-renders
    --manifest "$MANIFEST"
    --out-dir "$OUT_DIR"
    --run-eval
    --eval-out-dir "$EVAL_OUT_DIR"
  )
  if [[ -n "$ROUTE0_EVAL" ]]; then
    CMD+=(--route0-eval "$ROUTE0_EVAL")
  fi
  if [[ "$OVERWRITE_DERIVED" == "1" ]]; then
    CMD+=(--overwrite)
  fi
  "${CMD[@]}"
elif [[ "$STAGE" == "actual-real-renders" ]]; then
  if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: HIDE_REVEAL_MANIFEST is required for actual-real-renders stage." >&2
    exit 2
  fi
  CMD=(
    "$PYTHON_BIN" scripts/run_hide_reveal_poc.py actual-real-renders
    --manifest "$MANIFEST"
    --out-dir "$OUT_DIR"
    --eval-out-dir "$EVAL_OUT_DIR"
  )
  if [[ -n "$RESIDUAL_MANIFEST" ]]; then
    CMD+=(--residual-manifest "$RESIDUAL_MANIFEST")
  fi
  if [[ -n "$MATCHED_MANIFEST" ]]; then
    CMD+=(--matched-manifest "$MATCHED_MANIFEST")
  fi
  if [[ "$OVERWRITE_DERIVED" == "1" ]]; then
    CMD+=(--overwrite)
  fi
  if [[ "$COMPUTE_LPIPS" == "1" ]]; then
    CMD+=(--compute-lpips)
  fi
  "${CMD[@]}"
else
  echo "ERROR: HIDE_REVEAL_STAGE must be synthetic, real, derive-real-renders, or actual-real-renders. Got: $STAGE" >&2
  exit 2
fi
