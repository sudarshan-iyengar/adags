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

MODE="${DEPTH_OCCLUSION_MODE:-prepare}"
SOURCE_MANIFEST="${DEPTH_OCCLUSION_SOURCE_MANIFEST:-$REPO_ROOT/refine-logs/hide_reveal_real_windows.json}"
FRAME_MANIFEST="${DEPTH_OCCLUSION_FRAME_MANIFEST:-$REPO_ROOT/refine-logs/depth_occlusion_support/r031_da3_frame_manifest.json}"
DEPTH_OUT_DIR="${DEPTH_OCCLUSION_DEPTH_OUT_DIR:-$REPO_ROOT/refine-logs/depth_occlusion_support/r031_da3_depth}"
SUPPORT_OUT_DIR="${DEPTH_OCCLUSION_SUPPORT_OUT_DIR:-$REPO_ROOT/refine-logs/depth_occlusion_support/r031_depth_support}"
DATA_ROOT="${DEPTH_OCCLUSION_DATA_ROOT:-$PROJECT_ROOT/data/n3v}"
SCENES="${DEPTH_OCCLUSION_SCENES:-cut_roasted_beef flame_steak sear_steak}"
CAMERAS="${DEPTH_OCCLUSION_CAMERAS:-cam00}"
FRAME_STRIDE="${DEPTH_OCCLUSION_FRAME_STRIDE:-1}"
MAX_FRAMES_PER_SCENE="${DEPTH_OCCLUSION_MAX_FRAMES_PER_SCENE:-}"
DA3_REPO_DIR="${DA3_REPO_DIR:-$PROJECT_ROOT/repo/depth-anything-3}"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"
DA3_BATCH_SIZE="${DA3_BATCH_SIZE:-4}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-504}"
DA3_PROCESS_RES_METHOD="${DA3_PROCESS_RES_METHOD:-upper_bound_resize}"
DA3_DEVICE="${DA3_DEVICE:-cuda}"
DA3_WRITE_VIS="${DA3_WRITE_VIS:-1}"
DA3_MAX_IMAGES="${DA3_MAX_IMAGES:-}"
SUPPORT_MAX_COMPONENTS_PER_SCENE="${DEPTH_SUPPORT_MAX_COMPONENTS_PER_SCENE:-36}"
SUPPORT_MAX_PIXEL_FRACTION="${DEPTH_SUPPORT_MAX_PIXEL_FRACTION:-0.03}"
SUPPORT_BOUNDARY_DILATE="${DEPTH_SUPPORT_BOUNDARY_DILATE:-6}"
SUPPORT_MIN_COMPONENT_AREA="${DEPTH_SUPPORT_MIN_COMPONENT_AREA:-16}"
SUPPORT_MIN_SCORE="${DEPTH_SUPPORT_MIN_SCORE:-0.08}"
SUPPORT_TILE_SIZE="${DEPTH_SUPPORT_TILE_SIZE:-64}"
SUPPORT_TILE_STRIDE="${DEPTH_SUPPORT_TILE_STRIDE:-32}"
SUPPORT_USE_FLOW="${DEPTH_SUPPORT_USE_FLOW:-1}"
SUPPORT_FILL_COMPONENT_TILES="${DEPTH_SUPPORT_FILL_COMPONENT_TILES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DA3_PYTHON="${DA3_PYTHON:-}"
ENV_SCRIPT="${ADAGS_ENV_SCRIPT:-$PROJECT_ROOT/exp_index/leonardo_env.sh}"

if [[ -f "$ENV_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_SCRIPT"
fi

if [[ -n "$DA3_PYTHON" ]]; then
  PYTHON_BIN="$DA3_PYTHON"
elif [[ -x "$PROJECT_ROOT/envs/da3/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/envs/da3/bin/python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_EXTENSIONS_DIR="${DEPTH_TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/build/torch_extensions_jobs/${SLURM_JOB_ID:-manual}}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
if [[ -d "$DA3_REPO_DIR/src" ]]; then
  export PYTHONPATH="$DA3_REPO_DIR/src:$DA3_REPO_DIR:${PYTHONPATH:-}"
fi
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$DEPTH_OUT_DIR" "$SUPPORT_OUT_DIR" "$(dirname "$FRAME_MANIFEST")"

case "$MODE" in
  prepare)
    OUT_DIR="$(dirname "$FRAME_MANIFEST")"
    ;;
  infer)
    OUT_DIR="$DEPTH_OUT_DIR"
    ;;
  support)
    OUT_DIR="$SUPPORT_OUT_DIR"
    ;;
  *)
    echo "ERROR: DEPTH_OCCLUSION_MODE must be prepare, infer, or support. Got: $MODE" >&2
    exit 2
    ;;
esac
mkdir -p "$OUT_DIR"

{
  echo "timestamp: $(date -Iseconds)"
  echo "mode: $MODE"
  echo "source_manifest: $SOURCE_MANIFEST"
  echo "frame_manifest: $FRAME_MANIFEST"
  echo "depth_out_dir: $DEPTH_OUT_DIR"
  echo "support_out_dir: $SUPPORT_OUT_DIR"
  echo "data_root: $DATA_ROOT"
  echo "scenes: $SCENES"
  echo "cameras: $CAMERAS"
  echo "frame_stride: $FRAME_STRIDE"
  echo "max_frames_per_scene: ${MAX_FRAMES_PER_SCENE:-none}"
  echo "da3_repo_dir: $DA3_REPO_DIR"
  echo "da3_model_dir: $DA3_MODEL_DIR"
  echo "da3_batch_size: $DA3_BATCH_SIZE"
  echo "da3_process_res: $DA3_PROCESS_RES"
  echo "da3_process_res_method: $DA3_PROCESS_RES_METHOD"
  echo "support_fill_component_tiles: $SUPPORT_FILL_COMPONENT_TILES"
  echo "repo_root: $REPO_ROOT"
  echo "project_root: $PROJECT_ROOT"
  echo "env_script: $ENV_SCRIPT"
  echo "host: $(hostname)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "python: $(command -v "$PYTHON_BIN" 2>/dev/null || echo "$PYTHON_BIN")"
  "$PYTHON_BIN" -V || true
  echo "hf_home: $HF_HOME"
  echo "torch_extensions_dir: $TORCH_EXTENSIONS_DIR"
  echo "git_branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
    echo "git_dirty: true"
    git -C "$REPO_ROOT" status --short | sed 's/^/git_status: /'
  else
    echo "git_dirty: false"
  fi
  echo "---"
} | tee "$OUT_DIR/job_metadata.txt"

cd "$REPO_ROOT"

if [[ "$MODE" == "prepare" ]]; then
  CMD=(
    "$PYTHON_BIN" scripts/run_depth_occlusion_support.py prepare-frame-manifest
    --source-manifest "$SOURCE_MANIFEST"
    --out "$FRAME_MANIFEST"
    --scenes "$SCENES"
    --cameras "$CAMERAS"
    --data-root "$DATA_ROOT"
    --frame-stride "$FRAME_STRIDE"
  )
  if [[ -n "$MAX_FRAMES_PER_SCENE" ]]; then
    CMD+=(--max-frames-per-scene "$MAX_FRAMES_PER_SCENE")
  fi
elif [[ "$MODE" == "infer" ]]; then
  CMD=(
    "$PYTHON_BIN" scripts/run_depth_occlusion_support.py infer-da3-depth
    --frame-manifest "$FRAME_MANIFEST"
    --out-dir "$DEPTH_OUT_DIR"
    --model-dir "$DA3_MODEL_DIR"
    --da3-repo "$DA3_REPO_DIR"
    --batch-size "$DA3_BATCH_SIZE"
    --process-res "$DA3_PROCESS_RES"
    --process-res-method "$DA3_PROCESS_RES_METHOD"
    --device "$DA3_DEVICE"
  )
  if [[ "$DA3_WRITE_VIS" == "1" ]]; then
    CMD+=(--write-vis)
  fi
  if [[ -n "$DA3_MAX_IMAGES" ]]; then
    CMD+=(--max-images "$DA3_MAX_IMAGES")
  fi
elif [[ "$MODE" == "support" ]]; then
  CMD=(
    "$PYTHON_BIN" scripts/run_depth_occlusion_support.py build-support
    --source-manifest "$SOURCE_MANIFEST"
    --depth-manifest "$DEPTH_OUT_DIR/da3_depth_manifest.json"
    --out-dir "$SUPPORT_OUT_DIR"
    --max-components-per-scene "$SUPPORT_MAX_COMPONENTS_PER_SCENE"
    --max-pixel-fraction "$SUPPORT_MAX_PIXEL_FRACTION"
    --boundary-dilate "$SUPPORT_BOUNDARY_DILATE"
    --min-component-area "$SUPPORT_MIN_COMPONENT_AREA"
    --min-score "$SUPPORT_MIN_SCORE"
    --tile-size "$SUPPORT_TILE_SIZE"
    --tile-stride "$SUPPORT_TILE_STRIDE"
  )
  if [[ "$SUPPORT_USE_FLOW" != "1" ]]; then
    CMD+=(--no-flow)
  fi
  if [[ "$SUPPORT_FILL_COMPONENT_TILES" == "1" ]]; then
    CMD+=(--fill-component-tiles)
  fi
fi

printf "%q " "${CMD[@]}" | tee "$OUT_DIR/command_${MODE}.sh"
echo | tee -a "$OUT_DIR/command_${MODE}.sh"
"${CMD[@]}"
