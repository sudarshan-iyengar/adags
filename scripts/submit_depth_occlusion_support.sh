#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_depth_occlusion_support.sh --mode prepare [--dry-run]
  scripts/submit_depth_occlusion_support.sh --mode infer [--dry-run]
  scripts/submit_depth_occlusion_support.sh --mode support [--dry-run]

Submit DA3 depth occlusion support jobs through Slurm.

Options:
  --mode prepare|infer|support
  --source-manifest PATH
  --frame-manifest PATH
  --depth-out-dir PATH
  --support-out-dir PATH
  --dry-run

Environment overrides:
  DEPTH_OCCLUSION_SCENES="cut_roasted_beef flame_steak sear_steak"
  DEPTH_OCCLUSION_CAMERAS=cam00
  DEPTH_OCCLUSION_FRAME_STRIDE=1
  DA3_REPO_DIR=$WORK/proj_adags/repo/depth-anything-3
  DA3_MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE-1.1
  DA3_BATCH_SIZE=4
  DA3_PROCESS_RES=504
  DEPTH_SUPPORT_MAX_COMPONENTS_PER_SCENE=36
  DEPTH_SUPPORT_MAX_PIXEL_FRACTION=0.03
  PARTITION=boost_usr_prod
  ACCOUNT=euhpc_d21_034
  QOS=boost_qos_lprod
  TIME=00:30:00
  CPUS_PER_TASK=8
  GRES=gpu:1
EOF
}

MODE=""
SOURCE_MANIFEST=""
FRAME_MANIFEST=""
DEPTH_OUT_DIR=""
SUPPORT_OUT_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --source-manifest)
      SOURCE_MANIFEST="${2:-}"
      shift 2
      ;;
    --frame-manifest)
      FRAME_MANIFEST="${2:-}"
      shift 2
      ;;
    --depth-out-dir)
      DEPTH_OUT_DIR="${2:-}"
      shift 2
      ;;
    --support-out-dir)
      SUPPORT_OUT_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "prepare" && "$MODE" != "infer" && "$MODE" != "support" ]]; then
  echo "ERROR: --mode must be prepare, infer, or support." >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${ADAGS_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$ADAGS_PROJECT_ROOT"
elif [[ -n "${WORK:-}" ]]; then
  PROJECT_ROOT="$WORK/proj_adags"
else
  PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
fi
if [[ -z "${WORK:-}" ]]; then
  export WORK="$(cd "$PROJECT_ROOT/.." && pwd)"
fi

SOURCE_MANIFEST="${SOURCE_MANIFEST:-$REPO_ROOT/refine-logs/hide_reveal_real_windows.json}"
FRAME_MANIFEST="${FRAME_MANIFEST:-$REPO_ROOT/refine-logs/depth_occlusion_support/r031_da3_frame_manifest.json}"
DEPTH_OUT_DIR="${DEPTH_OUT_DIR:-$REPO_ROOT/refine-logs/depth_occlusion_support/r031_da3_depth}"
SUPPORT_OUT_DIR="${SUPPORT_OUT_DIR:-$REPO_ROOT/refine-logs/depth_occlusion_support/r031_depth_support}"

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"
if [[ -z "${TIME:-}" ]]; then
  if [[ "$MODE" == "infer" ]]; then
    TIME="02:00:00"
  else
    TIME="00:30:00"
  fi
fi
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
GRES="${GRES:-gpu:1}"

timestamp="$(date +%Y%m%d_%H%M%S)"
submit_manifest="$REPO_ROOT/refine-logs/depth_occlusion_support_${MODE}_jobs_${timestamp}.tsv"
log_dir="$REPO_ROOT/logs"
mkdir -p "$log_dir" "$(dirname "$submit_manifest")"

cmd=(
  sbatch --parsable
  -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
  -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK"
)
if [[ -n "$GRES" ]]; then
  cmd+=(--gres="$GRES")
fi
cmd+=(
  -t "$TIME"
  -o "$log_dir/depth_occlusion_${MODE}_%j.out"
  -e "$log_dir/depth_occlusion_${MODE}_%j.err"
  --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",DEPTH_OCCLUSION_MODE="$MODE",DEPTH_OCCLUSION_SOURCE_MANIFEST="$SOURCE_MANIFEST",DEPTH_OCCLUSION_FRAME_MANIFEST="$FRAME_MANIFEST",DEPTH_OCCLUSION_DEPTH_OUT_DIR="$DEPTH_OUT_DIR",DEPTH_OCCLUSION_SUPPORT_OUT_DIR="$SUPPORT_OUT_DIR"
  "$REPO_ROOT/scripts/run_depth_occlusion_support_job.sh"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY-RUN: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

job_id="$("${cmd[@]}")"
printf "job_id\tmode\tsource_manifest\tframe_manifest\tdepth_out_dir\tsupport_out_dir\tlog_stdout\tlog_stderr\n" > "$submit_manifest"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$job_id" "$MODE" "$SOURCE_MANIFEST" "$FRAME_MANIFEST" "$DEPTH_OUT_DIR" "$SUPPORT_OUT_DIR" \
  "$log_dir/depth_occlusion_${MODE}_${job_id}.out" \
  "$log_dir/depth_occlusion_${MODE}_${job_id}.err" \
  >> "$submit_manifest"

echo "Submitted depth occlusion $MODE job: $job_id"
echo "Manifest: $submit_manifest"
