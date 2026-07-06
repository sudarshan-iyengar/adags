#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_hide_reveal_poc.sh --stage synthetic [--dry-run]
  scripts/submit_hide_reveal_poc.sh --stage real --manifest refine-logs/real_windows.json [--dry-run]
  scripts/submit_hide_reveal_poc.sh --stage derive-real-renders --manifest refine-logs/real_windows.json \
    --route0-eval /path/to/test/ours_6000 [--dry-run]
  scripts/submit_hide_reveal_poc.sh --stage actual-real-renders --manifest refine-logs/real_windows.json \
    --residual-manifest refine-logs/r011_manifest.json --matched-manifest refine-logs/r012_manifest.json [--dry-run]

Submit proof-of-concept hide/reveal jobs through Slurm. Outputs go under
refine-logs/hide_reveal_poc/<stage>/ and scheduler logs go under logs/.

Options:
  --stage synthetic|real|derive-real-renders|actual-real-renders
                         PoC stage to run.
  --manifest PATH        Required for real, derive-real-renders, and actual-real-renders.
  --route0-eval PATH     Eval folder with renders/ and gt/ for derived renders.
  --residual-manifest PATH
                         Optional residual_uncertainty baseline manifest for actual-real-renders.
  --matched-manifest PATH
                         Optional matched_lifespan baseline manifest for actual-real-renders.
  --out-dir PATH         Override output directory.
  --eval-out-dir PATH    Output directory for real-eval after derived renders.
  --overwrite            Replace non-empty derived output render folders.
  --dry-run              Print the sbatch command without submitting.
  -h, --help             Show this help.

Environment overrides:
  HIDE_REVEAL_SEEDS="0 1 2"
  HIDE_REVEAL_CLIPS_PER_TYPE=8
  HIDE_REVEAL_COMPUTE_LPIPS=0
  PARTITION=boost_usr_prod
  ACCOUNT=euhpc_d21_034
  QOS=boost_qos_lprod
  TIME=00:20:00
  CPUS_PER_TASK=8
  GRES=gpu:1
EOF
}

STAGE=""
MANIFEST=""
OUT_DIR=""
ROUTE0_EVAL=""
EVAL_OUT_DIR=""
RESIDUAL_MANIFEST=""
MATCHED_MANIFEST=""
OVERWRITE_DERIVED=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="${2:-}"
      shift 2
      ;;
    --manifest)
      MANIFEST="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --route0-eval)
      ROUTE0_EVAL="${2:-}"
      shift 2
      ;;
    --residual-manifest)
      RESIDUAL_MANIFEST="${2:-}"
      shift 2
      ;;
    --matched-manifest)
      MATCHED_MANIFEST="${2:-}"
      shift 2
      ;;
    --eval-out-dir)
      EVAL_OUT_DIR="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE_DERIVED=1
      shift
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

if [[ "$STAGE" == "derive" ]]; then
  STAGE="derive-real-renders"
fi

if [[ "$STAGE" != "synthetic" && "$STAGE" != "real" && "$STAGE" != "derive-real-renders" && "$STAGE" != "actual-real-renders" ]]; then
  echo "ERROR: --stage must be synthetic, real, derive-real-renders, or actual-real-renders." >&2
  usage >&2
  exit 2
fi

if [[ "$STAGE" != "synthetic" && -z "$MANIFEST" ]]; then
  echo "ERROR: --manifest is required for $STAGE stage." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${ADAGS_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$ADAGS_PROJECT_ROOT"
elif [[ -n "${WORK:-}" ]]; then
  PROJECT_ROOT="$WORK/proj_adags"
else
  PROJECT_ROOT="$REPO_ROOT"
fi
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/refine-logs/hide_reveal_poc/$STAGE"
fi
if [[ -z "$EVAL_OUT_DIR" ]]; then
  EVAL_OUT_DIR="$REPO_ROOT/refine-logs/hide_reveal_poc/real"
fi

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"
TIME="${TIME:-00:20:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
GRES="${GRES:-gpu:1}"
COMPUTE_LPIPS="${HIDE_REVEAL_COMPUTE_LPIPS:-0}"
timestamp="$(date +%Y%m%d_%H%M%S)"
submit_manifest="$REPO_ROOT/refine-logs/hide_reveal_poc_${STAGE}_jobs_${timestamp}.tsv"

cmd=(
  sbatch --parsable
  -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
  -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK"
  --gres="$GRES"
  -t "$TIME"
  -o "$LOG_DIR/hide_reveal_${STAGE}_%j.out"
  -e "$LOG_DIR/hide_reveal_${STAGE}_%j.err"
  --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",HIDE_REVEAL_STAGE="$STAGE",HIDE_REVEAL_OUT_DIR="$OUT_DIR",HIDE_REVEAL_MANIFEST="$MANIFEST",HIDE_REVEAL_ROUTE0_EVAL="$ROUTE0_EVAL",HIDE_REVEAL_RESIDUAL_MANIFEST="$RESIDUAL_MANIFEST",HIDE_REVEAL_MATCHED_MANIFEST="$MATCHED_MANIFEST",HIDE_REVEAL_EVAL_OUT_DIR="$EVAL_OUT_DIR",HIDE_REVEAL_OVERWRITE="$OVERWRITE_DERIVED",HIDE_REVEAL_COMPUTE_LPIPS="$COMPUTE_LPIPS"
  "$REPO_ROOT/scripts/run_hide_reveal_poc_job.sh"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY-RUN: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

job_id="$("${cmd[@]}")"
mkdir -p "$(dirname "$submit_manifest")"
printf "job_id\tstage\tmanifest\tout_dir\tlog_stdout\tlog_stderr\n" > "$submit_manifest"
printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$job_id" "$STAGE" "${MANIFEST:-none}" "$OUT_DIR" \
  "$LOG_DIR/hide_reveal_${STAGE}_${job_id}.out" \
  "$LOG_DIR/hide_reveal_${STAGE}_${job_id}.err" \
  >> "$submit_manifest"

echo "Submitted hide/reveal $STAGE PoC job: $job_id"
echo "Manifest: $submit_manifest"
