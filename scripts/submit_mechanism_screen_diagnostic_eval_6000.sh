#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_mechanism_screen_diagnostic_eval_6000.sh [--dry-run] [--allow-missing-masks]

Submit eval-only Slurm jobs for the 15 mechanism-screen N3V models at chkpnt6000.pth.
The jobs reuse existing run directories and resume the original W&B run IDs so the
current mechanism-screen analysis manifest can pick up the added diagnostics.

Options:
  --dry-run              Print sbatch commands without submitting.
  --allow-missing-masks  Do not fail if <scene>/motion_priors/masks is absent.
                         This is only useful for checking command construction;
                         dynamic/static diagnostics will stay absent without masks.
  -h, --help             Show this help.

Environment overrides:
  MECHANISM_MANIFEST=exp_index/mechanism_screen_wandb_sync_20260617_173511.tsv
  CKPT_ITER=6000
  EVAL_TIME=00:50:00
  CPUS_PER_TASK=8
  WANDB_MODE=offline
  WANDB_PROJECT=adags
  WANDB_ENTITY=<optional>
  WANDB_GROUP=n3v
EOF
}

DRY_RUN=0
ALLOW_MISSING_MASKS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-missing-masks)
      ALLOW_MISSING_MASKS=1
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

if [[ -z "${WORK:-}" ]]; then
  export WORK="$(cd "$PROJECT_ROOT/.." && pwd)"
fi

CKPT_ITER="${CKPT_ITER:-6000}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
EVAL_TIME="${EVAL_TIME:-00:50:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-n3v}"
WANDB_MODE="${WANDB_MODE:-offline}"
MECHANISM_MANIFEST="${MECHANISM_MANIFEST:-$PROJECT_ROOT/exp_index/mechanism_screen_wandb_sync_20260617_173511.tsv}"

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"

if [[ ! -f "$MECHANISM_MANIFEST" ]]; then
  echo "ERROR: missing mechanism manifest: $MECHANISM_MANIFEST" >&2
  exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
submit_manifest="$PROJECT_ROOT/exp_index/mechanism_screen_diag_eval${CKPT_ITER}_jobs_${timestamp}.tsv"
mkdir -p "$PROJECT_ROOT/exp_index"

if [[ "$DRY_RUN" == "0" ]]; then
  printf "job_id\tscene\tmethod\tlabel\tsource_run_id\twandb_run_id\trun_dir\tconfig\tckpt\tmask_dir\tflow_dir\n" > "$submit_manifest"
fi

declare -A seen_run_ids=()
submitted=0
checked=0

while IFS=$'\t' read -r scene method label run_id payload path; do
  if [[ "$scene" == "scene" ]]; then
    continue
  fi
  if [[ -z "${run_id:-}" ]]; then
    continue
  fi
  if [[ -n "${seen_run_ids[$run_id]+x}" ]]; then
    continue
  fi
  seen_run_ids[$run_id]=1

  run_dir="${path%%/wandb/*}"
  cfg_path="$REPO_ROOT/configs/n3v/${label}.yaml"
  data_path="$DATASET_ROOT/$scene"
  ckpt_path="$run_dir/chkpnt${CKPT_ITER}.pth"
  mask_dir="$data_path/motion_priors/masks"
  flow_dir="$data_path/flow"

  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: missing run dir for $run_id: $run_dir" >&2
    exit 4
  fi
  if [[ ! -f "$cfg_path" ]]; then
    echo "ERROR: missing config for $run_id: $cfg_path" >&2
    exit 5
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    echo "ERROR: missing checkpoint for $run_id: $ckpt_path" >&2
    exit 6
  fi
  if [[ ! -d "$data_path" ]]; then
    echo "ERROR: missing dataset scene for $run_id: $data_path" >&2
    exit 7
  fi
  if [[ "$ALLOW_MISSING_MASKS" == "0" ]] && ! compgen -G "$mask_dir/*.png" > /dev/null; then
    echo "ERROR: missing dynamic-mask priors for $scene: $mask_dir" >&2
    echo "Run scripts/build_motion_priors.py for the scene before submitting diagnostics." >&2
    exit 8
  fi
  if [[ ! -d "$flow_dir" ]]; then
    echo "WARNING: missing flow prior directory for $scene: $flow_dir" >&2
  fi

  log_prefix="n3v_${scene}_${label}_diag_eval${CKPT_ITER}"
  wandb_extra_tags="experiment:mechanism_screen diagnostics:dynamic_static eval_ckpt:${CKPT_ITER} method:${method} budget:600k"
  cmd=(
    sbatch --parsable
    -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
    -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
    -t "$EVAL_TIME"
    -o "$PROJECT_ROOT/exp_index/${log_prefix}_%j.out"
    -e "$PROJECT_ROOT/exp_index/${log_prefix}_%j.err"
    --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="mechanism_diag",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$label",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="mechanism_screen",METHOD_FAMILY="$method",BUDGET_LABEL="600k"
    "$REPO_ROOT/scripts/run_leonardo.sh" eval
  )

  checked=$((checked + 1))
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN %s %s: ' "$scene" "$method"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    job_id="$("${cmd[@]}")"
    submitted=$((submitted + 1))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$job_id" "$scene" "$method" "$label" "$run_id" "$run_id" "$run_dir" "$cfg_path" "$ckpt_path" "$mask_dir" "$flow_dir" \
      >> "$submit_manifest"
    echo "Submitted $scene $label diagnostics at ${CKPT_ITER}: job $job_id"
  fi
done < "$MECHANISM_MANIFEST"

if [[ "$checked" -ne 15 ]]; then
  echo "ERROR: expected 15 unique mechanism-screen eval jobs, checked $checked." >&2
  exit 9
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete: checked $checked eval jobs."
else
  echo "Submitted $submitted eval jobs."
  echo "Manifest: $submit_manifest"
fi
