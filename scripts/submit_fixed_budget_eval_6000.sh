#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_fixed_budget_eval_6000.sh [--dry-run] [--append-wandb]

Submit eval-only Slurm jobs for the 18 fixed-budget N3V models at chkpnt6000.pth.

Options:
  --dry-run       Print the sbatch commands without submitting.
  --append-wandb  Resume the original W&B run ids. By default, use a new
                  "<source_run_id>_eval6000" W&B id to keep 9000 summaries intact.
  -h, --help      Show this help.

Environment overrides:
  SCENES="cut_roasted_beef flame_steak"
  FIXED_BUDGET_METHODS="lora_route0 scaffold_lora_route0_noreg scaffold_lora_route0_dyn"
  FIXED_BUDGETS="400k 600k 800k"
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
APPEND_WANDB=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --append-wandb)
      APPEND_WANDB=1
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
SCENES="${SCENES:-cut_roasted_beef flame_steak}"
FIXED_BUDGET_METHODS="${FIXED_BUDGET_METHODS:-lora_route0 scaffold_lora_route0_noreg scaffold_lora_route0_dyn}"
FIXED_BUDGETS="${FIXED_BUDGETS:-400k 600k 800k}"

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d36_068}"
QOS="${QOS:-boost_qos_lprod}"

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="$PROJECT_ROOT/exp_index/fixed_budget_eval${CKPT_ITER}_jobs_${timestamp}.tsv"
mkdir -p "$PROJECT_ROOT/exp_index"

if [[ "$DRY_RUN" == "0" ]]; then
  printf "job_id\tscene\tmethod\tbudget\trun_id\twandb_run_id\trun_dir\tconfig\tckpt\n" > "$manifest"
fi

submitted=0
checked=0

shopt -s nullglob
for method in $FIXED_BUDGET_METHODS; do
  for budget in $FIXED_BUDGETS; do
    cfg_name="fixed_budget_${method}_${budget}"
    cfg_path="$REPO_ROOT/configs/n3v/${cfg_name}.yaml"
    if [[ ! -f "$cfg_path" ]]; then
      echo "ERROR: missing config: $cfg_path" >&2
      exit 3
    fi

    for scene in $SCENES; do
      data_path="$DATASET_ROOT/$scene"
      if [[ ! -d "$data_path" ]]; then
        echo "ERROR: missing dataset scene: $data_path" >&2
        exit 3
      fi

      matches=("$PROJECT_ROOT/runs/$cfg_name"/*"_${scene}_${cfg_name}")
      if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "ERROR: expected exactly one run dir for $scene $cfg_name, found ${#matches[@]}" >&2
        printf '  %s\n' "${matches[@]}" >&2
        exit 4
      fi

      run_dir="${matches[0]}"
      source_run_id="$(basename "$run_dir")"
      ckpt_path="$run_dir/chkpnt${CKPT_ITER}.pth"
      if [[ ! -f "$ckpt_path" ]]; then
        echo "ERROR: missing checkpoint: $ckpt_path" >&2
        exit 5
      fi

      if [[ "$APPEND_WANDB" == "1" ]]; then
        wandb_run_id="$source_run_id"
      else
        wandb_run_id="${source_run_id}_eval${CKPT_ITER}"
      fi

      log_prefix="n3v_${scene}_${cfg_name}_eval${CKPT_ITER}"
      wandb_extra_tags="experiment:fixed_budget method:${method} budget:${budget} eval_ckpt:${CKPT_ITER} ckpt:${CKPT_ITER}"
      cmd=(
        sbatch --parsable
        -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
        -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
        -t "$EVAL_TIME"
        -o "$PROJECT_ROOT/exp_index/${log_prefix}_%j.out"
        -e "$PROJECT_ROOT/exp_index/${log_prefix}_%j.err"
        --export=ALL,SCENE="$scene",RUN_TAG="fixed_budget",RUN_ID="$wandb_run_id",RUN_DIR="$run_dir",RUN_LABEL="$cfg_name",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="fixed_budget",METHOD_FAMILY="$method",BUDGET_LABEL="$budget"
        "$REPO_ROOT/scripts/run_leonardo.sh" eval
      )

      checked=$((checked + 1))
      if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN %s %s %s: ' "$scene" "$method" "$budget"
        printf '%q ' "${cmd[@]}"
        printf '\n'
      else
        job_id="$("${cmd[@]}")"
        submitted=$((submitted + 1))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$job_id" "$scene" "$method" "$budget" "$source_run_id" "$wandb_run_id" "$run_dir" "$cfg_path" "$ckpt_path" \
          >> "$manifest"
        echo "Submitted $scene $cfg_name at ${CKPT_ITER}: job $job_id"
      fi
    done
  done
done

expected=0
for _method in $FIXED_BUDGET_METHODS; do
  for _budget in $FIXED_BUDGETS; do
    for _scene in $SCENES; do
      expected=$((expected + 1))
    done
  done
done

if [[ "$checked" -ne "$expected" ]]; then
  echo "ERROR: expected $expected eval jobs, checked $checked." >&2
  exit 6
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete: checked $checked eval jobs."
else
  echo "Submitted $submitted eval jobs."
  echo "Manifest: $manifest"
fi
