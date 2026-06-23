#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_lora_flow_smoke.sh [--dry-run] [--train-only]

Submit one LoRA flow-prior smoke test:
  - scene: cut_roasted_beef by default
  - config: fixed_budget_lora_route0_filemask_residual_flow_smoke_500
  - train: 500 iterations
  - eval: dependent eval at chkpnt500.pth

Options:
  --dry-run     Print sbatch commands without submitting.
  --train-only  Submit only training; skip dependent eval.
  -h, --help    Show this help.

Environment overrides:
  SCENE=cut_roasted_beef
  CANDIDATE=fixed_budget_lora_route0_filemask_residual_flow_smoke_500
  CKPT_ITER=500
  TRAIN_TIME=00:45:00
  EVAL_TIME=00:35:00
  CPUS_PER_TASK=8
  WANDB_MODE=offline
  WANDB_PROJECT=adags
  WANDB_ENTITY=<optional>
  WANDB_GROUP=lora_flow_smoke_YYYYMMDD
EOF
}

DRY_RUN=0
WITH_EVAL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --train-only)
      WITH_EVAL=0
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

SCENE="${SCENE:-cut_roasted_beef}"
CANDIDATE="${CANDIDATE:-fixed_budget_lora_route0_filemask_residual_flow_smoke_500}"
CKPT_ITER="${CKPT_ITER:-500}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
TRAIN_TIME="${TRAIN_TIME:-00:45:00}"
EVAL_TIME="${EVAL_TIME:-00:35:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-lora_flow_smoke_$(date +%Y%m%d)}"
WANDB_MODE="${WANDB_MODE:-offline}"

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"

cfg_path="$REPO_ROOT/configs/n3v/${CANDIDATE}.yaml"
data_path="$DATASET_ROOT/$SCENE"
flow_dir="$data_path/flow"
if [[ ! -f "$cfg_path" ]]; then
  echo "ERROR: missing config: $cfg_path" >&2
  exit 3
fi
if [[ ! -d "$data_path" ]]; then
  echo "ERROR: missing dataset scene: $data_path" >&2
  exit 4
fi
if [[ ! -d "$flow_dir" ]] || ! find "$flow_dir" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
  echo "ERROR: missing flow priors under: $flow_dir" >&2
  exit 5
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="$PROJECT_ROOT/exp_index/lora_flow_smoke_${timestamp}.tsv"
sync_script="$PROJECT_ROOT/exp_index/lora_flow_smoke_sync_after_eval_${timestamp}.sh"
mkdir -p "$PROJECT_ROOT/exp_index"

run_id="${timestamp}_${SCENE}_${CANDIDATE}"
run_dir="$PROJECT_ROOT/runs/$CANDIDATE/$run_id"
ckpt_path="$run_dir/chkpnt${CKPT_ITER}.pth"
log_prefix="n3v_${SCENE}_${CANDIDATE}"
method_family="lora_filemask_residual_flow_smoke"
wandb_extra_tags="phase:lora_flow_smoke screen:flow_smoke baseline:fixed_budget_lora_route0_600k comparator:fixed_budget_lora_route0_filemask_residual_600k candidate:flow_smoke method:${method_family} budget:500iter no_external_sync"

train_cmd=(
  sbatch --parsable
  -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
  -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
  -t "$TRAIN_TIME"
  -o "$PROJECT_ROOT/exp_index/${log_prefix}_train_%j.out"
  -e "$PROJECT_ROOT/exp_index/${log_prefix}_train_%j.err"
  --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$SCENE",RUN_TAG="lora_flow_smoke",RUN_ID="$run_id",RUN_LABEL="$CANDIDATE",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="lora_flow_smoke",METHOD_FAMILY="$method_family",BUDGET_LABEL="500iter"
  "$REPO_ROOT/scripts/run_leonardo.sh" train
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY-RUN train %s %s: ' "$SCENE" "$CANDIDATE"
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  train_job_id="<train_job_id>"
else
  printf "train_job_id\teval_job_id\tscene\tcandidate\tmethod_family\trun_id\trun_dir\tconfig\tckpt\tflow_dir\twandb_group\tsync_script\n" > "$manifest"
  if ! train_job_id="$("${train_cmd[@]}")"; then
    echo "ERROR: failed to submit train job for $SCENE $CANDIDATE" >&2
    exit 6
  fi
  if [[ -z "$train_job_id" ]]; then
    echo "ERROR: train sbatch returned an empty job id for $SCENE $CANDIDATE" >&2
    exit 6
  fi
  echo "Submitted train $SCENE $CANDIDATE: job $train_job_id"
fi

eval_job_id=""
if [[ "$WITH_EVAL" == "1" ]]; then
  eval_cmd=(
    sbatch --parsable
    --dependency="afterok:${train_job_id}"
    -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
    -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
    -t "$EVAL_TIME"
    -o "$PROJECT_ROOT/exp_index/${log_prefix}_eval${CKPT_ITER}_%j.out"
    -e "$PROJECT_ROOT/exp_index/${log_prefix}_eval${CKPT_ITER}_%j.err"
    --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$SCENE",RUN_TAG="lora_flow_smoke_eval",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$CANDIDATE",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags eval_ckpt:${CKPT_ITER}",EXPERIMENT_NAME="lora_flow_smoke",METHOD_FAMILY="$method_family",BUDGET_LABEL="500iter"
    "$REPO_ROOT/scripts/run_leonardo.sh" eval
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN eval %s %s: ' "$SCENE" "$CANDIDATE"
    printf '%q ' "${eval_cmd[@]}"
    printf '\n'
    eval_job_id="<eval_job_id>"
  else
    if ! eval_job_id="$("${eval_cmd[@]}")"; then
      echo "ERROR: failed to submit eval job for $SCENE $CANDIDATE after $train_job_id" >&2
      exit 7
    fi
    if [[ -z "$eval_job_id" ]]; then
      echo "ERROR: eval sbatch returned an empty job id for $SCENE $CANDIDATE" >&2
      exit 7
    fi
    echo "Submitted eval $SCENE $CANDIDATE after $train_job_id: job $eval_job_id"
  fi
fi

if [[ "$DRY_RUN" == "0" ]]; then
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "# Run this on the login node if you explicitly approve external W&B upload."
    echo "# It waits for eval job ${eval_job_id:-none}, then syncs only this run directory."
    printf 'ADAGS_APPROVE_WANDB_UPLOAD=1 WANDB_PROJECT=%q WANDB_ENTITY=%q %q/scripts/sync_wandb_run_dir.sh' "$WANDB_PROJECT" "$WANDB_ENTITY" "$REPO_ROOT"
    if [[ -n "$eval_job_id" ]]; then
      printf ' --wait-job %q' "$eval_job_id"
    fi
    printf ' --project %q' "$WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
      printf ' --entity %q' "$WANDB_ENTITY"
    fi
    printf ' %q\n' "$run_dir"
  } > "$sync_script"
  chmod +x "$sync_script"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$train_job_id" "$eval_job_id" "$SCENE" "$CANDIDATE" "$method_family" "$run_id" "$run_dir" "$cfg_path" "$ckpt_path" "$flow_dir" "$WANDB_GROUP" "$sync_script" \
    >> "$manifest"

  echo "Manifest: $manifest"
  echo "Login-node sync helper: $sync_script"
else
  echo "Dry run complete."
fi
