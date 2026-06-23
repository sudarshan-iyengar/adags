#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_lora_flow_6000_gate.sh [--dry-run] [--train-only] [--no-login-sync]

Submit the 3-scene LoRA flow-prior 6000-iteration mechanism screen:
  - scenes: cut_roasted_beef, flame_steak, sear_steak by default
  - config: fixed_budget_lora_route0_filemask_residual_flow_600k
  - train: 6000 iterations
  - eval: dependent eval at chkpnt6000.pth
  - sync: detached login-node watcher after eval, unless disabled

Options:
  --dry-run        Print sbatch and sync commands without submitting.
  --train-only     Submit only training; skip dependent eval and sync.
  --no-login-sync  Do not launch the detached login-node W&B sync watcher.
  -h, --help       Show this help.

Environment overrides:
  SCENES="cut_roasted_beef flame_steak sear_steak"
  CANDIDATE=fixed_budget_lora_route0_filemask_residual_flow_600k
  CKPT_ITER=6000
  TRAIN_TIME=05:00:00
  EVAL_TIME=00:45:00
  CPUS_PER_TASK=8
  WANDB_MODE=offline
  WANDB_PROJECT=adags
  WANDB_ENTITY=models-ku-leuven
  WANDB_GROUP=lora_flow_6000_YYYYMMDD
  EXCLUDE_NODES=lrdn1262,lrdn1386
EOF
}

DRY_RUN=0
WITH_EVAL=1
WITH_LOGIN_SYNC=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --train-only)
      WITH_EVAL=0
      WITH_LOGIN_SYNC=0
      shift
      ;;
    --no-login-sync)
      WITH_LOGIN_SYNC=0
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

SCENES="${SCENES:-cut_roasted_beef flame_steak sear_steak}"
CANDIDATE="${CANDIDATE:-fixed_budget_lora_route0_filemask_residual_flow_600k}"
CKPT_ITER="${CKPT_ITER:-6000}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
TRAIN_TIME="${TRAIN_TIME:-05:00:00}"
EVAL_TIME="${EVAL_TIME:-00:45:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-models-ku-leuven}"
WANDB_GROUP="${WANDB_GROUP:-lora_flow_6000_$(date +%Y%m%d)}"
WANDB_MODE="${WANDB_MODE:-offline}"

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

cfg_path="$REPO_ROOT/configs/n3v/${CANDIDATE}.yaml"
if [[ ! -f "$cfg_path" ]]; then
  echo "ERROR: missing config: $cfg_path" >&2
  exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="$PROJECT_ROOT/exp_index/lora_flow_6000_gate_${timestamp}.tsv"
sync_script="$PROJECT_ROOT/exp_index/lora_flow_6000_gate_sync_after_eval_${timestamp}.sh"
sync_log="$PROJECT_ROOT/exp_index/lora_flow_6000_gate_sync_after_eval_${timestamp}.log"
mkdir -p "$PROJECT_ROOT/exp_index"

method_family="lora_filemask_residual_flow"
wandb_extra_tags="phase:lora_flow_6000 screen:mechanism_flow baseline:fixed_budget_lora_route0_600k comparator:fixed_budget_lora_route0_filemask_residual_600k candidate:flow_6000 method:${method_family} budget:6000iter"

if [[ "$DRY_RUN" == "0" ]]; then
  printf "train_job_id\teval_job_id\tscene\tcandidate\tmethod_family\trun_id\trun_dir\tconfig\tckpt\tflow_dir\twandb_group\tsync_script\tsync_log\n" > "$manifest"
fi

eval_job_ids=()
run_dirs=()
common_sbatch_opts=()
if [[ -n "$EXCLUDE_NODES" ]]; then
  common_sbatch_opts+=(--exclude="$EXCLUDE_NODES")
fi

for scene in $SCENES; do
  data_path="$DATASET_ROOT/$scene"
  flow_dir="$data_path/flow"
  if [[ ! -d "$data_path" ]]; then
    echo "ERROR: missing dataset scene: $data_path" >&2
    exit 4
  fi
  if [[ ! -d "$flow_dir" ]] || ! find "$flow_dir" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
    echo "ERROR: missing flow priors under: $flow_dir" >&2
    exit 5
  fi

  run_id="${timestamp}_${scene}_${CANDIDATE}"
  run_dir="$PROJECT_ROOT/runs/$CANDIDATE/$run_id"
  ckpt_path="$run_dir/chkpnt${CKPT_ITER}.pth"
  log_prefix="n3v_${scene}_${CANDIDATE}"

  train_cmd=(
    sbatch --parsable
    -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
    "${common_sbatch_opts[@]}"
    -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
    -t "$TRAIN_TIME"
    -o "$PROJECT_ROOT/exp_index/${log_prefix}_train_%j.out"
    -e "$PROJECT_ROOT/exp_index/${log_prefix}_train_%j.err"
    --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="lora_flow_6000",RUN_ID="$run_id",RUN_LABEL="$CANDIDATE",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="lora_flow_6000",METHOD_FAMILY="$method_family",BUDGET_LABEL="6000iter"
    "$REPO_ROOT/scripts/run_leonardo.sh" train
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN train %s %s: ' "$scene" "$CANDIDATE"
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
    train_job_id="<train_job_id_${scene}>"
  else
    if ! train_job_id="$("${train_cmd[@]}")"; then
      echo "ERROR: failed to submit train job for $scene $CANDIDATE" >&2
      exit 6
    fi
    if [[ -z "$train_job_id" ]]; then
      echo "ERROR: train sbatch returned an empty job id for $scene $CANDIDATE" >&2
      exit 6
    fi
    echo "Submitted train $scene $CANDIDATE: job $train_job_id"
  fi

  eval_job_id=""
  if [[ "$WITH_EVAL" == "1" ]]; then
    eval_cmd=(
      sbatch --parsable
      --dependency="afterok:${train_job_id}"
      -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
      "${common_sbatch_opts[@]}"
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
      -t "$EVAL_TIME"
      -o "$PROJECT_ROOT/exp_index/${log_prefix}_eval${CKPT_ITER}_%j.out"
      -e "$PROJECT_ROOT/exp_index/${log_prefix}_eval${CKPT_ITER}_%j.err"
      --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="lora_flow_6000_eval",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$CANDIDATE",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags eval_ckpt:${CKPT_ITER}",EXPERIMENT_NAME="lora_flow_6000",METHOD_FAMILY="$method_family",BUDGET_LABEL="6000iter"
      "$REPO_ROOT/scripts/run_leonardo.sh" eval
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN eval %s %s: ' "$scene" "$CANDIDATE"
      printf '%q ' "${eval_cmd[@]}"
      printf '\n'
      eval_job_id="<eval_job_id_${scene}>"
    else
      if ! eval_job_id="$("${eval_cmd[@]}")"; then
        echo "ERROR: failed to submit eval job for $scene $CANDIDATE after $train_job_id" >&2
        exit 7
      fi
      if [[ -z "$eval_job_id" ]]; then
        echo "ERROR: eval sbatch returned an empty job id for $scene $CANDIDATE" >&2
        exit 7
      fi
      echo "Submitted eval $scene $CANDIDATE after $train_job_id: job $eval_job_id"
      eval_job_ids+=("$eval_job_id")
    fi
  fi

  run_dirs+=("$run_dir")

  if [[ "$DRY_RUN" == "0" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$train_job_id" "$eval_job_id" "$scene" "$CANDIDATE" "$method_family" "$run_id" "$run_dir" "$cfg_path" "$ckpt_path" "$flow_dir" "$WANDB_GROUP" "$sync_script" "$sync_log" \
      >> "$manifest"
  fi
done

if [[ "$DRY_RUN" == "0" && "$WITH_EVAL" == "1" && "$WITH_LOGIN_SYNC" == "1" ]]; then
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "# Auto-generated login-node sync watcher."
    echo "# Started detached by submit_lora_flow_6000_gate.sh; syncs only manifest runs after their eval jobs complete."
    printf 'MANIFEST=%q\n' "$manifest"
    printf 'PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
    printf 'REPO_ROOT=%q\n' "$REPO_ROOT"
    printf 'WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
    printf 'WANDB_ENTITY=%q\n' "$WANDB_ENTITY"
    echo 'export ADAGS_APPROVE_WANDB_UPLOAD=1'
    echo 'export WANDB_PROJECT WANDB_ENTITY'
    echo 'echo "sync watcher started at $(date -Iseconds) on $(hostname)"'
    echo 'status=0'
    echo 'while IFS=$'"'"'\t'"'"' read -r train_job eval_job scene candidate method_family run_id run_dir config ckpt flow_dir wandb_group sync_script sync_log; do'
    echo '  echo "waiting for eval job ${eval_job} (${scene})"'
    echo '  if "$REPO_ROOT/scripts/sync_wandb_run_dir.sh" --wait-job "$eval_job" --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" "$run_dir"; then'
    echo '    echo "synced ${scene} from ${run_dir}"'
    echo '  else'
    echo '    rc=$?'
    echo '    echo "ERROR: sync failed for ${scene} eval job ${eval_job} with status ${rc}" >&2'
    echo '    status=1'
    echo '  fi'
    echo 'done < <(tail -n +2 "$MANIFEST")'
    echo 'echo "sync watcher finished at $(date -Iseconds)"'
    echo 'exit "$status"'
  } > "$sync_script"
  chmod +x "$sync_script"

  nohup "$sync_script" > "$sync_log" 2>&1 &
  sync_pid="$!"
  disown "$sync_pid" 2>/dev/null || true
  echo "Started detached login-node W&B sync watcher: pid $sync_pid"
  echo "Sync script: $sync_script"
  echo "Sync log: $sync_log"
fi

if [[ "$DRY_RUN" == "0" ]]; then
  echo "Manifest: $manifest"
else
  echo "Dry run complete."
fi
