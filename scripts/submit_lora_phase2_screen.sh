#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_lora_phase2_screen.sh [--dry-run] [--train-only]

Submit the Phase 2 LoRA-only N3V screen:
  - phase2_lora_r16_a32_600k
  - phase2_lora_r8_a32_coeff00032_basis00004_600k

Each candidate is run on cut_roasted_beef, flame_steak, and sear_steak for
6000 iterations with scaffold and motion-aware densify disabled. By default,
the script also submits dependent eval-only jobs at chkpnt6000.pth so the
dynamic/static diagnostics and eval images are available for the gate.

Options:
  --dry-run     Print sbatch commands without submitting.
  --train-only  Submit only training jobs; skip dependent eval jobs.
  -h, --help    Show this help.

Environment overrides:
  SCENES="cut_roasted_beef flame_steak sear_steak"
  PHASE2_CANDIDATES="phase2_lora_r16_a32_600k phase2_lora_r8_a32_coeff00032_basis00004_600k"
  CKPT_ITER=6000
  TRAIN_TIME=05:30:00
  EVAL_TIME=00:50:00
  CPUS_PER_TASK=8
  WANDB_MODE=offline
  WANDB_PROJECT=adags
  WANDB_ENTITY=<optional>
  WANDB_GROUP=lora_failure_anatomy_phase2_20260619
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

SCENES="${SCENES:-cut_roasted_beef flame_steak sear_steak}"
PHASE2_CANDIDATES="${PHASE2_CANDIDATES:-phase2_lora_r16_a32_600k phase2_lora_r8_a32_coeff00032_basis00004_600k}"
CKPT_ITER="${CKPT_ITER:-6000}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/n3v}"
TRAIN_TIME="${TRAIN_TIME:-05:30:00}"
EVAL_TIME="${EVAL_TIME:-00:50:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-adags}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-lora_failure_anatomy_phase2_20260619}"
WANDB_MODE="${WANDB_MODE:-offline}"

PARTITION="${PARTITION:-boost_usr_prod}"
ACCOUNT="${ACCOUNT:-euhpc_d21_034}"
QOS="${QOS:-boost_qos_lprod}"

candidate_method_family() {
  case "$1" in
    phase2_lora_r16_a32_600k)
      echo "lora_phase2_r16_a32"
      ;;
    phase2_lora_r8_a32_coeff00032_basis00004_600k)
      echo "lora_phase2_r8_a32_coeff00032_basis00004"
      ;;
    *)
      echo "ERROR: unknown Phase 2 candidate: $1" >&2
      return 1
      ;;
  esac
}

candidate_tag() {
  case "$1" in
    phase2_lora_r16_a32_600k)
      echo "candidate:r16_a32"
      ;;
    phase2_lora_r8_a32_coeff00032_basis00004_600k)
      echo "candidate:r8_a32_coeff00032_basis00004"
      ;;
    *)
      echo "ERROR: unknown Phase 2 candidate: $1" >&2
      return 1
      ;;
  esac
}

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="$PROJECT_ROOT/exp_index/lora_phase2_screen_${timestamp}.tsv"
mkdir -p "$PROJECT_ROOT/exp_index"

if [[ "$DRY_RUN" == "0" ]]; then
  printf "train_job_id\teval_job_id\tscene\tcandidate\tmethod_family\trun_id\trun_dir\tconfig\tckpt\twandb_group\n" > "$manifest"
fi

submitted_train=0
submitted_eval=0
checked=0

for candidate in $PHASE2_CANDIDATES; do
  cfg_path="$REPO_ROOT/configs/n3v/${candidate}.yaml"
  if [[ ! -f "$cfg_path" ]]; then
    echo "ERROR: missing config: $cfg_path" >&2
    exit 3
  fi
  method_family="$(candidate_method_family "$candidate")"
  candidate_extra_tag="$(candidate_tag "$candidate")"

  for scene in $SCENES; do
    data_path="$DATASET_ROOT/$scene"
    if [[ ! -d "$data_path" ]]; then
      echo "ERROR: missing dataset scene: $data_path" >&2
      exit 4
    fi

    run_id="${timestamp}_${scene}_${candidate}"
    run_dir="$PROJECT_ROOT/runs/$candidate/$run_id"
    ckpt_path="$run_dir/chkpnt${CKPT_ITER}.pth"
    log_prefix="n3v_${scene}_${candidate}"
    wandb_extra_tags="phase:lora_failure_anatomy screen:phase2 baseline:fixed_budget_lora_route0_600k comparator:fixed_budget_lora_route0_dyn_600k ${candidate_extra_tag} method:${method_family} budget:600k"

    train_cmd=(
      sbatch --parsable
      -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
      -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
      -t "$TRAIN_TIME"
      -o "$PROJECT_ROOT/exp_index/${log_prefix}_train_%j.out"
      -e "$PROJECT_ROOT/exp_index/${log_prefix}_train_%j.err"
      --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="lora_phase2",RUN_ID="$run_id",RUN_LABEL="$candidate",DATASET_ROOT="$DATASET_ROOT",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags",EXPERIMENT_NAME="lora_failure_anatomy_phase2",METHOD_FAMILY="$method_family",BUDGET_LABEL="600k"
      "$REPO_ROOT/scripts/run_leonardo.sh" train
    )

    checked=$((checked + 1))
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN train %s %s: ' "$scene" "$candidate"
      printf '%q ' "${train_cmd[@]}"
      printf '\n'
      train_job_id="<train_job_id>"
    else
      if ! train_job_id="$("${train_cmd[@]}")"; then
        echo "ERROR: failed to submit train job for $scene $candidate" >&2
        exit 6
      fi
      if [[ -z "$train_job_id" ]]; then
        echo "ERROR: train sbatch returned an empty job id for $scene $candidate" >&2
        exit 6
      fi
      submitted_train=$((submitted_train + 1))
      echo "Submitted train $scene $candidate: job $train_job_id"
    fi

    eval_job_id=""
    if [[ "$WITH_EVAL" == "1" ]]; then
      eval_dependency="afterok:${train_job_id}"
      eval_cmd=(
        sbatch --parsable
        --dependency="$eval_dependency"
        -p "$PARTITION" -A "$ACCOUNT" --qos="$QOS"
        -N 1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --gres=gpu:1
        -t "$EVAL_TIME"
        -o "$PROJECT_ROOT/exp_index/${log_prefix}_eval${CKPT_ITER}_%j.out"
        -e "$PROJECT_ROOT/exp_index/${log_prefix}_eval${CKPT_ITER}_%j.err"
        --export=ALL,ADAGS_REPO_DIR="$REPO_ROOT",ADAGS_PROJECT_ROOT="$PROJECT_ROOT",SCENE="$scene",RUN_TAG="lora_phase2_eval",RUN_ID="$run_id",RUN_DIR="$run_dir",RUN_LABEL="$candidate",DATASET_ROOT="$DATASET_ROOT",CKPT_ITER="$CKPT_ITER",CONFIG="$cfg_path",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_GROUP="$WANDB_GROUP",WANDB_MODE="$WANDB_MODE",WANDB_EXTRA_TAGS="$wandb_extra_tags eval_ckpt:${CKPT_ITER}",EXPERIMENT_NAME="lora_failure_anatomy_phase2",METHOD_FAMILY="$method_family",BUDGET_LABEL="600k"
        "$REPO_ROOT/scripts/run_leonardo.sh" eval
      )

      if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY-RUN eval %s %s: ' "$scene" "$candidate"
        printf '%q ' "${eval_cmd[@]}"
        printf '\n'
        eval_job_id="<eval_job_id>"
      else
        if ! eval_job_id="$("${eval_cmd[@]}")"; then
          echo "ERROR: failed to submit eval job for $scene $candidate after $train_job_id" >&2
          exit 7
        fi
        if [[ -z "$eval_job_id" ]]; then
          echo "ERROR: eval sbatch returned an empty job id for $scene $candidate" >&2
          exit 7
        fi
        submitted_eval=$((submitted_eval + 1))
        echo "Submitted eval $scene $candidate after $train_job_id: job $eval_job_id"
      fi
    fi

    if [[ "$DRY_RUN" == "0" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$train_job_id" "$eval_job_id" "$scene" "$candidate" "$method_family" "$run_id" "$run_dir" "$cfg_path" "$ckpt_path" "$WANDB_GROUP" \
        >> "$manifest"
    fi
  done
done

expected=0
for _candidate in $PHASE2_CANDIDATES; do
  for _scene in $SCENES; do
    expected=$((expected + 1))
  done
done

if [[ "$checked" -ne "$expected" ]]; then
  echo "ERROR: expected $expected train jobs, checked $checked." >&2
  exit 5
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete: checked $checked train jobs."
else
  echo "Submitted $submitted_train train jobs and $submitted_eval eval jobs."
  echo "Manifest: $manifest"
fi
