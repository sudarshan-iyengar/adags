#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_wandb_run_dir.sh [options] RUN_DIR

Sync offline W&B payloads for one local ADAGS run directory.

Options:
  --dry-run             Print payloads and command shape without uploading.
  --wait-job JOB_ID     Wait until a Slurm job completes before syncing.
  --project NAME        W&B project. Defaults to WANDB_PROJECT, then "adags".
  --entity NAME         W&B entity/team. Defaults to WANDB_ENTITY if set.
  --include-synced      Pass --include-synced to wandb sync.
  --sync-tensorboard    Include TensorBoard event files during sync.
  --skip-console        Skip uploading console logs.
  -h, --help            Show this help.

Safety:
  Set ADAGS_APPROVE_WANDB_UPLOAD=1 to allow a real external upload.
  Without it, this script refuses to sync unless --dry-run is used.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

PROJECT="${WANDB_PROJECT:-adags}"
ENTITY="${WANDB_ENTITY:-}"
WANDB_BIN="${WANDB_BIN:-}"
DRY_RUN=0
WAIT_JOB=""
INCLUDE_SYNCED=0
SYNC_TENSORBOARD=0
SKIP_CONSOLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --wait-job)
      WAIT_JOB="${2:?ERROR: --wait-job requires a value}"
      shift 2
      ;;
    --project)
      PROJECT="${2:?ERROR: --project requires a value}"
      shift 2
      ;;
    --entity)
      ENTITY="${2:?ERROR: --entity requires a value}"
      shift 2
      ;;
    --include-synced)
      INCLUDE_SYNCED=1
      shift
      ;;
    --sync-tensorboard)
      SYNC_TENSORBOARD=1
      shift
      ;;
    --skip-console)
      SKIP_CONSOLE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 1 ]]; then
  echo "ERROR: expected exactly one RUN_DIR." >&2
  usage >&2
  exit 2
fi

RUN_DIR="$1"
if [[ -z "$WANDB_BIN" ]]; then
  if command -v wandb >/dev/null 2>&1; then
    WANDB_BIN="wandb"
  elif [[ -x "$WORKSPACE_ROOT/envs/adags/bin/wandb" ]]; then
    WANDB_BIN="$WORKSPACE_ROOT/envs/adags/bin/wandb"
  else
    echo "ERROR: could not find wandb. Set WANDB_BIN or activate the adags environment." >&2
    exit 127
  fi
fi

wait_for_job() {
  local job_id="$1"
  echo "Waiting for Slurm job ${job_id} to finish before W&B sync..."
  while true; do
    local state=""
    if command -v sacct >/dev/null 2>&1; then
      state="$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | awk 'NF {print $1; exit}')"
    elif command -v squeue >/dev/null 2>&1; then
      if squeue -h -j "$job_id" >/dev/null 2>&1 && [[ -n "$(squeue -h -j "$job_id")" ]]; then
        state="RUNNING"
      else
        echo "ERROR: job ${job_id} is no longer in squeue and sacct is unavailable; cannot prove successful completion." >&2
        return 11
      fi
    else
      echo "ERROR: neither sacct nor squeue is available; cannot wait for job ${job_id}." >&2
      return 12
    fi
    case "$state" in
      COMPLETED)
        echo "Job ${job_id} completed."
        return 0
        ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
        echo "ERROR: job ${job_id} ended with state ${state}; refusing to sync." >&2
        return 10
        ;;
      "")
        echo "Job state unavailable for ${job_id}; sleeping."
        ;;
      *)
        echo "Job ${job_id} state: ${state}; sleeping."
        ;;
    esac
    sleep 60
  done
}

if [[ -n "$WAIT_JOB" ]]; then
  wait_for_job "$WAIT_JOB"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run dir does not exist after wait: $RUN_DIR" >&2
  exit 1
fi

mapfile -d '' RUN_DIRS < <(
  find -L "$RUN_DIR" \
    -maxdepth 3 \
    -type f \
    -path '*/wandb/offline-run-*/run-*.wandb' \
    -print0 |
  while IFS= read -r -d '' wandb_file; do
    printf '%s\0' "$(dirname "$wandb_file")"
  done |
  sort -zu
)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No offline W&B payloads found under: $RUN_DIR"
  exit 0
fi

echo "Found ${#RUN_DIRS[@]} offline W&B payload(s) under: $RUN_DIR"
printf '%s\n' "${RUN_DIRS[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ "${ADAGS_APPROVE_WANDB_UPLOAD:-}" != "1" ]]; then
  echo "ERROR: refusing external W&B upload without ADAGS_APPROVE_WANDB_UPLOAD=1." >&2
  echo "This sync transfers local run configs/logs/metadata to W&B." >&2
  exit 20
fi

for payload_dir in "${RUN_DIRS[@]}"; do
  cmd=("$WANDB_BIN" sync "--project" "$PROJECT" "--append" "--no-sync-tensorboard")
  if [[ -n "$ENTITY" ]]; then
    cmd+=("--entity" "$ENTITY")
  fi
  if [[ "$INCLUDE_SYNCED" -eq 1 ]]; then
    cmd+=("--include-synced")
  fi
  if [[ "$SYNC_TENSORBOARD" -eq 1 ]]; then
    cmd=("$WANDB_BIN" sync "--project" "$PROJECT" "--append" "--sync-tensorboard")
    if [[ -n "$ENTITY" ]]; then
      cmd+=("--entity" "$ENTITY")
    fi
    if [[ "$INCLUDE_SYNCED" -eq 1 ]]; then
      cmd+=("--include-synced")
    fi
  fi
  if [[ "$SKIP_CONSOLE" -eq 1 ]]; then
    cmd+=("--skip-console")
  fi
  cmd+=("$payload_dir")

  echo
  echo "Syncing: $payload_dir"
  "${cmd[@]}"
done

echo
echo "W&B sync complete for: $RUN_DIR"
