#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_wandb_runs.sh [options] [runs_root]

Sync offline Weights & Biases runs stored under runs/**/wandb/offline-run-*.

Options:
  --dry-run             Print the W&B run directories that would be synced.
  --project NAME        W&B project. Defaults to WANDB_PROJECT, then "adags".
  --entity NAME         W&B entity/team. Defaults to WANDB_ENTITY if set.
  --no-append           Do not pass --append to wandb sync.
  --include-synced      Ask wandb to include runs already marked as synced.
  --sync-tensorboard    Include TensorBoard event files during sync.
  --skip-console        Skip uploading console logs.
  -h, --help            Show this help.

Environment:
  RUNS_ROOT             Default runs root when [runs_root] is omitted.
  WANDB_BIN             W&B executable to use.
  WANDB_API_KEY         Recommended for non-interactive login-node sync.

Examples:
  bash scripts/sync_wandb_runs.sh --dry-run
  WANDB_API_KEY=... bash scripts/sync_wandb_runs.sh --project adags
  bash scripts/sync_wandb_runs.sh --project adags --entity my-team ../../runs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

RUNS_ROOT="${RUNS_ROOT:-$WORKSPACE_ROOT/runs}"
PROJECT="${WANDB_PROJECT:-adags}"
ENTITY="${WANDB_ENTITY:-}"
WANDB_BIN="${WANDB_BIN:-}"
DRY_RUN=0
APPEND=1
INCLUDE_SYNCED=0
SYNC_TENSORBOARD=0
SKIP_CONSOLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --project)
      PROJECT="${2:?ERROR: --project requires a value}"
      shift 2
      ;;
    --entity)
      ENTITY="${2:?ERROR: --entity requires a value}"
      shift 2
      ;;
    --no-append)
      APPEND=0
      shift
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
      RUNS_ROOT="$1"
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "ERROR: unexpected extra arguments: $*" >&2
  usage >&2
  exit 2
fi

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

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "ERROR: runs root does not exist: $RUNS_ROOT" >&2
  exit 1
fi

mapfile -d '' RUN_DIRS < <(
  find -L "$RUNS_ROOT" \
    -maxdepth 5 \
    -type f \
    -path '*/wandb/offline-run-*/run-*.wandb' \
    -print0 |
  while IFS= read -r -d '' wandb_file; do
    printf '%s\0' "$(dirname "$wandb_file")"
  done |
  sort -zu
)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No offline W&B run payloads found under: $RUNS_ROOT"
  exit 0
fi

echo "Found ${#RUN_DIRS[@]} offline W&B run payload(s) under: $RUNS_ROOT"

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s\n' "${RUN_DIRS[@]}"
  exit 0
fi

if [[ -z "${WANDB_API_KEY:-}" && ( -z "${HOME:-}" || ! -f "${HOME}/.netrc" ) ]]; then
  echo "WARNING: WANDB_API_KEY is not set and no W&B login was detected." >&2
  echo "         If wandb is not already logged in, run: wandb login" >&2
fi

for run_dir in "${RUN_DIRS[@]}"; do
  cmd=("$WANDB_BIN" sync "--project" "$PROJECT")

  if [[ -n "$ENTITY" ]]; then
    cmd+=("--entity" "$ENTITY")
  fi
  if [[ $APPEND -eq 1 ]]; then
    cmd+=("--append")
  fi
  if [[ $INCLUDE_SYNCED -eq 1 ]]; then
    cmd+=("--include-synced")
  fi
  if [[ $SYNC_TENSORBOARD -eq 1 ]]; then
    cmd+=("--sync-tensorboard")
  else
    cmd+=("--no-sync-tensorboard")
  fi
  if [[ $SKIP_CONSOLE -eq 1 ]]; then
    cmd+=("--skip-console")
  fi

  cmd+=("$run_dir")

  echo
  echo "Syncing: $run_dir"
  "${cmd[@]}"
done

echo
echo "W&B sync complete."
