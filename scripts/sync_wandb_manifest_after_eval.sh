#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_wandb_manifest_after_eval.sh [options] MANIFEST.tsv

Wait for each eval job in an ADAGS manifest, then sync that row's run_dir to W&B.
The manifest must have columns matching submit_lora_flow_6000_gate.sh output.

Options:
  --project NAME    W&B project. Defaults to WANDB_PROJECT, then "adags".
  --entity NAME     W&B entity/team. Defaults to WANDB_ENTITY if set.
  --dry-run         Print sync targets without uploading.
  -h, --help        Show this help.

Safety:
  Real upload still requires ADAGS_APPROVE_WANDB_UPLOAD=1.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${WANDB_PROJECT:-adags}"
ENTITY="${WANDB_ENTITY:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:?ERROR: --project requires a value}"
      shift 2
      ;;
    --entity)
      ENTITY="${2:?ERROR: --entity requires a value}"
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
  echo "ERROR: expected exactly one MANIFEST.tsv." >&2
  usage >&2
  exit 2
fi

MANIFEST="$1"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest does not exist: $MANIFEST" >&2
  exit 1
fi

echo "manifest sync started at $(date -Iseconds) on $(hostname)"
echo "manifest: $MANIFEST"

status=0
while IFS=$'\t' read -r train_job eval_job scene candidate method_family run_id run_dir config ckpt flow_dir wandb_group sync_script sync_log; do
  [[ -n "${scene:-}" ]] || continue
  echo
  echo "scene=${scene} train_job=${train_job} eval_job=${eval_job}"
  echo "run_dir=${run_dir}"

  cmd=("$SCRIPT_DIR/sync_wandb_run_dir.sh" --wait-job "$eval_job" --project "$PROJECT")
  if [[ -n "$ENTITY" ]]; then
    cmd+=(--entity "$ENTITY")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry-run)
  fi
  cmd+=("$run_dir")

  if "${cmd[@]}"; then
    echo "sync complete for ${scene}"
  else
    rc=$?
    echo "ERROR: sync failed for ${scene} eval job ${eval_job} with status ${rc}" >&2
    status=1
  fi
done < <(tail -n +2 "$MANIFEST")

echo
echo "manifest sync finished at $(date -Iseconds) with status ${status}"
exit "$status"
