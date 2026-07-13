#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/test_wandb_online_compute.sh [--run]

Submit a tiny Leonardo compute-node job that verifies outbound HTTPS access
and performs a minimal W&B online run.

Options:
  --run       Internal mode used by the submitted Slurm job.
  -h, --help  Show this help.

Environment:
  WANDB_API_KEY     Optional if `wandb login` has written ~/.netrc.
  WANDB_PROJECT     Defaults to "adags".
  WANDB_ENTITY      Optional W&B entity/team.
  WANDB_TEST_GROUP  Defaults to "connectivity-test".

Example:
  export WANDB_API_KEY=...
  bash scripts/test_wandb_online_compute.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
EXP_INDEX="${WORKSPACE_ROOT}/exp_index"
ENV_FILE="${EXP_INDEX}/leonardo_env.sh"

setup_leonardo_env() {
  if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
  fi
  cd "$REPO_ROOT"
}

RUN_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_MODE" -eq 0 ]]; then
  setup_leonardo_env
  mkdir -p "$EXP_INDEX"

  JOB_ID="$(
    sbatch --parsable \
      --job-name=test_wandb_online \
      -p boost_usr_prod -A euhpc_d21_034 --qos=boost_qos_lprod \
      -N 1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 \
      -t 00:10:00 \
      -o "${EXP_INDEX}/wandb_online_test_%j.out" \
      -e "${EXP_INDEX}/wandb_online_test_%j.err" \
      --export=ALL \
      "$0" --run
  )"

  echo "Submitted W&B online connectivity test as job ${JOB_ID}"
  echo "stdout: ${EXP_INDEX}/wandb_online_test_${JOB_ID}.out"
  echo "stderr: ${EXP_INDEX}/wandb_online_test_${JOB_ID}.err"
  echo
  echo "Watch with:"
  echo "  squeue -j ${JOB_ID}"
  echo "  tail -f ${EXP_INDEX}/wandb_online_test_${JOB_ID}.out"
  exit 0
fi

echo "timestamp: $(date -Iseconds)"
echo "host: $(hostname)"
echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
echo "work: ${WORK:-unset}"
echo

setup_leonardo_env

echo "python: $(command -v python || true)"
python -V
echo

echo "Checking DNS for api.wandb.ai..."
getent hosts api.wandb.ai
echo

echo "Checking HTTPS reachability for https://api.wandb.ai ..."
if command -v curl >/dev/null 2>&1; then
  curl --fail --silent --show-error --location --max-time 20 \
    --output /dev/null \
    --write-out 'http_code=%{http_code} remote_ip=%{remote_ip} total_time=%{time_total}\n' \
    https://api.wandb.ai
else
  python - <<'PY'
import urllib.request

with urllib.request.urlopen("https://api.wandb.ai", timeout=20) as response:
    print(f"http_code={response.status}")
PY
fi
echo

if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
  echo "ERROR: no W&B credentials found, so the real W&B online run test cannot be performed." >&2
  echo "Set WANDB_API_KEY or run \`wandb login\` on the login node, then rerun this script." >&2
  exit 4
fi

echo "Running minimal W&B online init/log/finish test..."
python - <<'PY'
import os
import socket
import time

import wandb

project = os.getenv("WANDB_PROJECT", "adags")
entity = os.getenv("WANDB_ENTITY") or None
group = os.getenv("WANDB_TEST_GROUP", "connectivity-test")
job_id = os.getenv("SLURM_JOB_ID", "no-slurm-job")
name = f"compute-online-test-{job_id}"

run = wandb.init(
    project=project,
    entity=entity,
    group=group,
    name=name,
    mode="online",
    config={
        "host": socket.gethostname(),
        "slurm_job_id": job_id,
        "purpose": "compute node online connectivity test",
    },
)
run.log({"connectivity/ok": 1, "connectivity/unix_time": time.time()}, step=0)
run.finish()

print(f"W&B online test completed: {run.url}")
PY

echo
echo "SUCCESS: compute node can reach W&B in online mode."
