#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_phase9_depth_visibility.sh --run-id RUN_ID [--dry-run] [--retry]

Submits one registered Slurm entry from phase9_run_matrix_v1.json. The wrapper
refuses login/external entries, completed terminals, duplicate active jobs, and
unresolved prerequisites. It appends the captured job ID to the Phase 9
operational JOB_LEDGER.jsonl immediately after sbatch returns.
EOF
}

RUN_ID=""
DRY_RUN=0
RETRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?missing --run-id value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --retry) RETRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: --run-id is required." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${ADAGS_PROJECT_ROOT:-${WORK:?WORK must be set}/proj_adags}"
PYTHON_BIN="${PHASE9_PYTHON:-$PROJECT_ROOT/envs/adags/bin/python}"
MATRIX="${PHASE9_RUN_MATRIX:-$REPO_ROOT/configs/depth_visibility/phase9_run_matrix_v1.json}"
STATE_DIR="${PHASE9_STATE_DIR:-$PROJECT_ROOT/agent-control/phase9-depth-visibility-capacity}"
LEDGER="$STATE_DIR/JOB_LEDGER.jsonl"
JOB_SCRIPT="$REPO_ROOT/scripts/run_phase9_depth_visibility_job.sh"

if [[ ! -x "$PYTHON_BIN" || ! -f "$MATRIX" || ! -x "$JOB_SCRIPT" ]]; then
  echo "ERROR: Phase 9 Python, matrix, or job wrapper is unavailable." >&2
  exit 2
fi

mapfile -t RUN_FIELDS < <("$PYTHON_BIN" - "$MATRIX" "$RUN_ID" <<'PY'
import json,sys
matrix=json.load(open(sys.argv[1]))
matches=[item for item in matrix["runs"] if item["run_id"]==sys.argv[2]]
if len(matches)!=1:
    raise SystemExit(f"run entry count is {len(matches)}")
entry=matches[0]
scheduler=entry["scheduler"]
values=[
    entry["action"], scheduler["scheduler"], scheduler.get("partition") or "",
    scheduler.get("account") or "", scheduler.get("qos") or "",
    str(scheduler["nodes"]), str(scheduler["tasks_per_node"]),
    str(scheduler["cpus_per_task"]), str(scheduler["host_memory_gib"]),
    str(scheduler["gpus_per_node"]), str(scheduler["wall_minutes"]),
    scheduler.get("stdout") or "", scheduler.get("stderr") or "",
    entry["storage"]["output_root"] + "/resolved-execution.json",
    next(item["path"] for item in entry["expected_outputs"] if item["schema"]=="phase9-terminal-manifest-v1"),
    json.dumps(entry["prerequisites"],separators=(",",":")),
    "true" if entry.get("conditional") else "false",
]
print("\n".join(values))
PY
)
if [[ "${#RUN_FIELDS[@]}" -ne 17 ]]; then
  echo "ERROR: incomplete scheduler binding for $RUN_ID." >&2
  exit 2
fi
ACTION="${RUN_FIELDS[0]}"
SCHEDULER="${RUN_FIELDS[1]}"
PARTITION="${RUN_FIELDS[2]}"
ACCOUNT="${RUN_FIELDS[3]}"
QOS="${RUN_FIELDS[4]}"
NODES="${RUN_FIELDS[5]}"
TASKS="${RUN_FIELDS[6]}"
CPUS="${RUN_FIELDS[7]}"
MEMORY_GIB="${RUN_FIELDS[8]}"
GPUS="${RUN_FIELDS[9]}"
WALL_MINUTES="${RUN_FIELDS[10]}"
STDOUT_PATH="${RUN_FIELDS[11]}"
STDERR_PATH="${RUN_FIELDS[12]}"
EXECUTION_MANIFEST="${RUN_FIELDS[13]}"
TERMINAL_PATH="${RUN_FIELDS[14]}"
PREREQUISITES_JSON="${RUN_FIELDS[15]}"
CONDITIONAL="${RUN_FIELDS[16]}"

expand_path() {
  local value="$1"
  value="${value//\$WORK/$WORK}"
  printf '%s' "$value"
}
EXECUTION_MANIFEST="$(expand_path "$EXECUTION_MANIFEST")"
TERMINAL_PATH="$(expand_path "$TERMINAL_PATH")"
if [[ "$SCHEDULER" != "slurm" ]]; then
  echo "ERROR: $RUN_ID is a $SCHEDULER entry and must not be submitted with sbatch." >&2
  exit 2
fi
if [[ ! -f "$EXECUTION_MANIFEST" ]]; then
  echo "ERROR: resolved execution manifest is missing: $EXECUTION_MANIFEST" >&2
  exit 2
fi
if [[ -f "$TERMINAL_PATH" ]]; then
  terminal_status="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$TERMINAL_PATH")"
  if [[ "$terminal_status" == "completed" || "$terminal_status" == "not_applicable" ]]; then
    echo "ERROR: terminal already seals $RUN_ID as $terminal_status." >&2
    exit 2
  fi
  if [[ "$RETRY" != "1" ]]; then
    echo "ERROR: failed terminal exists; diagnose and pass --retry only for an admitted retry." >&2
    exit 2
  fi
fi

mkdir -p "$STATE_DIR" "$REPO_ROOT/logs"
touch "$LEDGER"
LAST_JOB_ID="$("$PYTHON_BIN" - "$LEDGER" "$RUN_ID" <<'PY'
import json,sys
latest=""
for raw in open(sys.argv[1]):
    try:
        row=json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit("malformed JOB_LEDGER.jsonl")
    if row.get("run_id")==sys.argv[2] and row.get("job_id") is not None:
        latest=str(row["job_id"])
print(latest)
PY
)"
if [[ -n "$LAST_JOB_ID" ]]; then
  queue_state="$(squeue -h -j "$LAST_JOB_ID" -o '%T' 2>/dev/null | head -n 1 || true)"
  accounting="$(
    sacct -n -X -j "$LAST_JOB_ID" --format=State,ExitCode 2>/dev/null       | awk 'NF{print $1" "$2; exit}' || true
  )"
  if [[ -n "$queue_state" ]]; then
    echo "ERROR: duplicate run blocked; job $LAST_JOB_ID is in squeue as $queue_state." >&2
    exit 2
  fi
  case "$accounting" in
    COMPLETED\ 0:0*) echo "ERROR: duplicate run blocked; job $LAST_JOB_ID completed successfully." >&2; exit 2 ;;
  esac
  if [[ "$RETRY" != "1" ]]; then
    echo "ERROR: prior job $LAST_JOB_ID exists with sacct='$accounting'; diagnose and use --retry only when justified." >&2
    exit 2
  fi
fi

DEPENDENCY_IDS="$("$PYTHON_BIN" - "$LEDGER" "$PREREQUISITES_JSON" <<'PY'
import json,os,subprocess,sys
ledger=[]
for raw in open(sys.argv[1]):
    if raw.strip():
        ledger.append(json.loads(raw))
ids=[]
for prerequisite in json.loads(sys.argv[2]):
    terminal=None
    for row in reversed(ledger):
        if row.get("run_id")==prerequisite:
            if row.get("terminal_path"):
                terminal=row["terminal_path"].replace("$WORK",os.environ.get("WORK",""))
            if row.get("job_id") is not None:
                job=str(row["job_id"])
                break
    else:
        job=None
    if terminal and os.path.isfile(terminal):
        try:
            if json.load(open(terminal)).get("status") in ("completed","not_applicable"):
                continue
        except Exception:
            pass
    if not job:
        raise SystemExit(f"unresolved prerequisite: {prerequisite}")
    state=subprocess.run(["squeue","-h","-j",job,"-o","%T"],text=True,capture_output=True).stdout.strip()
    if not state:
        raise SystemExit(f"prerequisite {prerequisite} job {job} is not active and has no successful terminal")
    ids.append(job)
print(":".join(ids))
PY
)"


"$PYTHON_BIN" - "$MATRIX" "$RUN_ID" "$EXECUTION_MANIFEST" "$DRY_RUN" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
matrix_path,run_id,execution_path,dry_run=sys.argv[1:]
matrix=json.load(open(matrix_path))
entry=next(item for item in matrix["runs"] if item["run_id"]==run_id)
execution=json.load(open(execution_path))
if execution.get("run_id")!=run_id or execution.get("action")!=entry["action"]:
    raise SystemExit("execution manifest run/action mismatch")
resolved=[]
for reference in entry["input_artifacts"]:
    path=reference["path"].replace("$WORK",os.environ["WORK"])
    source=pathlib.Path(path)
    if not source.is_file():
        raise SystemExit(f"required input artifact missing: {path}")
    if reference["schema"]=="phase9-terminal-manifest-v1":
        terminal=json.load(open(source))
        if terminal.get("run_id")!=reference["producer_run_id"]:
            raise SystemExit(f"terminal producer mismatch: {path}")
        if terminal.get("status") not in ("completed","not_applicable"):
            raise SystemExit(f"input terminal is not successful: {path}")
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        producer=next(item for item in matrix["runs"] if item["run_id"]==reference["producer_run_id"])
        terminal_ref=next(item for item in producer["expected_outputs"] if item["schema"]=="phase9-terminal-manifest-v1")
        terminal_path=terminal_ref["path"].replace("$WORK",os.environ["WORK"])
        terminal=json.load(open(terminal_path))
        if terminal.get("status") not in ("completed","not_applicable"):
            raise SystemExit(f"producer terminal is not successful: {terminal_path}")
        produced=[item for item in terminal.get("produced_artifacts",[]) if item.get("path")==reference["path"] or item.get("path")==path]
        if len(produced)!=1 or not produced[0].get("sha256"):
            raise SystemExit(f"producer terminal does not bind exact artifact: {path}")
        digest=produced[0]["sha256"]
    resolved.append({**reference,"sha256":digest,"status":"resolved_exact_before_submission"})
external=[]
for reference in entry["external_inputs"]:
    if not reference.get("path") or not reference.get("sha256"):
        raise SystemExit(f"required external input remains unresolved for {run_id}")
    path=reference["path"].replace("$WORK",os.environ["WORK"])
    source=pathlib.Path(path)
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=reference["sha256"]:
        raise SystemExit(f"external input missing or hash mismatch: {path}")
    external.append({**reference,"status":"resolved_exact_before_submission"})
execution["input_artifacts"]=resolved
execution["external_inputs"]=external
execution["input_binding_status"]="resolved_exact_before_submission"
encoded=(json.dumps(execution,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
if dry_run!="1":
    target=pathlib.Path(execution_path)
    descriptor,name=tempfile.mkstemp(prefix=target.name+".",dir=target.parent)
    with os.fdopen(descriptor,"wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name,target)
print(hashlib.sha256(encoded).hexdigest())
PY

days=$((WALL_MINUTES / 1440))
hours=$(((WALL_MINUTES % 1440) / 60))
minutes=$((WALL_MINUTES % 60))
if (( days > 0 )); then
  TIME_LIMIT="$(printf '%d-%02d:%02d:00' "$days" "$hours" "$minutes")"
else
  TIME_LIMIT="$(printf '%02d:%02d:00' "$hours" "$minutes")"
fi
SBATCH=(
  sbatch --parsable
  --job-name="$RUN_ID"
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --qos="$QOS"
  --nodes="$NODES"
  --ntasks-per-node="$TASKS"
  --cpus-per-task="$CPUS"
  --mem="${MEMORY_GIB}G"
  --time="$TIME_LIMIT"
  --output="$REPO_ROOT/$STDOUT_PATH"
  --error="$REPO_ROOT/$STDERR_PATH"
)
if (( GPUS > 0 )); then
  SBATCH+=(--gres="gpu:$GPUS")
fi
if [[ -n "$DEPENDENCY_IDS" ]]; then
  SBATCH+=(--dependency="afterok:$DEPENDENCY_IDS")
fi
SBATCH+=(
  --export="ALL,ADAGS_REPO_DIR=$REPO_ROOT,ADAGS_PROJECT_ROOT=$PROJECT_ROOT,PHASE9_RUN_ID=$RUN_ID,PHASE9_ACTION=$ACTION,PHASE9_EXECUTION_MANIFEST=$EXECUTION_MANIFEST"
  "$JOB_SCRIPT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY-RUN: '
  printf '%q ' "${SBATCH[@]}"
  printf '\n'
  exit 0
fi

JOB_ID="$("${SBATCH[@]}")"
JOB_ID="${JOB_ID%%;*}"
"$PYTHON_BIN" - "$LEDGER" "$RUN_ID" "$ACTION" "$JOB_ID" "$EXECUTION_MANIFEST" "$TERMINAL_PATH" "$RETRY" "$CONDITIONAL" <<'PY'
import datetime,json,sys
path,run_id,action,job_id,execution,terminal,retry,conditional=sys.argv[1:]
row={
 "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),
 "event":"submitted","run_id":run_id,"action":action,"job_id":int(job_id),
 "execution_manifest":execution,"terminal_path":terminal,
 "retry":int(retry),"conditional":conditional=="true",
 "squeue_state":"SUBMITTED","sacct_state":None,"exit_code":None,
}
with open(path,"a",encoding="utf-8") as handle:
    handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
PY
echo "Submitted $RUN_ID as Slurm job $JOB_ID"
echo "Ledger: $LEDGER"
