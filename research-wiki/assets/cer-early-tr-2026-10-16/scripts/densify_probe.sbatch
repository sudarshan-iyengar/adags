#!/bin/bash
#SBATCH --job-name=m4d-dprobe
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:50:00
#SBATCH --exclude=lrdn0066
#SBATCH --output=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.err

# Task B densification probe (plan research-wiki/operations/cer-multi4d-early-tr-plan-2026-10-16.md, section 9;
# exploratory diagnostic, never evidence). Runs agent-control/ars/densify_probe.py around the reviewed train.py of the
# multi4d-cer-early worktree (cer-early, pinned by EXPECT_COMMIT, clean), published schedule, bottle same-pose.
# MODE=early: train.py's own early CER (begin CER_BEGIN_ITER). MODE=shadow: CER off, labels attached by the probe at fine
# SHADOW_ITER without arming the gate. Loads the early extensions directory read-only (sources unchanged, no rebuild).
# Required --export: MODE, TREE, CER_SCENE, RUN_ID, EXPECT_COMMIT; early: CER_BEGIN_ITER; shadow: SHADOW_ITER.
W=/leonardo_work/EUHPC_D36_068/sri/proj_adags
ARS=$W/agent-control/ars
MODE="${MODE:?MODE (early|shadow)}"
TREE="${TREE:?TREE}"
CER_SCENE="${CER_SCENE:?CER_SCENE (canonical ars/wave1/<scene>)}"
RUN_ID="${RUN_ID:?RUN_ID}"
EXPECT_COMMIT="${EXPECT_COMMIT:?EXPECT_COMMIT}"
case "$MODE" in
  early) CER_BEGIN_ITER="${CER_BEGIN_ITER:?CER_BEGIN_ITER}"; [[ "$CER_BEGIN_ITER" =~ ^[1-9][0-9]*$ ]] || { echo "refusing: CER_BEGIN_ITER"; exit 1; }
         CER_ARGS=(--cer_scene "$CER_SCENE" --cer_begin_iter "$CER_BEGIN_ITER"); PROBE_ARGS=();;
  shadow) SHADOW_ITER="${SHADOW_ITER:?SHADOW_ITER}"; [[ "$SHADOW_ITER" =~ ^[1-9][0-9]*$ ]] || { echo "refusing: SHADOW_ITER"; exit 1; }
         CER_ARGS=(); PROBE_ARGS=(--shadow_iter "$SHADOW_ITER");;
  *) echo "refusing: MODE=$MODE"; exit 1;;
esac

export TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early
source "$W/agent-control/substrates/multi4d_env.sh"
set -uo pipefail

WT=$W/repo/multi4d-cer-early
git_state() {
  local h b s n
  h=$(git --no-optional-locks -C "$1" rev-parse HEAD) || return 1
  b=$(git --no-optional-locks -C "$1" rev-parse --abbrev-ref HEAD) || return 1
  s=$(git --no-optional-locks -C "$1" status --porcelain) || return 1
  if [ -z "$s" ]; then n=0; else n=$(printf '%s\n' "$s" | wc -l); fi
  echo "$h $b $n"
}
ST=$(git_state "$WT") || { echo "refusing: git failed on $WT"; exit 1; }
[ "$ST" = "$EXPECT_COMMIT cer-early 0" ] || { echo "refusing: $WT is [$ST], not $EXPECT_COMMIT cer-early clean"; exit 1; }
for x in diff_gaussian_rasterization_hybrid diff_gaussian_rasterization_hybrid_cer; do
  [ -f "$TORCH_EXTENSIONS_DIR/$x/$x.so" ] || { echo "refusing: extension $x not built in $TORCH_EXTENSIONS_DIR"; exit 1; }
done
RUN_DIR="$W/runs/multi4d_cer_early_probe/$RUN_ID"
if [ -e "$RUN_DIR" ]; then echo "refusing: $RUN_DIR exists"; exit 1; fi
mkdir -p "$RUN_DIR"
SCENE=$(basename "$TREE")
{
  echo "run_id=$RUN_ID"; echo "exploratory=task B densification probe"; echo "mode=$MODE"
  echo "slurm_job_id=$SLURM_JOB_ID"; echo "node=$(hostname)"; echo "started=$(date -Is)"
  echo "tree=$TREE"; echo "cer_scene=$CER_SCENE"; echo "substrate_state=$ST"
  echo "probe_sha256=$(sha256sum "$ARS/densify_probe.py" | cut -d' ' -f1)"
  echo "trainpy_sha256=$(sha256sum "$WT/train.py" | cut -d' ' -f1)"
  echo "cer_args=${CER_ARGS[*]:-}"; echo "probe_args=${PROBE_ARGS[*]:-}"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
} > "$RUN_DIR/provenance.txt"
cat "$RUN_DIR/provenance.txt"

cd "$WT" || exit 1
python "$ARS/densify_probe.py" --repo "$WT" --out "$RUN_DIR/probe" --scene "$CER_SCENE" --mode "$MODE" \
  ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"} -- \
  -s "$TREE" --expname "${SCENE}_dprobe_${MODE}" --configs arguments/dynerf.py \
  --model_path "$RUN_DIR/model" --saving_folder "$RUN_DIR/results/" \
  ${CER_ARGS[@]+"${CER_ARGS[@]}"}
rc=$?
ST_AFTER=$(git_state "$WT") || ST_AFTER="git failed"
[ "$ST_AFTER" = "$ST" ] || { echo "FAIL: the worktree changed"; [ "$rc" = 0 ] && rc=1; }
# the exit code is recorded only after every check, so a recorded 0 means the job's own exit status is 0
{ echo "finished=$(date -Is)"; echo "substrate_state_after=$ST_AFTER"; echo "exit_code=$rc"; } >> "$RUN_DIR/provenance.txt"
exit $rc
