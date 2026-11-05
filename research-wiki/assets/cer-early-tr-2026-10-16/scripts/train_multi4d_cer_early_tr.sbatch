#!/bin/bash
#SBATCH --job-name=m4d-cer-tr
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=lrdn0066
#SBATCH --output=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.err

# Multi4D + CER, early gate with transient copies (plan research-wiki/operations/cer-multi4d-early-tr-plan-2026-10-16.md,
# section 3; EXPLORATORY). A copy of train_multi4d_cer_early.sbatch with these differences only: the multi4d-cer-early-tr
# worktree (branch cer-early-tr), its own extensions directory, runs under runs/multi4d_cer_early_tr, REQUIRES
# CER_TRANSIENT_GATE (0 or 1; passed as --cer_transient_gate and recorded), and pins the cer-early tree (7b30ef8) as well.
# The original header follows.
# Multi4D + CER, early gate (plan research-wiki/operations/cer-multi4d-early-gate-plan-2026-10-06.md, section 3). The
# train_multi4d_cer.sbatch launcher with these differences only: runs from the multi4d-cer-early worktree (branch
# cer-early), JIT-builds both extensions in its own extensions directory (never the one the cer or baseline jobs load),
# writes under runs/multi4d_cer_early, REQUIRES CER_BEGIN_ITER (no default; passed as --cer_begin_iter and recorded),
# records the sha256 of scene/gaussian_model.py and test_cer_early.py, and checks the early worktree, the cer tree and the
# pinned baseline tree before training (refuses on any mismatch or any failing git call) and after it (records them, exit
# code non-zero if any changed). The final-render wrapper is reused unchanged. One indivisible job.
# Required --export: TREE, CER_SCENE, RUN_ID, REP, CER_BEGIN_ITER, EXPECT_COMMIT. Optional: EXTRA (shortened schedule;
# only --iterations, --phase3_start_iter, --densify_until_iter, each with a positive integer; one line), RENDER_AT.
# Only the validated word arrays are executed, never the raw strings.

W=/leonardo_work/EUHPC_D36_068/sri/proj_adags
ARS=$W/agent-control/ars
TREE="${TREE:?TREE must be set}"
CER_SCENE="${CER_SCENE:?CER_SCENE (canonical ars/wave1/<scene>) must be set}"
REP="${REP:?REP must be set}"
RUN_ID="${RUN_ID:?RUN_ID must be set}"
CER_BEGIN_ITER="${CER_BEGIN_ITER:?CER_BEGIN_ITER (the begin iteration of the early gate) must be set: no default}"
if ! [[ "$CER_BEGIN_ITER" =~ ^[1-9][0-9]*$ ]]; then
  echo "refusing: CER_BEGIN_ITER=$CER_BEGIN_ITER is not a positive integer (-1 would silently run the legacy arm)"; exit 1
fi
CER_TRANSIENT_GATE="${CER_TRANSIENT_GATE:?CER_TRANSIENT_GATE (0 or 1) must be set: no default}"
case "$CER_TRANSIENT_GATE" in 0|1) ;; *) echo "refusing: CER_TRANSIENT_GATE=$CER_TRANSIENT_GATE"; exit 1;; esac
RENDER_AT="${RENDER_AT:-}"
EXTRA="${EXTRA:-}"
# one line only: a newline would be word-split differently by the validation and by an expansion
case "$EXTRA$RENDER_AT" in
  *$'\n'*|*$'\r'*) echo "refusing: EXTRA or RENDER_AT contains a newline or carriage return"; exit 1;;
esac
if [ -n "$RENDER_AT" ] && ! [[ "$RENDER_AT" =~ ^[0-9]+( [0-9]+)*$ ]]; then
  echo "refusing: RENDER_AT must be iteration numbers separated by spaces"; exit 1
fi
# EXTRA may only shorten the schedule: the options below take a positive integer each; every other option, the reserved
# ones (--cer_begin_iter, --cer_scene, --model_path, --expname, -s, --configs, --checkpoint_iterations) included, and the
# --opt=value spelling, are refused, so that nothing after --cer_begin_iter can override it
read -r -a EXTRA_WORDS <<< "$EXTRA"
i=0
while [ "$i" -lt "${#EXTRA_WORDS[@]}" ]; do
  case "${EXTRA_WORDS[$i]}" in
    --iterations|--phase3_start_iter|--densify_until_iter)
      if ! [[ "${EXTRA_WORDS[$((i+1))]:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "refusing: EXTRA option ${EXTRA_WORDS[$i]} needs a positive integer"; exit 1
      fi
      i=$((i+2));;
    *)
      echo "refusing: EXTRA option '${EXTRA_WORDS[$i]}' is not allowed (allowed: --iterations, --phase3_start_iter, --densify_until_iter; reserved: --cer_begin_iter --cer_scene --model_path --expname -s --configs --checkpoint_iterations)"; exit 1;;
  esac
done

export TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early_tr
source "$W/agent-control/substrates/multi4d_env.sh"
set -uo pipefail

RUN_DIR="$W/runs/multi4d_cer_early_tr/$RUN_ID"
SCENE=$(basename "$TREE")
EXPNAME="${SCENE}_cer_rep${REP}"
WT=$W/repo/multi4d-cer-early-tr
EARLYT=$W/repo/multi4d-cer-early
EXPECT_EARLY=7b30ef842fe59713da989322663fcc601bf31db6
CERT=$W/repo/multi4d-cer
PINNED=$W/repo/multi4d
EXPECT_CER=00357aa59f6ead1c746373547f67b843c5e35d85
EXPECT_PINNED=c483e82
# "HEAD BRANCH PORCELAIN-LINES" of a tree by read-only git calls; a failing git call is a failure, never "clean"
git_state() {
  local h b s n
  h=$(git --no-optional-locks -C "$1" rev-parse HEAD) || return 1
  b=$(git --no-optional-locks -C "$1" rev-parse --abbrev-ref HEAD) || return 1
  s=$(git --no-optional-locks -C "$1" status --porcelain) || return 1
  if [ -z "$s" ]; then n=0; else n=$(printf '%s\n' "$s" | wc -l); fi
  echo "$h $b $n"
}
# EXPECT_COMMIT (required; the submission scripts export it): refuse before creating anything unless the early worktree is
# still that commit, on branch cer-early and clean, the cer tree is at $EXPECT_CER on branch cer, clean, and the pinned
# baseline tree is at $EXPECT_PINNED*, clean, when the job starts (it may wait in the queue; the scorer would only catch
# it afterwards)
EXPECT_COMMIT="${EXPECT_COMMIT:?EXPECT_COMMIT (the reviewed cer-early commit) must be set}"
ST_EARLY=$(git_state "$WT") || { echo "refusing: git failed on $WT"; exit 1; }
ST_EARLYT=$(git_state "$EARLYT") || { echo "refusing: git failed on $EARLYT"; exit 1; }
ST_CER=$(git_state "$CERT") || { echo "refusing: git failed on $CERT"; exit 1; }
ST_PINNED=$(git_state "$PINNED") || { echo "refusing: git failed on $PINNED"; exit 1; }
[ "$ST_EARLY" = "$EXPECT_COMMIT cer-early-tr 0" ] || { echo "refusing: $WT is [$ST_EARLY], not $EXPECT_COMMIT on branch cer-early-tr with a clean tree"; exit 1; }
[ "$ST_EARLYT" = "$EXPECT_EARLY cer-early 0" ] || { echo "refusing: $EARLYT is [$ST_EARLYT], not $EXPECT_EARLY cer-early clean"; exit 1; }
[ "$ST_CER" = "$EXPECT_CER cer 0" ] || { echo "refusing: $CERT is [$ST_CER], not $EXPECT_CER on branch cer, clean"; exit 1; }
case "$ST_PINNED" in "$EXPECT_PINNED"*" 0") ;; *) echo "refusing: $PINNED is [$ST_PINNED], not $EXPECT_PINNED* clean"; exit 1;; esac
if [ -e "$RUN_DIR" ]; then echo "refusing: $RUN_DIR exists"; exit 1; fi
mkdir -p "$RUN_DIR"

cd "$WT" || exit 1
TRACKED_DIFF=$(git diff --stat HEAD) || { echo "refusing: git diff failed"; exit 1; }
if [ -z "$TRACKED_DIFF" ]; then TRACKED_DIFF_LINES=0; else TRACKED_DIFF_LINES=$(printf '%s\n' "$TRACKED_DIFF" | wc -l); fi
{
  echo "run_id=$RUN_ID"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "started=$(date -Is)"
  echo "tree=$TREE"
  echo "cer_scene=$CER_SCENE"
  echo "scene=$SCENE"
  echo "rep=$REP"
  echo "expname=$EXPNAME"
  echo "cer_begin_iter=$CER_BEGIN_ITER"
  echo "cer_transient_gate=$CER_TRANSIENT_GATE"
  echo "exploratory=yes"
  echo "substrate_branch=$(echo "$ST_EARLY" | cut -d' ' -f2)"
  echo "substrate_commit=$(echo "$ST_EARLY" | cut -d' ' -f1)"
  echo "substrate_porcelain=$(echo "$ST_EARLY" | cut -d' ' -f3)"
  echo "substrate_tracked_diff=$TRACKED_DIFF_LINES"
  echo "config_sha256=$(sha256sum arguments/dynerf.py | cut -d' ' -f1)"
  echo "trainpy_sha256=$(sha256sum train.py | cut -d' ' -f1)"
  echo "cerpy_sha256=$(sha256sum scene/cer.py | cut -d' ' -f1)"
  echo "cergate_sha256=$(sha256sum utils/cer_gate.py | cut -d' ' -f1)"
  echo "gaussian_model_sha256=$(sha256sum scene/gaussian_model.py | cut -d' ' -f1)"
  if [ -f test_cer_early_tr.py ]; then echo "test_cer_early_tr_sha256=$(sha256sum test_cer_early_tr.py | cut -d' ' -f1)"; fi
  echo "hybrid_forward_cu_sha256=$(sha256sum diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu | cut -d' ' -f1)"
  echo "event_spec_sha256=$(sha256sum "$CER_SCENE/event_spec.json" | cut -d' ' -f1)"
  echo "fused_ply_sha256=$(sha256sum "$TREE/fused.ply" | cut -d' ' -f1)"
  echo "poses_bounds_sha256=$(sha256sum "$TREE/poses_bounds.npy" | cut -d' ' -f1)"
  echo "wrapper_sha256=$(sha256sum "$ARS/m4d_final_render.py" | cut -d' ' -f1)"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
  echo "cer_tree_state=$ST_CER"
  echo "early_tree_state=$ST_EARLYT"
  echo "pinned_tree_state=$ST_PINNED"
  echo "render_at=${RENDER_AT:-final}"
  echo "extra=$EXTRA"
  echo "python=$(python -V 2>&1)"
  echo "torch=$(python -c 'import torch;print(torch.__version__)')"
} > "$RUN_DIR/provenance.txt"
cat "$RUN_DIR/provenance.txt"

# build (or load) both extensions now, so a build problem fails at t = 0, not at the first mass pass
PYTHONDONTWRITEBYTECODE=1 python -c "from scene import cer; exec('with cer.instrument_kernel(): pass'); print('instrument build ready')" || exit 1

RA=()
if [ -n "$RENDER_AT" ]; then read -r -a RA_WORDS <<< "$RENDER_AT"; RA=(--render_at "${RA_WORDS[@]}"); fi
python "$ARS/m4d_final_render.py" --repo "$WT" --out_dir "$RUN_DIR/final" ${RA[@]+"${RA[@]}"} -- \
  -s "$TREE" \
  --expname "$EXPNAME" \
  --configs arguments/dynerf.py \
  --model_path "$RUN_DIR/model" \
  --saving_folder "$RUN_DIR/results/" \
  --checkpoint_iterations 20000 \
  --cer_scene "$CER_SCENE" \
  --cer_begin_iter "$CER_BEGIN_ITER" --cer_transient_gate "$CER_TRANSIENT_GATE" ${EXTRA_WORDS[@]+"${EXTRA_WORDS[@]}"}
rc=$?

# after: the three trees must be exactly as before (read-only); a failing git call or any change fails the job
ST_EARLY_AFTER=$(git_state "$WT") || ST_EARLY_AFTER="git failed"
ST_CER_AFTER=$(git_state "$CERT") || ST_CER_AFTER="git failed"
ST_EARLYT_AFTER=$(git_state "$EARLYT") || ST_EARLYT_AFTER="git failed"
ST_PINNED_AFTER=$(git_state "$PINNED") || ST_PINNED_AFTER="git failed"
TREES_CHANGED=0
[ "$ST_EARLY_AFTER" = "$ST_EARLY" ] && [ "$ST_EARLYT_AFTER" = "$ST_EARLYT" ] && [ "$ST_CER_AFTER" = "$ST_CER" ] && [ "$ST_PINNED_AFTER" = "$ST_PINNED" ] || TREES_CHANGED=1
{
  echo "finished=$(date -Is)"
  echo "substrate_state_after=$ST_EARLY_AFTER"
  echo "cer_tree_state_after=$ST_CER_AFTER"
  echo "early_tree_state_after=$ST_EARLYT_AFTER"
  echo "pinned_tree_state_after=$ST_PINNED_AFTER"
  echo "trees_changed=$TREES_CHANGED"
  # the begin iteration as Multi4D itself resolved it (train.py writes cfg_args from the parsed arguments)
  echo "cfg_args_cer_begin_iter=$(grep -o 'cer_begin_iter=[-0-9]*' "$RUN_DIR/model/cfg_args" 2>/dev/null | head -n 1 | cut -d= -f2)"
  echo "cfg_args_cer_transient_gate=$(grep -o 'cer_transient_gate=[-0-9]*' "$RUN_DIR/model/cfg_args" 2>/dev/null | head -n 1 | cut -d= -f2)"
} >> "$RUN_DIR/provenance.txt"
cat "$RUN_DIR/model/cer_gates.json" 2>/dev/null
if [ "$TREES_CHANGED" != 0 ]; then echo "FAIL: a tree changed during the job"; [ "$rc" = 0 ] && rc=1; fi
# the exit code is recorded only after every check of the job, so a recorded 0 means the job itself exits 0
echo "exit_code=$rc" >> "$RUN_DIR/provenance.txt"
exit $rc
