#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0066
#SBATCH --output=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/cer_m4d_tests_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/cer_m4d_tests_%j.err

# CER Multi4D early gate with transient copies: tests F0-F9 (plan research-wiki/operations/cer-multi4d-early-tr-plan-2026-10-16.md
# section 4; exploratory). A copy of cer_tests_m4d_early.sbatch: runs from the multi4d-cer-early-tr worktree (branch
# cer-early-tr, EXPECT_EARLYTR required) with its own extensions directory; TEST_SCRIPT defaults to test_cer_early_tr.py;
# the cer-early tree is exported as CER_TREE_EARLY and pinned to 7b30ef8; T7_ARGS also allows --f9_run --f9_metrics.
# The original header follows.
# CER Multi4D early-gate tests (plan research-wiki/operations/cer-multi4d-early-gate-plan-2026-10-06.md section 4).
# cer_tests_m4d.sbatch with these differences only: runs from the multi4d-cer-early worktree and JIT-builds in its own
# extensions directory; TEST_SCRIPT chooses the script (default test_cer_early.py for E1-E9; test_cer.py for E0, the
# existing T0-T6 on this tree); the `cer` tree at $W/repo/multi4d-cer is exported as CER_TREE_LEGACY for E1(b) and E1(c),
# which run it as a read-only subprocess whose JIT builds go to a fresh directory under OUT (never to a shared cache).
# The early worktree (EXPECT_EARLY, required), the cer tree and the pinned tree are checked before the tests and again
# after them (read-only git calls, each call's exit status checked: a failing call is a refusal before and a failure
# after, never "clean"); the job refuses to start, and fails, on any difference.
# SCENES: space-separated; TESTS: colon-separated; REP: base run (default 1); RUN overrides the run dir;
# T7_ARGS: extra arguments for the script, one line, only from this list (each takes one value): --e9_run --e9_metrics
# --t7_iter --t7_metrics --begin_iter; every other option (--run --tree --cer_scene --pinned --out --tests --cer_tree ...)
# is refused. Only the validated word array is executed, never the raw string.
W=/leonardo_work/EUHPC_D36_068/sri/proj_adags
S=/leonardo_scratch/large/userexternal/siyengar/ars/wave1
SCENES="${SCENES:?}"; TESTS="${TESTS:?}"; OUT="${OUT:?}"; REP="${REP:-1}"
EXPECT_EARLYTR="${EXPECT_EARLYTR:?EXPECT_EARLYTR (the commit of the early-tr worktree under test) must be set}"
EXPECT_EARLY=7b30ef842fe59713da989322663fcc601bf31db6
TEST_SCRIPT="${TEST_SCRIPT:-test_cer_early_tr.py}"
case "$TEST_SCRIPT" in test_cer_early_tr.py|test_cer_early.py|test_cer.py) ;; *) echo "refusing: TEST_SCRIPT=$TEST_SCRIPT"; exit 1;; esac
# one line each: a newline would be word-split differently by the validation and by an expansion
case "${T7_ARGS:-}$SCENES$TESTS$REP" in
  *$'\n'*|*$'\r'*) echo "refusing: T7_ARGS, SCENES, TESTS or REP contains a newline or carriage return"; exit 1;;
esac
SCENES_RE='^[A-Za-z0-9_ -]+$'
[[ "$SCENES" =~ $SCENES_RE ]] || { echo "refusing: SCENES"; exit 1; }
[[ "$TESTS" =~ ^[A-Za-z0-9_:,-]+$ ]] || { echo "refusing: TESTS"; exit 1; }
[[ "$REP" =~ ^[0-9]+$ ]] || { echo "refusing: REP"; exit 1; }
EXPECT_CER=00357aa59f6ead1c746373547f67b843c5e35d85
EXPECT_PINNED=c483e82
EARLYTR=$W/repo/multi4d-cer-early-tr; EARLY=$W/repo/multi4d-cer-early; CERT=$W/repo/multi4d-cer; PINNED=$W/repo/multi4d
read -r -a ARG_WORDS <<< "${T7_ARGS:-}"
i=0
while [ "$i" -lt "${#ARG_WORDS[@]}" ]; do
  case "${ARG_WORDS[$i]}" in
    --e9_run|--e9_metrics|--t7_iter|--t7_metrics|--begin_iter|--f9_run|--f9_metrics)
      if [ -z "${ARG_WORDS[$((i+1))]:-}" ] || [[ "${ARG_WORDS[$((i+1))]}" == -* ]]; then echo "refusing: T7_ARGS option ${ARG_WORDS[$i]} needs a value"; exit 1; fi
      i=$((i+2));;
    *) echo "refusing: T7_ARGS option '${ARG_WORDS[$i]}' is not allowed"; exit 1;;
  esac
done
# "HEAD BRANCH PORCELAIN-LINES" of a tree by read-only git calls; returns non-zero if any git call fails
tree_state() {
  local h b s n
  h=$(git --no-optional-locks -C "$1" rev-parse HEAD) || return 1
  b=$(git --no-optional-locks -C "$1" rev-parse --abbrev-ref HEAD) || return 1
  s=$(git --no-optional-locks -C "$1" status --porcelain) || return 1
  if [ -z "$s" ]; then n=0; else n=$(printf '%s\n' "$s" | wc -l); fi
  echo "$h $b $n"
}
check_trees() {  # prints the four states; returns 1 unless each tree is exactly what it must be and every git call worked
  local t e c p ok=0
  t=$(tree_state "$EARLYTR") || { echo "MISMATCH early-tr: git failed"; t="git failed"; ok=1; }
  e=$(tree_state "$EARLY") || { echo "MISMATCH early: git failed"; e="git failed"; ok=1; }
  c=$(tree_state "$CERT") || { echo "MISMATCH cer: git failed"; c="git failed"; ok=1; }
  p=$(tree_state "$PINNED") || { echo "MISMATCH pinned: git failed"; p="git failed"; ok=1; }
  echo "trees $1: early-tr=[$t] early=[$e] cer=[$c] pinned=[$p]"
  [ "$t" = "$EXPECT_EARLYTR cer-early-tr 0" ] || { echo "MISMATCH early-tr: want $EXPECT_EARLYTR cer-early-tr 0"; ok=1; }
  [ "$e" = "$EXPECT_EARLY cer-early 0" ] || { echo "MISMATCH early: want $EXPECT_EARLY cer-early 0"; ok=1; }
  [ "$c" = "$EXPECT_CER cer 0" ] || { echo "MISMATCH cer: want $EXPECT_CER cer 0"; ok=1; }
  case "$p" in "$EXPECT_PINNED"*" 0") ;; *) echo "MISMATCH pinned: want $EXPECT_PINNED* 0"; ok=1;; esac
  return $ok
}
export TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early_tr
source "$W/agent-control/substrates/multi4d_env.sh"
export TORCH_EXTENSIONS_DIR_PINNED=$W/build/torch_extensions_multi4d
export CER_TREE_LEGACY=$CERT
export CER_TREE_EARLY=$EARLY
set -uo pipefail
cd "$EARLYTR" || exit 1
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) script=$TEST_SCRIPT tests=$TESTS scenes=$SCENES rep=$REP ext=$TORCH_EXTENSIONS_DIR"
check_trees before || { echo "refusing: a tree is not what the tests are pinned to"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader
rc=0
for sc in $SCENES; do
  echo "=== $sc"
  PYTHONDONTWRITEBYTECODE=1 python "$TEST_SCRIPT" --run "${RUN:-$W/runs/multi4d/ars_w1_${sc}_multi4d_rep${REP}}" \
    --tree "$S/trees/multi4d/$sc" --cer_scene "$S/$sc" --pinned "$PINNED" \
    --out "$OUT/$sc" --tests "${TESTS//:/,}" ${ARG_WORDS[@]+"${ARG_WORDS[@]}"} || rc=1
done
check_trees after || { echo "FAIL: a tree changed during the tests"; rc=1; }
echo "end=$(date -Is) rc=$rc"
exit $rc
