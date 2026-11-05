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
#SBATCH --output=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.err

# Test of the task-B densification probe (plan cer-multi4d-early-tr-plan-2026-10-16.md section 9). Runs
# agent-control/ars/test_densify_probe.py from repo/multi4d-cer-early (pinned by EXPECT_COMMIT, clean before and after) on
# the bottle same-pose base run 1, loading the early extensions directory read-only. Required --export: OUT, EXPECT_COMMIT.
W=/leonardo_work/EUHPC_D36_068/sri/proj_adags
ARS=$W/agent-control/ars
S=/leonardo_scratch/large/userexternal/siyengar/ars/wave1
OUT="${OUT:?OUT}"; EXPECT_COMMIT="${EXPECT_COMMIT:?EXPECT_COMMIT}"
export TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early
source "$W/agent-control/substrates/multi4d_env.sh"
set -uo pipefail
WT=$W/repo/multi4d-cer-early
st() { echo "$(git --no-optional-locks -C $WT rev-parse HEAD) $(git --no-optional-locks -C $WT status --porcelain | wc -l)"; }
[ "$(st)" = "$EXPECT_COMMIT 0" ] || { echo "refusing: $WT is [$(st)]"; exit 1; }
mkdir -p "$OUT"
cd "$WT" || exit 1
echo "job=$SLURM_JOB_ID node=$(hostname) probe_sha=$(sha256sum $ARS/densify_probe.py | cut -d' ' -f1) test_sha=$(sha256sum $ARS/test_densify_probe.py | cut -d' ' -f1)"
PYTHONDONTWRITEBYTECODE=1 python "$ARS/test_densify_probe.py" --probe "$ARS/densify_probe.py" \
  --run "$W/runs/multi4d/ars_w1_ARS-L1-A0_multi4d_rep1" --tree "$S/trees/multi4d/ARS-L1-A0" --scene "$S/ARS-L1-A0" \
  --out "$OUT/test_densify_probe.json"
rc=$?
[ "$(st)" = "$EXPECT_COMMIT 0" ] || { echo "FAIL: worktree changed"; rc=1; }
exit $rc
