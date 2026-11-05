#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0066
#SBATCH --output=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D36_068/sri/proj_adags/repo/adags/logs/%x_%j.err
# Descriptive diagnostic (2026-10-16, plan cer-multi4d-early-tr-plan-2026-10-16): when are the transient rows that paint the
# object interior before the gap centred in time, and do they also paint the gap region? 9 base runs + the 3 early-arm run 1s.
# Read-only from repo/multi4d-cer-early-tr (its own built extensions directory); saved models only, nothing trained.
W=/leonardo_work/EUHPC_D36_068/sri/proj_adags; A=$W/agent-control/ars; O=$A/cer/return_diag/transient_time
export TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer_early_tr
source $W/agent-control/substrates/multi4d_env.sh >/dev/null 2>&1
cd $W/repo/multi4d-cer-early-tr || exit 1
echo "commit=$(git rev-parse HEAD) porcelain=$(git status --porcelain | wc -l)"
RUNS=""
for sc in ARS-L1-A0 ARS-L2-A0 ARS-L3-A0; do for r in 1 2 3; do RUNS="$RUNS $W/runs/multi4d/ars_w1_${sc}_multi4d_rep$r"; done
  RUNS="$RUNS $W/runs/multi4d_cer_early/ars_w1_${sc}_m4dearly_rep1"; done
mkdir -p $O
PYTHONDONTWRITEBYTECODE=1 python $A/cer/transient_time_interior.py --out $O/transient_time_interior.json --scratch $O/scratch --runs $RUNS
echo "rc=$? porcelain_after=$(git status --porcelain | wc -l)"
