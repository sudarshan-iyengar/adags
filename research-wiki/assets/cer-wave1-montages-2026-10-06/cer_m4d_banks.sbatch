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
# CER wave 1, Multi4D: what each bank paints on cam00 (descriptive, looking material, never a score).
# SCENE=<ARS-Lx-A0> re-renders the 3 base and 3 CER runs of that scene; MONTAGE=1 instead builds the montages.
W=/leonardo_work/EUHPC_D36_068/sri/proj_adags; ARS=$W/agent-control/ars
OUT=$ARS/cer/m4d_banks
export TORCH_EXTENSIONS_DIR=$W/build/torch_extensions_multi4d_cer
source $W/agent-control/substrates/multi4d_env.sh >/dev/null 2>&1
set -uo pipefail
if [ "${MONTAGE:-0}" = 1 ]; then
  python $ARS/cer/cer_montages_m4d.py --banks $OUT/renders --out $ARS/reports/wave1/cer_compare_m4d
  echo "rc=$?"; ls -la $ARS/reports/wave1/cer_compare_m4d; exit
fi
cd "$W/repo/multi4d-cer" || exit 1
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is) commit=$(git rev-parse HEAD) porcelain=$(git status --porcelain | wc -l)"
echo "script_sha256=$(sha256sum $ARS/cer/cer_m4d_banks.py | cut -d' ' -f1)"
RUNS=""
for r in 1 2 3; do RUNS="$RUNS $W/runs/multi4d/ars_w1_${SCENE}_multi4d_rep$r"; done
for r in 1 2 3; do RUNS="$RUNS $W/runs/multi4d_cer/ars_w1_${SCENE}_m4dcer_rep$r"; done
PYTHONDONTWRITEBYTECODE=1 python $ARS/cer/cer_m4d_banks.py --out $OUT/renders --scratch $OUT/scratch \
  --frames 57 63 75 87 92 --runs $RUNS
rc=$?
echo "end=$(date -Is) rc=$rc porcelain_after=$(git status --porcelain | wc -l)"
exit $rc
