#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.err
# fmd dy: FREEZE the DyCheck region masks (sha256 manifest + read-only) BEFORE any pilot result is read, then run the
# Pilot 2 day-0 band-share gate on the SAVED MoSca (2026-10-19) and Shape of Motion (2026-10-19/20) renders.
X=/leonardo_scratch/large/userexternal/siyengar/fmd; R=$X/data/dycheck/regions_v4_final
cp $X/agent/dy/clicks_v4.json $R/clicks_v4.json; cp $X/agent/dy/dy_sam2_regions.py $R/
cd $R && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 && echo "files=$(wc -l < MANIFEST.sha256) manifest_sha256=$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)" | tee FROZEN.txt && date -Is >> FROZEN.txt
chmod -R a-w $R
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7
source $X/envs/mosca/bin/activate
python $X/agent/dy/band_gate.py $R $X/runs/dy/band_gate_day0.json
echo "end=$(date -Is)"
