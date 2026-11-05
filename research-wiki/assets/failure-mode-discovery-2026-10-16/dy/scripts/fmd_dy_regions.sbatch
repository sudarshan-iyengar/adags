#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.err
# fmd dy: region metrics (region_score.py) on the frozen masks for all finished p0_/d1_ runs (skips scored runs).
X=/leonardo_scratch/large/userexternal/siyengar/fmd
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7
source $X/envs/mosca/bin/activate
python $X/agent/dy/region_score.py p0_ & python $X/agent/dy/region_score.py d1_ & wait
echo "end=$(date -Is)"
