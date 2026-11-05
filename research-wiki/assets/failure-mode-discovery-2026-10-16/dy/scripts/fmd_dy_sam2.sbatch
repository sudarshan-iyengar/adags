#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.err
# fmd dy P0: DyCheck held-out region masks with SAM2.1 (dy_sam2_regions.py), all 7 scenes. --export: OUT (dir), CLICKS (json path)
P=/leonardo_work/EUHPC_D36_068/sri/proj_adags; X=/leonardo_scratch/large/userexternal/siyengar/fmd
source $P/agent-control/s2/s2_env.sh; set -uo pipefail
cd $P/repo/sam2
for s in apple block paper-windmill space-out spin teddy wheel; do
  python $X/agent/dy/dy_sam2_regions.py $X/data/dycheck/mosca/data/iphone $CLICKS $OUT $s 2>&1 | grep -v Warn | tail -n 4
done
echo "end=$(date -Is)"
