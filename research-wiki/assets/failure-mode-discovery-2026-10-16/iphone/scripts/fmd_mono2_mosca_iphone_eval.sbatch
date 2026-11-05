#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/%x_%j.err
# fmd mono2: MoSca's own DyCheck metric step (eval_mosca_iphone.py) for every finished iPhone run whose tto_test is complete.
X=/leonardo_scratch/large/userexternal/siyengar/fmd
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/mosca/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
export TORCH_HOME=$X/cache/torch XLA_PYTHON_CLIENT_PREALLOCATE=false TORCH_EXTENSIONS_DIR=$X/cache/torch_ext/$SLURM_JOB_ID
cd $X/repo/mosca
for s in ${SCENES:-apple block paper-windmill space-out spin teddy wheel}; do
  WS=$X/runs/mosca_iphone/$s; LD=$(ls -d $WS/logs/iphone_fit_* 2>/dev/null | tail -1)
  [ -n "$LD" ] && [ -d $LD/tto_test ] || { echo "SKIP $s (no tto_test)"; continue; }
  echo "=== $s $LD tto_test=$(ls $LD/tto_test | wc -l) test_images=$(ls $WS/test_images | wc -l) $(date -Is)"
  python $X/agent/eval_mosca_iphone.py $WS $LD 2>&1 | grep -v Warn | grep -E "jax devices|COLUMNS|MEANS|mpsnr|mPSNR|Error|Traceback" | tail -n 8
done
echo "end=$(date -Is)"
