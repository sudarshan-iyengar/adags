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

# fmd mono2 lane: MoSca (JiahuiLei/MoSca @02404d2, code UNMODIFIED) on a real monocular sweep, published iPhone
# profiles: mosca_precompute.py profile/iphone/iphone_prep.yaml (dep_mode sensor = the prepared depth; RAFT flow,
# epipolar error, BootsTAPIR tracks) then mosca_reconstruct.py profile/iphone/iphone_fit.yaml (init_gt_camera True =
# KNOWN cameras, refined by MoSca's own small pose lr as published). The run ends with MoSca's own iPhone evaluation
# (pose-only test-time alignment, renders in <logdir>/tto_test). Then render_mosca.py renders WITHOUT that alignment
# (<logdir>/test). Only override: --epi_num_threads=8 (process count, 8 CPUs allocated).
# Required --export: SCENE, RUN_ID.
set -uo pipefail
# fmd mono2: render-only for a finished MoSca run (render_mosca.py, no test-time pose alignment) while the training job
# is still in MoSca's own TTO evaluation. Reads the saved photometric_*.pth only. Required --export: RUN_ID.
X=/leonardo_scratch/large/userexternal/siyengar/fmd; WS=$X/runs/mosca/${RUN_ID:?}
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/mosca/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6" TORCH_HOME=$X/cache/torch
LOGDIR=$(ls -d $WS/logs/iphone_fit_* | tail -1); echo "logdir=$LOGDIR job=$SLURM_JOB_ID"
[ -f $LOGDIR/photometric_s_model_native_add3.pth ] || { echo "NO FINAL MODEL"; exit 4; }
cd $X/repo/mosca && python $X/agent/render_mosca.py $WS $LOGDIR ./profile/iphone/iphone_fit.yaml
