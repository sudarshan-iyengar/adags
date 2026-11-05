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
X=/leonardo_scratch/large/userexternal/siyengar/fmd
SCENE="${SCENE:?}"; RUN_ID="${RUN_ID:?}"
WS=$X/runs/mosca/$RUN_ID
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/mosca/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
export TORCH_HOME=$X/cache/torch HF_HOME=$X/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "job=$SLURM_JOB_ID node=$(hostname) scene=$SCENE run=$RUN_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -e "$WS" ] && { echo "REFUSE: $WS exists"; exit 3; }
mkdir -p $X/runs/mosca
python $X/agent/prep_mosca.py $SCENE $X/data/mono_real_som/$SCENE $WS || exit 1
cd $X/repo/mosca
{ echo "run_id=$RUN_ID"; echo "job=$SLURM_JOB_ID"; echo "scene=$SCENE"; echo "commit=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain --untracked-files=no | wc -l)"; echo "torch=$(python -c 'import torch;print(torch.__version__)')"
  echo "prep_cfg_sha=$(sha256sum profile/iphone/iphone_prep.yaml | cut -c1-16) fit_cfg_sha=$(sha256sum profile/iphone/iphone_fit.yaml | cut -c1-16)"; } > $WS/provenance.txt
python mosca_precompute.py --ws $WS --cfg ./profile/iphone/iphone_prep.yaml --epi_num_threads=8 || { echo "PRECOMPUTE FAILED"; exit 2; }
echo "precompute done $(date -Is)"
python mosca_reconstruct.py --ws $WS --cfg ./profile/iphone/iphone_fit.yaml
rc=$?; echo "reconstruct_exit=$rc finished=$(date -Is)" | tee -a $WS/provenance.txt
LOGDIR=$(ls -d $WS/logs/iphone_fit_* | tail -1); echo "logdir=$LOGDIR" >> $WS/provenance.txt
[ -f $LOGDIR/photometric_s_model_native_add3.pth ] || { echo "NO FINAL MODEL"; exit 4; }
python $X/agent/render_mosca.py $WS $LOGDIR ./profile/iphone/iphone_fit.yaml || exit 5
echo "end=$(date -Is)"
