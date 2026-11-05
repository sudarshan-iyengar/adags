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

# fmd mono2 lane: Shape of Motion (vye16/shape-of-motion @579753e, code UNMODIFIED) on a real monocular sweep.
# STAGE=prep   : SoM input tree + moving-content masks (prep_som.py tree) + Depth Anything + alignment (prep_som.py depth)
# STAGE=tracks : BootsTAPIR all-pairs 2D tracks (SoM's compute_tracks_torch.py, query frames T0..T1-1)
# STAGE=train  : run_training.py data:iphone (published iPhone/DyCheck route) with --data.camera-type original
#                (KNOWN cameras from camera/*.json), then render_som.py at held-out cam00/05/17, all 300 frames.
# Required --export: STAGE, SCENE. tracks: T0, T1. train: RUN_ID.
set -uo pipefail
X=/leonardo_scratch/large/userexternal/siyengar/fmd
STAGE="${STAGE:?}"; SCENE="${SCENE:?}"
D=$X/data/mono_real_som/$SCENE
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/som/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
export TORCH_EXTENSIONS_DIR=$X/cache/torch_ext/$SLURM_JOB_ID  # per-job gsplat JIT build (concurrent builds in ~/.cache raced)
export HF_HOME=$X/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8
echo "job=$SLURM_JOB_ID node=$(hostname) stage=$STAGE scene=$SCENE start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $STAGE in
  prep)
    python -c 'import cuml, cupy; print("cuml import OK", cuml.__version__)' || echo "CUML IMPORT FAILED ON GPU NODE"
    python $X/agent/prep_som.py $SCENE $D tree || exit 1
    python $X/agent/prep_som.py $SCENE $D depth || exit 1 ;;
  tracks)
    python $X/agent/compute_tracks_range.py --image_dir $D/flow3d_preprocessed/rgb_preproc \
      --mask_dir $D/flow3d_preprocessed/track_anything/1x --out_dir $D/flow3d_preprocessed/2d_tracks/1x \
      --model_type bootstapir --ckpt_dir $X/weights/som --t0 ${T0:?} --t1 ${T1:?} || exit 1 ;;
  train)
    RUN=$X/runs/som/${RUN_ID:?}
    if [ "${RESUME:-0}" = "1" ] && [ -f "$RUN/checkpoints/last.ckpt" ]; then
      echo "RESUME from $RUN/checkpoints/last.ckpt (SoM's own checkpoint resume; init not recomputed)"; echo "resumed_by_job=$SLURM_JOB_ID $(date -Is)" >> $RUN/provenance.txt
    else
      [ -e "$RUN" ] && { echo "REFUSE: $RUN exists"; exit 3; }
    fi
    mkdir -p $RUN
    n=$(ls $D/flow3d_preprocessed/2d_tracks/1x | wc -l); echo "track files: $n"; [ "$n" -eq 90000 ] || { echo "TRACKS INCOMPLETE"; exit 4; }
    cd $X/repo/shape-of-motion
    { echo "run_id=$RUN_ID"; echo "job=$SLURM_JOB_ID"; echo "scene=$SCENE"; echo "data=$D"
      echo "commit=$(git rev-parse HEAD)"; echo "dirty=$(git status --porcelain --untracked-files=no | wc -l)"
      echo "cmd=python run_training.py --work-dir $RUN data:iphone --data.data-dir $D --data.camera-type original"
      echo "torch=$(python -c 'import torch;print(torch.__version__)') gsplat=$(python -c 'import gsplat;print(gsplat.__version__)')"
    } >> $RUN/provenance.txt
    python run_training.py --work-dir $RUN data:iphone --data.data-dir $D --data.camera-type original
    rc=$?; echo "train_exit=$rc finished=$(date -Is)" >> $RUN/provenance.txt
    [ $rc -eq 0 ] || exit $rc
    python $X/agent/render_som.py $SCENE $D $RUN $RUN/heldout || exit 5 ;;
esac
echo "end=$(date -Is)"
