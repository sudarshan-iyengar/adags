#!/bin/bash
#SBATCH --account=euhpc_d36_068
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/siyengar/fmd/logs/dy/%x_%j.err
# fmd dy (P0 noise floor + pilots D1/H1): one MoSca fit on DyCheck iPhone under its published route.
# The workspace symlinks the scene's PUBLISHED preprocessing (iphone_prep.yaml, run 2026-10-19 in runs/mosca_iphone/<scene>),
# so only the fit varies between reruns/variants. Then mosca_reconstruct.py with iphone_fit.yaml (+ EXTRA overrides),
# whose own test_main does the pose-only TTO renders (tto_test/); then eval_mosca_iphone.py (MoSca's metric step, jax
# initialised before torch) and count_prims.py (active primitive counts).
# --export: SCENE, RUN_ID; optional REPO (code copy; default original), EXTRA (MoSca CLI overrides, colon-separated
# key=value list, e.g. gs_nn_fusion=8), FITCFG.
set -uo pipefail
X=/leonardo_scratch/large/userexternal/siyengar/fmd
SCENE="${SCENE:?}"; RUN_ID="${RUN_ID:?}"; REPO="${REPO:-$X/repo/mosca}"; FITCFG="${FITCFG:-./profile/iphone/iphone_fit.yaml}"
PREP=$X/runs/mosca_iphone/$SCENE; WS=$X/runs/dy/$RUN_ID
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/mosca/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
export TORCH_HOME=$X/cache/torch XLA_PYTHON_CLIENT_PREALLOCATE=false TORCH_EXTENSIONS_DIR=$X/cache/torch_ext/$SLURM_JOB_ID
echo "job=$SLURM_JOB_ID node=$(hostname) scene=$SCENE run=$RUN_ID repo=$REPO extra=${EXTRA:-} start=$(date -Is)"
[ -e "$WS" ] && { echo "REFUSE: $WS exists"; exit 3; }
mkdir -p $WS; for f in $PREP/*; do b=$(basename $f); case $b in logs|provenance.txt) ;; *) ln -s $f $WS/$b ;; esac; done
cd $REPO
OVR=""; [ -n "${EXTRA:-}" ] && OVR=$(echo "$EXTRA" | tr ':' '\n' | sed 's/^/--/' | tr '\n' ' ')
{ echo "run=$RUN_ID scene=$SCENE job=$SLURM_JOB_ID repo=$REPO commit=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
  echo "prep=$PREP (symlinked)"; echo "cmd=python mosca_reconstruct.py --ws $WS --cfg $FITCFG $OVR"
  [ -f $REPO/PATCH.diff ] && echo "patch_sha=$(sha256sum $REPO/PATCH.diff | cut -c1-16)"; } > $WS/provenance.txt
python mosca_reconstruct.py --ws $WS --cfg $FITCFG $OVR
echo "reconstruct_exit=$? $(date -Is)" | tee -a $WS/provenance.txt
LOGDIR=$(ls -d $WS/logs/iphone_fit_* | tail -1); echo "logdir=$LOGDIR" >> $WS/provenance.txt
[ -f $LOGDIR/photometric_d_model_native_add3.pth ] || { echo "NO FINAL MODEL"; exit 4; }
[ "$(ls $LOGDIR/tto_test 2>/dev/null | wc -l)" -eq "$(ls $WS/test_images | wc -l)" ] || { echo "TTO RENDERS INCOMPLETE"; exit 5; }
python $X/agent/eval_mosca_iphone.py $WS $LOGDIR 2>&1 | grep -E "jax devices|COLUMNS|MEANS|Error|Traceback" | tail -n 6
python $X/agent/dy/count_prims.py $LOGDIR | tee -a $WS/provenance.txt
rm -rf $LOGDIR/tto_test_viz $LOGDIR/mosca_photo_viz_* $LOGDIR/*.mp4 2>/dev/null
echo "end=$(date -Is)"
