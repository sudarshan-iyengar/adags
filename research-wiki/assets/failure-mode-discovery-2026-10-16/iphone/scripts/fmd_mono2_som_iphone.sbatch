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

# fmd mono2: Shape of Motion (vye16/shape-of-motion @579753e, code UNMODIFIED) on UNTOUCHED DyCheck iPhone, as its
# README: run_training.py data:iphone (all defaults: refined COLMAP cameras, aligned Depth Anything, 500 epochs), on
# Shape of Motion's released processed iPhone data; held-out renders are written by its own validator to
# <run>/results/rgb; then scripts/evaluate_iphone.py (masked co-visible mPSNR/mSSIM/mLPIPS + tracking metrics).
# Required --export: SCENE.
set -uo pipefail
X=/leonardo_scratch/large/userexternal/siyengar/fmd
SCENE="${SCENE:?}"; D=$X/data/dycheck/som/scenes; RUN=$X/runs/som_iphone/$SCENE
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/som/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
export TORCH_EXTENSIONS_DIR=$X/cache/torch_ext/$SLURM_JOB_ID HF_HOME=$X/cache/hf HF_HUB_OFFLINE=1
echo "job=$SLURM_JOB_ID node=$(hostname) scene=$SCENE start=$(date -Is)"
if [ -n "${ZIPFILE:-}" ]; then echo "unzip $ZIPFILE $(date -Is)"; unzip -q -o "$ZIPFILE" -d $D/ && rm -f "$ZIPFILE" && echo "unzipped $(date -Is)" || { echo "UNZIP FAILED"; exit 6; }; fi  # login-node unzip of the 30-50 GB zips gets killed
if [ "${RESUME:-0}" != "1" ]; then [ -e "$RUN" ] && { echo "REFUSE: $RUN exists"; exit 3; }; fi
mkdir -p $RUN; cd $X/repo/shape-of-motion
{ echo "scene=$SCENE job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
  echo "data=Shape of Motion release (Drive folder 1xJaFS_3027crk7u36cue7BseAX80abRe) $SCENE"
  echo "cmd=python run_training.py --work-dir $RUN data:iphone --data.data-dir $D/$SCENE"; } >> $RUN/provenance.txt
python run_training.py --work-dir $RUN data:iphone --data.data-dir $D/$SCENE
rc=$?; echo "train_exit=$rc finished=$(date -Is)" | tee -a $RUN/provenance.txt; [ $rc -eq 0 ] || exit $rc
ls $RUN/results/rgb | wc -l
PYTHONPATH=. python scripts/evaluate_iphone.py --data_dir $D --result_dir $X/runs/som_iphone --seq_names $SCENE 2>&1 | tee $RUN/eval_iphone.txt
echo "end=$(date -Is)"
