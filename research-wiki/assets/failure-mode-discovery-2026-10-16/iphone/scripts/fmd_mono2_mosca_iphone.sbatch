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

# fmd mono2: MoSca (JiahuiLei/MoSca @02404d2, code UNMODIFIED) on UNTOUCHED DyCheck iPhone, exactly as its reproduce.sh:
# mosca_precompute.py profile/iphone/iphone_prep.yaml, then mosca_reconstruct.py profile/iphone/iphone_fit.yaml (the
# run ends with MoSca's own iPhone evaluation: pose-only test-time alignment + masked co-visible mPSNR/mSSIM/mLPIPS).
# Data = MoSca's released processed iPhone data (README link), copied per scene into the run workspace.
# Only override: --epi_num_threads=8 (process count; 8 CPUs allocated). Required --export: SCENE.
set -uo pipefail
X=/leonardo_scratch/large/userexternal/siyengar/fmd
SCENE="${SCENE:?}"; WS=$X/runs/mosca_iphone/$SCENE
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
source $X/envs/mosca/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
export TORCH_HOME=$X/cache/torch HF_HOME=$X/cache/hf HF_HUB_OFFLINE=1 TORCH_EXTENSIONS_DIR=$X/cache/torch_ext/$SLURM_JOB_ID
echo "job=$SLURM_JOB_ID node=$(hostname) scene=$SCENE start=$(date -Is)"
[ -e "$WS" ] && { echo "REFUSE: $WS exists"; exit 3; }
mkdir -p $X/runs/mosca_iphone && cp -r $X/data/dycheck/mosca/data/iphone/$SCENE $WS || exit 1
cd $X/repo/mosca
{ echo "scene=$SCENE"; echo "job=$SLURM_JOB_ID"; echo "commit=$(git rev-parse HEAD)"; echo "dirty=$(git status --porcelain --untracked-files=no | wc -l) (lib_mosca/camera.py symlink restored)"
  echo "data=MoSca release zip 1sSvVi5Bid_KQsuguVGuqUVgWPzVdb9jM data/iphone/$SCENE"; } > $WS/provenance.txt
python -c "import imageio,glob; f=sorted(glob.glob('$WS/images/*'))[0]; print('train image', imageio.imread(f).shape); f=sorted(glob.glob('$WS/test_images/*'))[0]; print('test image', imageio.imread(f).shape)"
python mosca_precompute.py --ws $WS --cfg ./profile/iphone/iphone_prep.yaml --epi_num_threads=8 || { echo "PRECOMPUTE FAILED"; exit 2; }
echo "precompute done $(date -Is)"
python mosca_reconstruct.py --ws $WS --cfg ./profile/iphone/iphone_fit.yaml
rc=$?; echo "reconstruct_exit=$rc finished=$(date -Is)" | tee -a $WS/provenance.txt
LOGDIR=$(ls -d $WS/logs/iphone_fit_* | tail -1); echo "logdir=$LOGDIR" >> $WS/provenance.txt; ls $LOGDIR
cat $LOGDIR/*dycheck*.txt $LOGDIR/*metric* 2>/dev/null | head -20
echo "end=$(date -Is)"
