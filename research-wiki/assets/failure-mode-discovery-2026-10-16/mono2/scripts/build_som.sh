#!/usr/bin/env bash
# fmd mono2 lane: Shape of Motion env (vye16/shape-of-motion, code unmodified) on SCRATCH (work area is at 99.6 % of quota).
# Run on the LOGIN node. torch 2.4.1+cu121 pinned through a constraints file; gsplat prebuilt wheel for pt24cu121;
# RAPIDS cuml-cu12 for the KMeans/HDBSCAN used by flow3d/init_utils.py; preprocessing extras (transformers for Depth
# Anything, mediapy/einshape for tapnet_torch); BootsTAPIR checkpoint + Depth Anything (v1 large, HF) weights.
set -uo pipefail
X=/leonardo_scratch/large/userexternal/siyengar/fmd
E=$X/envs/som; R=$X/repo/shape-of-motion
LOG=$X/logs/env_som.log; mkdir -p $X/logs; exec > >(tee -a "$LOG") 2>&1
echo "=== SOM ENV START $(date -Is) ==="
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 PIP_CACHE_DIR=$X/cache/pip HF_HOME=$X/cache/hf
C=$X/envs/constraints_som.txt
printf 'torch==2.4.1+cu121\ntorchvision==0.19.1+cu121\nnumpy<2\nsetuptools<70\n' > $C
[ -d $E ] && { echo "REFUSE: $E exists"; exit 3; }
python -m venv $E || exit 1; source $E/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
P="python -m pip install -c $C"
python -m pip install --upgrade pip wheel "setuptools<70" || exit 1
$P --extra-index-url https://download.pytorch.org/whl/cu121 "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" || exit 1
$P viser opencv-python-headless imageio imageio-ffmpeg matplotlib tensorboard scikit-learn jaxtyping roma ninja pytorch-msssim fsspec loguru nerfview torchmetrics splines==0.3.2 pyyaml tyro trimesh ipdb || exit 1
$P gsplat==1.4.0 --extra-index-url https://docs.gsplat.studio/whl/pt24cu121 || $P gsplat==1.4.0 || exit 1
$P --extra-index-url https://pypi.nvidia.com "cuml-cu12==24.6.*" || echo "CUML INSTALL FAILED"
$P transformers==4.44.2 mediapy einshape gdown "huggingface-hub<0.26" || exit 1
mkdir -p $X/weights/som; cd $X/weights/som
[ -f bootstapir_checkpoint_v2.pt ] || wget -q https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
python -c "from huggingface_hub import snapshot_download as s; print(s('LiheYoung/depth-anything-large-hf'))"
sha256sum bootstapir_checkpoint_v2.pt
python - <<'PY'
import importlib
for m in ["torch","gsplat","cuml","roma","viser","nerfview","pytorch_msssim","transformers","mediapy","einshape","tyro","splines","imageio","cv2"]:
    try: importlib.import_module(m); print("  OK  ", m)
    except Exception as e: print("  FAIL", m, type(e).__name__, str(e)[:200])
import torch; print("torch", torch.__version__, torch.version.cuda)
PY
python -m pip freeze > $X/logs/freeze_som.txt
echo "=== SOM ENV DONE $(date -Is) ==="
