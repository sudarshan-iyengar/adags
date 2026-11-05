#!/usr/bin/env bash
# fmd mono2 lane: MoSca env (JiahuiLei/MoSca @02404d2, code unmodified) on SCRATCH. Run on the LOGIN node.
# Deviations from install.sh (conda, py3.10, torch 2.1.0+cu118): Leonardo has only python 3.11 + CUDA 12.2 modules,
# so torch 2.1.2+cu121 (constraints file), PyG wheels for torch-2.1.0+cu121, pytorch3d 0.7.8 copied from the lane's
# MAtCha env (same python/torch/CUDA build), cupy-cuda12x instead of cupy-cuda11x, py3.11-compatible pins for
# matplotlib/scipy/scikit-image; jax (DyCheck metric only), mmcv (Metric3D only) and GUI/mesh extras skipped.
set -uo pipefail
X=/leonardo_scratch/large/userexternal/siyengar/fmd
E=$X/envs/mosca; R=$X/repo/mosca
LOG=$X/logs/env_mosca.log; exec > >(tee -a "$LOG") 2>&1
echo "=== MOSCA ENV START $(date -Is) ==="
module purge; module load profile/base; module load gcc/12.2.0; module load python/3.11.7; module load cuda/12.2
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 PIP_CACHE_DIR=$X/cache/pip
C=$X/envs/constraints_mosca.txt
printf 'torch==2.1.2+cu121\ntorchvision==0.16.2+cu121\nnumpy==1.26.4\nsetuptools<66\n' > $C
[ -d $E ] && { echo "REFUSE: $E exists"; exit 3; }
python -m venv $E || exit 1; source $E/bin/activate
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$GCC_LIBDIR/libstdc++.so.6"
P="python -m pip install -c $C"
python -m pip install --upgrade pip wheel "setuptools<66" || exit 1
$P --extra-index-url https://download.pytorch.org/whl/cu121 "torch==2.1.2+cu121" "torchvision==0.16.2+cu121" numpy==1.26.4 || exit 1
$P pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html || exit 1
$P torch_geometric fvcore iopath || exit 1
$P "pytorch-lightning==2.0.2" "huggingface-hub>=0.22.0,<0.26" "diffusers>=0.20.1,<0.31" "accelerate>=0.22.0,<1.1" "transformers>=4.32.1,<4.45" kornia "matplotlib<3.8" plyfile open3d "opencv-python==4.10.0.84" "opencv-contrib-python==4.10.0.84" "imageio==2.25.0" imageio-ffmpeg "scipy<1.12" "scikit-image<0.23" rtree tensorboardX tensorboard transforms3d munch ipython rich moviepy "shapely<2" "trimesh<4" pydot omegaconf numba hydra-core openpyxl cython ninja gdown "timm==0.9.2" configargparse easydict "Pillow==9.5.0" einops ipympl "mediapy==1.2.2" einshape cupy-cuda12x lpips pandas evo submitit flow_vis ipdb torchmetrics scikit-learn || exit 1
$P "xformers==0.0.23.post1" --extra-index-url https://download.pytorch.org/whl/cu121 || echo "xformers skipped"
# pytorch3d: copy the build from the lane's MAtCha env (python 3.11, torch 2.1.2+cu121, sm80)
SP=$(python -c "import site;print(site.getsitepackages()[0])")
MSP=$(ls -d $X/envs/matcha/lib/python3.11/site-packages 2>/dev/null || ls -d /leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/envs/matcha/lib/python3.11/site-packages)
cp -r $MSP/pytorch3d $MSP/pytorch3d-*info $SP/ || echo "PYTORCH3D COPY FAILED"
cd $R
for e in simple-knn diff-gaussian-rasterization-alphadep-add3 diff-gaussian-rasterization-alphadep gof-diff-gaussian-rasterization; do
  ls lib_render/$e/third_party/glm >/dev/null 2>&1 && echo "glm ok $e"
  $P --no-build-isolation ./lib_render/$e || echo "BUILD FAILED $e"
done
cd $R; python - <<'PY'
import importlib, sys
sys.path.insert(0, ".")
for m in ["torch","torch_geometric","torch_scatter","pytorch3d.ops","simple_knn._C","diff_gaussian_rasterization","lpips","evo","cupy","mediapy","kornia",
          "lib_mosca.dynamic_gs","lib_mosca.static_gs","lib_moca.camera","lib_render.render_helper","lib_prior.moca_processor","eval_utils.campose_alignment"]:
    try: importlib.import_module(m); print("  OK  ", m)
    except Exception as e: print("  FAIL", m, type(e).__name__, str(e)[:300])
PY
python -m pip freeze > $X/logs/freeze_mosca.txt
echo "=== MOSCA ENV DONE $(date -Is) ==="
