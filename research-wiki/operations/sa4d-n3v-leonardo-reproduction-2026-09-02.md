# SA4D on Leonardo: N3V reimplementation record (2026-09-02)

Status: exploratory reproduction of a third-party method (Segment Any 4D Gaussians,
arXiv 2407.04504, code https://github.com/Marine318/sa4d), not a claim-grade ADAGS
experiment. Evidence boundary: artefacts under
`/leonardo_work/EUHPC_D36_068/sri/proj_adags/{repo/sa4d,runs/sa4d,agent-control/sa4d}`.

## What was built

| Item | Decision | Why |
|---|---|---|
| Data | `data/n3v/<scene>` rsynced from `EUHPC_D21_034/proj_adags/data/n3v`, `flow/` excluded | user request; audit: file counts and bytes equal for all six scenes, 24/24 sha256 spot checks OK (job 55586771) |
| Repo | `repo/sa4d` clone at commit `9b46359`, standalone (no git ops) | user request |
| Env | `envs/sa4d` venv: python 3.11.7 module, torch 2.5.1+cu121, gcc/12.2.0 + cuda/12.2 modules, libstdc++ fix copied from `exp_index/leonardo_env.sh` | reuse the proven adags pattern; SA4D's torch 2.0.1/py3.9 is not reproducible on Leonardo's module python |
| COLMAP | official container `colmap/colmap:20241206.1709` (COLMAP 3.11.1, CUDA 12.6) run through `bin/colmap` -> `singularity exec --nv`, with NVIDIA's `cuda-compat-12-6` forward-compat user-mode driver bound in front of the injected host `libcuda` | no colmap module exists; pip pycolmap wheels are CPU-only (a separate `pycolmap-cuda12` exists but is a Python API, not the CLI). Three container failures were diagnosed in sequence: (a) SIF creation fails on every Lustre tmp (xattr `lustre.lov`) and login `/tmp` is 6 GB, `SINGULARITY_TMPDIR=$HOME/.sing_tmp` works; (b) `latest` is COLMAP 4.2, which renamed `SiftExtraction.max_image_size` and rejects SA4D's custom text model (needs rigs/frames), so a 3.11-era date tag was pinned; (c) on the 535 driver (CUDA 12.2) the container's CUDA 12.6 code fails every GPU SIFT extraction/matching call (`Feature matching failed ... insufficient GPU memory` on an idle A100, an instant catch-all failure); binding cuda-compat 560.35.05 fixes it (probe: 3 pairs, 886 inliers). A native source build (COLMAP 3.11.1 against cuda/12.2 + gcc/12.2, `CMAKE_CUDA_ARCHITECTURES=80`, vendored gflags/glog/Eigen/Ceres/lz4/FLANN/SQLite/FreeImage under `opt/colmapdeps`) was started as a backup; outcome in the run log |
| Weights | `models/sa4d_weights/{DEVA-propagation.pth, sam_vit_h_4b8939.pth}` symlinked into `Tracking-Anything-with-DEVA/saves/` | README download script |
| Extensions | diff-gaussian-rasterization, diff-gaussian-rasterization_contrastive_f (editable), simple-knn (non-editable: its setup.py declares no package), pytorch3d 0.7.8 from source (no py311/cu121/torch2.5 wheel) built in a GPU job (55587555) | |

## Source adaptations (recorded diff: `agent-control/sa4d/patches/sa4d_changes.diff`)

1. `database.py` restored from hustvl/4DGaussians (SA4D's `colmap.sh` calls it but the file is missing).
2. Hardcoded `CUDA_VISIBLE_DEVICES` = "0"/"1"/"2" in `train_4dgs.py`, `train_ie.py`, `render_4dgs.py`, `render_ie.py` and `colmap.sh` changed to respect the Slurm-provided value.
3. `render_ie.py`: `makedirs` before writing `video_mask.mp4` (the upstream makedirs are commented out and the directory never exists on the README's dynerf path).
4. `prepare_pseudo_label.sh`: optional third argument for the labelled reference camera (default cam15). coffee_martini has no cam15; the nearest rig camera by centre distance (1.08 vs 1.13 for cam16) is cam14.
5. `scene/deformation.py`: dead `from tkinter import W` removed (module python has no `_tkinter`).
6. `utils/point_utils.py`: `torch_cluster` import guarded (only used by an unreachable helper; no torch-2.5.1 wheel exists).
7. `data/dynerf/<scene>` are staging directories with only `poses_bounds.npy` and `camXX.mp4` symlinks: SA4D's `Scene` tests `transforms_train.json` before `poses_bounds.npy`, and the N3V dirs carry both, so a plain symlink would route into the Blender loader. All derived artefacts (frames, COLMAP, labels) land in the staging dirs, keeping the raw N3V tree untouched.
8. Env: mmcv 1.7.2 (needs `setuptools<70` for `pkg_resources`), numpy 1.26.4 (`database.py` uses removed-in-2.0 APIs), scikit-learn and timm (module-level imports in `point_utils.py`/`render_ie.py` and DEVA's MobileSAM).

9. Run-time downloads: DEVA builds its ResNet-18/50 backbones with `pretrained=True` (torchvision model-zoo URLs); compute nodes have no internet, so both weight files are cached under `cache/torch/hub/checkpoints` (`TORCH_HOME`). Nothing else on the training path downloads (the LPIPS model construction is commented out upstream).

10. `utils/general_utils.safe_state`: `torch.cuda.set_device(torch.device("cuda"))` is rejected by torch 2.5 (needs an index); changed to `cuda:0` as in upstream 3DGS.

Everything else in the `dynerf` path (frame extraction to 1352x1014, `llff2colmap`, dense COLMAP, downsampling, DEVA labelling, per-scene `arguments/dynerf/<scene>.py`) runs as shipped.

## Pipeline

`agent-control/sa4d/sa4d_pipeline.sbatch <scene> <cam>`: preprocess -> colmap -> downsample -> label -> train_4dgs (14k) -> render_4dgs (test cam00 + spiral video) -> train_ie (5k) -> render_ie. Each stage is verified on artefacts (frame counts and size, fused.ply, downsampled point count, 300 masks, `iteration_14000/scene_point_cloud.ply`, 300+300 rendered PNGs, `<cam>/` classifier+mlp, two `video_mask.mp4`) and marked DONE in `status/<scene>.stages`; reruns skip DONE stages and prune partial extraction/label outputs. A fresh-context review of the scripts found the three blockers in items 6-8 and the Blender-branch trap (item 7) before any job ran.

## Runs

Round 1 (5.5 h jobs, before the 2026-09-02 08:00 to 09-04 08:00 full-system maintenance `fix_0209`):
frame extraction DONE on all six scenes in about 24 min each (300 frames per camera at 1352x1014, verified by count and size).
Rounds 1a/1b failed at COLMAP (4.2 CLI rename; then the CUDA/driver mismatch), round 1c completed COLMAP + downsampling on all six
(fused points: coffee_martini 318,864; cook_spinach 402,658; cut_roasted_beef 386,337; flame_salmon_1 365,053; flame_steak 406,571;
sear_steak 408,102; downsampled to 36k to 39k) and failed at labelling on the missing backbone download. Round 1d (4 h) completed labelling on all six (300 masks each, DEVA semi-online with SAM ViT-H at short side 480, about 6 min per scene for both passes) and failed at the first training call on the torch 2.5 `set_device` incompatibility; round 1e (3 h 50 min) runs training onward; round 2 continuation jobs (24 h, `--begin 2026-09-04T09:00`, `afterany` on round 1) resume whatever is not DONE.

(results table filled in below as stages complete)

Per-scene artefacts under `runs/sa4d/<scene>` (all at `iteration_14000`; test camera = cam00, 300 frames; spiral video = 300 poses;
identity encoder trained on the reference camera's DEVA masks for 5,000 iterations; renders verified by file count):

| scene | cams | ref cam | test PSNR (L1) @14k | train PSNR | fused / init points | renders (test/video) | mask videos (train/video) | model dir |
|---|---|---|---|---|---|---|---|---|
| coffee_martini | 18 | cam14 | 28.35 (0.0220) | 31.10 | 318,864 / 36,304 | 300/300 | yes/yes | 1.1 GB |
| cook_spinach | 21 | cam15 | 32.34 (0.0157) | 34.80 | 402,658 / 38,600 | 300/300 | yes/yes | 891 MB |
| cut_roasted_beef | 20 | cam15 | 30.04 (0.0213) | 35.28 | 386,337 / 37,269 | 300/300 | yes/yes | 891 MB |
| flame_salmon_1 | 19 | cam15 | 28.90 (0.0198) | 32.18 | 365,053 / 39,076 | 300/300 | yes/yes | 1.1 GB |
| flame_steak | 21 | cam15 | 33.39 (0.0150) | 35.28 | 406,571 / 37,646 | 300/300 | yes/yes | 874 MB |
| sear_steak | 21 | cam15 | 33.11 (0.0160) | 35.75 | 408,102 / 37,858 | 300/300 | yes/yes | 877 MB |

Every scene reached `ALL_DONE` in round 1e (jobs 55589649-55589654, 03:48 to 05:19 CEST on 2026-09-02): 4DGS training
about 30 min (fast nodes) to 56 min, `render_4dgs` 2 to 3 min, `train_ie` 17 to 21 min, `render_ie` about 1 min. PSNR is
SA4D's own training-time test evaluation (cam00, all 300 frames, no clamping), reported here as the reproduction's
health check, not as a benchmark number. The post-maintenance continuation jobs were cancelled as unneeded.
Cost: about 12 A100 slot-hours across all rounds including diagnostics. Storage: `runs/sa4d` 5.7 GB, staging dirs
(`repo/sa4d/data/dynerf`) 51 GB, containers 4.9 GB; the work area stayed at 96% (191 GB free).

Not done / caveats: the SA4D README's `render_binary.py` / `train_binary.py` and the notebooks were not run (the README's
Train section lists only the four commands above). The native COLMAP source build (backup) stopped at CMake because the
spack Boost module lacks `boost_graph`; it is not needed while the container route works. The stale rsync temp file
`data/n3v/coffee_martini/images/.cam19_0121.png.iSl5E5` from an earlier interrupted copy was left in place.

Recorded source diff: [sa4d-leonardo-patches-2026-09-02.diff](sa4d-leonardo-patches-2026-09-02.diff) (also at
`agent-control/sa4d/patches/sa4d_changes.diff` on Leonardo). Job ledger: `agent-control/sa4d/ledger.txt`.
