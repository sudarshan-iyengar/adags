# fmd mono2: Shape of Motion and MoSca on the real monocular sweeps (2026-10-19)

All Leonardo paths are on SCRATCH: `/leonardo_scratch/large/userexternal/siyengar/fmd/` (`X` below).
The work area (`/leonardo_work/EUHPC_D36_068`) was at 99.6 % of its 4 TB quota (`cindata`, 2026-10-19 08:40), so new
environments, repositories, derived data and runs went to scratch. The input sweeps are read-only from work
(`fmd/data/mono_real/<scene>/multi4d`).

## Code (unmodified)
- Shape of Motion: github vye16/shape-of-motion @ 579753e1c7ba96f60cd7690e5b835627bd1935e9 (cloned --recursive), `X/repo/shape-of-motion`.
- MoSca: github JiahuiLei/MoSca @ 02404d2841b4e0dc2d9074860950bce5da579656 (cloned --recursive), `X/repo/mosca`.
- Each run writes `git rev-parse HEAD` and a dirty count to its provenance.txt.

## Environments (`X/envs/`)
- `som`: python 3.11.7 (Leonardo module; SoM asks for 3.10), torch 2.4.1+cu121 pinned by `constraints_som.txt`,
  gsplat 1.4.0, RAPIDS cuml-cu12 24.6 (KMeans/HDBSCAN in flow3d/init_utils.py; import verified on a GPU node),
  transformers 4.44.2 (Depth Anything pipeline). Build script `X/agent/build_som.sh`, log `X/logs/env_som.log`,
  freeze `X/logs/freeze_som.txt`.
- `mosca`: python 3.11.7, torch 2.1.2+cu121 (install.sh: conda, py3.10, torch 2.1.0+cu118; no conda/py3.10/cu118 on
  Leonardo), PyG wheels for torch-2.1.0+cu121, pytorch3d 0.7.8 copied from the lane's MAtCha env (same python, torch,
  CUDA build), cupy-cuda12x for cupy-cuda11x, py3.11-compatible pins for matplotlib/scipy/scikit-image; jax (DyCheck
  metric only), mmcv (Metric3D only) and GUI/mesh extras not installed. Four lib_render CUDA extensions built from the
  repo. Build script `X/agent/build_mosca.sh`, log `X/logs/env_mosca.log`.
- gcc 12.2.0 libstdc++ LD_PRELOADed in every job (lane rule).

## Weights (official sources)
- BootsTAPIR `bootstapir_checkpoint_v2.pt` from storage.googleapis.com/dm-tapnet (sha256 8493c7a6...), `X/weights/som/`.
- Depth Anything v1 large (`LiheYoung/depth-anything-large-hf`, Hugging Face, snapshot 27ccb09), `X/cache/hf/` (offline on compute nodes).
- MoSca weights zip from the README's Google Drive link (RAFT raft-things, BootsTAPIR v2, SpaTracker), unzipped into `X/repo/mosca/weights/`.

## Shared inputs and adapters
- Sweep = `fmd/data/mono_real/<scene>/multi4d` (built by `agent-control/fmd/build_real_sweep.py`): 300 frames, one
  static N3V camera at a time, 17 cameras left to right; held-out cam00/05/17 never in the sweep. AUTHORED absence:
  the wine bottle is removed from frames 60-89 in every camera (fixture), and the sweep shows the empty surface.
- Cameras: the KNOWN cameras (`camera/*.json`: orientation = world-to-camera rotation, position = centre, OpenCV axes),
  never re-estimated.
- Moving-content masks (stand-in for SoM's Track-Anything clicks): per sweep frame, |fixture frame - temporal median of
  that static camera's fixture footage (stride 2)| > 20/255 (max channel), opened r=3, dilated r=7
  (`X/agent/prep_som.py`). Marks the cook, hands, pan, flames and the dog, AND the vacated surface during the gap
  (it differs from the median) - the method-favourable choice.
- Depth: Depth Anything v1 large disparity via SoM's own `preproc/compute_depth.py` (`save_disp_from_dir`), aligned
  with SoM's own scale/shift formula (`align_monodepth_with_colmap`, copied verbatim) against the sweep's starting cloud
  (`points.npy`, COLMAP dense of frame 0 over 4 sweep cameras) projected through the KNOWN cameras (z-buffered on 3-px
  cells, moving-content pixels excluded), instead of a COLMAP sparse model. Per-frame log `depth_align_log.json`.

## Shape of Motion run
- Tree `X/data/mono_real_som/<scene>/` in the iPhone layout (`iphone_dataset.py`): RGBA copies (alpha 255) of the
  sweep frames, camera JSONs, splits (train = 300 sweep frames; val empty), extra.json (fps 30), masks
  (`track_anything/1x`), aligned depth (`aligned_depth_anything_colmap/1x`).
- 2D tracks: SoM's `preproc/compute_tracks_torch.py` (BootsTAPIR, 256x256, grid 4, queries inside the masks, all
  pairs), verbatim except a `--t0/--t1` query-frame range so 5 GPU jobs split one sequence (`X/agent/compute_tracks_range.py`).
- Train: `python run_training.py --work-dir RUN data:iphone --data.data-dir TREE --data.camera-type original`
  (published iPhone route, all other defaults: 500 epochs, 40k fg / 100k bg, 10 motion bases, 3DGS).
  `camera-type original` = known cameras (the published iPhone runs use COLMAP-refined cameras).
- Render: `X/agent/render_som.py`, SceneModel.render(t, w2c, K) at held-out cameras, cameras normalised exactly as the
  dataset normalises them (scene_norm_dict), white background (trainer default), all 300 frames, full resolution.

## MoSca run
- Workspace `X/runs/mosca/<run>/` (`X/agent/prep_mosca.py`): images downsampled 2x to 676x507 (DyCheck iPhone trains
  at 2x; MoSca holds every buffer of every frame on the GPU); `cameras/0_NNNNN.json` (known cameras, focal and principal
  point halved); `test_cameras/{1,2,3}_NNNNN.json` = cam00/05/17 at full resolution; `test_images/` = symlinks to the
  held-out fixture frames; `sensor_depth/` = the SoM aligned depth (1/disparity), nearest-downsampled.
- World scale ADAPTER: cameras and depth divided by S = 8 (N3V units are ~8x metres: median cloud depth from cam00 = 16
  units, the cook stands ~2 m from the rig), because the published iPhone config fixes sizes in metres
  (mosca_unit_world 0.02, gs_radius_max 0.03).
- `mosca_precompute.py --cfg profile/iphone/iphone_prep.yaml --epi_num_threads=8` (dep_mode sensor, RAFT, epipolar
  error, BootsTAPIR), then `mosca_reconstruct.py --cfg profile/iphone/iphone_fit.yaml` (init_gt_camera True; MoSca
  refines the known poses with its published small lr; focal fixed).
- Renders: (a) MoSca's own end-of-run iPhone evaluation renders `<logdir>/tto_test/` after a pose-only test-time
  alignment against the held-out images; (b) `X/agent/render_mosca.py` calls MoSca's `test_main(tto_flag=False)`:
  Sim(3) alignment of the known held-out cameras by the training trajectory, no image-based optimisation, `<logdir>/test/`.

## Scoring
- `D:/adags-ars/fmd/mono2/pull_score.sh LABEL SCENE REMOTE_DIR KIND`: frames 40-99 of cam00/05/17 into
  `D:/adags-ars/fmd/real/<label>/render/`, plus fix/orig/mask copied from `real/m4d_monoreal_<scene>_rep1/`;
  `ghost_ring_real.py` (unchanged) -> `scores/real_<label>.json`; whole-frame PSNR vs fixture -> `scores/wholeframe_<label>.json`.

## Fixes and incidents during the run (2026-10-19)
- MoSca checkout: `lib_mosca/camera.py` is committed upstream as a 21-byte text file containing `../lib_moca/camera.py`
  (a symlink stored as a file). Replaced by the intended symlink; no code changed. `git status` therefore shows one
  typechange (the runs' "dirty=1").
- MoSca env: kornia pinned to 0.7.3 (0.8.3 fails to import with torch 2.1.2).
- MoSca: the iPhone-route camera reader needs `image_size` in every JSON (written) and single-digit test camera ids
  (1 = cam00, 2 = cam05, 3 = cam17).
- Shape of Motion: the first training jobs (61345137, 61345147) built the initialisation and saved
  `checkpoints/last.ckpt`, then died when torchmetrics' LPIPS tried to download AlexNet weights (no network on
  compute nodes). AlexNet weights (torchvision `alexnet-owt-7be5be79.pth`, sha256 7be5be79...) cached in
  `~/.cache/torch/hub/checkpoints`. The runs were resumed from that checkpoint (SoM's own resume path; `RESUME=1`
  in the sbatch), first attempt (61370608/11) died on a concurrent gsplat JIT build in `~/.cache/torch_extensions`
  (stale file handle); second attempt (61374491/502) with a per-job `TORCH_EXTENSIONS_DIR`. The roast-beef run
  (61345122) reached the LPIPS step after the weights were cached and ran straight through.
- First preprocessing jobs failed on two bugs in my own `prep_som.py` (a symlink-name assertion; an absolute path
  joined twice). Their never-started dependents were dropped by Slurm, not cancelled. Ledger has notes.
- MoSca render-only jobs (`fmd_mono2_moscarender_*`) rendered `<logdir>/test` while the main jobs were still in
  MoSca's own evaluation; the main jobs then re-rendered the same files (deterministic).

## Diagnostics
- `X/agent/static_only_mosca.py`: MoSca static model alone / dynamic model alone / both at cam00 f45, f75 (render_test's
  own alignment). `X/agent/static_only_som.py`: SoM background Gaussians alone / foreground alone / both.
  Outputs `D:/adags-ars/fmd/mono2/diag/`.
- `D:/adags-ars/fmd/mono2/persist.py`: registration-free persistence (render change in the vacated region, gap vs
  pre-gap; share of pixels closer to the method's own pre-gap render than to the fixture).
