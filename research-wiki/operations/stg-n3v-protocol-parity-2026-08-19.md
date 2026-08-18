# STG / N3V protocol parity, and a correction to how one ADAGS number may be read (2026-08-19)

Operational/engineering record. EXPLORATORY. Append-only; nothing on
[[sota-sweep-2026-08]] is rewritten — the correction below is stated here and
that page is to be read with it.

**Provenance, stated first.** The STG-side facts come from a bounded read-only
pass over the official arXiv paper and the official
`oppo-us-research/SpacetimeGaussians` repository, through a summarizing fetch
layer. They are high-fidelity transcriptions, not byte-exact reads, and the
primary did **not** independently re-fetch them. The load-bearing config was
read twice, independently, and agreed exactly. **The ADAGS-side facts in
section 2 were verified by the primary directly in this repository** and are
decision-grade.

## 1. The correction, and it is the reason this page exists

[[sota-sweep-2026-08]] records ADAGS route0 at 6000 iterations / 600k points on
`cut_roasted_beef` as **34.37 dB**, and reads that as *"inside the competitive
band for that scene"* against STG's published **33.52 dB**.

**That reading is not supported, and the largest reason is resolution.**

**Primary-verified in source:**

* `scripts/n3v2blender.py:273` — `img.resize((img.width // 2, img.height // 2))`.
  The materialization on disk is **already halved**, native 2704x2028 to
  **1352x1014**.
* `utils/camera_utils.py:22` — `resolution = round(orig_w / (resolution_scale *
  args.resolution))`. With `configs/n3v/lane_l0_route0.yaml`'s
  `ModelParams.resolution: 2`, the loader halves the **already-halved** PNGs
  again.

So the recorded ADAGS number was measured at **676x507 — one quarter of the
pixels** at which STG trains, evaluates and publishes. PSNR is not invariant to
that: fewer, larger pixels average away exactly the high-frequency error the
metric is most sensitive to.

Three further mismatches on the same comparison:

* **temporal extent.** STG's published per-scene number is a **50-frame**
  model; 300 frames is six independently trained models arranged in series
  (paper §6.1 / Appendix C.8, and every N3V config sets `"duration": 50`). The
  ADAGS figure is one model over all 300 frames.
* **PSNR pooling — primary-verified, and it inflates ADAGS.**
  `utils/image_utils.py:17-19` computes
  `((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1)`. Both ADAGS eval call
  sites — `main.py:1979` and `utils/mesh_utils.py:106` — pass an **unbatched
  `(3, H, W)`** tensor, so `shape[0]` is **3** and the view splits by
  **channel**: the function returns three per-channel PSNRs and the caller
  averages them. STG (and 3DGS's own `metrics.py`) pass `(1, 3, H, W)` and pool
  correctly. The bias is `10*log10(AM/GM)` over the per-channel MSEs, which is
  **always >= 0**. A worker's measurement on a real degraded
  `cut_roasted_beef` frame put it at **+0.031 dB** at a channel-MSE ratio near
  1.3, rising past +0.17 dB at ratio 2.0. Small, one-directional, and free to
  fix — one line at each of two call sites.
  `scripts/eval_diva360_heldout.py:236-238` already batches correctly, and
  `scripts/eval_lrv1_event.py` pools over channels and pixels by construction.
* **LPIPS convention.** ADAGS uses torchmetrics with `normalize=True`
  (i.e. [-1,1] internally); STG calls its vendored `lpipsPyTorch` on raw
  clamped **[0,1]**. This repository has already measured the gap on identical
  images — **0.14685 vs 0.12398** ([[diva360-protocol-parity-audit]]). The
  [0,1] convention gives the smaller number, so the naive comparison of ADAGS's
  0.0524 against STG's 0.036 **overstates** the gap.
* **initialization.** STG concatenates 50 per-frame `point_triangulator` sparse
  clouds with a normalized time channel and, for `ours_full`, applies no
  filtering. ADAGS runs ONE frame-0 COLMAP plus **dense MVS**
  (`image_undistorter` -> `patch_match_stereo` -> `stereo_fusion`) to 366,366
  points with no time channel. A confound of unknown sign.

**None of this says ADAGS is worse.** It says the two numbers were never on the
same scale and the "competitive band" reading has no support. Cell B1
(experiment 166) is the measurement that puts the ADAGS side on the published
raster and segment.

## 2. The official STG protocol, as extracted

| item | value |
|---|---|
| scenes | six N3V, one config each in `configs/n3d_lite/` and `configs/n3d_full/` |
| frames per model | **50**; 300 frames = 6 models in series |
| held-out | **cam00**, selected POSITIONALLY — `readColmapSceneInfo` sorts by `image_name` and takes `cam_infos[:duration]`, with an assert that exactly one camera appears |
| train cameras on `cut_roasted_beef` | **19** — the scene ships 20 cameras, `cam04` absent from the official release |
| raster | `"resolution": 2` on native 2704x2028 -> **1352x1014** for train AND eval |
| frame extraction | **OpenCV, not ffmpeg** — `cv2.VideoCapture` + `cv2.imwrite`, PNG, default compression |
| initialization | per-frame COLMAP `point_triangulator` only (poses come from LLFF `poses_bounds.npy`); no `mapper`, no dense MVS |
| budget | 30,000 iterations, batch **2**, `0.8*L1 + 0.2*(1-SSIM)`, densify 500->9000 interval 100, opacity reset every 3000, at most 6 clone/split events |
| metric iteration | per-scene `test_iteration`; **25000** for `cut_roasted_beef` |
| metrics | PSNR on batched `(1,3,H,W)` in [0,1]; 3DGS Gaussian SSIM **and** skimage SSIM; LPIPS Alex **and** VGG on **[0,1]** |
| published `cut_roasted_beef` | **33.52** (full) / **33.72** (lite) |
| hardware | 40-60 min per 50-frame model on an A6000 (paper text; Table 10 implies ~16.7 min — unresolved upstream) |

Note the inversion worth remembering: on this specific scene the **lite** model
beats the full model by +0.20 dB, though full is +0.46 dB better on the 6-scene
average.

## 3. Blockers to reproducing STG locally — why B0 is not a config change

A local public-code STG run is a separate build, not a separate cell:

* **five vendored CUDA extensions** must build against torch 2.0.1 / CUDA 11.8
  for sm_70 and sm_90 (`gaussian_rasterization_ch9`, `..._ch3`, `forward_full`,
  `forward_lite`, `simple-knn`). They are vendored directories, so a
  `--recursive` clone does not fetch them;
* **`mmcv` built from source** (~30 min per STG's own `setup.sh`); mmcv 1.x does
  not build cleanly against torch 2.0.1. It supplies the CUDA `knn` that only
  the **lite** variant needs, so `ours_full` may sidestep it — unverified;
* **`simple_knn` import-name collision**: this repository installs `simple_knn`
  and so does STG. Both are stock 3DGS so the collision is benign in content,
  but two packages claiming one import name must not share an environment —
  **STG needs its own image**;
* **`test.py` crashes as shipped**: `sk_ssim(..., multichannel=True)` on float
  input with no `data_range`; `multichannel` was removed in scikit-image 0.23
  and `data_range` is required for float input from 0.19. scikit-image is not
  installed in the ADAGS image at all;
* STG pins Python 3.7.13 / torch 1.12.1 / cudatoolkit 11.6 against the ADAGS
  image's torch 2.0.1+cu118;
* preprocessing is **POSIX-only** (`os.system("rm -r ...")`, symlinks) and
  always decodes all 300 frames of all cameras regardless of the requested
  range.

**Consequence:** B0 is a user decision about building a second image, not
something to improvise inside this lane. It is NOT attempted in this block.

## 4. The matched protocol, if B0 is ever funded

One 50-frame segment, frames 0-49, `cut_roasted_beef`:

| item | who moves |
|---|---|
| raster 1352x1014 | **ADAGS** (`resolution: 1`) |
| 50-frame segment | **ADAGS** (`time_duration` upper bound must exceed 49/30 = 1.63333 and fall below 50/30 = 1.66667) |
| held-out cam00, 19 train cameras | neither — already identical |
| GT pixel pipeline | neither — both PIL-bicubic-halve the same H.264 frames; verify once with a byte diff |
| initialization | **ADAGS** should adopt STG's time-channelled per-frame ply, or the comparison runs twice and discloses |
| training budget | **neither** — each at its own published budget, with the resource ledger reported |
| PSNR pooling | **ADAGS** — one line at each of two call sites |
| LPIPS | report **both** input conventions, labelled |
| seeds | **both**, >= 2 per arm — no same-arm spread has been measured at this configuration |

## 5. What B1 (experiment 166) can and cannot settle

It puts ADAGS on the published raster and segment, so its number is comparable
to a **published** figure on the axes that were previously mismatched. That is
a **screening** measurement:

* if ADAGS lands far below 33.52, no reproduction subtlety explains the gap and
  substrate competitiveness is an active blocker;
* if it lands close, the difference is inside the space of reproduction
  detail and only a local STG run can arbitrate;
* it is **not** a reproduction of STG and must never be reported as one.

## 6. Open, and not determined from primary sources

* How STG aggregates six per-segment metric sets into one published per-scene
  number. `script/post.py`'s `__main__` is `pass`; no averaging script exists;
  three repository issues ask and none is answered.
* Whether the paper's DSSIM is `(1-SSIM)` or `(1-SSIM)/2` — so the published
  SSIM for this scene cannot be inverted with confidence.
* STG's published learning rates: the phrase does not appear in the paper, and
  the code ships a 25k/30k schedule against the paper's only stated figure of
  12K.
* The two recorded ADAGS baselines for this scene — 34.366 dB (round-1 L0) and
  a 30.5021 dB three-scene route0 mean — sit on different pages under different
  protocols and were **not** reconciled. Neither should be cited in a
  comparison until they are.
