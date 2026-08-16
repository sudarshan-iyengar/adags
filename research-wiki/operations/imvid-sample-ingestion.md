# ImViD "Opera" sample — acquisition, calibration and ingestion facts

Date: 2026-08-16. Status: **ACQUIRED AND VERIFIED; calibration read;
loader constraint identified.** EXPLORATORY. No ImViD training has run
and no ImViD number exists.

Authority: the ImViD lane of the 2026-08-16 directive (preflight,
acquisition, ingestion smoke, and at most one explicitly frozen
exploratory baseline).

Sources: arXiv **2604.09473**, `Metaverse-AI-Lab-THU/ImViD`,
`https://sheng-qi.github.io/IVV/`.

## Acquisition — done, and reproducible

`scripts/fetch_imvid_sample.py`, run as a zero-GPU-slot Determined
command. This is deliberately a tracked script: the DiVa-360 `chess_long`
pilot is blocked precisely because its tranche was collected by hand
through a Dropbox UI and "there is no reproducible acquisition path in
the repository" ([[operations/elgs-m1-evidence-wiring-record]]). ImViD
does not have to inherit that.

```
url    https://github.com/Metaverse-AI-Lab-THU/ImViD/releases/download/v0.2/scene1_opera.zip
bytes  1,001,763,804                       (server Content-Length agreed)
sha256 7cc2c5eba67da6a993e151c60418f79a446ef485122cae4e51917fe9fdbd682b   VERIFIED
dest   /apollo/users/sri/proj_adags/data/imvid/raw/scene1_opera.zip   (0444, read-only)
```

**The SAMPLE is ungated**: a plain public HTTPS asset, CC BY 4.0
(attribution), no form, no login, no click-through, no account. The
**FULL** dataset is different — an application form emailed to the
authors plus manual approval — and the fetcher deliberately cannot
request it. That remains a user decision and a user action.

Apollo free space at acquisition: **31.165 TiB**, so storage is not a
binding constraint for this dataset at any plausible extraction.

### Archive inventory (central directory only, nothing extracted)

```
entries 41   compressed 1,001,756,570   uncompressed 1,001,801,403   ratio 1.00x
  .mp4  n=39   0.933 GiB
  .txt  n= 2   6,610 bytes
```

Ratio 1.00 because H.264 is already compressed; the zip is a container,
not a compressor. **The uncompressed archive is under 1 GiB**, so the
earlier resolution-arithmetic projection (~557 GB raw RGB) describes
DECODED FRAMES, not the archive, and no extraction decision rests on it.

## Calibration — READ from the data, not inferred

`cameras.txt` carries **one shared camera model for all 39 views**:

```
2 OPENCV 5312 2988
  fx 2603.3326864600399   fy 2602.2436600602796
  cx 2656                 cy 1494
  k1 -0.024546867645992888  k2 0.0035148158874614976
  p1 -0.00045079985723632071  p2 -0.00023832152424359775
```

This closes the preflight's UNAVAILABLE item: the distortion model is
COLMAP **`OPENCV`** (fx, fy, cx, cy, k1, k2, p1, p2) — two radial and two
tangential terms, not FISHEYE and not SIMPLE_RADIAL. Distortion is mild
(k1 ≈ -0.025) and the principal point sits exactly at the image centre
(2656 = 5312/2, 1494 = 2988/2), consistent with an idealized intrinsic
rather than a per-camera calibrated one — every camera shares
`CAMERA_ID 2`.

`images.txt` carries **39 entries**, one per camera, each
`IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME` with `NAME = camXX.png`
and COLMAP's world-to-camera quaternion+translation convention. The
per-image `POINTS2D[]` line is **empty for every image**, so **no sparse
3D point cloud ships with the sample** — the README's instruction to run
`point_triangulator` is not optional, it is the only way to get one.
That matters for initialization: unlike DiVa-360 (which at least ships a
synthesized frustum cloud) ImViD ships nothing, and this project's own
visual-hull initializer is not directly applicable either because **no
masks ship** with the sample.

One rig fact worth stating: 39 poses for 39 cameras, and the Opera scene
is `#Strategy = 1`, fixed-point only. So the rig is STATIC for this
sample and a single pose per camera is complete. ImViD's moving-rig
capability is real but is not exercised here and cannot be validated
from this sample.

## The loader constraint — measured, and it is not the obvious one

**The MP4s cannot be stream-decoded from inside the zip.** `ffprobe` on
a 4 MiB prefix returns `moov atom not found`: the sample's files are not
`faststart`, so the index sits at the END of each file. Any decode
therefore needs the whole MP4 present, which means extracting the 39
files (0.93 GiB total — trivial), not a streaming reader.

Decoder availability in the Apollo image, probed directly:

```
cv2 4.10.0      PRESENT
av              MISSING
imageio         MISSING
decord          MISSING
```

So decoding goes through `cv2.VideoCapture`, which is sequential-access;
random frame seeking in H.264 is a decode-from-keyframe operation, not a
constant-time one. Any ImViD loader must either decode sequentially into
a cache or extract frames ahead of time.

## Method audit of ImViD's own baseline — what is actually available

**No training or evaluation code is released.** The repository's TODO
`Open-source the code after the paper submission is completed` is
unchecked, and the only Python in the repo is a thin ffmpeg wrapper
(`scripts/extract_frames.py`, which targets PNG). Every method fact below
is paper-only and cannot be checked against an implementation.

VERIFIED FROM PAPER: flow-guided initialization (VideoFlow magnitude
against threshold `εf = 0.1` classifies triangulated SfM points
static/dynamic); per-primitive velocity `vᵢ`, temporal centre `tᵢ` and
temporal extent `τᵢ` with `μᵢ(t) = μᵢ + vᵢ(t − tᵢ)`; rasterized 2D
velocity supervised by flow endpoint error; Depth-Anything-V2 depth
aligned to sparse SfM points as an ℓ1 loss; a per-camera learnable
bilateral grid for appearance; a per-camera temporal offset `Δγₖ` shared
across that camera's frames, jointly optimized with an L2 regularizer;
Adam for **30 epochs** where an epoch is one permutation of the training
frames; learning rates 2e-3 (velocity), 1e-5 (temporal centre), 3e-2
(temporal extent); loss weights λdssim 0.2, λcolor 1.0, λperc 0.1,
λdepth 1.0, λflow 1.0, λreg 1e-4.

NOT SPECIFIED, and deliberately not filled in from another paper's
defaults: SH degree; the numeric learning-rate schedule for
position/rotation/scale/SH/opacity ("the same ... as vanilla 3DGS" is
referenced, not restated); any densification or pruning algorithm,
threshold or point budget; total gradient steps or batch size; temporal
window size.

### The 24.43 → 30.98 figure, with its conditions

**Table VII**, ImViD dataset, held-out **Cam 10** — deliberately a
PERIPHERAL view, not ImViD's usual held-out Cam 0, chosen because "the
central view ... benefits from stronger multi-view supervision" and so
cannot reveal the constraints' contribution.

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| no flow, no depth | 24.43 | 0.729 | 0.287 |
| flow only | 30.03 | 0.748 | 0.153 |
| depth only | 30.45 | 0.760 | 0.134 |
| flow + depth | **30.98** | 0.772 | 0.124 |

So the headline 6.55 dB is measured on a deliberately hard held-out view,
and **either constraint alone buys ~5.6-6.0 dB while the pair adds only
~0.5 dB more** — the two are largely redundant, not additive. Table
VII's own caption does not name a scene; Opera is strongly implied by
adjacency to Tables V/VI but is NOT stated, and is recorded here as
circumstantial rather than verified.

## What this sample cannot establish

Full-dataset performance; 1-5 minute scalability (the sample is 300
frames = **5 seconds**, against Opera's full 3 min 22 s / 226 GB);
moving-rig validity (Opera is fixed-point only); and
disappearance/reactivation supply (no masks, no identity-preserving
evidence, and no measurement has been made).

Do NOT transplant ImViD's temporal-opacity representation into EL-GS. A
single smooth Gaussian visibility bump around one temporal centre `tᵢ` is
not a substitute for EL-GS's latched multi-episode presence; it is the
single-interval family EL-GS's novelty record already lists as occupied.

## COLMAP is IN the Determined runtime — and its defaults would destroy the calibration

Probed directly at zero GPU slots, so this is the installed authority
rather than documentation:

```
/usr/bin/colmap                     PRESENT
colmap -h banner                    COLMAP 3.6
dpkg                                ii  colmap  3.6+dev2+git20191105-1build1
point_triangulator                  PRESENT
model_analyzer, model_converter     PRESENT
pycolmap                            MISSING
/usr/bin/ffmpeg, /usr/bin/ffprobe   PRESENT
```

`colmap --version` is NOT a valid flag in 3.6 (`ERROR: Command
'--version' not recognized`); the version comes from the `-h` banner and
from dpkg.

**Apollo-side execution is therefore the route**, and the workstation
fallback (local `colmap.bat`, hashed transfer, destination manifest) is
NOT needed. The data is already on Apollo and provenance stays in one
place. `ffmpeg` being present also settles the decode question — frames
come out through ffmpeg rather than through `cv2.VideoCapture`, which
was the only option identified before this probe.

### The trap, verified from the installed help text

`point_triangulator` runs a bundle adjustment, and in 3.6 its DEFAULTS
refine the intrinsics:

```
--Mapper.ba_refine_focal_length     arg (=1)    <-- WOULD ALTER fx, fy
--Mapper.ba_refine_extra_params     arg (=1)    <-- WOULD ALTER k1,k2,p1,p2
--Mapper.ba_refine_principal_point  arg (=0)
```

The ImViD sample's supplied intrinsics and extrinsics are FIXED
AUTHORITY. Running `point_triangulator` at its defaults would silently
return a *different* camera than the one shipped, and the resulting
cloud would be consistent with that different camera rather than with
the calibration the renderer will use. All three must be set to `0`
explicitly. Pose fixing is structural — `point_triangulator` triangulates
under given poses rather than estimating them, unlike `mapper` — but
intrinsic fixing is a flag, not a property, and the default is the wrong
way round.

Two further defaults matter and are recorded before any run:

* `feature_extractor` defaults to `--ImageReader.camera_model
  SIMPLE_RADIAL` with `--ImageReader.single_camera 0`. Left alone it
  would create 39 separate SIMPLE_RADIAL cameras and none of them would
  be the shipped `2 OPENCV ...` camera. `--ImageReader.existing_camera_id`
  (default `-1`) is the flag that binds extracted images to an existing
  camera entry instead.
* `--SiftExtraction.max_image_size` defaults to **3200**, while ImViD
  frames are **5312x2988**. At the default, features would be detected
  on a downscaled image while the supplied intrinsics describe the full
  raster — a scale mismatch between the correspondences and the
  calibration. This has to be raised to the native width or the
  consequence has to be handled explicitly.
* `SiftExtraction.use_gpu` and `SiftMatching.use_gpu` both default to
  `1`, so a zero-slot cell cannot run them unmodified.

### Stream validation — and a correction to "60 FPS"

All 39 videos were `ffprobe`d before any frame was decoded:

```
39/39   5312x2988   nb_frames 300   codec h264   pix_fmt yuv420p
        r_frame_rate 60000/1001
```

Every declared figure holds — 39 cameras, 300 frames, 5312x2988 — with
one correction. The rate is **60000/1001 = 59.94 FPS**, not 60. The
paper and README say "60 FPS" and this page repeated it; the container
metadata says NTSC-rate 59.94. The difference is 0.1%, which is
irrelevant to a 300-frame sample viewed as an index range and
potentially relevant to anything that converts frame index to seconds —
`index/60` drifts from `index * 1001/60000` by about 5 ms over 300
frames. Recorded so a later timestamping step uses the measured rate
rather than the advertised one.

`pix_fmt yuv420p` is worth noting too: chroma is subsampled 2x in each
direction, so colour detail at 5312x2988 is really carried at
2656x1494. That is a property of the source, not of any processing here,
but it bounds what colour-based initialization could recover.

### RECONCILIATION — the runtime is 3.6 and it does NOT behave like 3.9+

An independent documentation mapping was obtained and is thorough, but
it inspected **COLMAP 3.11.1** (`C:\Users\sucar\colmap\bin\colmap.exe`,
commit `682ea9a`) on the WORKSTATION, tracing the actual Ceres
parameterization in the pinned source. Its conclusions are correct for
3.11.1. **They are not transferable to the runtime**, and its own
version-boundary section says so, naming COLMAP 3.9 (2024-01-06) as the
divide. A targeted re-probe of `/usr/bin/colmap` confirms the divide is
real and lands on the wrong side for us:

| | 3.11.1 (workstation) | **3.6 (Determined runtime)** |
|---|---|---|
| `--refine_intrinsics` | exists, default `0` | **ABSENT** — passing it is a hard error |
| pose fixing | `fix_existing_images = true` hardcoded in `RunPointTriangulatorImpl` | **`--Mapper.fix_existing_images arg (=0)`** — a live flag defaulting to NOT fixing poses |
| `--clear_points` | default `1`, with filename-based `TranscribeImageIdsToDatabase` | **default `0`**, help text has no transcription clause |
| `ba_refine_focal_length` | overridden internally, inert | listed, `(=1)`, no known override in 3.6 |
| `model_comparer` | present | **ABSENT** |

**The pose row is the dangerous one.** The 3.11 analysis concluded
"extrinsics cannot be modified at all, by construction, no flag needed".
On 3.6 that is FALSE: `fix_existing_images` defaults to `0`, so the
post-triangulation global bundle adjustment is free to move the supplied
poses. Trusting the newer version's guarantee is exactly how supplied
calibration gets silently replaced — the failure the directive names.

**The 3.6 invocation must therefore be explicit about all four:**

```
colmap point_triangulator \
  --database_path DB --image_path IMAGES \
  --input_path MODEL_IN --output_path MODEL_OUT \
  --Mapper.fix_existing_images 1 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0
```

Whether `ba_refine_*` is live or inert on 3.6's `point_triangulator` is
NOT established from here — the 3.11 override may or may not exist in
3.6, and reading 3.6's source was not done. Passing `0` is correct under
both readings, which is why it is passed rather than reasoned about.

**Image-ID matching is a 3.6 problem that does not exist in 3.11.** With
`--clear_points` defaulting to `0` and no transcription clause in the
3.6 help, image IDs in the supplied `images.txt` cannot be assumed to be
remapped by filename. The plan is therefore to read the database's
`image_id`/`name` table after feature extraction and REWRITE the input
model's `images.txt` to those IDs, changing ONLY the ID column and
preserving every pose and camera reference — a deterministic, auditable
remap rather than a hope that the versions agree.

**Verification is by direct numeric diff**, not `model_comparer` (absent
in 3.6) — parse the 8 OPENCV parameters and every image's
`QW,QX,QY,QZ,TX,TY,TZ` from input and output, match by NAME, and assert
the maximum absolute difference is zero. The 3.11 analysis recommended
this as the rigorous check independently of the version question, and it
is the only option here.

The workstation's 3.11.1 remains available as a fallback with better
semantics, but Apollo-side execution is preferred (the directive's
instruction, and the data is already there), so the 3.6 constraints are
the ones that bind.

### Discipline for this lane

Any COLMAP step that COULD alter poses or intrinsics runs on a
disposable copy first, and its output camera is compared numerically
against the shipped `cameras.txt` before anything is promoted. No
altered calibration enters the ImViD baseline without explicit review
and authorization. Unrestricted `mapper` / bundle adjustment / pose
estimation is not run at all.

## Open

The ingestion smoke past calibration — extraction of the 39 MP4s, a
bounded loader decode, numeric and visual reprojection validation, a
train/test split, and the `point_triangulator` step needed for any
initialization — has NOT run. No exploratory ImViD baseline has been
launched, and per the directive one may not be until the loader and
calibration checks pass and a focused review is obtained.
