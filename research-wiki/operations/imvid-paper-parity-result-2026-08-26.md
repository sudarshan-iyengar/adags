# RESULT — ImViD Opera/Puppy paper-protocol parity (2026-08-26)

EXPLORATORY, `evidence_bearing: false`. Frozen protocol:
[[imvid-paper-parity-freeze-2026-08-26]]. Read that page first — it fixes the
windows, the split, the metrics, the arms and the endpoints, and every claim
below is bounded by its §1 parity table and its §10 evidence boundary.

**Status: IN PROGRESS.** Sections marked `PENDING` have not run. Nothing in
this page may be read as a completed comparison until §7 carries numbers.

## 1. Provenance

| | |
|---|---|
| branch | `apollo/imvid-paper-parity`, forked from `22daf58` |
| remote | `github.com/sudarshan-iyengar/adags`, verified to resolve to the exact local SHA at each push |
| pool | Apollo `dgx` / V100 (resolves the open hardware contradiction; see freeze §9.2) |
| image (final) | `sudarshaniyengar/adags@sha256:02ad9cb41d0b613db73c0cee3777e547899c42dd2b93220edd30317d7f04b1e6`, tag `apollo-v100-searaft-afc4200` |
| SEA-RAFT | `princeton-vl/SEA-RAFT` @ `886fb094fe21d4fa5ff675da18362b27b023ccc3`, cloned INTO the image at that commit |
| flow checkpoint | `Tartan-C-T-TSKH-spring540x960-M.pth`, sha256 `adcc169244e99d4e6fe645b60aa8eaf3e4263698a3e870b8fbae618e3d2acc28`, read from `/apollo` and hash-verified per run |

Two earlier image digests exist and are recorded rather than hidden, because
one of them produced real artifacts:

| digest | tag | what it was |
|---|---|---|
| `sha256:44a373ec…62a61` | `…-577dee6` | first working build; **produced the window extractions**. Differs from the final image only by `pytest`, a `chmod`, and a newer copy of the seed script — none of which touch pixel output |
| `sha256:f65e0543…f8d4` | `…-61c847b` | added `pytest`; still failed the runtime SEA-RAFT gate on file permissions |

## 2. Dependency closure — what was actually missing, and why a green build was not enough

SEA-RAFT's `requirements.txt` omits **two of its own hard imports**:
`core/raft.py` does `from huggingface_hub import PyTorchModelHubMixin` at
module top level (`RAFT` inherits from it), so the import fails before a
checkpoint is touched, local or not; `h5py` is imported by
`core/datasets.py` and `core/utils/frame_utils.py`. Conversely `einops`,
`matplotlib` and `tensorboard` ARE in that file and are imported nowhere in
the checkout — they were deliberately not installed.

**No custom CUDA extension is required.** The only reference,
`core/corr.py`'s `alt_cuda_corr`, sits inside a bare `try/except: pass` and
is never called; the correlation block is pure PyTorch. Nothing in the source
requires torch > 2.0 — no `torch.compile`, no
`scaled_dot_product_attention`, no `torch.func`, no bf16 path — so
CUDA 11.8 / torch 2.0.1 / sm_70 is compatible.

Four defects were found and repaired, each by a check rather than by reading:

1. **Implicit relative imports.** `core/raft.py:7` is
   `from update import BasicUpdateBlock`, not `from .update import …`, so
   `core/` must be on `sys.path` in its own right. Upstream hides this by
   running every entrypoint from the repo root; importing SEA-RAFT as a
   library does not inherit that. **Caught by the build-time gate**, which
   constructs `RAFT` behind a poisoned proxy.
2. **An unconditional ImageNet download.** `RAFT.__init__` → `ResNetFPN`
   `_init_weights` calls `resnet34(weights=IMAGENET1K_V1)` at construction,
   before the flow checkpoint loads — weights the checkpoint then completely
   overwrites. It cannot change a result and can only fail a task at start.
   Seeded into the image.
3. **A stale baked copy.** The runtime `TORCH_HOME` opt-in lives in
   `searaft_sys_path()`, and the image had been built from a commit predating
   it. **The build passed anyway**, because the build step sets `TORCH_HOME`
   on its own command line — so the gate proved something the runtime did not
   enjoy. Caught only by running the gate again *without* that variable.
4. **Root-owned weights.** The build runs as root; a Determined task does
   not. The seeded file was present, `TORCH_HOME` pointed at it, and the task
   still died on a `PermissionError` — surfacing several frames inside
   torchvision's loader, where it looks like anything but a permissions
   problem.

**Method note worth carrying:** items 3 and 4 were both invisible to a
successful `docker build`. A build-time gate proves the build environment,
not the runtime one, whenever it supplies any variable the runtime will not.

## 3. Container validation — every branch that skips on the workstation

Run inside the final image (`OVERALL_FAIL=0`). The workstation has no
`torch`, `cv2` or `plyfile`, so these branches are only ever exercised here.

```
imvid_to_blender.py --mode self-test      18 checks, checks_skipped: []   (cv2 branch live)
imvid_extract_window.py --self-test       14 checks
imvid_framewise_init.py --self-test        5 checks
imvid_flow_searaft.py --self-test          7 checks
imvid_build_population.py --self-test      7 checks
pytest tests/test_imvid_point_cloud_time_extent.py    6 passed
imvid_searaft_seed.py --build-check       RAFT constructed with no network, 19,663,876 params
```

## 4. Window recovery — the one place a published gap was closed by measurement

Full record and the numbers: freeze §3.1. **Opera's public 300-frame sample
is frames 0-299 of the full take**, offset exactly 0, agreed independently by
`cam00` and `cam20` across 14,916 candidate offsets. Puppy's window is
unrecoverable and is declared by an outcome-blind rule.

## 5. Inputs and preprocessing

### 5.1 Takes, verified directly at zero GPU slots

| | Opera | Puppy |
|---|---|---|
| frames | 15,215 | 5,936 |
| duration | 253.836917 s | 99.032267 s |
| rate | `60000/1001` | `60000/1001` |
| raster | 5312x2988 `yuv420p` h264 | 5312x2988 `yuv420p` h264 |
| camera model | **OPENCV** | **OPENCV**, with its OWN intrinsics |

**Puppy's camera model had never been read** by this project; it is read from
the data here. Its `p1` is OPPOSITE IN SIGN to Opera's, so each scene derives
its own undistortion maps — reusing Opera's would displace every feature
while leaving poses and intrinsics superficially correct.

### 5.2 Opera window extraction — VERIFIED

```
schema          imvid-window-extract-v1
window          start 0, count 300, end_inclusive 299, of a 15,215-frame take
agreed_stream   5312x2988 yuv420p 60000/1001, all 39 cameras agreeing
renumbering     source 0..299 -> output 0..299; offset recorded in the manifest, never a filename
fps             60000/1001, time_duration [0.0, 4.988316666666667], suppressed_offset 0.0
verification    images_expected 11700, images_written 11700, 300 files for every one of 39 cameras
                raster checked from the PNG IHDR of EVERY written file
total_bytes     137,142,068,651
```

### 5.3 Undistortion to the 2656x1494 PINHOLE raster

The converter runs its own 19-check self-test before converting; inside the
image all 19 pass with `checks_skipped: []`, so the cv2 branch — the
undistortion maths itself — is exercised rather than skipped as it is on the
workstation.

Measured throughput: **0.35 images/s single-threaded** (9.3 h per window),
**2.12 images/s** threaded. PNG is lossless at every level and each image is
independent, so neither the encoder effort nor the worker count changes a
single output byte.

## 5A. Evaluation path — PROVEN before anything depends on it

The frozen `--val` convention was run end to end on an existing checkpoint,
so that the LPIPS weight fetch and the checkpoint-restore path could not fail
for the first time at the end of a 12,000-iteration run:

```
PSNR 20.558157   SSIM 0.761628   LPIPS 0.584353   num_GS 20157   static 0
stats/validation.json written
```

**This is a PATH PROOF, not a result.** It is the 600-iteration plumbing
smoke, on the superseded 4-camera split, and its PSNR may not be compared
with anything.

## 5B0. Puppy conversion — DONE, and it derived its OWN camera

```
images        11,700/11,700
split         profile paper_cam00, held_out ['cam00'], 38 training cameras
derived cam   PINHOLE 2656x1494  fx 1341.9557193  fy 1345.6522931  cx 1327.75  cy 746.75
invalid_fraction  0.0
required_config   eval true, resolution 1, motion_track_dt 1001/60000,
                  time_duration [0.0, 4.988316666666667]
```

**Puppy's focal lengths differ from Opera's** (1301.6663 / 1301.1218), which
is the check that matters here: the freeze recorded that Puppy ships its own
OPENCV intrinsics with `p1` opposite in sign, and reusing Opera's maps would
have displaced every feature while leaving poses and intrinsics superficially
correct. The two scenes demonstrably went through their own derivation.

`cx`/`cy` coincide at 1327.75 / 746.75 because both scenes share the same
5312x2988 raster and both put the principal point at the image centre; that
is the raster, not a shared camera.

## 5C. Framewise triangulation — the cost model was wrong, and correcting it improved the plan

Planning assumed SIFT feature count scales with pixel area, so halving the
raster would cut exhaustive matching ~16x (it is O(F^2) per camera pair).
**It does not**: COLMAP caps detections at `max_num_features` = 8192 and a
detailed scene saturates that cap at BOTH rasters, so the dominant term never
moved. Measured, one frame on all 80 cores: **>= 14 minutes against a
projected 90 s.**

The cap is the lever the raster is not, and unlike `--max-image-size` it does
not touch keypoint localisation, so the frozen 2.0 px native reprojection gate
is unaffected:

| `max_num_features` | s/frame | cameras contributing | mean track length |
|---:|---:|---:|---:|
| 8192 (default) | >= 840 | — | — |
| 2048 | 89 | 38 | 4.152 |
| **1024** | **33** | **38** | **4.110** |

Thinner, not degenerate — all 38 training cameras contribute at every cap.
Instantiated per freeze A5: **1024 features, stride 3, 100 frames**, which is
MORE temporal coverage than the stride it superseded (100 frames vs 50), paid
for in points-per-frame. For an arm whose mechanism IS per-frame temporal
assignment, frames are the axis that carries the mechanism.

Production, first frames: `points 2428 / 2734 / 2857` at ~60 s each with 3
workers — projecting ~270k points over 100 frames, below the 300k initial cap
so no subsample binds.

**A silent defect surfaced in the same probe.** The point collector globbed
for `points3D.txt` and matched `model_in/` — the EMPTY INPUT file
imvid_sparse_init.py writes, which sorts before `model_txt/`. It reported
**0 points from a reconstruction with 38 contributing cameras and a mean track
length of 4.11**. Unrepaired it yields one cloud file per frame, an
initializer with no geometry, and success codes throughout. The output model
is now named rather than discovered, and zero points from a successful COLMAP
run is a refusal.

### 5C.1 Flow and triangulation must be SERIALISED, measured not assumed

Run concurrently on one node, triangulation managed ~1.0 frames/min and flow
fell from 3.82 to ~1.2 pairs/s. With flow paused, triangulation measured
**2.99 frames/min — a 3.0x speedup.** Both stages are CPU-and-NFS hungry
(COLMAP saturates every core; flow copies two 2 MB PNGs per pair), so
overlapping them costs more than it buys:

```
parallel:   ~90 min for both
serial:     ~23 min triangulation + ~24 min flow = ~47 min
```

Recorded because the obvious intuition — flow is GPU work, triangulation is
CPU work, so overlap them — is wrong here, and the pipeline defaults should
reflect the measurement rather than the intuition.

## 5B. The initialization seam — PROVEN through the CUDA path

The one link nothing upstream could verify: that a cloud carrying per-point
`time` and `t_extent` is actually consumed by `create_from_pcd`, under the
`paper_cam00` split, and trains. A cloud was built from the real
20,157-point union with its rows split evenly across the three declared
support bands, assembled into an arm root, and trained for 20 iterations;
`_scaling_t` was then read back out of the checkpoint (capture-tuple index
14) and converted to a temporal standard deviation:

```
train cams 38   test ['cam00']
Number of points at initialisation : 20157      <- NOT the 1,000,000 num_pts fallback
rows 20157 | distinct temporal centres 16636
recovered temporal std  min 0.12979  med 1.00572  max 2.59613
  within 5% of compact (0.13347 s):  6719 rows
  within 5% of broad   (2.49416 s):  6719 rows
  within 5% of default (0.99942 s):  6719 rows
VERDICT: ALL THREE SUPPORT BANDS LANDED
```

6,719 is exactly 20,157/3, the construction. The seam is verified end to
end: writer -> `fetchPly` -> `BasicPointCloud` -> `create_from_pcd` ->
`_scaling_t` -> checkpoint.

**This probe is also what confirmed the reviewer's most speculative finding.**
Before the repair it failed with `given numpy array strides not a multiple of
the element byte size` — the reviewer had predicted exactly this from the
35-byte record stride and could not test it without torch. It is a genuine
production failure on a path no cloud in this repository had ever exercised,
and it would have stopped every arm at initialisation.

**Renders at 1/sqrt(2):** per
[[temporal-marginal-applied-twice-2026-08-26]] the rendered supports are
0.09437 / 0.70669 / 1.76364 s, not the stored 0.13347 / 0.99942 / 2.49416.

## 5B1. Puppy triangulation — DONE

```
frames_requested 100   frames_ok 100   frames_failed []
total_points     389,188          cameras_used 38   excluded ['cam00']
max_num_features 1024             stride 3
points/frame     min 3,694  median 3,898  max 4,038
```

Denser than Opera (median 3,898 against ~2,800) at the same feature cap,
consistent with a more textured scene.

**389,188 EXCEEDS the 300,000 initial cap of freeze A1.1**, so Puppy's NF
population WILL be subsampled — uniformly, without replacement, at a fixed
seed, with both the pre-cap and post-cap counts recorded. Opera's 282,672 sat
below the cap and was not subsampled. That difference is between SCENES, not
between arms, so it cannot move NF relative to FG within either scene; it
does mean the two scenes' NF arms are not capacity-matched to each other, and
no cross-scene capacity comparison may be drawn from them.

## 5D. Opera NF initializer — BUILT AND VERIFIED

```
framewise clouds      100/100 frames (stride 3), 282,672 points, 38 training cameras
points per frame      2,428 / 2,734 / 2,828 / 2,967 / 2,657 / 2,962 / 3,001 / 2,823 (sampled)
written points        282,672  (below the 300,000 cap, so no subsample binds)
distinct timestamps   100
window_span_seconds   4.988316666666667   <- EXACTLY the config's time_duration
sampled_span_seconds  4.954950            <- frames 0..297
support bands (s)     broad 2.4941583  default 0.9994153  compact 0.1334667
out_ply_sha256        0290431490880133932165077c56215289b8a0c3062fba3201fb8a143ad49994
held-out provenance   verified: true, upstream cameras_used = cam01..cam38, upstream_excluded = cam00
arm roots             arm_nf (38 train / cam00 test), arm_nf_dev (37 train / cam10 test, cam00 absent)
```

The abstain band, 0.9994153 s, is exactly `(time_duration / 5) ** 0.25` — the
value the trainer would use with no `t_extent` column at all. That equality is
the point of the band, and it is the thing the defect below broke.

### 5D.1 A defect the exit code did not show

The FIRST NF build returned 0 and wrote a complete cloud with support bands
**three times too narrow**: `broad 0.826 s` where the window implies 2.494,
and an abstain band of `0.758 s` against the trainer's own 0.999.

`span` was derived from the NUMBER OF TRIANGULATED CLOUDS rather than from the
window. At stride 1 those coincide, so it was invisible until striding was
introduced; at stride 3, 100 clouds spanning frame indices 0..297 gave
1.652 s against the true 4.988 s. The per-point TIMESTAMPS were correct
throughout, so two descriptions of the same window disagreed with nothing to
flag it.

Now derived from the declared window, with a refusal if the clouds span more
than it, and a self-test pinning the 3.02x discrepancy the stride produces.

**This is the fourth defect this block whose only symptom was a wrong number
inside an otherwise-successful run**, alongside the reader dropping
`t_extent`, the collector reading the empty input model, and the degeneracy
guards that could not fire. The pattern is consistent enough to state as
method: on this pipeline the return code carries almost no information, and
every stage is checked by reading its emitted manifest.

## 5E. Opera FG initializer — BUILT, and its engagement preconditions PASS

Classified over all 282,672 candidate observations (100 frames x ~2,800):

```
static    264,879   93.71%
abstain    14,820    5.24%
dynamic     2,973    1.05%
points seen by NO training camera: 0
```

**Every frozen precondition of freeze §6.1 holds.** The three-way split is
real and far from the 99% degeneracy floor; abstention is preserved rather
than forced; the classification is not all-static, all-dynamic or
all-abstain; and every candidate point projects into at least one training
view, so no classification rests on absent evidence.

**The reviewer's F11 worry did not materialise, and inverted.** The concern
was that a maximum over 38 views with no occlusion test would flag nearly
everything dynamic (`1 - 0.95^38 = 0.86`). Measured, dynamic is **1.05%**.
Adjacent-frame motion at 59.94 fps is simply too small — mean 0.17 px — for
the union to saturate.

### 5E.1 The written population, and the capacity asymmetry it creates

```
written   static 2,032 (reference frame only) + dynamic 2,973 + abstain 14,820
          = 19,825 points
NF                                            = 282,672 points
```

**FG starts with 7% of NF's geometry — a 14x difference.** That is the
literal consequence of the two constructions the directive specifies, and it
is legitimate ("the point population that construction legitimately
produces") — but it is large enough that it must be read alongside every
metric rather than discovered afterwards.

`static_duplication_reduction` is **0.992329**, and it must be read beside
`static_dropped_non_reference` = **262,847**. Those 262,847 are not all
redundant copies: static surfaces first revealed after the reference frame
have NO initial primitive in FG and do have one in NF. Part of any FG-vs-NF
difference is therefore initial coverage, not the flow mechanism.

**It also makes the arms differ in COST, not just in content.** NF's 282,672
points measure ~4 s/iteration against roughly 1 s for a ~20k population, so
the same 12,000-iteration schedule is a very different amount of compute per
arm.

### 5E.2 Flow direction provenance — stated precisely

The production flow's OWN direction check had nothing to evaluate. The shards
were paused and resumed (see 5C.1), and on resume every existing pair is
skipped before a direction record can be collected, so `direction_records`
was empty and the check was correctly skipped rather than passed.

**Direction is therefore established by the dedicated probe, not by the
production run**: cam01 at gap 8, reversed/forward warp-error ratios
**1.358 and 1.399** on 83,742 and 89,263 evidence pixels, same scene, same
model, same checkpoint. That is sound evidence, and it is recorded as what it
is rather than implied to be a self-check of the production artifact.

## 6. Flow, and the initializer engagement checks

### 6.1 Measured flow magnitudes (an input property, not an outcome)

Adjacent-frame motion on Opera at the training raster, in full-raster pixels:
**mean 0.12-0.23, p99 0.93-3.69, max 7-12.** Fields verified
`(747, 1328, 2)` float16, finite, `stored_raster [1328, 747]`,
`source_raster [2656, 1494]`.

### 6.2 Direction is measured — and the first test was the thing that was wrong

The warp test failed at ratio **1.083** (gap 1) and **1.132** (gap 8) against
a declared 1.2 floor, for a field that is in fact correctly oriented. Cause:
the test averaged over the whole frame, and an opera stage is ~95% static —
static pixels warp to the same place under `+flow` and `-flow` and dominate
the mean. Restricted to pixels carrying >= 1.0 px of displacement, with a
minimum evidence count below which the test ABSTAINS loudly:

```
cam01 f1 gap 8  evidence 83,742 px  forward 9.772  reversed 13.275  ratio 1.358
cam01 f2 gap 8  evidence 89,263 px  forward 9.151  reversed  ...     ratio 1.399
```

**Flow is FORWARD.** The held-out exclusion was exercised rather than
trivially satisfied — `cam00` was staged alongside the training camera and
the run recorded it as excluded and unread.

### 6.2a Throughput, and why flow is sharded

Measured on one V100: **0.96 adjacent pairs/s**, so one scene's
38 x 299 = 11,362 pairs is **3.2 h**. Split across four GPUs by camera the
measured rate is **3.82 pairs/s** — a clean 4.0x — bringing a scene to ~50
minutes.

The shard list NARROWS and can never widen: the held-out exclusion is applied
first and independently, and naming a held-out camera inside a shard is
refused rather than quietly honoured. The four shards partition all 38
training cameras exactly, with `cam00` absent from every one.

### 6.3 Engagement preconditions

The frozen anti-vacuity preconditions of freeze §6.1 are evaluated before any
FG score is read; an FG run failing one is INVALID rather than null. After
the independent review the guards were rebuilt: degeneracy is now a SHARE of
commensurable counts rather than exact equality over a per-frame count and
two per-window ones, a collapsed static set refuses, and the held-out check
asserts against the framewise manifest's recorded camera set rather than
restating a structural guarantee. PENDING the first FG build.

## 7. Results at 6k and 12k

**PENDING — both arms are training.** Run identities, verified from their own
emitted manifests rather than from the submission command:

```
295  imvid_opera_nf   dgx  commit b9e89bf861f32716bf1735a5cdc6ac223dd396c2
     image  sha256:02ad9cb41d0b613db73c0cee3777e547899c42dd2b93220edd30317d7f04b1e6
     seed 0   evidence_bearing false   projected_gpu_hours 14.0
     run_dir /apollo/users/sri/proj_adags/runs/elgs/20260826T020643Z_imvid_opera_nf_0_b9e89bf
     input.ply 9,893,798 B  (282,672 points)

296  imvid_opera_fg   dgx  same commit, same image digest, seed 0
     run_dir /apollo/users/sri/proj_adags/runs/elgs/20260826T020708Z_imvid_opera_fg_0_b9e89bf
     input.ply   694,152 B  (19,825 points)
```

The two `input.ply` sizes are the cheapest available proof that each arm
received its OWN initializer rather than a shared one.

Each carries `archive_sha256`, `config_canonical_hash`,
`rendered_config_sha256`, an O_EXCL claim path and a ledger line. Both are
`evidence_bearing: false`, matching the freeze's status.

**No number may be entered here from the training-time path.** The 6k and 12k
figures come from a separate `--val` pass over `chkpnt6000.pth` and
`chkpnt12000.pth` in the clamped / pooled-PSNR / LPIPS-AlexNet convention
frozen in §7 of the freeze.

## 7B. FIRST Cam 00 NUMBER — Opera FG at 6,000 iterations

Determined experiment **299**, the frozen `--val` path over all 300 held-out
Cam 00 frames, reading `chkpnt6000.pth` of experiment 298:

```
PSNR 22.961224   SSIM 0.830659   LPIPS 0.531246   num_GS 347,600
```

Published Opera row for context — external anchors, never tuning targets:

| | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Gaussian4D | 25.61 | 0.873 | 0.206 |
| STG | 26.30 | 0.899 | 0.169 |
| IVV (Ours) | 33.51 | 0.916 | 0.070 |
| **this, FG @ 6k** | **22.96** | **0.831** | **0.531** |

**Four things this number is NOT:**

1. **Not a completed arm.** 6,000 of 12,000, with the LR decay horizon and
   `densify_until_iter` both set for 12,000 — it is a mid-trajectory reading,
   not an endpoint.
2. **Not a comparison.** The NF counterpart does not exist; every NF attempt
   so far died of memory (§7C). The lane's actual question is untouched.
3. **Not the frozen configuration.** Experiment 298 ran under the 400,000
   ceiling of amendment A6, which is being superseded because the V100 cannot
   reach it either. This measures THAT setup.
4. **Not comparable to the published table** beyond the crudest ordering: no
   method parity, a different schedule, and — for LPIPS especially — a
   convention that has not been cross-checked against whatever the authors
   used.

**The number worth watching is LPIPS.** It moved 0.584 -> 0.531 between the
600-iteration plumbing smoke and 6,000 iterations, while PSNR moved
20.56 -> 22.96 over the same interval. A perceptual metric that barely
responds while PSNR climbs is the signature of a reconstruction that is
getting the low frequencies right and not the detail — which is what one
would expect of an arm that begins with 19,825 primitives. It is stated as an
observation, not a diagnosis; the 12k reading and the NF arm are what would
settle it.

## 7C. TWO FINDINGS FROM THE FG READINGS, and both outrank the numbers themselves

| run | arm | iter | pool | ceiling | PSNR | SSIM | LPIPS | points |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 299 | FG | 6,000 | V100 | 400k | 22.961 | 0.8307 | 0.5312 | 347,600 |
| 302 | FG | 12,000 | V100 | 400k | **19.758** | 0.7993 | 0.5446 | 399,709 |
| 303 | FG | 6,000 | H100 | 600k | 24.347 | 0.8380 | 0.5346 | 350,525 |

### 7C.1 Held-out quality DEGRADES from 6k to 12k, by 3.203 dB

Same run (298), two checkpoints. Its own training trajectory:

```
total_points  1:19,825  2k:52,760  4k:181,740  6k:347,600  8k:399,780
              10k:399,710  11k:399,710  12k:399,710      <- FROZEN from ~8k
train loss    6k:0.13663  8k:0.13461  10k:0.10845        <- still improving
train l1      6k:0.01719  10k:0.01472                    <- still improving
HELD-OUT      6k:22.961            12k:19.758            <- WORSE by 3.203 dB
```

**Training loss falls while held-out PSNR falls.** That is overfitting, and
there is a mechanism for why it is severe here: `main.py:1658` gates the
ENTIRE densification block on the point count, so once the population reaches
the ceiling at ~8,000 iterations, **pruning and opacity reset stop as well as
densification**. From there the model has a frozen topology of 399,710
primitives and can only fit them harder to the 38 training views.

**Consequence for the frozen endpoint: 12,000 iterations is PAST the optimum
for this configuration.** That is a statement about the user-directed
endpoint and is raised, not acted on. It is also NOT yet known to generalise:
the 600,000-ceiling runs have far more headroom (301 was at 350,525 at 6k)
and may never freeze, in which case the mechanism above would not apply.

### 7C.2 A cross-hardware replicate difference of 1.385 dB

299 and 303 are the SAME arm at the SAME iteration with the same seed, and
**neither had reached its ceiling** (347,600 against 400,000; 350,525 against
600,000), so up to 6,000 iterations the two runs should be behaviourally
identical. They differ by **1.385 dB PSNR** and by 0.8% in point count.

**Any NF-vs-FG difference smaller than ~1.4 dB is therefore not resolvable by
single runs on this protocol.** This is the ImViD instance of a result this
project already holds on N3V — that the substrate is chaotic and
densification is the amplifier
([[operations/b1f-flow-postmortem-2026-08-23]]) — and it means the eventual
NF/FG comparison needs its difference reported against this floor, not
against zero.

The two runs also differ in ceiling and in hardware, so the 1.385 dB is an
UPPER bound on pure run-to-run variance and cannot be attributed to any one
of the three. Separating them would need replicates, which the block has not
paid for.

## 7D. THE COMPLETE MATCHED PAIR, at both frozen endpoints

All four cells below are the SAME pool (hopper/H100), the SAME ceiling
(600,000), the SAME seed and the SAME frozen 12,000-iteration schedule. Only
the initial population differs, which is the one variable this lane exists to
vary.

| run | arm | iter | PSNR | SSIM | LPIPS | points |
|---|---|---:|---:|---:|---:|---:|
| 300/304 | NF | 6,000 | 26.805 | 0.8783 | 0.3853 | 599,333 |
| 300/306 | **NF** | **12,000** | **26.855** | **0.8868** | **0.3673** | 599,305 |
| 301/303 | FG | 6,000 | 24.347 | 0.8380 | 0.5346 | 350,525 |
| 301/305 | FG | 12,000 | 24.083 | 0.8414 | 0.5026 | 599,698 |

**NF - FG at 6,000:** +2.458 dB PSNR, +0.0403 SSIM, -0.1493 LPIPS.
**NF - FG at 12,000:** +2.772 dB PSNR, +0.0454 SSIM, -0.1353 LPIPS.

Both clear the 1.385 dB replicate floor of §7C.2 by roughly a factor of two,
in the same direction, on all three metrics, at both endpoints.

### 7D.1 The capacity confound is present at 6k and ABSENT at 12k

This is the load-bearing observation of the whole comparison, and it was not
designed for -- it fell out of the ceiling being reached by both arms.

At 6,000 iterations NF carries **1.710x** FG's primitives (599,333 against
350,525), exactly the confound §5E.1 predicted when the populations were
built: FG starts with 7% of NF's points, so part of any 6k difference is
initial coverage rather than the flow mechanism.

At 12,000 iterations **both arms have converged to the same budget** --
599,305 against 599,698, a difference of 393 points, or **0.066%**. Capacity
is matched to within a fifteenth of one percent, and NF still leads by
**2.772 dB**.

**A capacity explanation of the NF advantage is therefore available at 6k and
NOT available at 12k.** The advantage is slightly LARGER at the matched
endpoint than at the confounded one, which is the opposite of what a
capacity-driven gap would do.

What this still does NOT establish: that flow-guided initialization is
harmful in general. It establishes that ON THIS WINDOW, at this schedule and
this ceiling, the flow-guided initial population reaches a worse held-out
optimum than uniform seeding does, and that the deficit is not explained by
the primitive budget. The FG population is 35,107 points against NF's
1,000,000-capped 300,000; a starting population two orders of magnitude
smaller may simply never recover, which is a statement about THIS
construction of FG, not about flow priors.

### 7D.2 12,000 iterations is the right endpoint at the 600k ceiling

NF's best held-out PSNR is at iteration 12,000 (`best_val_iter: 12000` in its
own summary), and it IMPROVES from 6k to 12k on all three metrics
(+0.050 dB, +0.0085 SSIM, -0.0180 LPIPS). FG loses 0.264 dB but improves on
both SSIM and LPIPS.

**This WITHDRAWS the concern raised in §7C.1** that 12,000 iterations is past
the optimum. That concern was raised from the 400,000-ceiling run and does
not survive contact with the 600,000-ceiling runs, exactly as §7C.1 itself
flagged it might not.

### 7D.3 CORRECTION, append-only: the frozen-topology mechanism in §7C.1 is REFUTED

§7C.1 attributed the -3.203 dB collapse of the 400k run to `main.py:1658`
freezing the entire densification block -- pruning and opacity reset included
-- once the point count reaches the ceiling. **That mechanism does not
operate, and the error was mine.**

`densify_and_prune` caps its own growth internally
(`scene/gaussian_model.py:2567-2568`):

```python
if max_total_points is not None and max_total_points >= 0:
    remaining_new_points = max(0, int(max_total_points) - int(self.get_xyz.shape[0]))
```

so the population **asymptotes just below the ceiling and never crosses it**.
Every run in this lane ends below its own cap -- 399,709 of 400,000;
599,305 and 599,698 of 600,000 -- so the outer gate's
`get_xyz.shape[0] < opt.densify_until_num_points` stays TRUE throughout and
the topology is never frozen. Pruning and opacity reset ran for the whole
schedule in all four runs.

The numbers in §7C.1 are unaffected; only the explanation is withdrawn. The
surviving candidate mechanism is **starved densification**: once the internal
cap binds, `remaining_new_points` is 0, so each opacity-reset cycle continues
to fire with no ability to regrow what it suppresses. That account is
CONSISTENT with the 400k run and INCONSISTENT with NF, which saturated its
budget by 6k and still gained. It is recorded as an open hypothesis and is
**not needed for anything this lane concludes** -- the three runs differ in
ceiling, hardware and arm simultaneously, so none of them can separate it.

**Carry as method:** the freeze reading was taken from a point-count
trajectory that plateaued, and a plateau at a cap looks identical whether the
cap is enforced inside the operator or outside it. Reading the gate condition
in source is what distinguished them, and it should have come first.


## 7A. What may NOT be concluded from this lane, restated before any number exists

- Not an exact reproduction of the ImViD paper: method parity is unavailable
  (no released implementation) and window parity holds only for Opera, and
  only in the weaker `window-constrained-by-measurement` sense.
- Not a statement about the published table. STG and IVV are external
  anchors; nothing here was tuned against them, and 12,000 iterations is not
  the paper's 30 epochs.
- Not a same-capacity comparison. FG begins with 7% of NF's points, and part
  of any difference is initial coverage rather than the flow mechanism
  (§5E.1).
- Not an independently verified rig status: Opera and Puppy remain
  SUPPLIER-DECLARED per the freeze's §10 evidence boundary.

## 8. Deviations, failures and cost

Every amendment is recorded append-only in
[[imvid-paper-parity-freeze-2026-08-26]] §Amendments 1-8. This section
records only what they cost and what they taught.

**Cells that produced no science (~7.8 slot-hours discarded):**

| exp | arm | pool | outcome |
|---|---|---|---|
| 295 | NF | V100 | OOM at iteration 2,866 / 516,990 points (600k ceiling) |
| 296 | FG | V100 | cancelled with 295 when the ceiling was amended |
| 297 | NF | V100 | OOM at iteration 2,032 / 399,865 points -- **at** the 400k ceiling |

Both OOMs died without a Python traceback, which is the signature of a
SIGKILL or a CUDA abort rather than an allocator failure inside PyTorch. The
parameter tensors are only ~580 MB at that count; the memory goes to the
rasterizer's per-Gaussian sort and tile buffers at 2656x1494. **Lowering the
ceiling from 600k to 400k did not fix it -- 297 died AT the lower ceiling** --
which is what settled that the V100's 32 GB is the binding constraint and
sent this lane to the H100.

**Two engineering findings that cost real time and are worth carrying:**

1. **A build-time gate proves the BUILD environment, not the RUNTIME one**,
   whenever it supplies a variable the runtime will not. The SEA-RAFT
   construction check passed inside the image build because the build step
   set `TORCH_HOME` itself; the first real run died on a `PermissionError`
   against root-owned weights. Fixed by baking `ADAGS_TORCH_HOME` and
   `chmod -R a+rX /opt/adags`.

2. **The triangulation cost model was wrong by ~16x** because I assumed SIFT
   feature count scales with pixel area. COLMAP caps at
   `max_num_features 8192` and both rasters saturate the cap, so halving the
   raster bought nothing. Measured feature-cap curve: 1024 -> 33 s/frame,
   2048 -> 89 s, 8192 -> >=840 s, all with 38 cameras contributing and mean
   track length 4.11-4.15. Correcting the model improved the plan rather than
   just the estimate.

**One defect found in the point collector, and it reported a FALSE ZERO.** It
globbed for `points3D.txt` and matched `model_in/` -- the empty INPUT model,
which sorts before `model_txt/` -- and so reported 0 points from a
reconstruction with 38 contributing cameras. Fixed by naming the output model
explicitly and refusing on zero points. A collector that returns 0 from a
healthy reconstruction is the failure mode that would have silently produced
an empty initial population.

**Two claims of mine were corrected during the lane, both append-only:** that
NF's ~4 s/iteration had been measured, when the preflight that would have
printed it was killed first; and the frozen-topology mechanism of §7C.1,
refuted in §7D.3 above.

**Provenance gap, recorded rather than papered over:** flow shard manifests
1-3 were never written, because the direction guard refused when resumed
shards carried too few fresh pairs. The flow DATA is complete (11,362 pairs,
all 38 cameras x 299) but per-shard provenance is not, so **the FORWARD
direction of the SEA-RAFT assets rests on the dedicated probe (ratios
1.358 / 1.399 on 83,742 / 89,263 evidence pixels), not on the production
run's own self-check.**

## 9. Recommended starting point for the later gating pair

**Recommendation: freeze the NF configuration --
`configs/imvid/opera_paper12k_nf.yaml` -- unchanged, on hopper/H100.**

Concretely: 600,000-point ceiling, `densify_until_iter: 10_000`,
`num_pts: 1_000_000`, 12,000 iterations, `position_lr_max_steps: 12_000`,
seed 0, image `sudarshaniyengar/adags@sha256:0d5771688c9b...`, Cam 00 held
out, 2x-downsampled 300-frame window at frames 0-299.

Four reasons, in order of weight:

1. **NF is the better substrate on every metric at both endpoints**, by a
   margin that clears the replicate floor twice over, AT MATCHED CAPACITY
   (§7D.1). A gating experiment run on the FG substrate would be building on
   a 2.772 dB deficit that has nothing to do with gating.

2. **12,000 iterations is where NF is best** (§7D.2), so the endpoint the
   user directed is also the endpoint the data supports. No amendment needed.

3. **The V100 is not viable for this window** and the H100 is (§8). This is
   settled, not a preference.

4. **NF has no initializer machinery in the loop**, so an episode-gating
   result on top of it is attributable to gating alone. FG adds a second
   moving part -- flow-derived support bands -- whose interaction with
   temporal gating is unknown and would confound the very thing the later
   pair is built to measure.

**What must be paid for before that pair is claim-grade**, and none of it was
in this lane's scope:

- **Replicates.** The 1.385 dB floor is an UPPER bound confounded by ceiling
  and hardware. A gating effect smaller than that is currently unresolvable,
  and the N3V programme already holds the general form of this result
  ([[operations/b1f-flow-postmortem-2026-08-23]]). At least 3 same-config
  replicates at one seed are needed before any gating delta is quoted.
- **A rendered-vs-declared temporal-width decision.** The marginal is applied
  twice ([[temporal-marginal-applied-twice-2026-08-26]]), so every authored
  support width renders at 1/sqrt(2) of its stored value. That is harmless
  for arm-vs-arm ratios and NOT harmless for any gating spec that authors an
  episode duration in seconds.
- **Puppy.** Experiments 307 and 308 are running as this is written; a
  single-scene recommendation should not be frozen across scenes until they
  return.
