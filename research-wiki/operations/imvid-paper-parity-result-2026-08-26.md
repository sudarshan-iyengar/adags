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

PENDING.

## 8. Deviations, failures and cost

PENDING.

## 9. Recommended starting point for the later gating pair

PENDING.
