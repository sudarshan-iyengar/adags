# FROZEN — ImViD Opera exploratory baseline: split, evaluation, budget

Date: 2026-08-17. Status: **FROZEN, NOT LAUNCHED.** Recorded before any
ImViD training. Three of the remaining gates close here (split,
evaluation conventions, compute/storage estimate); the visual
distortion check and the focused review remain open, and the baseline
may not launch until they close.

EXPLORATORY, `evidence_bearing: false`. Hopper/H100 is reserved for this
lane and has not been used.

## What is already closed

| gate | status |
|---|---|
| acquisition + integrity | CLOSED — sha256 verified, 41 files, read-only |
| decoding | CLOSED — 117 frames at 5312x2988, PNG IHDR-checked fail-closed |
| calibration, numerically | CLOSED — fixed-pose triangulation at 1.17-1.21 px over three frames, 39/39 cameras each |
| sparse initialization | CLOSED — 23,999-point union, hashed and manifested |
| initialization vs fallback | CLOSED — COLMAP cloud 12.8x more concentrated than a uniform fill of its own p01-p99 box |
| **split** | **CLOSED HERE** |
| **evaluation conventions** | **CLOSED HERE** |
| **compute / storage estimate** | **CLOSED HERE** |
| visual distortion check | OPEN |
| focused review | OPEN |

## Split — declared, and it is NOT ImViD's own

ImViD's paper holds out **one** camera (Camera 0 for ImViD/N3V/MeetRoom).
A single held-out view is too thin to carry a photometric claim, and this
project has already been burned once by ranking on a thin split
([[operations/diva360-scissor-sweep-matrix-v1]]: a +0.909 dB development
win that reversed on the official split).

**Frozen split for this lane:**

* **held-out (test): 4 cameras**, chosen by the same outcome-blind rule
  used throughout this project — evenly spaced through the sorted camera
  id list, reading ids only, never a metric. With 39 cameras
  (`cam00`-`cam38`) that is `np.linspace(0, 38, 4)` rounded =
  **`cam00, cam13, cam25, cam38`**.
* **training: the remaining 35 cameras.**
* `cam00` is deliberately INCLUDED in the held-out set so that a
  single-camera number comparable to ImViD's own protocol can be reported
  as a SUBSET of ours, without ever having trained on it.

**Prohibited held-out information**, explicitly: imagery, COLMAP
observations, triangulated points, or any metric from `cam00`, `cam13`,
`cam25`, `cam38` may not influence training, initialization, weights,
stopping or selection.

**This invalidates the current initialization and it must be rebuilt.**
The frozen sparse clouds were triangulated using ALL 39 cameras
([[operations/imvid-sample-ingestion]]), so they carry observations from
the four held-out views. That is initialization-time leakage. Before the
baseline runs, `scripts/imvid_sparse_init.py` must be re-run on the
35 training cameras only, and the union rebuilt. The 39-camera artifacts
are PRESERVED as the calibration-validation evidence they were built for
— they are sound for that purpose and unsound as an initializer.

Recorded rather than quietly fixed, because the 39-camera clouds already
exist and it would have been easy to reuse them.

## Evaluation conventions — frozen

* Metrics: PSNR, SSIM, LPIPS, reported under **both** convention sets and
  never averaged across them, per
  [[operations/diva360-protocol-parity-audit]]. Convention deltas are
  MEASURED per model, never applied as constants — that page shows the
  SSIM gap varies with quality regime and the LPIPS backbone delta
  changes sign.
* Resolution: the training resolution, declared per run. ImViD's own
  baseline evaluates at 2x downsample; whatever this lane uses is stated
  in its config header and not changed afterwards.
* Background: ImViD ships no masks, so there is NO alpha compositing and
  NO black-background convention here. Full-frame metrics only. This is
  a real difference from the DiVa-360 lane and any cross-dataset
  comparison must respect it.
* Temporal: all 300 frames of the sample, or a declared contiguous
  subset stated in the config.
* The held-out four are scored **once**, at the end, on the final model.

## Compute and storage — estimated from measurements, not guesses

Storage, measured:

```
archive (read-only, preserved)                     0.93 GiB
extracted mp4s                                     0.93 GiB
decoded frames, 3 frozen frames x 39 cameras       0.856 GiB  (7.5 MiB/frame measured)
sparse reconstructions + manifests                 < 0.1 GiB
```

Full-decode projection, from the MEASURED 7.5 MiB per frame:

```
300 frames x 39 cameras x 7.5 MiB = 85.7 GiB
```

That supersedes the preflight's ~557 GB raw-RGB arithmetic, which
described uncompressed RGB rather than PNG. Apollo had **31.165 TiB**
free, so storage is not a constraint. A decode of all 300 frames is
therefore affordable but is NOT authorized here; the baseline's frame
coverage is declared in its own config.

Compute: NOT estimated from ImViD's reported numbers, deliberately. Their
baseline trains 30 epochs over full permutations of the training frames
on an A100 with unreleased code, which is not this trainer. The honest
estimate comes from this project's own measurements: the DiVa-360
benchmark baseline ran 6000 iterations at 1160x550 over 4,935 training
units in ~1.9 h on a V100. ImViD at 5312x2988 is **~23x the pixels per
image**. Any first ImViD run therefore starts from a declared downscale
and a bounded iteration count, and its cost is MEASURED on a short
preflight before a full run is submitted — the same measure-then-commit
discipline used for the oracle and the acceptance path.

## Hardware

DGX/V100 is NOT used for this lane. Hopper/H100 is reserved for it, and
the whole ImViD comparison stays on one hardware class. H100 runtime is
never compared against V100 runtime as though hardware were controlled.
To record when a run happens: exact H100 model, resource pool, container
image digest, CUDA and dependency versions, numerical settings, seed,
runtime, peak memory, experiment and task ids.

## What this baseline cannot establish

Full-dataset performance; 1-5 minute scalability (the sample is 300
frames = 5 seconds); moving-rig validity (Opera is fixed-point only);
disappearance/reactivation supply. And ImViD's reported flow/depth gains
are NOT an expected gain for this lane, for scissor, or for N3V — they
were measured on a different implementation, a different representation,
and a deliberately peripheral held-out camera.

---

# PILOT APPENDIX (2026-08-18) — the 35-camera rebuild, the undistortion measured, and the access reconciliation

Append-only. Nothing above is rewritten. This appendix closes the "visual
distortion check" gate, closes the initialization-leakage item the split
section opened, and records a **blocking access finding** that changes what the
ImViD lane can do next.

## A1. THE ACCESS FINDING — there is no full-release access path in this environment

The 2026-08-18 strategy document states that full Google-Drive access to ImViD
now exists: 8 folders, 7 scene folders with one take each at ~105–120 GB, a
separate `moving_rig` folder, ~0.9 TB accessible against a published ~2.07 TB.
The execution block was directed to mirror that release scene-wise.

**It was not possible, and the reason is not a permissions error — it is that
no such path exists here.** Verified directly:

| probe | result |
|---|---|
| `rclone listremotes` | exactly two remotes, `leonardo:` and `apollo:`, **both `type = sftp`**. No Google Drive remote of any kind. |
| `gdown`, `gcloud`, `gsutil` on PATH | none installed; `import gdown` fails |
| repository + `agent-control/` grep for `drive.google`, `folders/1`, `uc?id=`, `gdown` | **zero hits** |
| local filesystems | `C:` and `D:` only; no mounted Drive |
| earlier handovers | no Drive listing, folder id, or share link recorded anywhere |
| GitHub Releases API for `Metaverse-AI-Lab-THU/ImViD` | **two releases, `v0.1` and `v0.2`, each with a SINGLE asset `scene1_opera.zip` (~1.0 GB).** No full scene is publicly downloadable. |

So the only ImViD data reachable by any tool available here is the 300-frame
Opera sample already on Apollo. The Drive access the strategy document
describes may well exist in the user's own browser session; it is not reachable
programmatically, and the standing safety rules forbid manufacturing a path
from browser credentials.

**This is recorded as a conflict resolved in favour of verified state**, per the
directive's own precedence rule. Consequences:

* **D1 release reconciliation cannot be completed.** The published-versus-
  accessible discrepancy (~0.9 TB vs ~2.07 TB) cannot be explained from here,
  because the accessible listing cannot be enumerated. The original page's
  statement stands and is the only verified one: the full dataset requires an
  application form emailed to the authors plus manual approval.
* **D3 raw acquisition did not start** and no bytes were transferred. Nothing
  is half-downloaded and there is no transfer to resume.
* **D5, the first complete-take pilot on Meeting, is blocked** at its first
  step. Its frozen checks are specified and unchanged; what is missing is the
  take.
* **What DID proceed** is everything that needs only the existing sample:
  sections A2–A4 below.

**Exactly what would unblock it** — a user action, not an agent action: either
(a) configure an authorized rclone Drive remote and name it, after which
`rclone copy` of one scene folder at a time is a single command with a manifest
and per-file hashes; or (b) supply the accessible folder listing (names, file
counts, sizes) so the reconciliation can be done on metadata alone.

`moving_rig` remains **EXCLUDED from every lane** in this block regardless, and
that exclusion is a scope decision, not a permanent rejection: moving-rig
material is future robustness work.

## A2. THE 35-CAMERA INITIALIZATION INPUTS — leakage removed, verified against the artifacts

The split section above identified that the frozen sparse clouds were
triangulated on **all 39 cameras** and therefore carry observations from the
four held-out views `cam00, cam13, cam25, cam38` — initialization-time
leakage — and required a rebuild on the 35 training cameras.

`scripts/imvid_pilot_prepare.py --mode subset`, Determined experiment **155**
(`imvid_pilot_subset` r0, commit `00d2a32`, admitted image, `dgx`,
`evidence_bearing: false`, 23 s):

```
n_training_cameras   35
images written       105          (3 frozen frames x 35 cameras)
excluded per frame   cam00.png cam13.png cam25.png cam38.png
model_35/cameras.txt sha256 889a999a651c7dc74eea7bf391fcdbc7da52d3577878d2bbefb58ea0a63833b8
model_35/images.txt  sha256 6dba64bfecace321323ddc113b7bca2eb16fe346d58ab5476874f62d508b49b7
leakage check        PASSED -- no held-out camera name in any prepared input
destination          data/imvid/pilot35/  (790.5 MiB, 107 objects)
```

Two properties of the construction, both deliberate:

* **`cameras.txt` is copied byte-identically** and the copy's digest is checked
  against the source. The supplied intrinsics are fixed authority and the
  subset must not perturb them, not even through a reformat.
* **The leakage assertion is made against the WRITTEN artifacts**, not against
  intent: the check walks every written PNG and the written `images.txt` and
  refuses if any held-out camera name appears. An assertion about what the code
  meant to do would not have caught a filter that silently matched nothing.

The 39-camera clouds under `data/imvid/sparse/` are **preserved untouched** as
the calibration-validation evidence they were built for.

## A3. THE UNDISTORTION — measured, and it CORRECTS this record's "mild"

`scripts/imvid_pilot_prepare.py --mode undistort`, Determined experiment
**156**, scale 0.5 (the declared 2x downsample). Derived PINHOLE camera:

```
PINHOLE 2656 1494   fx 1301.66634323002   fy 1301.1218300301398
                    cx 1327.75            cy 746.75
```

The principal point follows COLMAP's integer-pixel-centre convention,
`(c + 0.5) * scale - 0.5`, rather than a naive `c * scale`; the two differ by a
quarter pixel and the choice is recorded because it is the kind of half-pixel
error that survives every downstream check.

**The measurement, over a 7,540-point grid at native resolution:**

| statistic | pixels |
|---|---:|
| min displacement | 0.00018 |
| **median** | **14.72** |
| p99 | 76.32 |
| **max** | **90.53** |
| round-trip error, median | 2.1e-08 |
| round-trip error, max | 1.7e-05 |

**This corrects the calibration section above.** That section reads
"Distortion is mild (k1 ≈ -0.025)". The coefficient IS small; the resulting
displacement is not — a median of 14.7 px and a maximum of 90.5 px on a
5312-wide raster, i.e. 1.7% of image width at the periphery. "Mild" was a
judgement about the coefficient magnitude and it does not transfer to the pixel
domain. Any step that treated the OPENCV camera as approximately pinhole would
be wrong by tens of pixels near the frame edge — including, specifically, any
attempt to feed raw ImViD frames to a loader that assumes PINHOLE.

The round-trip error at 1.7e-05 px shows the map itself is numerically sound,
so the correction is about magnitude, not about correctness.

**Scope of what was built:** the derived PINHOLE `cameras.txt` and the
measurement. **No undistorted images were written**, and the 35-camera sparse
initialization is being built in the SUPPLIED OPENCV frame. A PINHOLE-frame
initialization would be a separate build; that is stated rather than left for a
later reader to discover from a filename.

## A4. The 35-camera sparse rebuild

Determined experiments **158 / 159 / 160**, one per frozen frame
(`imvid_init35_frame_000000` / `_000150` / `_000299`, r0 each, commit
`06aea96`), running `scripts/imvid_sparse_init.py` unchanged against the
35-camera inputs, native resolution, CPU SIFT, all four COLMAP 3.6 calibration
guarantees passed explicitly.

*Results to be recorded when terminal.* The comparison that matters is against
the 39-camera figures already on this page (6,075 / 9,256 / 8,668 points at
1.2127 / 1.1697 / 1.2061 px, 39/39 camera coverage): the 35-camera rebuild
should show **fewer points and 35/35 coverage at a similar residual**. A
materially WORSE residual would indicate the held-out views were doing
structural work, which would itself be worth knowing.

### A4 RESULT — frame 0 (experiment 158, COMPLETED)

The 35-camera rebuild succeeds, and it behaves exactly as the comparison
predicted before it ran.

| quantity | 39 cameras (leaky) | **35 cameras (this rebuild)** |
|---|---:|---:|
| cameras registered | 39 / 39 | **35 / 35** |
| cameras with observations | 39 | **35** |
| points | 6,075 | **5,140** |
| observations | 25,220 | 20,366 |
| mean track length | 4.151 | 3.962 |
| **mean reprojection error** | 1.2127 px | **1.1953 px** |

**Fewer points, full 35/35 coverage, and a residual that is not worse — it is
marginally better.** So the four held-out views were contributing observations,
not holding the reconstruction together: removing them costs 935 points and
nothing in geometric consistency. Had the residual degraded materially, that
would have meant the held-out cameras were doing structural work and the split
itself would have needed reconsidering.

**Calibration preserved, verified against the BINARY model:**

```
intrinsics max abs delta   0.0            (EXACT equality required, no tolerance)
pose max abs delta         1.110e-16      (one ULP, cam34.png; tolerance 1e-12)
```

All four COLMAP 3.6 guarantees passed explicitly
(`fix_existing_images=1`, `ba_refine_focal_length=0`,
`ba_refine_principal_point=0`, `ba_refine_extra_params=0`), and the check reads
`cameras.bin`/`images.bin` rather than the lossy text export, per the
reconciliation in [[imvid-sample-ingestion]].

**This closes the per-camera reprojection limb of the S5 validation for frame
0: 1.1953 px against a 2 px gate, with every training camera contributing.**
Observations per camera range 43 (cam34) to 1,787 (cam01) — the spread is large
but no camera is empty.

Cost, measured: `feature_extractor` 79.8 s, **`exhaustive_matcher` 2,692.8 s**,
`point_triangulator` 13.4 s, converter + analyzer 2.3 s — about **47 minutes**,
essentially all CPU matching, and inflated by three of these cells running
concurrently on one node (the third cell's feature extraction took 189 s
against this one's 80 s, which is the contention showing).

**Frames 150 and 299 (experiments 159 / 160) also COMPLETED**, and the pattern
holds across the whole clip:

| frame | cameras | points (35-cam) | points (39-cam) | mean reprojection (35-cam) | (39-cam) |
|---:|---|---:|---:|---:|---:|
| 0 | 35/35 | 5,140 | 6,075 | **1.1953 px** | 1.2127 |
| 150 | 35/35 | 7,803 | 9,256 | **1.1361 px** | 1.1697 |
| 299 | 35/35 | 7,214 | 8,668 | **1.1808 px** | 1.2061 |

Intrinsics delta **exactly 0.0** and pose delta **1.110e-16** (one ULP) on all
three. The 35-camera residual is BETTER than the 39-camera residual at every
frame — so the supplied calibration is self-consistent on the training subset
at the start, middle and end of the clip, and the held-out views were adding
observations rather than constraining geometry.

The union rebuild (`scripts/imvid_build_initialization.py` over the three) has
NOT been run; that is a single cheap follow-up cell.

**Still open for S5**: the loader has not been exercised on ImViD data — the
trainer's loader reads PINHOLE and ImViD ships OPENCV, and the conversion step
does not exist. That is the remaining gate, and section A3's measurement is why
it cannot be skipped.

## A5. Cost preflight for one 300-frame Opera baseline — an ESTIMATE, and labelled as one

Built from this project's own measurements rather than from ImViD's reported
numbers, which came from unreleased code on different hardware.

**Storage**, from the MEASURED 7.5 MiB per decoded frame:

```
300 frames x 35 training cameras x 7.5 MiB  =  76.9 GiB   (native, PNG)
300 frames x  4 held-out cameras x 7.5 MiB  =   8.8 GiB   (needed only at final eval)
undistorted 2x (1/4 the pixels)             = ~21 GiB     (projected, NOT measured)
```

Apollo has 31.174 TiB free, so storage does not bind. **Only 3 of the 300
frames are currently decoded**; a full-segment decode has not been run and is
not required by anything authorized.

**Compute.** The only directly comparable in-house measurement is the DiVa-360
benchmark baseline: 6,000 iterations at 1160x550 over 4,935 training units in
~1.9 h on a V100.

```
pixels per image   2656x1494 / 1160x550        = 6.22x
training units     300x35 / 4935               = 2.13x   (affects loading, not iteration count)
V100 -> H100       assumed 2.0-3.0x faster     ASSUMPTION, not measured here
6,000 iterations   1.9 h x 6.22 / 2.5          ~ 4.7 h
15,000 iterations                              ~ 11.8 h
```

**The H100 factor is an assumption and it is the dominant uncertainty**, so the
range is stated as **~4–7 GPU-h at 6,000 iterations and ~10–18 GPU-h at
15,000**, and a **measured 500-iteration preflight is mandatory before any full
run is submitted**. That requirement is not boilerplate: the renderer lane's
own projection understated actual by 2.4x this week because it counted training
time only ([[renderer-integrity-admission-2026-08-18]] Appendix C).

## A6. What the pilot still does not establish

* **No ImViD number exists**, and no training has run.
* **The loader has not been exercised** on ImViD data end to end. The
  distortion measurement above tells us why that matters: the trainer's loader
  reads PINHOLE and ImViD ships OPENCV, so a conversion step is mandatory, not
  optional.
* **The fixed-rig property is verified only for the SAMPLE**, at frames
  0/150/299. Nothing here verifies any full take, and metadata alone cannot:
  a moving take registered at frame 0 produces an identical `images.txt`.
* **Nothing about event supply.** ImViD ships no masks and no identity ground
  truth; see [[dataset-admission-matrix-2026-08-18]]'s append-only narrowing.
* **Comparability to published ImViD numbers.** Their segment and take are
  unspecified, their method code is unreleased, and a like-for-like comparison
  needs a public-baseline (STG) reproduction on the same declared segment plus
  a measured ADAGS seed spread. None of that exists.

---

# APPENDIX B (2026-08-24, append-only) — the loader gate audited; one record CORRECTED; the dangerous path is the one that fails OPEN

Nothing above is rewritten. This appendix closes the "loader has not been
exercised" item at the level of *what would have to be true*, corrects one
stale statement in section A4, and records a hazard no previous page names.

## B1. CORRECTION — the union rebuild HAS been run

Section A4 above reads: *"The union rebuild
(`scripts/imvid_build_initialization.py` over the three) has NOT been run;
that is a single cheap follow-up cell."*

**That is stale.** It ran as Determined experiment **164**
(`imvid_init35_union`, r0, commit `c4ff0d4`), and the artifact was verified
this block directly against primary Apollo storage:

```
data/imvid/init35/points3d_colmap_union.ply    911,796 bytes  2026-08-18 22:37:54
  union_points  20,157
  sha256        d5b10be099b05c85fe63c04336a66b42561a1b3c7bf193dace504417351fdf71
  per-frame     0 -> 5,140 | 150 -> 7,803 | 299 -> 7,214   (sums to 20,157 exactly)
```

Claim index `r0` is consumed; the next free index for that cell is `r1`.

**The useful consequence:** the decisive calibration verification needs
**no new decode and no new triangulation**. Everything it consumes already
exists and is sealed.

## B2. The loader gap, cited — and it is not where the record points

The record has treated the `OPENCV`-vs-`PINHOLE` acceptance check as *the*
ImViD blocker. Verified in code, that check **fails closed** and is
therefore harmless:

* `scene/dataset_readers.py:676` — `assert False, "Colmap camera model not
  handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras)
  supported!"`
* an earlier, stricter `assert model == "PINHOLE"` in
  `scene/colmap_loader.py:159` fires first on the text path and rejects
  even `SIMPLE_PINHOLE`.

**The dangerous path is the Blender route, and it fails OPEN.**
`readCamerasFromTransforms` (`scene/dataset_readers.py:367-466`) — the route
N3V and DiVa-360 actually use, and the only one structurally compatible with
ImViD — reads `fl_x / fl_y / cx / cy` from JSON at `:433-451` with **no
camera-model field, no distortion field, and no check of any kind.**
Distorted images paired with pinhole intrinsics would train **silently**,
wrong by a median of 14.72 px and a maximum of 90.53 px.

Both halves of that mistake already sit adjacent on Apollo: experiment 156
wrote a derived PINHOLE `cameras.txt`, while the union and every decoded
frame remain in the supplied **OPENCV** frame. Nothing in the code would
object to combining them.

## B3. The COLMAP route is structurally impossible — on counts that DIFFER between the sample and the full release

Verified in code:

1. **`cam10` is hard-coded as the held-out camera** (`:574-575`:
   `train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]`).
   ImViD's frozen split holds out `cam00, cam13, cam25, cam38`. **This
   applies to sample and full release alike and is fatal to the COLMAP route
   for this lane.**
2. **`uid = intr.id`** (`:662`), with the assertion at `:588` requiring the
   test and train uid sets to be disjoint. **This applies to the Apollo
   SAMPLE only.** The sample's `cameras.txt` declares one shared camera
   `2 OPENCV 5312 2988 ...`, so all 39 views take `uid = 2`, and the
   assertion compares `2` against `[2]` and fails by construction. **The
   FULL release does NOT have this defect** — its `scene1_opera/cameras.txt`
   was read in full this block (6,309 bytes) and carries **39 entries with
   `CAMERA_ID` 1..39**, all with identical OPENCV parameters. That is valid
   COLMAP, and it also resolves the "39 identical lines would be malformed
   COLMAP" inconsistency left open in
   [[dataset-admission-matrix-2026-08-18]]'s 2026-08-19 appendix.
3. **`/30` timestamp division** (`:700`) — see B4.

## B4. FPS — there is no `60` to fix; the hard-coded rate is `30`, and that is 2x wrong

The concern on the record was 60 versus the measured `60000/1001` = 59.94, a
0.1% error. **A repo-wide search finds no site that hard-codes 60.** What is
hard-coded is **30**, at `scene/dataset_readers.py:200`
(`timestamp = frame_idx / 30.0`) and `:700`, plus 73 configs carrying
`motion_track_dt: 0.0333333333`.

For ImViD that is wrong by a factor of **2.002** — three orders of magnitude
worse than the question being asked. The correct period is
`1001/60000 = 0.0166833...` s. **Any ImViD lane must set the frame period
from the measured stream rate, never inherit the N3V constant.**

## B5. The 2 px gate has a raster trap

The recorded reprojection gate is **2 px**, and the passing values
(1.1953 / 1.1361 / 1.1808 px) are **at native 5312x2988**. Residuals scale
with the raster, so the same geometry evaluated on the 0.5-scale 2656x1494
raster reads **half**.

> **State the gate as `mean <= 2.0 px AT NATIVE`, i.e. `<= 1.0 px` when
> measured on the 0.5-scale raster, and make every reported residual say
> which raster it is in.** Comparing a 0.5-scale residual against a native
> 2 px gate passes trivially and measures nothing.

## B6. The sparse initialization is REUSABLE unchanged — and re-triangulating would be worse

The 20,157-point union is valid in the undistorted PINHOLE frame with no
modification, because undistortion replaces only the camera-to-pixel map:
with no rectification rotation the world-to-camera transform `(R_i, t_i)` is
unchanged, and a world-frame 3D point is invariant under a change to the
projection model alone. The consistency requirement is that the intrinsic
written to disk is the same `K_new` used to build the resampling map.

Re-triangulating would be actively worse rather than merely wasteful:
`scripts/imvid_sparse_init.py` pins `--SiftExtraction.max_image_size` to the
native 5312 while an undistorted raster is 2656 wide, so features would be
detected at full raster on half-resolution content, changing the residual
scale — at a further ~47 min x 3 of CPU matching.

**One thing does have to change: the file NAME.** The Blender reader looks
for `points3d.ply` exactly (`:481`), the artifact is
`points3d_colmap_union.ply`, and a mis-named point cloud is silently
substituted by a random uniform fill **with no error raised** (`:481-491`) —
the exact silent-initialization failure this project already paid for once
on DiVa-360.

## B7. Schedule — "6k" does not mean on ImViD what it means on N3V

| protocol | frames | train cams | units | pres/unit @6k | @12k |
|---|---:|---:|---:|---:|---:|
| N3V 50-frame | 50 | **19** (cam04 absent from the release) | 950 | 12.632 | 25.263 |
| ImViD Opera full | 300 | 35 | 10,500 | **1.143** | **2.286** |

**An 11.05x exposure gap.** Matching N3V's per-unit exposure on the full
Opera split would need ~66,300 iterations — 5.5x the authorized 12k ceiling.
Even 12k lands at 2.29 presentations/unit, essentially the ~2.1 that
[[b0c-canonical-300f-2026-08-20]] explicitly dismisses as pre-peak.

**Consequence:** a meaningful sub-12k ImViD pilot must use a frozen,
event-selected **frame tranche**, not the full 300. A 50-frame ImViD subset
reaches N3V-equivalent exposure at **11,053 iterations**, which fits under
the ceiling. Presentations per unit must be reported alongside iterations in
every ImViD result.

## B8. Submission readiness

**No allowlist change is needed.** `scripts/imvid_pilot_prepare.py` is
already at `scripts/submit_apollo.py:104` and already carries a `--mode`
dispatcher, so extending it is the zero-diff route. Non-`main.py`
entrypoints take their whole CLI from `--extra-arg` (values starting with
`-` need the `--extra-arg=--flag` form) and receive no generated run dir.

## B9. Still NOT established

No ImViD number exists and no ImViD training has run. The undistortion has
been *measured* but no undistorted image has been *written*. The
reprojection gate of B5 has been *specified* but not *executed*. And the
fixed-rig property remains verified only for the 300-frame sample at frames
0/150/299 — metadata cannot certify a full take, because a moving take
registered at frame 0 produces an identical `images.txt`.

## B10 (2026-08-24) — the FULL Opera take is 15,215 frames, and that makes B7's exposure gap 50x worse

Measured on the first fully-downloaded full-take file, not inferred:

```
scene1_opera/cam00.mp4      3,224,052,860 bytes   (== inventory expected, == manifest observed)
  codec h264   5312x2988   pix_fmt yuv420p
  r_frame_rate 60000/1001  avg_frame_rate 60000/1001
  duration     253.836917 s
  nb_frames    15,215
```

**End-to-end transfer integrity verified independently:** the SHA-256
recomputed on Apollo from the landed bytes equals the downloader's
recorded manifest hash exactly
(`764d9c72cd98ccae3d3042f41978735ceb43655c7b77a02840d72c946133234a`), and
a real frame decoded to a 90,166-byte PNG — which also proves the `moov`
atom arrived and the file is not truncated.

### The correction

B7 computed the exposure gap from the **300-frame SAMPLE** dimensions
(300 x 35 = 10,500 units, an 11.05x gap). The **full take is 15,215
frames**, i.e. **50.7x** the sample, so:

| protocol | frames | train cams | units | ratio vs N3V-50f |
|---|---:|---:|---:|---:|
| N3V 50-frame | 50 | 19 | 950 | 1x |
| ImViD Opera, 300-frame sample | 300 | 35 | 10,500 | 11.05x |
| **ImViD Opera, FULL take** | **15,215** | 35 | **532,525** | **560.6x** |

**Consequence, and it is now binding rather than advisory: the full Opera
take cannot be trained at any authorized schedule.** At the 12,000-iteration
absolute ceiling it receives roughly 1/560th of the per-unit exposure that
the N3V 50-frame protocol gets at 6k. B7's conclusion — that a meaningful
sub-ceiling pilot needs a frozen, event-selected frame tranche rather than
the whole take — stands, and the margin by which it stands is 50x larger
than B7 knew.

**Also corrected:** [[imvid-sample-ingestion]] records Opera's full take as
"3 min 22 s / 226 GB". The measured duration is **253.84 s = 4 min 14 s**,
and the accessible folder's 39 files total 125,649,776,270 bytes
(117.02 GiB). Both figures on that page are superseded by measurement;
neither was verified there and both are marked as such in the original.

**NOT verified in this probe:** the read-only (0444) promotion of completed
raw files. The permission listing was emitted but filtered out of the
captured output, so it is recorded here as unchecked rather than as passing.
