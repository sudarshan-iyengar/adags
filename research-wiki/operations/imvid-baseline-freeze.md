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
