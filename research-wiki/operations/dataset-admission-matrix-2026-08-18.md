# Dataset admission matrix (2026-08-18)

**Provenance caveat, stated first.** The external facts below come from a
bounded read-only research pass over official sources on 2026-08-18. The
primary agent reviewed the reasoning and the internal citations to this
repository's own records, but **did not independently re-fetch the
external URLs.** Items the researcher could not verify are carried
through as UNVERIFIED rather than smoothed over. Treat the rig-geometry
findings as decision-grade and the sizes/licences as good-faith but
single-sourced.

No dataset was downloaded. No access control or licence gate was touched.

## 1. The binding criterion

Not resolution, not scene count, not camera count. For EL-GS the question
is:

> Can this dataset supply — **and can we MEASURE that it supplies** —
> genuine full-applicable-camera disappearance and return of a trackable
> identity, measured **independently of tracker failure**?

The second clause is what M1 and the absence diagnostic turned into a
hard requirement. DiVa-360's masks are the only reason its absence claims
could be checked at all — and checking them **refuted** the tracker
rather than confirming it (0 of 597 windows corroborated).

## 2. Verdicts

| dataset | rig geometry | tracker-independent GT? | verdict |
|---|---|---|---|
| **DiVa-360** | 53 cameras, TRUE 360° surround | **YES — fg/bg masks** | **primary for event-supply.** The only one where the question has actually been tested |
| **N3V** | frontal/narrow arc — measured, not assumed: 93.4% of 3.07M bins rejected for insufficient multi-camera co-support | no | photometric substrate + motion-supervision only. Already retired for event supply on this project's own evidence |
| **Google Immersive** | **46 cameras inside a 92 cm dome**, ~18 cm apart | no | **NOT ADMITTED** |
| **MeetRoom** | 13 Azure Kinect on a 100×75 cm panel; authors call it sparser than N3DV | no — **depth deliberately disabled at capture** to save USB bandwidth | **NOT ADMITTED** |
| **ImViD** | 39 cameras on a **rig that physically relocates between and within takes** | no | **NOT ADMITTED** for event supply |
| **MPEG-GSC** | n/a | n/a | **NOT ADMITTED — not a dataset** |

### Why each disqualification is structural, not fixable

**Google Immersive.** All 46 viewpoints sit inside a sub-metre sphere. It
is one dense light-field *eye*, not a distributed array: anything
occluding the subject from the dome occludes it from all 46 cameras
near-simultaneously. Azimuthal diversity is essentially zero, which makes
it **worse** than N3V's frontal arc for corroboration — there is no
second direction from which to rule out "merely occluded from here".

**MeetRoom.** Same family as N3V — a compact frontal panel. The sharpest
detail is that the hardware *could* have given depth and the authors
switched it off, so the one auxiliary signal that might have supported
tracker-independent corroboration does not exist in the release.

**ImViD.** The decisive finding, and one the paper does not address: the
entire 39-camera array is mounted on a mobile platform that moves. So a
subject vanishing from all 39 cameras could equally mean the rig moved
away. **The "applicable camera set" is not geometrically stable over
time** — which is strictly worse than a merely frontal fixed rig, because
the confound is in the instrument rather than in the scene. This
disqualifies it for event supply independently of its missing masks.

**MPEG-GSC.** Not a dataset but an ISO/IEC JTC1/SC29 standardization
activity. Its finalized common-test material is **static only** — no
camera rig, no temporal axis, so the disappearance question is
inapplicable by construction. The call for dynamic test material is still
open, closing 2026-10-15. Access needs registration as an MPEG expert
through a national body; there is no individual-researcher path.
Recommend excluding it until a finalized dynamic corpus with a real
access path exists, if ever.

## 3. What this actually changes

**R4 "dataset extension" is now a much weaker option than it looked.**
None of the four candidates improves on DiVa-360 for event supply, and
two are disqualified on geometry alone. The decision memo's route table
should be read with that in mind.

**The synthetic option is NOT hypothetical.** The most useful new finding
is that the "measured independently of tracker" requirement is achievable
— just not by any of the real-world candidates:

* **MOVi-MC-AC** (Kubric-generated, CC BY 4.0, ~1.49 TB) ships exact
  renderer-derived ground truth: modal AND amodal segmentation, amodal
  RGB, per-instance IDs, depth, collision metadata, 45.2% mean occlusion.
  The labels come from the simulator, not a tracker. Wrong domain
  (generic clutter, not hand-object manipulation) and only 6 cameras, not
  surround.
* **CMU Panoptic Studio** — 480 VGA + 31 HD cameras on a true ~5 m
  geodesic dome, real humans. Triangulated 3D pose could corroborate
  presence semi-independently of any single 2D tracker. Pose-based rather
  than object-identity-based, and the sequences are short social
  interactions unlikely to contain scripted leave-and-return.
* **Kubric itself** — the generator behind MOVi-MC-AC. The only route to
  a purpose-built surround-rig, hand-object-domain, scripted
  disappearance-and-return scene with exact ground truth. An engineering
  effort, not an acquisition.

This materially strengthens decision 3 of
[[user-decision-memo-2026-08-18]]: a synthetic controlled event scene is
the only way to exercise C1/C2 reactivation mechanics against *truth*
while real supply is unresolved, and there is now a concrete route to one.

## 4. Apollo inventory (measured)

| path | contents | size |
|---|---|---|
| `data/n3v` | 6 scenes | 390.84 GiB |
| `data/diva360` | 25 of 54 sequences + zips | 593.56 GiB |
| `data/diva360_derived` | tracks, hull inits, sweep artifacts | 114.86 GiB |
| `data/imvid` | Opera sample only | 2.98 GiB |
| free space | | **31.85 TiB** |

Nothing for Google Immersive, MeetRoom or MPEG-GSC exists on Apollo.
**Storage is not a constraint** — which means dataset decisions should be
made on the measurability criterion alone, not on cost.

## 5. Loader compatibility — a shared, unclosed gap

`scene/dataset_readers.py:666-676` accepts only `SIMPLE_PINHOLE` /
`PINHOLE`. Consequences:

* Google Immersive is fisheye → incompatible as shipped;
* ImViD is OPENCV-distorted → the same gap, already open in the ImViD
  lane, where the triangulation script decodes OPENCV itself but the
  trainer's loader cannot;
* MeetRoom ships LLFF `poses_bounds.npy`, which no branch reads at all.

So every new candidate needs a conversion step that does not exist today.
That cost is real but small; it is not what disqualifies any of them.

## 6. Recommended next dataset action

**A bounded loader/calibration smoke on MeetRoom `Discussion`** — ranked
above Google Immersive because its content (people who can leave and
re-enter a room) is at least the right *kind* of behaviour, and it is an
order of magnitude cheaper to test.

**But state plainly what it can and cannot buy.** The disqualifications
in §2 are geometric facts that no loader test can change. A smoke would
only rule out "we missed something obvious" and keep the loader path
exercised against a new format family. It cannot advance the event-supply
question.

Success: `prepare_dataset.py` runs; a ~50–100 line conversion (analogous
to `diva360_to_blender.py`) maps `poses_bounds.npy` into the `Blender` or
COLMAP convention; the scene loads unmodified; reprojection under 2 px,
matching the bar already met by ImViD (1.17–1.21 px) and DiVa-360.

**Not run tonight**, because it is not on the critical path and the
critical path is M-2 plus the user's route decision.

## 7. UNVERIFIED items carried forward

* Google Immersive has no `LICENSE` file found and no terms on the
  project page — open access is not the same as a clear licence.
* MeetRoom's dataset licence is unstated anywhere found (the BSD-2-Clause
  covers the StreamRF *code*).
* MeetRoom's scene list comes from downstream citing papers, not from
  StreamRF's own paper body.
* ImViD's application-form fields could not be rendered; whether the
  gated full dataset carries terms stricter than the sample's CC BY 4.0
  is unknown.
* ImViD distortion models for the 6 non-Opera scenes were not read from
  data.
* The three dynamic MPEG-GSC exploration sequences have unverified
  provenance and availability.

---

## APPEND-ONLY NARROWING (2026-08-18) — the ImViD rig verdict, and what survives it

Nothing above is rewritten. Two of this page's three structural
disqualifications are narrowed in scope by facts that arrived after it was
written; the third stands unchanged. The **event-supply** verdicts are
untouched in every case.

### ImViD — "the rig physically relocates" is NARROWED, not withdrawn

The section-2 row and the section-"why" paragraph treat rig mobility as a
property of the whole dataset. The paper's own capture-strategy table
distinguishes per scene: **Opera and Meeting are captured fixed-point only**,
while Laboratory, Classroom, Rendition, Puppy and Playing have both fixed and
moving takes. So the confound is a property of a TAKE, not of the dataset, and
per-take verification is the correct granularity.

**Recorded as reported, not as verified here.** The per-scene strategy table is
taken from the 2026-08-18 strategy document's reading of the paper. This block
does not claim to have re-read the paper's table; what IS verified here is the
sample-level fact already on record — the Opera sample carries 39 poses for 39
cameras with a single static pose each, and fixed-pose triangulation succeeds
at 1.17–1.21 px across frames 0/150/299 ([[imvid-sample-ingestion]]).

**What does NOT change.** ImViD remains **NOT ADMITTED for event supply**, for
the reason that never depended on the rig: it ships no masks and no identity
ground truth, so there is no tracker-independent instrument and no
coverage statistic can be defined on it. A rig fixed for one take does not
supply an event instrument.

**What DOES change.** ImViD is admissible for temporal/photometric
reconstruction and held-out-view generalization, entered through the Opera
sample. That work is under way and recorded in
[[imvid-baseline-freeze]]'s pilot appendix.

**A necessary-not-sufficient caution that must travel with this narrowing:**
metadata cannot certify a fixed take. A moving take registered at frame 0
produces exactly the same 39-pose `images.txt` as a fixed one. The only
sufficient test is the fixed-rig test — fixed-pose triangulation residual at
frames 0/mid/end — and that needs the whole take, which is not acquired.

### Google Immersive — "NOT ADMITTED" stands for event supply; the geometry claim is CONFIRMED and sharpened

The dome argument is unchanged and remains the reason it cannot supply
absence evidence: no second azimuth, so "merely occluded from here" cannot be
ruled out.

Newly VERIFIED here, by direct request rather than from prose: the raw
distribution is **not gated at all**. The `deepview_video_raw_data` bucket is
publicly listable through the GCS JSON API, holds **15 scene archives
totalling 65,461,026,250 bytes**, and every object serves an unauthenticated
GET. The smallest is `15_Branches.zip` at 179,620,533 bytes and the largest
`10_Alexa_Meade_Face_Paint_1.zip` at 10,587,538,727. The dataset is therefore
two orders of magnitude cheaper to obtain than the earlier "acquisition cost"
framing implied — which changes nothing about its admissibility for event
supply and does change its cost as a reconstruction/generalization anchor.
Inventory: [[immersive-inventory-2026-08-18]].

### MeetRoom and MPEG-GSC

Unchanged.

---

## APPEND-ONLY (2026-08-18, block 4) — ImViD access is OPEN, and the Immersive camera-count gap is resolved

Nothing above is rewritten. Two facts arrived that change what is *reachable*
and what a prior page *claimed*; neither changes an event-supply verdict.

### C1. ImViD: the full release IS reachable — the block-3 "no access path" record is CORRECTED

`overnight-handover-18aug-block3.md` recorded, on five probes, that the full
ImViD release "is not reachable from this environment at all". **That was
true of the probes and false of the world:** that session did not have the
release URL. With the user-supplied folder id `1TrhrOrmFdvw-wTRPiVqlyWUWZrJJgHZe`
the folder is **world-readable and needs no credential, no OAuth and no
cookie**. The correction is recorded here rather than by editing the earlier
finding.

Verified by the primary directly (`curl` on the public
`embeddedfolderview` endpoint returned 8 top-level entries), and enumerated by
a bounded worker two independent ways — a stdlib crawl and a pinned
`gdown==5.2.0` in an isolated throwaway venv — which agreed on **325 files
exactly**, so nothing was lost to pagination. Exact sizes came from 1-byte
HTTP `Range` probes (HTTP 206 + `Content-Range`), i.e. **no bulk bytes were
transferred**; ~340 requests drew zero 403 and zero rate-limiting.

**The structure is flat: 8 folders, one per scene plus `moving_rig`, no
take-level subdivision.**

| folder | files | mp4 | bytes | GiB |
|---|---:|---:|---:|---:|
| `moving_rig` | 39 | 39 | 131,492,109,120 | 122.46 |
| `scene1_opera` | 41 | 39 | 125,649,776,270 | 117.02 |
| `scene2_laboratory` | 41 | 39 | 81,340,649,443 | 75.75 |
| `scene3_classroom` | 40 | **38** | 409,317,428,086 | 381.21 |
| `scene4_meeting` | 41 | 39 | 122,672,447,671 | 114.25 |
| `scene5_rendition` | 41 | 39 | 113,041,967,198 | 105.28 |
| `scene6_puppy` | 41 | 39 | 115,934,621,018 | 107.97 |
| `scene7_playing` | 41 | 39 | 81,627,960,479 | 76.02 |
| **total** | **325** | **310** | **1,181,076,959,285** | **1,099.96** |

**Four discrepancies against the published release, recorded rather than
reconciled away:**

1. **9 of the 16 published takes are absent.** The README documents 16 takes
   totalling 2,069.3 GiB; the folder exposes one folder per scene — **7 takes,
   977.50 GiB, 47.2% of the published bytes**. With exactly 39 files per
   folder (one per camera) there is no room for a second take unless takes
   were concatenated. `scene4_meeting` is the anchor that fixes the reading:
   it is the only 1-take scene and its 114.25 GiB matches the README's 114.0
   to 0.2%, which also establishes that the README's "GB" column is **GiB**.
2. **`scene3_classroom` is missing `cam38.mp4`** — 38 videos against a
   39-line calibration.
3. **`moving_rig` (122.46 GiB) is not in the README total** (226 + 137.3 +
   497 + 114 + 516 + 359 + 220 = 2,069.3 exactly). It is extra, uncalibrated
   by the authors' own statement, and its source scene is not identified.
4. **The paper says 46 cameras; the README and the shipped data say 39.**

**No checksums are obtainable.** The anonymous endpoints return no ETag, no
Content-MD5 and no `X-Goog-Hash`; Drive's `md5Checksum` needs the
authenticated API. Any transfer must be verified against the byte counts
above and nothing stronger.

**Calibration, read from data** (one 6,309-byte probe of
`scene1_opera/cameras.txt`): 39 identical lines, model **`OPENCV`**,
5312×2988, `fx` 2603.333, `fy` 2602.244, `cx` 2656.0, `cy` 1494.0,
`k1` −0.0245469, `k2` +0.0035148, `p1` −0.00045080, `p2` −0.00023832.
Non-zero distortion, so undistortion is required — consistent with the
pilot's own measured 14.7 px median / 90.5 px max displacement.

**One inference, flagged as such and NOT acted on:** `scene4_meeting`'s
`cameras.txt` is 70 bytes and `scene7_playing`'s 72, against 6,309–6,348 for
the three known 39-line `OPENCV` files. A single `PINHOLE` line at the same
float precision computes to exactly 70 bytes. So Meeting and Playing very
likely ship **one** camera line, distortion-free — but that is arithmetic on
a file size, not a read, and it must be read before any Meeting pilot is
designed.

**Transfer feasibility, for the two priority takes.** Opera
125,649,776,270 B + Meeting 122,672,447,671 B = **248,322,223,941 B
(231.27 GiB)**. `Accept-Ranges: bytes` is served and arbitrary offsets were
empirically honoured (HTTP 206 at both 1e9 and 3.224e9), so `curl -C -`
resumes correctly and non-destructively. The unmeasured risk is Drive's
undocumented per-file daily download cap, which typically surfaces as a 403
partway through a multi-file take; the resumable loop tolerates it, parallel
connections trip it sooner. **Transfers should run cluster-side, not through
this workstation, to avoid a 248 GB double hop.**

**What this does NOT change.** ImViD's event-supply verdict is untouched:
still **NOT ADMITTED for event supply**, for the reasons in the narrowing
section above. Access being open changes cost and reachability, not
admissibility.

### C2. Google Immersive: 46 videos vs 45 calibrations — RESOLVED, and it is not a held-out convention

**Verified by the primary against the sealed Apollo artifacts**, not from
prose: the exp-157 archive inventory and the exp-162 extracted `models.json`
(sha256 `199afc790c274f4782b7786fd6014137286d05eec152d845e31d92ddc8ea8908`)
were enumerated and diffed directly.

* zip mp4 entries: **46**, `camera_0001` … `camera_0046`, contiguous, **1-based
  — there is no `camera_0000`**;
* `models.json` entries: **45**, `camera_0001` … `camera_0045`;
* in the zip but uncalibrated: **exactly `camera_0046`**;
* calibrated with no video: **none**.

**It is a per-scene calibration failure, not a designated held-out camera.**
The official README states plainly that some scenes have a small number of
missing cameras, and the missing index varies by scene (`01_Welder` →
`camera_0036`, a mid-sequence gap; `04_Truck` → `camera_0003`), with
`12_Cave` showing the other failure mode — 45 videos and 45 matching
calibrations. **The conventional held-out camera is `camera_0001`, and it is
present and calibrated here**, so the earlier working hypothesis that the
uncalibrated file *was* the held-out view is refuted.

STG absorbs the mismatch silently by construction: frame extraction and the
symlink stage enumerate the **directory** (all 46), while undistortion and
COLMAP conversion iterate **`models.json`** (45). The uncalibrated camera's
extracted frames become an orphan directory that never reaches the database
or the trainer, and **no assertion checks that the two counts agree**.

**Consequence for any ADAGS visibility bookkeeping on this dataset:** key on
`models.json` `name` fields and treat the mp4 set as a superset. A ledger
built by enumerating `camera_*.mp4` will mis-index, and *will do so
differently on different scenes*.

### C3. The loader-incompatibility blocker recorded at §5 is NOT a blocker

Section 5 above records that `scene/dataset_readers.py` accepts only
`SIMPLE_PINHOLE`/`PINHOLE`, so "Google Immersive is fisheye → incompatible as
shipped". True of the raw data, **but it does not gate use**: STG's
preprocessing performs the fisheye→pinhole conversion offline
(`cv2.fisheye.initUndistortRectifyMap` + `cv2.remap`, Kannala–Brandt with
Google's two radial terms in `k1,k2` and `k3=k4=0`) and writes the COLMAP
camera line itself as `PINHOLE`. The trainer never sees a fisheye model.
**What Immersive is disqualified for is event supply, not loader
compatibility.**

Two further protocol facts that bear on cost and on any reproduction:

* **Poses are never solved.** COLMAP runs `feature_extractor` →
  `exhaustive_matcher` → `point_triangulator` against poses written from
  `models.json` into `manual/`. There is no `mapper` and no pose bundle
  adjustment; `models.json` `orientation` is axis-angle world→camera and
  `position` is the camera centre, converted as `(q, −R·C)`.
* **Extraction cost is invariant to the requested frame range.** The frame
  extractor is called with the path only, so its `startframe`/`endframe`
  defaults apply and **every run decodes frames 0–299 of all 46 videos at full
  2560×1920 as PNG** regardless of the `--startframe/--endframe` passed to the
  preprocessing script. Estimated **≈131 GB** of raw PNG for `02_Flames`
  alone, and STG's own published protocol (`duration: 50`, `colmap_0`) still
  pays it: **≈170 GB total**, against ≈270–470 GB for a full 300-offset run.
  These are estimates over an unconfirmed PNG compression level and are marked
  as such.

### C4. The dome limitation, now with a number

The parallax bound is the quantitative form of the existing verdict. The rig
is a hemispherical dome of radius ≈0.46 m (corroborated by `models.json`'s
first camera at 0.420 m from the origin), so for a scene point at distance
*d* the maximum angular separation between any two viewing rays across the
**entire 46-camera rig** is ≈ 2·arctan(0.46/*d*):

| *d* | max parallax, whole rig |
|---|---|
| 0.5 m | ≈85° |
| 1 m | ≈50° |
| 2 m | ≈26° |
| 5 m | ≈10.5° |

DiVa-360's inward-facing 53-camera surround reaches 180°. Because the dome is
**convex and outward-facing**, the cameras that see any exterior point form a
contiguous patch with near-parallel axes: adding cameras raises sampling
density, not viewpoint diversity. So the corroboration that the ELGS absence
instrument needs — an eligible foreground component holding the anchor *from a
different azimuth* — has no geometric support here. The test does not fail on
this rig; **it is undefined on it.** The NOT-ADMITTED verdict for event supply
is unchanged and now has a stated mechanism.

### C5. Provenance and residual uncertainty

The STG protocol map and the ImViD enumeration were produced by two bounded
read-only workers. The primary independently verified the two load-bearing
claims — the ImViD folder listing, and the `camera_0046` diff against the
sealed Apollo artifacts. **Not verified by the primary, and carried as the
workers' reporting:** the STG line-level protocol details, the published
per-scene PSNR table (transcribed via an HTML rendering rather than the CVF
PDF, and marked medium-confidence by the worker), the ImViD per-file byte
counts beyond the folder-level totals, and the decoded-size estimates. Two
malformed-command defects the worker reports in STG's own source
(a `SiftExtraction.max_image_size` flag missing its `--`, and an unreachable
`04_Trucks`/`04_Truck` branch) are recorded as reported and unconfirmed.

---

## APPENDIX (2026-08-19, append-only) — Meeting and Playing ship a distortion-free PINHOLE intrinsic, so §5's loader gap does not apply to them

Nothing above is rewritten. §5 records a shared, unclosed loader gap:
`scene/dataset_readers.py:666-676` accepts only `SIMPLE_PINHOLE` / `PINHOLE`,
and ImViD Opera ships `OPENCV`, so "every new candidate needs a conversion step
that does not exist today". That is an **Opera** fact and it does **not**
generalise across ImViD scenes.

The two `cameras.txt` files were read in full (70 and 72 bytes; the only bulk-free
fetch this needed) and each contains **exactly one line**:

```
scene4_meeting   1 PINHOLE 5338 2991 2722.5516678678127 2721.4363233225208 2669 1495.5
scene7_playing   1 PINHOLE 5411 2999 2626.5693056599121 2636.5945038889649 2705.5 1499.5
```

sha256 `73c29251...` and `0ca2e1d5...` respectively.

Three consequences:

* **The standing UNVERIFIED inference is CONFIRMED exactly.** The byte
  arithmetic works out (69 chars + newline = 70; 71 + newline = 72), and one
  line means one camera entry, so all 39 views of each scene reference
  `CAMERA_ID 1` — a single shared intrinsic, not per-camera calibration.
* **`PINHOLE` carries no distortion parameters at all.** Meeting and Playing are
  therefore **directly loader-compatible as shipped**, with no conversion step.
  The A3 undistortion measurement (median 14.7 px, max 90.5 px) is an Opera
  measurement and must not be transferred to them.
* **The principal point is exactly centred in both** (5338/2 = 2669,
  2991/2 = 1495.5; 5411/2 = 2705.5, 2999/2 = 1499.5) and the rasters differ per
  scene. That reads as the authors having already rectified these two scenes and
  not Opera — an interpretation, not a measurement.

**What this does NOT change.** ImViD remains **NOT ADMITTED for event supply**.
The disqualification in §2 is that the 39-camera array is mounted on a platform
that moves, so the applicable-camera set is not geometrically stable over time;
a loader fact cannot touch that. And the fixed-rig question for Meeting remains
**structurally undeterminable from metadata** — the only sufficient test is a
fixed-pose triangulation residual at frames 0 / mid / end, which needs the whole
114.25 GiB take. Reading Meeting's `images.txt` would show 39 poses and would
prove nothing about whether the rig moved.

### Two corrections to the inventory record

* **Apollo free space is 30.841 TiB**, measured this session via `rclone about`.
  The figures carried above (31.174 TiB) and in the block-4 handover
  (31.85 TiB) are stale by 0.3-1.0 TiB. Storage still does not bind.
* **An unresolved inconsistency, not closed.** The Opera *sample* already on
  Apollo has a `cameras.txt` describing ONE shared `2 OPENCV 5312 2988 ...`
  camera, while the Drive `scene1_opera/cameras.txt` is recorded as 39
  *identical* OPENCV lines at 6,309 bytes. Thirty-nine identical lines would be
  malformed COLMAP, because the ID column must vary. Whether the Drive file uses
  `CAMERA_ID 1..39` or repeats one id is **not determined** — it is a 6.3 KB
  text read and it decides what `images.txt` references. Worth closing before
  any full-take Opera pilot.

Per-folder FILE COUNTS were re-verified independently and match the recorded
table exactly, including `scene3_classroom` missing `cam38.mp4` and `moving_rig`
carrying no `.txt` at all. Per-folder BYTE totals were **not** re-verified:
Drive serves no ETag, no Content-MD5 and no `X-Goog-Hash`, and a HEAD returns a
`Content-Length: 0` virus-scan interstitial, so sizes are obtainable only by
1-byte Range GETs — 325 of them for the whole release. All byte figures remain
inherited.
