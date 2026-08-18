# Google Immersive Light Field Video — verified inventory and pilot acquisition

Date: 2026-08-18. Status: **inventory VERIFIED by direct request; pilot scene
ACQUIRED and digest-verified; calibration read from the data.** EXPLORATORY. No Immersive training has run, no
Immersive number exists, and no preprocessing beyond the archive inventory has
been performed.

Authority: the Google-Immersive lane of the 2026-08-18 execution directive,
which supersedes the earlier inventory-only limit and authorizes **one** pilot
scene acquisition plus a bounded preprocessing smoke.

Reads [[dataset-admission-matrix-2026-08-18]] (whose dome-geometry
disqualification for **event supply** is unchanged by anything here).

## 1. The distribution is not gated, and that was measured

Every fact in this section came from a request, not from a page describing the
dataset.

```
bucket            deepview_video_raw_data
listing           https://storage.googleapis.com/storage/v1/b/deepview_video_raw_data/o
object            https://storage.googleapis.com/deepview_video_raw_data/<scene>.zip
auth required     NONE -- unauthenticated GET returns 200
range support     yes -- a Range: bytes=0-0 probe returns 206
objects           15
total             65,461,026,250 bytes (60.97 GiB)
```

No form, no login, no click-through, no credentials, no account. This is a
materially different acquisition situation from ImViD's full release (section 4
of [[imvid-baseline-freeze]]'s pilot appendix) and from DiVa-360's
browser-collected tranche.

**Digests.** Google publishes **no sha256**. Each object carries an MD5 (base64
in the GCS metadata) and a CRC32C. `scripts/fetch_immersive_scene.py` therefore
gates on the **publisher's MD5** and records a self-computed sha256 alongside
it. Gating on a digest we computed ourselves would verify only that the file
did not change between two of our own reads, which is not integrity
verification; that distinction is why this script's digest handling differs
from `scripts/fetch_imvid_sample.py`'s hard-coded sha256.

## 2. The 15 released scenes, sizes verified 2026-08-18

| scene archive | bytes | GiB |
|---|---:|---:|
| `01_Welder.zip` | 10,554,565,078 | 9.83 |
| **`02_Flames.zip`** | **5,474,948,990** | **5.10** |
| `03_Dog.zip` | 1,398,414,399 | 1.30 |
| `04_Truck.zip` | 1,811,297,781 | 1.69 |
| `05_Horse.zip` | 3,462,199,574 | 3.22 |
| `06_Goats.zip` | 1,617,389,119 | 1.51 |
| `07_Car.zip` | 3,122,877,276 | 2.91 |
| `08_Pond.zip` | 1,780,220,296 | 1.66 |
| `09_Alexa_Meade_Exhibit.zip` | 3,500,462,687 | 3.26 |
| `10_Alexa_Meade_Face_Paint_1.zip` | 10,587,538,727 | 9.86 |
| `11_Alexa_Meade_Face_Paint_2.zip` | 3,578,161,493 | 3.33 |
| `12_Cave.zip` | 4,578,867,676 | 4.26 |
| `13_Birds.zip` | 4,887,272,734 | 4.55 |
| `14_Puppy.zip` | 8,927,189,887 | 8.31 |
| `15_Branches.zip` | 179,620,533 | 0.17 |
| **total** | **65,461,026,250** | **60.97** |

`02_Flames.zip` publisher digests: MD5 `b0pj+tNbWxPsY6Qj8S89Fg==`, CRC32C
`bfLeCQ==`, last modified 2021-04-21T19:01:40.372Z.

The fetcher hard-codes this table and **refuses to acquire** if a remote size
differs from the recorded value, so a changed remote is detected rather than
silently accepted. An inventory run on 2026-08-18 reported zero drift and no
missing objects.

**Storage is not a constraint.** Apollo has 31.174 TiB free; the entire
dataset is 0.19% of that. The earlier framing that treated Immersive
acquisition as expensive is superseded on cost — and not on admissibility.

## 3. Pilot scene selection — BY RULE

**`02_Flames`**, because it appears in SpacetimeGaussians' published 7-scene
Immersive protocol and in that repository's own example invocation. The choice
is deliberately **not** based on expected disappearance/return content, which
nothing here has measured. Recorded explicitly because selecting a scene by
its name's suggestion of dynamic content is exactly the reasoning
[[dataset-admission-matrix-2026-08-18]] rejects.

Acquisition: Determined experiment **157**, cell `immersive_fetch_flames` r0,
commit `06aea96`, admitted image, pool `dgx`, `evidence_bearing: false`,
destination `/apollo/users/sri/proj_adags/data/immersive/raw` (read-only after
verification). Result recorded in section 6.

## 4. What is NOT done, and what each step needs

| step | status |
|---|---|
| read-only inventory | **DONE** (section 1–2) |
| pilot acquisition of `02_Flames` | **DONE**, exp 157, digest-verified |
| archive central-directory inventory | **DONE**, exp 157 |
| `models.json` fisheye format read from the data | **DONE** — section 6 |
| STG `pre_immersive_undistorted.py` route read from source | **NOT DONE** |
| fisheye -> perspective conversion | **NOT DONE**, and must go through STG's official script rather than a reimplementation |
| loader compatibility | **NOT DONE** |
| decoded-size projection from measured frames | **NOT DONE** |
| held-out camera (centre) protocol | **NOT DONE** |
| training | **NOT AUTHORIZED** |

The directive's E3 requires the official STG route rather than an independent
fisheye conversion. That route has **not** been read in this block: three
attempts to delegate a read of `pre_immersive_undistorted.py` and the
published per-scene numbers failed on transient API errors, and the primary
agent prioritized the acquisition and the DiVa/ImViD lanes over doing it by
hand. **So no claim is made here about what that script does, what it
requires, or what STG's per-scene Immersive numbers are.** The
[[loop2-sweep-2026-08]] record's STG 7-scene average (29.2 / 0.042 / 0.081)
is the only figure on record and it is an average, not per-scene.

## 5. What this dataset can and cannot be used for

**CAN** (subject to the preprocessing smoke passing): temporal/photometric
reconstruction and held-out-view generalization, as a deferred external SOTA
anchor under STG's published protocol.

**CANNOT**: supply disappearance/return events. All 46 viewpoints sit inside a
sub-metre sphere, so anything occluding the subject occludes it from
essentially all cameras at once and there is no second azimuth from which to
rule out "merely occluded from here". That disqualification is structural and
is **not** revisited by this page. Any future event screening on Immersive
would be a scene-ranking diagnostic only, never claim-grade admission, and
would need its own dataset-specific preregistration — DiVa's floors do not
transfer, and the coverage instrument cannot even be defined without masks.

## 6. Acquisition result

Experiment **157** (`immersive_fetch_flames` r0) COMPLETED in ~5 min.

```
path            /apollo/users/sri/proj_adags/data/immersive/raw/02_Flames.zip
bytes           5,474,948,990          (matches the recorded size exactly)
publisher MD5   b0pj+tNbWxPsY6Qj8S89Fg==   MATCH
sha256          0209febf06d7989a016fa38164e6ebc38472bb0637da0d7bd3a64c614feb468b
read-only       true
free before     34,198,435,921,920 bytes (31.1 TiB)
```

Archive central directory, nothing extracted at this stage:

```
entries      47      = 46 x .mp4 (5,474,302,514 bytes) + 1 x .json (25,985 bytes)
uncompressed 5,474,328,499        ratio 1.0001 -- a container, not a compressor
largest      camera_0016.mp4 127,043,635 bytes
naming       02_Flames/camera_00NN.mp4
```

### The calibration, read FROM THE DATA (experiment 162)

`02_Flames/models.json` extracted by name under a 4 MiB per-member cap
(sha256 `199afc790c274f4782b7786fd6014137286d05eec152d845e31d92ddc8ea8908`). It
is a JSON **list**, one entry per camera:

```json
{"name": "camera_0001",
 "position": [0.00655, 0.00148, 0.42002],
 "orientation": [-0.02831, 0.02742, -0.03381],
 "focal_length": 1113.591793482135,
 "pixel_aspect_ratio": 1.0,
 "principal_point": [1286.024, 930.536],
 "width": 2560.0, "height": 1920.0,
 "radial_distortion": [0.09911, -0.01876, 0.0],
 "projection_type": "fisheye"}
```

So: **fisheye projection, a single focal length with a pixel aspect ratio, a
2-parameter radial distortion** (the third term is 0.0 here), and orientation
as a 3-vector — an axis-angle convention, NOT a quaternion and NOT a matrix.
Raster 2560x1920.

**A DISCREPANCY, recorded rather than smoothed:** the archive holds **46**
`.mp4` files but `models.json` describes **45** cameras. One video has no
calibration entry, or one entry is dropped. Nothing here establishes which, and
any preprocessing step must resolve it explicitly rather than zipping the two
lists together by position — that is precisely the kind of off-by-one that
produces a silently mis-calibrated scene.

**The rig confirms the dome disqualification quantitatively:** camera positions
in this file sit within a fraction of a metre of the origin (the first entry is
0.42 m out), which is the sub-metre sphere
[[dataset-admission-matrix-2026-08-18]] describes.

**Still not done:** STG's `pre_immersive_undistorted.py` has not been read, so
nothing here states how these fields map onto its expected inputs, and no
conversion, loader check or decoded-size measurement has been performed.
