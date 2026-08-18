# Google Immersive Light Field Video — verified inventory and pilot acquisition

Date: 2026-08-18. Status: **inventory VERIFIED by direct request; pilot scene
acquisition submitted.** EXPLORATORY. No Immersive training has run, no
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
| pilot acquisition of `02_Flames` | submitted, exp 157 |
| archive central-directory inventory | part of exp 157 (`--inspect`) |
| `models.json` fisheye format read from the data | **NOT DONE** — requires the extracted archive |
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

*To be completed when experiment 157 is terminal.* The record must carry: the
publisher MD5 match, the self-computed sha256, the byte count, the archive's
central-directory inventory (entry count, uncompressed total, compression
ratio, per-suffix breakdown), and whether `models.json` is present.
