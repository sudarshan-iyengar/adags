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
