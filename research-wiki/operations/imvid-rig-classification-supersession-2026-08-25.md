# SUPERSESSION (append-only) — the ImViD fixed-rig requirement becomes bounded input-integrity checks on a supplier-declared classification (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Written BEFORE any candidate is
generated on the completed takes**, so that no rig rule can be chosen
after seeing which candidates a take happens to offer.

This page **supersedes one prerequisite** in
[[imvid-event-definition-2026-08-24]] §4. It does not rewrite, weaken, or
delete that section, and the superseded rule remains readable there in its
original form. Everything else in that spec — §2 (the six conditions),
§2.1 (the frozen thresholds `C_min=3`, `W_pre=15`, `W_gap=20`,
`W_post=15`), §3 (the three-way A/B/C classification requiring POSITIVE
evidence), §5 (synchronization), §6 (the ordered gate and the zero-events
stop rule), §7, §8, and the 2026-08-24 amendment — is **UNCHANGED and
still binding**.

## 1. What §4 required, quoted so the change is legible

> **No ImViD scene may enter the event census until its take passes the
> fixed-rig test.** Metadata cannot certify it — a moving take registered
> at frame 0 produces an identical `images.txt`. The only sufficient test
> is a **fixed-pose triangulation residual at frames 0 / mid / end** of
> the actual take, with intrinsics and extrinsics held fixed, meeting the
> recorded gate (**mean ≤ 2.0 px AT NATIVE**).

That rule was written when the only thing known about rig mobility was a
scene-level claim in the paper's Table II, and when
[[dataset-admission-matrix-2026-08-18]] had disqualified ImViD partly on
the ground that the 39-camera array is mounted on a platform that moves.
Under those premises the rule was correct: with no way to tell a fixed
take from a moving one except by measurement, measurement was the only
sufficient test.

## 2. THE NEW EVIDENCE — the supplier separates the two classes itself

The premise has changed. The release supplier's own email states:

> "The following link contains our dataset, which includes 7 scenes (with
> calibration results) and 1 scene captured with a moving camera rig."

and the supplier's Drive layout **matches that sentence structurally**:
seven named `scene*` folders each carrying calibration, plus a **separate
`moving_rig` folder** holding the moving-rig material.

This is a materially different kind of evidence from Table II. Table II
describes the larger **capture campaign** and assigns scene-level capture
strategies across it; the delivered subset **physically separates** the
moving-rig recording into its own folder, outside the seven calibrated
`scene*` folders. The distinction the original §4 could not obtain from
metadata — *which takes are fixed* — is supplied directly by the party
that performed the capture, and is expressed in the file layout rather
than only in prose.

**Verified on Apollo this block, read-only:** there is **no `moving_rig`
directory anywhere** under `/apollo/users/sri/proj_adags/data/`. The only
ImViD raw folders present are `scene1_opera`, `scene6_puppy`, and a
partial `scene5_rendition`. `moving_rig` was never transferred and is
**out of scope**; it is not acquired to satisfy this page, and no
artificial exclusion or admission task is created for data that is not
present.

**Explicitly forbidden inference, recorded so it cannot be made later:**
Table II's scene-level "strategy 2" entry may **not** be used to conclude
that the accessible `scene6_puppy` folder is a moving take. Table II
describes the campaign; the supplied subset separates the moving-rig
material into its own folder. Where the two could be read against each
other, the delivered folder separation governs.

## 3. WHAT REPLACES THE RULE — bounded input-integrity checks

For a take delivered inside a calibrated `scene*` folder, the fixed-rig
prerequisite of §4 is replaced by the following bounded checks. All are
cheap, and all are checks on the **inputs**, not on a score:

1. **Calibration parses.** `cameras.txt` and `images.txt` parse under the
   COLMAP text convention, with the camera model read FROM DATA and never
   assumed (§6 item 2 of the original spec is unchanged and still governs
   this: Opera is `OPENCV`; other takes may ship a distortion-free
   `PINHOLE` line, and applying Opera's undistortion to those would
   corrupt them).
2. **Every expected camera is represented exactly once.** No missing, no
   duplicated, no invented camera. (`scene3_classroom`'s known defect —
   38 MP4s against a 39-line calibration — remains the worked example of
   what this check is for.)
3. **Dimensions and stream metadata agree** across cameras: resolution,
   codec, frame count, and measured rational rate consistent, with any
   disagreement named rather than averaged away.
4. **Sampled frames decode**, at recorded indices, with the decoded frame
   verified against an independent decode of the same index.
5. **Camera IDs map consistently** between the MP4 basenames,
   `cameras.txt`, and `images.txt`.

**If any of those checks fails, or if a later reconstruction reveals a
concrete pose or calibration contradiction, the affected take STOPS and
is investigated.** That is the residual safety net, and it is the reason
this supersession is not merely a relaxation: the original rule was a
*prospective* measurement, and what replaces it is a *prospective*
integrity screen plus a *standing* stop condition on any contradiction
that later appears.

## 4. THE EVIDENCE BOUNDARY — stated plainly, because it narrowed

**Fixed-rig status for Opera and Puppy is now SUPPLIER-DECLARED, not
independently measured by this project.** Any page, table, or claim that
rests on rig fixity must say so in those words. This is a genuine
reduction in evidential strength relative to the superseded §4, and it is
accepted deliberately and recorded here rather than absorbed silently.

**The partial independent corroboration that does exist, and its exact
extent.** Experiments 270 / 271 / 272 ran the reprojection gate on Opera
at frames 0, 150 and 299 with intrinsics and extrinsics held fixed, and
all three passed at **1.215 / 1.162 / 1.214 px mean at NATIVE** against a
**2.0 px NATIVE** gate, with **35/35** cameras contributing
([[block-2026-08-24-handover]] §2.1). A moving rig would not generally
hold a fixed-pose residual at that level across those frames, so this is
real corroborating evidence for Opera — **over the span it covers.**

That span is now known to be small. The full Opera take is **15,215
frames** (`60000/1001`, 253.84 s). Frames 0–299 are **1.97% of the take**
— **5.005 s of 253.84 s**. The three tested frames are the start, middle
and end **of the 300-frame sample**, not of the take. The superseded §4
asked for 0 / mid / end **of the actual take**, and that has **NOT** been
run on the full take for either scene.

| take | independent fixed-pose evidence | coverage | status |
|---|---|---|---|
| `scene1_opera` | exps 270/271/272, 1.215/1.162/1.214 px @ NATIVE, 35/35 cameras | frames 0–299 = **1.97%** of 15,215 | supplier-declared + **partially** corroborated |
| `scene6_puppy` | **none** | — | supplier-declared, **uncorroborated** |

**The asymmetry is deliberate and must travel with any Puppy result.**
Puppy has no independent pose evidence of any kind at the time of writing.

## 5. RESIDUAL RISK — preserved, not discharged

1. **A supplier misfiling is undetectable by these checks.** If a moving
   take were placed in a calibrated `scene*` folder by mistake, checks
   1–5 would all pass: a moving take registered at frame 0 produces an
   identical `images.txt`, which is exactly the point the original §4
   made and which remains true.
2. **Partial-span corroboration does not extend.** Opera's evidence
   covers 1.97% of its take. A rig that is fixed for the first five
   seconds and is disturbed later would pass everything recorded here.
3. **Puppy is uncorroborated.** Its fixed-rig status rests entirely on
   the supplier declaration and the folder separation.
4. **The consequence of being wrong is specific, not diffuse.** If a take
   is not fixed, the "applicable camera set" of E2 is not geometrically
   stable over time, and a class-C rig-induced visibility change can
   masquerade as a class-A absence. §3 already requires POSITIVE
   multi-view free-space evidence for class A, which is the guard that
   would have to fail as well; but the two failures are correlated,
   because both are computed in the same assumed-static frame.
5. **Cheap later strengthening exists and is NOT run here.** Extending
   the fixed-pose residual to frames spanning the full take — or at
   minimum to the frozen event window plus a late-take frame — would
   restore most of the lost strength at small cost. It is recorded as
   available, deliberately not performed at this point, and **required**
   before any Puppy or Opera result is promoted beyond exploratory.

## 6. Why an overconservative gate was the wrong instrument here

The superseded rule would have required a fresh feature-triangulation
experiment on each complete take purely to **re-prove a classification
the supplier already publishes and expresses in its folder layout**. That
is not the direction this project's errors have run. Every event failure
recorded in [[imvid-event-definition-2026-08-24]] §1 — DiVa-360's
0-of-597, the N3V dev masks, R034/R035 — was an instrument reporting
events that were **not there**. None was caused by trusting a supplier's
own capture classification. Spending a take-scale reconstruction on this
particular question would consume the block's scarce compute on the one
premise that has an independent documentary source, while the conditions
that actually produced this project's false positives — §3's positive
class-A evidence and §5's synchronization measurement — remain untouched
and fully funded.

**This is a judgment about where to spend evidence, and it is labelled as
one.** It is not a finding that the rig question is unimportant.

## 7. Permitted and forbidden under this supersession

**Permitted.** To run the bounded checks of §3 and, on passing them,
enter a delivered calibrated `scene*` take into the §6 event census. To
describe Opera's rig status as supplier-declared with partial
independent corroboration over frames 0–299. To describe Puppy's as
supplier-declared and uncorroborated.

**Forbidden.** To describe either take's fixed-rig status as
independently measured or as "verified". To cite exps 270/271/272 as a
full-take fixed-rig result. To infer Puppy's rig class from Table II. To
acquire `moving_rig` to satisfy this page. To treat a passing integrity
screen as evidence about rig motion — it is evidence about file
integrity, and nothing else. To promote any ImViD result beyond
exploratory without first running the §5 item-5 strengthening.

## 8. Provenance

* Supplier release email, quoted in §2, naming 7 calibrated scenes plus 1
  moving-rig scene.
* Supplier Drive layout: seven `scene*` folders with calibration, plus a
  separate `moving_rig` folder; recorded inventory
  `imvid_drive_inventory.json` (325 files, 1,181,076,959,285 bytes,
  enumerated live twice from different hosts —
  [[block-2026-08-24-handover]] §4).
* Apollo raw state verified read-only 2026-08-25: `scene1_opera` 41 files
  / 39 MP4 / 125,649,776,270 bytes; `scene6_puppy` 41 files / 39 MP4 /
  115,934,621,018 bytes; `scene5_rendition` 6 MP4 / 17,392,225,463 bytes
  (partial, out of scope); **0 `.part`, 0 `.lock`, 0 stale locks**; no
  `moving_rig` present anywhere under `data/`.
* Structural completeness only. Opera carries transfer-manifest SHA-256
  records for **15** objects; the manually transferred Puppy folder has
  **none**. This is **not** publisher-hash identity, and no page may
  describe it as such.
