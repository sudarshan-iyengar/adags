# SPEC (FROZEN) — the EL-GS event definition and scene-admission gate for
# ImViD (2026-08-24)

EXPLORATORY, `evidence_bearing: false`. **Frozen BEFORE any ImViD proxy is
generated and before any candidate is scored.** Written now, while the
release is still downloading, precisely so that no definition can be
chosen after seeing which candidates a scene happens to offer.

Authority: the 2026-08-24 block directive, Lane I ("Freeze an
EL-GS-relevant event definition"). Inherits the hard-won constraints of
[[elgs-absence-diagnostic-result]], [[crb300-event-mask-curation-2026-08-23]],
[[nonoracle-timing-t2-result-2026-08-23]] and
[[dataset-admission-matrix-2026-08-18]].

## 1. Why a definition has to be frozen first

This project has been wrong about events three times, each time in the
same direction — an instrument reported events, and the events were not
there:

* **DiVa-360**: **0 of 597** scored true-absence windows were corroborated
  as genuine full-multiview disappearance. In 96.6% an eligible foreground
  component was demonstrably occupying the anchor while the tracker's
  report merely failed to qualify. The tracker's visibility flag turned
  out to be a **per-point self-occlusion signal, not an existence
  signal**.
* **N3V `cut_roasted_beef`**: a ground-truth-only curation of all 300
  frames found **essentially ONE** clean occlude-and-return event on
  dynamic content. The frozen 0-49 dev masks were afterwards found **NOT
  confirmed by ground truth**, and one box appears to label the
  occluder-PRESENT window rather than the absence.
* **R034/R035**: a synthetic fixture scored AUC 1.0 and then predicted
  nothing about real admission (0 of 72 accepted). The standing rule from
  that failure: **fixture passage must never again be a Go criterion.**

So the definition below is deliberately restrictive, and it is written to
be *hard to satisfy accidentally*.

## 2. THE DEFINITION

A candidate is an **EL-GS event** if and only if ALL SIX hold. Any one
failing makes it **not an event** for this project's purposes, however
visually striking it is.

**E1 — Same identity present before.** A spatially-coherent object is
visible and localizable in at least **`C_min` training cameras** for at
least **`W_pre` consecutive frames** immediately preceding the gap.

**E2 — A meaningful absence interval.** For at least **`W_gap`
consecutive frames**, the object is absent from **every applicable
camera**, where "applicable" is fixed per candidate by a sealed
geometric rule declared before scoring (the D3 lesson: an
applicable-camera set must never silently include cameras that were never
queried).

**E3 — Same identity present after.** The **same** object — not merely a
similar one — is visible again for at least **`W_post`** consecutive
frames, in at least `C_min` training cameras.

**E4 — Sufficient training-view evidence.** The pre-gap and post-gap
segments are each observed by at least `C_min` **training** cameras, with
`C_min` counted over cameras that are in the training split and were
actually queried.

**E5 — A sealed held-out return endpoint.** At least one **held-out**
camera observes the return segment, and that camera's imagery is
untouched by every upstream step — curation, proxy generation,
initialization, tuning and model selection.

**E6 — The absence is not explained by occlusion or by the rig.** See §3.
This is the condition that killed 597 DiVa-360 windows and it is the one
that does the work.

### 2.1 The free parameters, frozen now

| symbol | value | why |
|---|---:|---|
| `C_min` | **3** training cameras | 2 permits a coincident pair; the N3V T1 estimator's zero-false-activation property was measured at a 3-of-4 camera bar, and relaxing it to 2 was only ever validated where boundary agreement was independently fully specific |
| `W_pre` | **15** frames | ≈0.25 s at 59.94 FPS; long enough that the object is established, short enough not to exclude brief appearances |
| `W_gap` | **20** frames | ≈0.33 s. Below this a "gap" is indistinguishable from motion blur, a dropped detection, or sync slop |
| `W_post` | **15** frames | symmetric with `W_pre`; the return is the scored endpoint and needs enough frames to be measurable |

**These are declared judgments, not derived quantities**, and they are
fixed here so that a scene cannot be admitted by moving them.

## 3. E6 — the three-way classification, and it must be POSITIVE

Every candidate is classified into exactly one of three classes, and
**only class A is an event**:

* **A — genuine scene absence.** The object has left the represented
  volume. Positive evidence required.
* **B — ordinary occlusion.** The object is still present but hidden by
  something else from the applicable cameras.
* **C — rig- or camera-induced visibility change.** The object left the
  frustum, the camera moved, exposure/white balance changed, or the applicable
  set itself changed.

**The classification must be POSITIVE for A, never residual.** "We could
not find an occluder, therefore it is absent" is exactly the inference
that produced 0-of-597. The required positive evidence for class A is a
**multi-view geometric argument**: across the applicable cameras, the
volume the object occupied is observed as *free* — not merely
un-associated — for the duration of the gap.

**Tracker visibility flags are inadmissible as evidence of absence**, at
any threshold. This is not a tuning preference: the flag was measured to
be binary in a prior instrument, which makes every threshold in `(0,1]`
equivalent and the lower-the-threshold repair vacuous.

**A class-B candidate is not worthless** — occlusion-and-reveal is real
and is what N3V's one clean event actually is. It is recorded, and
recorded *as class B*, and it may not be reported under a heading that
implies absence.

## 4. THE RIG CONDITION — a hard precondition, checked per take

[[dataset-admission-matrix-2026-08-18]] disqualified ImViD for event
supply on the ground that the 39-camera array is mounted on a platform
that moves, so the applicable-camera set is **not geometrically stable
over time**. The append-only narrowing established that mobility is a
property of a **TAKE**, not of the dataset: Opera and Meeting are
captured fixed-point only.

> **No ImViD scene may enter the event census until its take passes the
> fixed-rig test.** Metadata cannot certify it — a moving take registered
> at frame 0 produces an identical `images.txt`. The only sufficient test
> is a **fixed-pose triangulation residual at frames 0 / mid / end** of
> the actual take, with intrinsics and extrinsics held fixed, meeting the
> recorded gate (**mean ≤ 2.0 px AT NATIVE** — see
> [[imvid-baseline-freeze]] Appendix B5 for the raster trap in that
> number).

A take that fails this is excluded from Lane I entirely. `moving_rig` is
downloaded but **not admitted**, and its admission is a separate question
requiring per-frame pose.

## 5. SYNCHRONIZATION — the condition most likely to be fatal, and it is measurable in advance

ImViD reports **~10-20 ms** synchronization uncertainty. At the measured
`60000/1001` = 59.94 FPS that is **0.6-1.2 frames**.

**This sits directly on the critical path, and the reason is measured.**
On the LRV3 fixture a **2-frame** timing error cost **−2.386 dB** on the
return — *worse than not gating at all* — while exact timing gained
+1.05 dB. The ordering established there is: correct gate > no gate >>
mistimed gate. A sub-frame-to-1.2-frame per-camera offset is therefore not
a rounding concern; it is within a factor of two of the error magnitude
already shown to invert the sign of the mechanism's value.

**Frozen decision.** Before any ImViD gating experiment:

1. the census reports, for every multi-camera candidate, the **per-camera
   timing spread in FRAMES and in MILLISECONDS**, computed from the
   measured rational rate, never from an assumed 30 or 60;
2. if the measured spread of a candidate exceeds **1 frame**, an exact
   shared frame boundary **may not be assumed**, and that candidate
   requires either per-camera temporal offsets or timestamp-aware
   boundary inference before any gating cell is authorized;
3. a candidate whose spread cannot be measured is treated as failing (2),
   not as passing it.

**No ImViD gating experiment is authorized under this spec while a
candidate's timing spread is unmeasured.**

## 6. THE SCENE-ADMISSION GATE, and the stop rule

For each scene, in this order — cheapest kill first:

1. **Stream integrity** — camera count matches mp4 count; resolution,
   frame count, codec, rate consistent across cameras; malformed or
   missing cameras named. (`scene3_classroom` is already known to be
   missing `cam38.mp4`: **38 mp4s against a 39-line calibration.** That
   defect is preserved and reported; no camera is invented or duplicated.)
2. **Calibration** — supplied intrinsics/extrinsics preserved exactly;
   camera model read FROM DATA, never assumed (Opera is `OPENCV`; Meeting
   and Playing ship a single distortion-free `PINHOLE` line, so applying
   Opera's undistortion to them would corrupt them).
3. **Fixed-rig test** — §4.
4. **Event census** — §2, on low-resolution proxies.
5. **Sync measurement** — §5.

**If a scene yields ZERO class-A candidates, that is a RESULT and it is
reported as one.** It is not a reason to relax `C_min`, `W_gap`, or the
class-A positive-evidence requirement. The N3V dev scene has essentially
one event in 300 frames and DiVa-360 has zero corroborated in 597
windows; a scene with none is the expected outcome, not an anomaly to be
tuned away.

### 6.1 The oracle ceiling comes FIRST — and it is a stop rule

Per the directive, and consistent with everything this project has
measured:

> **The first scientific comparison on any admitted ImViD event window is
> ordinary temporal representation vs AUTHORED/ORACLE episode support and
> membership, under an identical training protocol and evaluator.**
>
> **If the oracle episodic arm does not improve the frozen return
> endpoint, that scene's mechanism lane STOPS.** Non-oracle membership
> cannot rescue an absent oracle ceiling.

This ordering is what made the LRV3 work cheap and decisive, and its
absence is what made the 2026-08-19 negative uninterpretable until the
wiring was fixed.

## 7. What this spec does NOT do

It does not admit ImViD for event supply — [[dataset-admission-matrix-2026-08-18]]'s
**NOT ADMITTED** verdict stands, and it stands for a reason no census can
change: **ImViD ships no masks and no identity ground truth**, so there is
no tracker-independent instrument of the kind DiVa-360's masks provided.
What this spec defines is the *strongest curation available without such
an instrument*, and every candidate it yields is therefore
**human/GT-curated evidence at best, never a measured supply statistic**.

It does not authorize any training. It does not set the frame tranche,
the schedule, or the split — those live in [[imvid-baseline-freeze]] and
its Appendix B.

## 8. Permitted and forbidden

**Permitted.** To report candidates under this definition, classed A/B/C,
with camera support and timing spread. To report that a scene has zero
class-A candidates. To proceed to an oracle-ceiling comparison on an
admitted class-A window.

**Forbidden.** To call a candidate an event without the positive class-A
evidence of §3. To use a tracker visibility flag as absence evidence. To
assume an exact shared frame boundary before §5's measurement. To admit a
take that has not passed §4. To relax any threshold in §2.1 after seeing
a scene's candidate list. To report a class-B occlusion under a heading
implying absence. To let a proxy-derived candidate list stand as ground
truth — it is a scouting instrument and must be labelled as one.

---

## AMENDMENT (2026-08-24, append-only) — the proxy census CANNOT satisfy §5 at scouting rates, and that is a constraint on the instrument, not a relaxation of the spec

Nothing above is rewritten. §5 requires that, before any ImViD gating
experiment, the census report each multi-camera candidate's per-camera
timing spread **in frames and in milliseconds**, and forbids assuming a
shared frame boundary when that spread exceeds one frame.

`scripts/imvid_event_proxy.py` is now built and self-tested (91/91, with
end-to-end ffmpeg verification on synthetic media, including a byte-exact
check that a strided proxy frame is identical to an independently decoded
source frame at the same index). **It cannot satisfy §5 at its scouting
defaults, and it says so in its own manifest.**

At the default 2 fps proxy rate the sampling step is **30 source frames =
500.5 ms** — **25x coarser** than ImViD's stated 20 ms upper bound
(~1.2 source frames). The reported `spread_ms` column at that rate is
dominated by proxy sampling, not by camera synchronization. The manifest
therefore carries `sync_uncertainty_resolvable_at_this_proxy_rate: false`,
and only a near-native proxy rate would flip it.

**The consequence is a two-stage requirement, and §5 is UNCHANGED by it:**

1. **Scouting** — the low-rate proxy census locates candidate windows.
   Its timing numbers are *localization brackets*, not synchronization
   measurements, and may not be cited as satisfying §5.
2. **Sync measurement** — before any gating cell on a selected candidate,
   the per-camera timing spread must be measured at a rate fine enough to
   resolve ~1 frame, on the *narrow* window the scouting stage selected.

That two-stage split is what makes the requirement affordable: a
near-native-rate measurement over a 15,215-frame take is prohibitive,
while the same measurement over one selected ~50-frame window is cheap.

### Other census limitations, recorded now so no later reading over-claims

* **Localization is one proxy step.** A "rise at frame 120" means
  *somewhere in (90, 120]*. Every candidate carries its bracket.
* **Polarity is a signal direction, not a semantic.** A rise is equally
  consistent with an occluder arriving, content leaving, a light change,
  or auto-exposure. This instrument has **no return-fidelity gate**, which
  is the thing the N3V curation needed to separate occlude-and-return from
  permanent change. It therefore cannot by itself assign the A/B/C classes
  of §3 — it can only propose candidates for that classification.
* **Global-mean signals are blind to small objects.** At a 480 px long
  edge on a 5312 px source (11x down) a small object cannot move a
  whole-frame mean. The N3V pass needed a box search at quarter resolution
  and still put sub-8-px features near its detection floor.
* **The window's own temporal median is the template**, so content
  occluded for more than half a window becomes the template and inverts
  the polarity — the exact contamination the N3V curation hit and fixed
  with hand-chosen exposed-reference windows.
* **Cross-camera clustering is greedy and anchored on its first member**,
  so a chain spaced just under tolerance fragments differently depending
  on camera-name order.
* **No occlusion reasoning at all.** A cluster with 3-of-39 support may be
  one real event seen from three views or three coincidences. Support
  count is evidence of scene-level structure, **never proof** — which is
  precisely why §3 requires positive geometric evidence for class A.

**None of the above relaxes any threshold in §2.1 or any requirement in
§3 or §5.** It records what the available instrument can and cannot
measure, so that a candidate list is read as what it is: a scouting
output for human/ground-truth curation.
