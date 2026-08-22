# 300-frame event-mask curation for `cut_roasted_beef` — record and
# findings (2026-08-23)

Curation record for `configs/n3v/ladder_event_masks_crb0_299.json`.
Produced by a bounded read-only worker and reviewed by the primary.
EXPLORATORY. The comparison these masks serve is DEFERRED
([[crb300-b0r-b1-spec-2026-08-23]]); the masks are frozen now so it is
launch-ready.

## 1. Method — ground truth only

Input was **exclusively** `data/synthetic/b0c_eval/gt/00000.png …
00299.png`, the 300 ground-truth cam00 frames at 1352×1014. Model
renders (`renders36k/`, `uncap/`) were excluded from mask SELECTION so
the event definition is method-independent.

Pipeline: per-frame absolute difference and temporal-standard-deviation
activity maps to localize motion; occluder detection by distance from a
temporal-median template, cross-checked under three template scopes
(global, rolling ±30 frames, and hand-chosen exposed-reference windows —
the last used for all final numbers because global and rolling medians
get contaminated wherever an occluder dwells); an exhaustive box search
over a 12-px grid × 4 box sizes with hysteresis; and three gates —
**texture** (mean gradient, which eliminated a large family of
detections that were only a hand moving across a featureless black
apron), **coverage** (≥40% of the box at peak), and **return fidelity**
(post-occlusion distance ≈ pre-occlusion distance, the discriminator
between occlude-and-return and permanent content change). Every
surviving candidate was then verified frame-by-frame by eye, and several
high-scoring automated candidates were rejected only at that step.

## 2. Independence disclosure, and the primary's assessment

The worker disclosed, unprompted, that its first orienting command also
read `data/synthetic/b0c_eval/b0c_36k_slices.json`, which turned out to
be a RENDER-METRICS file (per-frame PSNR and a worst-decile frame list
for the B0-C 36k arm). It stopped immediately and states no mask
decision derived from it.

**Primary's assessment, recorded rather than waved through.** There IS
overlap between the curated intervals and that file's worst-decile
frames (249-270, 297-299, 39-51): event `I` at [252,258], part of `E`
at [263,282], and the tail of `J` at [286,297] fall inside it. Three
things bound the risk: the overlap is physically EXPECTED, because
frames are hard for the same reason they contain events; the file
describes ONE arm (B0-C, no reserved parity, 36k schedule) which is
neither arm of the deferred comparison; and a mask's placement affects
both arms of a PAIRED comparison equally, so a paired delta is not
biased by where the mask sits. The strongest curated event, `F` at
[190,209], is absent from the worst-decile list.

**Mitigation, and it is free because the comparison is deferred: an
independent ground-truth-only re-verification of the curated intervals
is a PRECONDITION of any 300-frame launch.** This is recorded as a
requirement, not as a reassurance.

## 3. What the sequence contains

A cook slices roasted beef at a butcher-block counter under a window
with a roman blind; a small dog sits on a stool at frame-left and moves
throughout (excluded — self-motion of a deformable object, not surface
occlusion, and return fidelity could not be established).

| frames | content |
|---|---|
| 0-90 | wrist articulation only; a thin blade tip oscillates over x≈745-810. Very little occlusion amplitude anywhere |
| 90-152 | the cook straightens; head rises into the blind, arm moves onto the counter. This is the OCCLUDING half-cycle — no reveals |
| 152-190 | a wide flat blade is laid across the beef pile, hiding it entirely, while the head drops and re-exposes the blind |
| 188-212 | the blade withdraws and the beef pile re-appears in the same place, slightly fanned. **The strongest occlude-and-return in the clip** |
| 200-250 | repeated blade passes; the left sleeve covers the left sill continuously |
| 250-299 | the cook straightens fully; the left sill clears after ~150 frames hidden; a final blade pass sweeps the upper-right board and withdraws |

**Occlusion activity is heavily back-loaded.** Reveal-onset counts per
50-frame segment across the whole box search: **0-49: 2, 50-99: 8,
100-149: 10, 150-199: 94, 200-249: 136, 250-299: 143.**

## 4. The finding that matters most for the paper

**Frames 0-299 of `cut_roasted_beef` contain essentially ONE large,
clean occlude-and-return event on DYNAMIC content** — `F`, the blade
laid across the beef pile (occluded 158-187, revealed 190-209, 66% peak
coverage, template distance 5→48→6.6).

Every other high-confidence event is **static background revealed by the
cook's body**: the roman blind behind the head (`E`), the window mullion
behind the sleeve (`D`), the window sill behind the upper arm (`I`), the
butcher-block edge (`H`), the empty board (`J`). Those are rigid
surfaces with exact returns — and they are arguably EASIER to
reconstruct than food surfaces.

This bears directly on the open real-data event-supply question. It is
the N3V analogue of the DiVa-360 supply problem: the events the method
targets are not merely hard to measure here, they are **nearly absent**
from the development scene. Recorded as a scene-level fact, not as a
verdict on any method.

## 5. Consequence: the endpoint is CLASS-SPLIT, and the primary decided it

The B1 operator relocates capacity into DYNAMIC-MASK regions. A static
background reveal is not such a region, so B1 has no mechanism by which
it would act there. Pooling both classes into one "event union" endpoint
would therefore dilute the quantity the operator is supposed to move,
and would let a background result masquerade as an event result.

Each event carries a `class`, and:

* **PRIMARY endpoint = the `dynamic_content` class**
  (`F`, plus the inherited `A`/`B`/`C` — which are excluded from the
  primary as low-confidence, see §6, leaving `F` alone as confirmed);
* **SECONDARY diagnostic = the `static_background` class**, reported
  separately and never pooled with the primary.

Class-pooled PSNR is recovered exactly from the per-event outputs of
`scripts/event_ray_metrics.py`, which reports `psnr_pooled` and
`pixel_times` per event:
`pooled_mse = Σ_e n_e · 10^(−psnr_e/10) / Σ_e n_e`, with `n_e = 3 ·
pixel_times_e`. One run therefore suffices; the tool's own
`all_events_union` mixes classes and is NOT the primary endpoint.

Validated structurally this block: all bounding boxes in raster bounds,
all intervals inside `[0, 299]`, names unique. Coverage:
`dynamic_content` 343,956 pixel-times over 56 scored view-frames;
`static_background` 503,904 over 110.

## 6. The frozen 0-49 dev masks are NOT confirmed by ground truth

Measured inside the frozen boxes against a 0-49-local exposed template:

| box | occluder coverage over 0-49 | mean distance ON the frozen scored frames | OFF |
|---|---|---:|---:|
| `A [655,845,745,905]` | 2-17% | 8.54 | 7.93 |
| `B [700,880,790,955]` | 0-7% | 4.51 | 4.62 |
| `C [795,845,890,955]` | 1-9% | 7.91 | 5.74 |

`A`'s distance drifts monotonically 4→15 across 0-49 (a hand slowly
entering) rather than cycling. `B`'s box shows essentially nothing
happening. **`C`'s frozen `[34,39]` window sits at a local deviation
MAXIMUM — i.e. more occluded than the segment average.** An occlusion
kymograph of the board band confirms all occluder activity in 0-49 is
confined to x≈745-810 in two blobs at f≈19-27 and f≈31-41, so `[34,39]`
— the interval assigned to BOTH `B` and `C` — falls INSIDE the second
occlusion blob rather than after it.

**Reading:** the 0-49 intervals for `B` and `C` appear to label the
occluder-PRESENT window rather than the occluder-withdrawn one, and all
three boxes are larger than the region that actually changes, so they
score mostly static pixels.

**This does NOT retract any recorded number.** The ladder's measured
event-union deltas (+0.077 / +0.345) stand exactly as recorded. What
weakens is their INTERPRETATION as an event-region effect: on masks that
mostly score static pixels, a small event-union gain is consistent with
a small global effect, and the recorded magnitude is unsurprising in
that light. Append-only correction; the frozen 0-49 file is unchanged
and the three events are carried into the 300-frame spec VERBATIM so
the spec remains a strict superset and old numbers stay comparable —
marked `low_inherited` and excluded from the primary endpoint.

## 7. Verdict on the previously-proposed segments

* **130-179 QUALIFIES but the interval is MISALIGNED.** It contains the
  best event in the clip, but the occlusion runs 158-187 and the reveal
  does not begin until 188-189, so a window ending at 179 would score
  almost entirely OCCLUDED frames. The correct window is ≈[190,209],
  which is what the frozen file uses.
* **195-244 QUALIFIES, second tier.** It holds the tail of `F`'s reveal
  and all of `H`, but no large-amplitude event of its own.
* **100-149 is EMPTY of qualifying reveals** — it is the occluding
  half-cycle. **0-49 contains none beyond the unconfirmed `A`/`B`/`C`.**
  If a design needs six equally-weighted 50-frame event segments, **this
  clip's imagery cannot supply them.**

## 8. Rejected candidates, recorded so they are not re-proposed

* `[676,784,748,840]`, `[728,816,792,864]`, `[664,800,728,848]`,
  `[640,820,712,876]`, `[856,800,920,848]`, `[832,772,904,828]` — a hand
  or fingers moving against a **featureless black apron or cream wall**.
  The "revealed surface" is flat; there is nothing to reconstruct. This
  was the single largest false-positive class and is why the texture
  gate exists.
* `[748,894,822,950]` — genuine blade occlusion, but coverage falls only
  45%→22% and never returns to baseline: the cook **moved beef into**
  the box. Content change, not occlude-and-return. Correctly caught by
  the return-fidelity gate.
* `[810,430,900,520]`, `[840,432,916,528]` — a very clean occlusion from
  f≈255 that is **still occluded at frame 299**. No return in the window.
* `[790,600,900,700]`, `[730,610,830,700]`, `[748,484,824,542]` — slow
  monotonic arm drift with no clean cycle, or only partial clearing.
* The dog — highest raw cycle count in the frame, deliberately excluded.

## 9. Caveats carried forward

1. **Return is not pixel-exact for the working-area events.** `F`, `H`
   and `J` retain a residual (scored-frame distance 5.6-7.0 against a
   pre-occlusion 2.4-3.7): the beef pile is slightly FANNED after the
   blade withdraws and there is new debris on the board. Only the blind
   events have genuinely exact returns. A reconstruction scored on `F`
   is being asked to reproduce a mildly perturbed surface.
2. **`D` has no clean pre-occlusion exposure inside the window** — the
   sleeve arrives from before frame 0, so "return to a previously
   observed state" cannot be verified within the clip for that box.
3. **`I`'s reveal is only 7 frames**, though it follows the longest
   absence in the clip (~150 frames) and is therefore the most
   interesting episodic case; it is statistically thin.
4. **Box edges are ±15 px**, the same tolerance as the dev spec. Only
   `F` was refined by a nested-box sweep (peak coverage 37%→48%→66% as
   the box tightened; the tightest was taken).
5. **Detection ran at quarter resolution** (338×254) with full-resolution
   visual verification, so sub-8-px occluder features — the thin tongs
   wire, the blade tip in 0-49 — sit near the detection floor. This is
   part of why 0-49 reads as empty, and a finer pass is not excluded
   from finding a small event there.
