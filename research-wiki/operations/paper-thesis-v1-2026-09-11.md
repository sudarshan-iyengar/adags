---
title: "Paper thesis v1 — Occlusion is not absence: per-primitive multi-interval presence in 4D Gaussians"
date: 2026-09-11
evidence_bearing: false
status: draft-for-adversarial-review
---

# Paper thesis v1 (2026-09-11) — "Occlusion is not absence"

Working title: **Occlusion is not absence: per-primitive multi-interval
presence in 4D Gaussians.** This page states the four claims the paper
would make, the evidence on the record for each, the limitations the
record forces, and the closest work. Every number is copied from the
wiki page named beside it; nothing here is new evidence. The page is the
input to a kill-argument review and to the wave-2 spec; it is not the
paper. Written before wave 2, so that the claims cannot drift toward
whatever wave 2 returns.

## 0. The thesis in one paragraph

4D Gaussian methods give each primitive a temporal support (a lifespan,
a temporal marginal, a flat-top window). That support is learned from
photometric residuals, and a photometric residual cannot tell an object
that has LEFT the scene from an object that is merely HIDDEN: in both
cases the object's pixels are gone from every training view. On a real
occlusion the primitive that renders the object also renders its
occluder over the same location, so there is nothing for a presence
mechanism to remove ([[absence-fixture-lane-2026-09-10]] §7A;
[[realdata-gating-lane-2026-09-09]] §4). On a true absence the same
primitive keeps painting the object as a ghost during the gap and comes
back degraded at the return. We give each primitive a multi-interval
presence function with exact zeros, infer the absence window from the
training views alone, train WITH the gate, and show on real footage
with an authored absence that this yields exact absence in the gap and
a better return, while the same gate applied after training does the
opposite. The hard, unsolved part is deciding WHICH primitives belong to
the absent object; we quantify it with membership controls and declare
it.

## 1. Claim 1 — a training-time total presence gate gives exact absence and a faithful return; post-hoc gating is the opposite

**Evidence (real footage, authored absence).** Wine bottle removed from
`cut_roasted_beef` for frames 60–89 on all 20 cameras by an SA4D edit;
six arms on four independent 6k prefixes continued to 12k; every cell
re-rendered with its own gate live over frames 0–299 on the held-out
camera ([[absence-fixture-lane-2026-09-10]] §7B, spec v1.2.0). Paired
within-prefix medians (dB, min..max, sign consistency):

| contrast | P1 ghost core [63,87] | P2 return [92,99] | S1 settled [100,109] | H1 pre-gap [30,57] | H2 whole frame | C1 control [230,259] |
|---|---|---|---|---|---|---|
| G-oracle − U | **+4.16 (3.16..4.92) 4/4** | **+2.86 (2.61..3.24) 4/4** | +0.58 4/4 | +1.20 4/4 | −0.01 | −0.22 4/4 |
| G-mis − U (same rows, gap moved) | −4.58 4/4 | −1.73 4/4 | −0.99 | −1.12 | −0.02 | −0.23 |
| G-ones − U (same rows, late gap) | −4.57 4/4 | −1.61 4/4 | −0.90 | −1.07 | −0.02 | 0.00 |

G-oracle has the lower core MSE than U on 25 of 25 frames in every pair
(§7B). The two timing shams lose in every pair, so the effect is the
window, not the code path or the row set.

**The post-hoc direction.** On the ungated 6k model of the same fixture,
switching the same rows off at render time costs **5.9 dB** on the ghost
core (31.22 → 25.32) while gaining **+3.2 dB** at the return
(27.34 → 30.50) ([[absence-fixture-lane-2026-09-10]] §5): the bottle rows
also paint the counterfactual background in the gap. Trained with the
gate on, the other rows learn that background and both windows gain.
**A gate must be trained with, not applied after** (§7B reading 1).

**The occlusion side.** On the real occlusion of the same scene (beef
under a hand and knife, frames 158–187) the render-time gate on a fixed
model is **−3.4 dB** in the occluded box and −0.14 dB over frames
140–230 ([[realdata-gating-lane-2026-09-09]] §4); trained with the gate
(deferred seeding, ~3,100 rows per prefix) every paired contrast sits
inside ~0.1 dB (G − U −0.074 on the gap, +0.050 on the return; floor
0.040/0.068) ([[absence-fixture-lane-2026-09-10]] §7A). The gated rows
are behind the hand; gating them changes the render by nothing
measurable.

**The synthetic identity check.** On the LRV3 fixture the localized
total gate beat the matched temporal control by **+1.0496 dB** on
`event_return` at 1,126 fewer primitives, **+2.0 dB** on the first return
frame, and rendered exact absence (zero error) on **21 of 27** gap frames
([[lrv3-local-presence-corrected-cell-2026-08-20]]). A 2-frame timing
error costs −2.39 dB, below not gating at all; the measured ordering is
+1.05 > 0 > −2.39 >> −17.16 ([[paper-path-decision-2026-08-23b]] §2).

## 2. Claim 2 — absence windows are recoverable from training views alone, with zero false activations on real footage

The estimator (T1) ablates voxel groups of primitives in training-view
renders only, never reading a held-out image, mask or annotation, and
abstains unless the required number of cameras agree.

| substrate | groups | gated | boundaries | truth | false activations | page |
|---|---:|---:|---|---|---:|---|
| LRV3 synthetic | 417 | 2 | onset 30, offset 57 | 30 / 57 | 0 | [[nonoracle-episode-timing-result-2026-08-23]] |
| real occlusion, `cut_roasted_beef` | 1,218 | 1 | first absent 159, first present 188 | curated 158–187 | 0 | [[realdata-gating-lane-2026-09-09]] §5.1 |
| authored absence, same scene | 1,245 | 2 | offset 60 (both), onsets 90 / 92 | 60–89 | 0 | [[absence-fixture-lane-2026-09-10]] §4 |

On the fixture the estimated gap [60,91] has temporal IoU 0.9375 with the
authored [60,89], and training on the estimated gap (G-est) matches
G-oracle to **0.04 dB** on P1 and −0.04 dB on P2 in every pair (§7B).
Limits stated with the claim: recall is low by design (2 of 8
event-overlapping groups on LRV3; on the fixture T1's own cells against
the truth row set give precision 0.771, recall 0.737, Jaccard 0.605, §5),
and the estimator's selectivity is what makes it unable to exercise
shape-level operators ([[gap_map]] G13 update 2026-08-24).

## 3. Claim 3 — no public multi-view capture supplies the phenomenon, so counterfactual-absence fixtures and an event-window metric are the evaluation

**Supply on the record.** Frames 0–299 of `cut_roasted_beef` contain
essentially ONE clean occlude-and-return event on dynamic content, and it
is an occlusion, not an absence ([[crb300-event-mask-curation-2026-08-23]]).
DiVa-360: **0 of 597** scored true-absence windows corroborated as
genuine full-multiview disappearance ([[elgs-absence-diagnostic-result]]).
ImViD `scene6_puppy`: two real absence-and-return events with gaps of
6.7 s and 6.2 s, found by a human in an hour; the automated census could
not support a recall claim, and the full take is 15,215 frames
([[imvid-event-census-result-and-closure-2026-08-25]],
[[imvid-acquisition-quota-2026-08-24]]). The 2026-09-10 literature check
(session record; to be promoted to a wiki page) found no ≥8-camera
public video dataset that annotates full-multiview absence and return:
N3V/DyNeRF and Technicolor are occlusion-only, DNA-Rendering and
ActorsHQ are human-centric with props held throughout, CMU Panoptic has
incidental entries and exits but is unsuited to photometric evaluation,
and Ego-Exo4D, BEHAVE and HOI4D fail the camera bar.

**The fixture.** Real N3V footage in which one segmented, rigid object is
removed for a window by compositing SA4D's render of the scene without
that object's Gaussians into the real frames inside the segmenter's
silhouette; only window frames are edited; the derived scene is verified
byte-identical outside the edit ([[absence-fixture-lane-2026-09-10]] §0,
§1.3, §2). The edit costs the ungated substrate 0.10–0.28 dB whole-frame
at 6k (§4). Every arm is trained on the same edited data, so the
comparison is between representations, not between data.

**The metric.** Pooled PSNR over a frame window on a per-frame region:
the supported, eroded core of the absent object's silhouette during the
gap (P1), the object box on the early return (P2) and settled return
(S1), a pre-gap harm guard (H1), the whole frame (H2), and an untouched
control window (C1), with anchors frozen before any number exists
(`configs/n3v/absfix_gate_spec_v1.json` v1.2.0). Masked dynamic-region
PSNR on N3V exists in the literature; a temporal event-window protocol
does not (2026-09-10 literature check).

## 4. Claim 4 — membership is the open problem, and it is quantified

**On the fixture.** A count-matched RANDOM row set with the authored gap
(G-wrongmem, ~1% overlap with the truth set) gains **+0.65 dB
(0.17..0.78)** on P1 and +0.33 dB on P2 against U, over the frozen
0.5 dB floor in 3 of 4 pairs; the pre-registered rule required the
shams silent, so the frozen verdict is **NOT_MET** under spec v1.2.0
([[absence-fixture-lane-2026-09-10]] §7B). The membership-specific share
is G-oracle − G-wrongmem: **+3.78 dB (2.38..4.20)** on P1 and
**+2.32 dB (2.14..2.99)** on P2, 4 of 4 pairs. Membership ESTIMATION was
not tested on the fixture: the construction masks are the DEVA
silhouettes, so the estimated and truth row sets coincide by
construction (§5).

**On LRV3.** The spatial-partition estimator binds membership at
precision **0.0446 / recall 0.1786** on the fresh seeding cloud; the
recall cap is geometric (the object occupies 6.6% of the cells that
cover it) ([[lrv3-membership-diagnostic-2026-08-23]]). Gating on that
membership with EXACT timing costs **−2.469 dB** on the return against
not gating; the ordering is fully gated 28.19 > not gated 27.14 >>
partially gated 24.67 ([[nonoracle-timing-t2-result-2026-08-23]]).

**On the real occlusion.** Seeding-time membership bound 389 rows of
366k and clone/split grew them only to ~500 of 600k
([[realdata-gating-lane-2026-09-09]] §6-v2 results); deferred seeding
binds ~3,100 rows ([[absence-fixture-lane-2026-09-10]] §3, §7A); the
closed-form training-view vote is stable across cameras (3,326 members,
LOCO Jaccard ≥ 0.985) but is a visual hull, not an object
([[realdata-gating-lane-2026-09-09]] §5.2c).

## 5. Declared limitations (verbatim from the record)

* "an oracle-controlled diagnostic on a teacher-rendered counterfactual
  absence (`evidence_bearing: false`); the ground truth inside [60,89]
  is SA4D's background render, the red cap was left floating by the
  edit, and the object was hand-selected. Nothing here is evidence
  about the physical world" ([[absence-fixture-lane-2026-09-10]] §7B
  reading 5). The return windows P2/S1 and the guards H1/H2/C1 read
  real, unedited frames; only P1 reads teacher pixels.
* "Membership is authored in every positive cell"
  ([[paper-path-decision-2026-08-23b]] §3); on the fixture, estimated
  membership equals truth by construction and is not a test.
* "The verdict under the frozen spec v1.2.0 is NOT_MET"
  ([[absence-fixture-lane-2026-09-10]] §7B); the membership-specific
  contrast is a re-reading and would need a new frozen spec (wave 2).
* One scene, four prefixes, descriptive statistics at n = 4 pairs
  (paired sd 0.73 dB on P1); no external baseline has been trained on
  the fixture (the ungated arm is our own substrate at STG parity,
  33.5050 dB against STG's published 33.52 on frames 0–49,
  [[stg-n3v-protocol-parity-2026-08-19]]).
* The gate costs 0.16–0.28 dB in the untouched control window C1 in
  every pair (§7B reading 4) and 0.3–0.4 dB `ordinary_all` on LRV3.
* "the gate helps exactly when the object is absent from every camera
  and does nothing when it is merely occluded" (§7B reading 5); the
  real-occlusion negatives stay in front of the fixture positive.
* Every `--val` metric of every EL-GS cell before 2026-09-10 was
  rendered with the gate off; only gate-faithful re-evaluations are
  cited here (§6).

## 6. Closest work (from the 2026-09-10 literature check and the wiki paper notes)

1. **PersistGS** (arXiv 2606.03479, CVPRW 2026;
   [[papers/ramlal2026_persistgs]]). States the problem verbatim:
   Gaussians of a fully occluded object receive no photometric gradient
   and the reconstruction must recreate the object from scratch on
   re-emergence. Mechanism: differentiable rigid-body simulation
   supplying the SE(3) trajectory during the gap; per-object Gaussians;
   synthetic scenes; 5 cameras; held-out views that see the object
   during the gap; no event-window metric. Owns the problem statement,
   not the inferred-window per-primitive gate.
2. **SharpTimeGS** (arXiv 2602.02989, CVPR 2026;
   [[papers/liao2026_sharptimegs]]). Learnable per-primitive lifespan
   with a flat-top visibility profile; single interval; no absence and
   return. Makes per-primitive temporal support standard, so novelty
   cannot rest on temporal opacity.
3. **FreeTimeGS / FreeTimeGS++** (arXiv 2506.05348 / 2605.03337).
   Per-primitive temporal opacity and motion; unimodal support.
4. **RetimeGS** (arXiv 2603.13783). Primitives appear and disappear via
   short-tailed temporal opacity; single interval.
5. **CubifyGS** (arXiv 2606.28720, IROS 2026;
   [[papers/ren2026_cubifygs]]). Object-level asset pruning on
   disappearance and rigid re-alignment on reappearance; entity-level
   rule-based existence state, not per-primitive, RGB-D SLAM setting.
6. **SA4D** (arXiv 2407.04504; [[sa4d-read-2026-08-24]]). Temporal
   identity field for segmentation and editing; no presence inference.
   It is the fixture's editing tool, not a competitor.

Terminology: this page and the paper say **resumption** or **return of
the same primitives**; the word "reactivation" is avoided because
arXiv 2510.19653 ("Re-Activating Frozen Primitives") uses it for a
static-3DGS optimization mechanism.

## 7. What wave 2 must add before the claims above can be made in a paper

Stated here so the kill-argument review can attack the plan, not only
the record: (i) a new frozen spec whose primary is the
membership-specific contrast; (ii) two further scenes; (iii) an
estimated-membership arm whose membership comes from a segmenter
independent of the construction masks, with a precision/recall
precondition declared before any score; (iv) at least one external
per-primitive-lifespan baseline trained on the fixture; (v) the
literature check promoted to a wiki page.

## 8. Amendments after the kill-argument review (2026-09-11, append-only)

The review ([[paper-thesis-v1-kill-argument-2026-09-11]]) returned FAIL
with four unresolved points. The text above is left as reviewed (its
sha256 is recorded on the review page); the following corrections and
narrowings apply to every later use of this page.

1. **§3, ImViD sentence is WRONG and is corrected here.** The source
   page [[imvid-event-census-result-and-closure-2026-08-25]] records
   two "occlude-and-return" events in `scene6_puppy` (a 5,936-frame
   take), NOT classified as full-multiview absence; the 15,215-frame
   figure is the Opera take
   ([[imvid-acquisition-quota-2026-08-24]]). ImViD therefore supplies
   no adjudicated physical absence-and-return event either.
2. **§0 and §1, "exact absence" is narrowed.** On the fixture the
   mechanism predicate establishes exact-ZERO PRESENCE of the gated
   rows inside [63,87]; no frame of any arm reaches zero error there
   because the teacher background is never rendered bit-exactly
   ([[absence-fixture-lane-2026-09-10]] §7B). "Zero-error absence"
   applies only to the synthetic LRV3 fixture (21 of 27 gap frames).
   The paper's wording is "an exact-zero presence gate and a
   +4.2 dB lower ghost-core error".
3. **§2, T1 is renamed.** T1 estimates a VISIBILITY GAP (a window in
   which a group's contribution to the training views vanishes); it
   cannot and does not distinguish an object that has left the scene
   from one that is hidden. Absence-versus-occlusion classification is
   out of scope. The real-occlusion result is cited as evidence that
   the gate is inert on an occlusion, not as absence detection.
4. **Scope.** The paper claims a training-time presence-gate result on
   real footage under an AUTHORED absence, with the real-occlusion null
   reported alongside; it does not claim physical absence discovery.
   Title to be narrowed at paper-plan time.
5. Every remaining unresolved point (independent membership,
   generic-deletion share, replication, a trained baseline) is an arm,
   a precondition or an out-of-scope line in the wave-2 spec v2.0.0.
