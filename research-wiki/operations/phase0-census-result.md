# Phase 0 Census Result — PHASE0_NO_GO

Date: 2026-07-29
Branch: `csvl-vpl-v2-phase0`
Preregistration: [[operations/phase0-census-preregistration]] (floors frozen at
commit `01324f3`, before execution)
Smoke job (non-scientific, 3 frames): `50762194`, COMPLETED `0:0`, 1:43
Scientific job: `50762703`, COMPLETED `0:0`, elapsed 32:36, MaxRSS 9.9 GiB
Output: `$WORK/proj_adags/runs/phase9-depth-visibility-capacity/phase0-census-v1/`
(`census-v1.json`, canonical scientific SHA-256 recorded in the artifact)
Outcome: **PHASE0_NO_GO** under the frozen floors. Phase 1 is not
scientifically authorized.

## Floor verdicts (applied exactly as preregistered)

| Floor | Requirement | Observed | Verdict |
|---|---|---|---|
| F1 abundance | occluded-with-witness >= 0.5% of evaluable tuples | **4.397%** (40,587,132 of 923,149,439) | PASS |
| F2 reveals | >= 5,000 reveal pairs, >= 10 end frames, >= 5 cameras | **415,282 pairs**, 297 end frames, 17 cameras | PASS |
| F3 non-degeneracy | per-camera median in [0.1%, 40%], max <= 60% | median 4.41%, max 7.99% | PASS |
| F4 control separation | valid pairs >= 2.0x shuffle pairs | ratio **0.950** (415,282 vs 437,206) | **FAIL** |
| F5 evidence validity | conflict <= 15% AND map pass fraction >= 90% | conflict **0.024%** (3/12,457); map pass **89.47%** | **FAIL** |

## Attribution (per the preregistered failure-attribution rule)

**F5 (evidence validity, marginal).** The 89.47% map pass fraction is exactly
17/19: cameras `cam12` and `cam19` appear in only **one** P01 group per frame,
so the >= 3-member consensus rule excludes them in all 300 frames. This is a
structural property of the sealed P01 group composition, not of depth quality
— every other camera has median >= 3 members and ~0.8 valid-pixel fraction
(the 0.8 ceiling is the preregistered 20th-percentile confidence gate). The
cross-view conflict fraction of 0.024% simultaneously shows the consensus
depth that does exist is internally consistent to a degree the P03 route never
demonstrated.

**F4 (control separation, decisive).** Frame-shuffling the evidence depth did
not reduce — in fact slightly increased — completed-reveal counts (shuffle
occluded fraction 5.47% vs valid 4.40%; shuffle events 2,274,670 vs valid
1,308,731). Interpretation: in a mostly static kitchen, per-pixel consensus
depth is nearly time-invariant, so per-frame margin-state transitions are
driven by primitive motion and margin-level flicker, which are invariant to
the shuffle — not by temporal structure in the evidence. The naive per-frame
margin-transition rule therefore cannot certify that any counted reveal is a
genuine dynamic disocclusion event. The margin-sensitivity variants agree
with a flicker-dominated head: occluded fraction moves 13.3% -> 4.4% -> 2.4%
as tau_rel goes 0.01 -> 0.03 -> 0.05, and the run-length histogram has a large
geometric-looking short-run head (336,196 completed runs of exactly 3) plus
long structural tails (runs up to 299 frames — static-parallax occlusions
that persist for the whole sequence in one camera).

## What this result does and does not establish

Established (positive knowledge, new relative to Stage 1C):

- The primitive-centric E1 representation contains the target phenomenon in
  abundance: 4.4% of a ~10.7M (primitive, camera) universe is in
  occluded-with-witness state at typical times, across 17 healthy cameras,
  vs the sealed P03 route's **zero** cross-order candidates. The evidence
  substrate is dense and internally consistent (0.024% conflict).
- The preregistered shuffle control works as designed and caught, at a cost
  of one 33-minute CPU-scale job, the same class of non-specificity that
  Stage 1 discovered only after a full association implementation.

Not established:

- That any specific counted reveal is a genuine dynamic disocclusion event.
- Elevated event density in the two historical cut R009 windows: 83,593 and
  55,632 completed events in frames 95-110 and 140-155 respectively, which is
  the same order as the uniform-rate expectation (~70,000 per 16-frame
  window). Descriptive only, and an honest null.

## Binding restrictions

No floor may be adjusted and no re-run may be launched against this cycle's
authority: revising the counting rule after seeing the outcome and re-running
under the same preregistration would be outcome-conditioned tuning. Any
census-v2 requires a new preregistration, new floors, and user approval.

## Smallest justified next action (recommendation, requires user approval)

A census-v2 preregistration whose reveal-certification rule is designed to
separate dynamic occlusion from static parallax and margin flicker:

1. occluder-gap magnitude: `z - d` must exceed `k x margin` (k >= 3), i.e.
   something with substantial depth must be in front, not margin noise;
2. occluder temporal coherence: the occluding surface's depth at the pixel
   must itself change coherently at onset and offset (the shuffle control
   destroys exactly this, so a correct rule should collapse under shuffle);
3. hysteresis on entry/exit states;
4. static-parallax exclusion: (primitive, camera) pairs occluded in their
   baseline state at sequence start never count as reveal candidates;
5. cam12/cam19: either restore their membership upstream in the P01 grouping
   or preregister their exclusion explicitly.

## Dataset contingency assessment (bounded, read-only, from already-read sources)

The Phase 0 authorization asked for a dataset assessment if N3V lacked
sufficient or conclusive opportunities. The census outcome is *inconclusive
certification*, not absence: F1-F3 show the phenomenon is present at scale.
Datasets used by the closest works (from the wiki paper pages):
TAD-GS: N3DV (= N3V) + Interdigital + VRU Basketball; 4C4D: Neural3DV
(ships configs for our exact scenes); STG/Ex4DGS lineage: N3DV + Technicolor;
VAD-GS: Waymo/nuScenes (rigid urban — unsuitable for a non-rigid claim);
PersistGS: own 5-camera rigid-object capture; PackUV: own 50+-camera dataset;
GauSTAR: multiview studio capture; RiGS / Mono4DGS-HDR / USplat4D: monocular.
Viable multiview-dynamic alternates if N3V were ever abandoned: Technicolor/
Interdigital light-field video, CMU Panoptic / PanopticSports (a smoke config
exists in-repo; cross-dataset work remains contract-deferred).
**Recommendation: stay on N3V.** The failure is in the certification rule,
which is cheap to fix and cheap to re-falsify under a new preregistration;
the comparator field evaluates on the same data; and no alternate dataset
would evade the F4 problem, which is a property of the rule, not the scene.

## Links

- [[operations/phase0-census-preregistration]]
- [[operations/phase9-csvl-vpl-v2-direction]]
- [[operations/phase9-csvl-vpl-stage1c-result]] (the contrast: 0 candidates)
