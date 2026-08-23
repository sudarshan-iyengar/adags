# DECISION — the paper path is REPRESENTATION-FIRST, and the membership
# limb is a declared limitation rather than a pending repair
# (2026-08-23, block 2)

Supersedes nothing; it updates the recommendation using this block's
three diagnostics. Inputs:
[[lrv3-membership-diagnostic-2026-08-23]],
[[lrv4-lo-distribution-result-2026-08-23]],
[[b1f-flow-postmortem-2026-08-23]].

## 1. The decision

**Representation-first.** The paper's claim is per-primitive multi-episode
temporal presence with a TOTAL opacity gate, demonstrated on an authored
fixture, with **authored membership declared as a limitation** and the
non-oracle *boundary* result carried as a positive.

The three candidate paths, and why the other two are not selected:

| path | status |
|---|---|
| **representation-first** | **SELECTED** |
| representation + repaired membership | **NOT AVAILABLE.** Non-oracle recall is hard-capped at 0.1786 by the estimator, and every downstream repair lands in the partial-membership regime measured at 24.67 against 27.14 ungated. |
| representation + an admitted flow/supporting mechanism | **NOT AVAILABLE.** The birth prior is rejected, and the post-mortem shows the 6k/50-frame channel cannot resolve an effect of this size at all. |

## 2. What the paper can claim

* **Per-primitive discontinuous support with a total opacity gate** —
  `+1.0496 dB` on `event_return` over the matched temporal control at
  *lower* realized capacity, `+2.0 dB` on the first return frame, and
  **exact absence (infinite PSNR, zero error) on 21 of 27 gap frames**,
  which a temporal marginal structurally cannot reach.
* **Episode boundaries are recoverable EXACTLY from training views
  alone** — 2 of 417 groups gated, both on the event object, onset and
  offset 0 frames in error, zero false activations, 99.52% abstention,
  0.188 slot-h. Membership came from the cloud's own bounding box, never
  the oracle sphere.
* **The substrate is not the blocker** — 33.5050 dB against STG's
  published 33.52 at a quarter of the schedule.
* **Timing precision is the binding requirement**, with the measured
  ordering `+1.05 > 0 > −2.39 >> −17.16`.

## 3. What the paper must declare as limitations

* **Membership is authored in every positive cell.** Quantified rather
  than hand-waved: the obvious spatial-partition estimator achieves
  **4.46% precision / 17.86% recall** at the moment it binds, and gating
  is then **−2.469 dB against not gating** even with exact timing.
* **The 8³ partition is refuted, and the ceiling is geometric** — the
  sphere occupies 6.6% of the 8 cells that cover it, so perfect recall
  caps precision at 5.71%.
* **The fixture's absence is a clean ray-trace removal**, not occlusion.
  Real data will not supply that step.
* **Real-data event supply on the development scene is nearly absent** —
  frames 0-299 of `cut_roasted_beef` contain essentially ONE clean
  occlude-and-return event on dynamic content.
* **Consolidation has no demonstrated payload**, and the fixture built to
  test the observation-supply explanation cannot host the experiment.

## 4. The single next scientific action

**Measure the same-code replicate floor at the 50-frame 6k protocol, then
decide whether that protocol can support any N3V utility claim at all.**

This is now the binding methodological question, and it is upstream of
every remaining lane. The flow post-mortem measured that two runs of
*identical code at an identical seed* separate by ~0.09–0.45 relative RMS
training loss before iteration 1000, with densification at iteration 500
as the amplifier, and that arm-to-arm endpoint differences are the same
size as seed-to-seed differences. If the replicate floor is comparable to
the 0.341 dB event-union seed spread, then **no 50-frame two-seed
comparison in this family can resolve the effects the ladder was built to
measure**, and the utility claim needs either a longer protocol, more
seeds, or a different endpoint.

It is cheap — three identical 6k cells at one seed, ~7.5 slot-h — and it
either rescues or retires the entire N3V utility lane. Nothing else
should be scaled until it returns.

## 5. Explicitly NOT reopened

B1-D, the old B2 DC edit, the six-scene benchmark, the deferred 300-frame
B0-R vs B1 comparison, broad literature review, and DiVa-360 claim-grade
instrumentation. No primary evidence in this block changes any of their
gates.
