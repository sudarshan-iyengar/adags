# Block decisions — 2026-08-23 (payload falsified, timing inference
# works, event supply is nearly absent)

EXPLORATORY tier throughout. Decisions strongest-first, each
evidence-labelled. Experiments this block: **233-250**
(`agent-control/elgs-apollo/experiment-ledger.jsonl` is the id/retry
authority). Schedule authority:
[[block-2026-08-23-schedule-amendment]] — 12,000-iteration absolute
ceiling, 6k default, 24 slot-hour block ceiling, the 18k proposal
historical and unauthorized.

## 1. Non-oracle episode boundaries are recoverable EXACTLY — the block's strongest result

Verified from the primary artifact
([[nonoracle-episode-timing-result-2026-08-23]], experiment 235): a
training-view-only estimator gated **2 of 417** candidate groups, both on
the event object, with **onset and offset both 0 frames in error**, **zero
false activations**, 99.52% abstention, in **0.188 slot-h**.

This addresses what the supervisor brief calls the single largest
scientific risk. The risk was real: a measured 2-frame error costs
−2.39 dB, *below* not gating at all. The frozen gate therefore
disqualified any accepted boundary off by ≥2 frames, and was met by exact
recovery rather than by a tolerance.

**Load-bearing design fact:** membership came from a voxel grid over the
trained cloud's own bounding box, **NOT** the oracle sphere — which is
the event object's true geometry and would have faked the result.
Held-out cameras were never touched. Abstention reused the existing
`family_id = -1` path, so an abstaining group keeps the ordinary temporal
marginal bit-for-bit.

**Limits, recorded as limits:** recall is 2 of 8 event-overlapping groups
— a high-precision, low-recall instrument. LRV3's absence is genuine
removal from the ray-trace, not occlusion, so this does not transfer to
real data.

**Decision: phase T2 was justified, built, RUN — and it FAILED**
([[nonoracle-timing-t2-result-2026-08-23]], experiments 245/246/249/250).
Retraining on the inferred program costs **−2.469 dB** against not
gating, matching the 2-frame mistiming arm, **with timing exact to
~1e-14**. The gate fires at the right times, helps during presence, and
destroys the return. **Fully gated 28.19 > NOT gated 27.14 >> PARTIALLY
gated 24.67 — partial membership is worse than both.**

**So the T1 positive does NOT carry downstream on its own**, and the
honest headline of this lane is now two-sided: boundaries ARE
recoverable exactly, and that is not sufficient. Per-row MEMBERSHIP is a
second precision requirement of comparable severity, and voxel-cell
membership at 8³ is refuted — the boundary output remains exact and
reusable. **Next step is membership, scored against ground truth BEFORE
any retraining**, reusing the ordering that made T1 cheap and decisive.

## 2. NO consolidation payload has headroom — and the finding is about the FIXTURE

Verified ([[payload-headroom-result-2026-08-23]], experiments 233/236):
with an oracle-correct link, every transferable per-row quantity on LRV3
sits at or below the same-identity floor — appearance 1.43 (already
falsified), opacity 1.22 activated, `_scaling_t` 0.77, position 1.09,
extent 1.04, orientation 1.17. Position **discriminates identity**
(10.5-35.4) while having **no headroom** (1.09): a working instrument
finding nothing to recover. The oracle-correct opacity edit is **actively
harmful (−1.19 dB)**, not neutral, and the certificate correctly rejected
a −6.05 dB wrong-identity edit.

Geometry was eliminated BEFORE compute: `EVENT_SPHERE_CENTRE` is a module
constant applied in both episodes, so an oracle-correct geometry transfer
is the identity map.

**Decision, and it is the frozen rule's:** consolidation currently has no
useful payload, and **the representation-only pivot is the recommendation
on the record.** The certificate machinery works; the payload does not
exist.

**APPENDED after the permutation control ran (experiment 244) — the
attribution changed and the negative got stronger.** A fresh adversarial
review identified that the L3 no-op could not attribute the harm, because
it edits donor rows whose support ends before the scored frames. The
control it named — a within-recipient permutation, identity destroyed,
window preserved — ran in under a minute and returned **−0.9685 dB at a
pre-edit distance within 0.6% of the oracle link's, while editing HALF as
many rows**. The three metric-visible links are monotone in edit
magnitude (7.14 → −0.97, 7.18 → −1.19, 10.05 → −6.05).

So the −1.19 dB says **nothing about identity**; "actively harmful" was
right about the sign and wrong about the cause, and that reading is
withdrawn. **The payload negative is thereby STRENGTHENED**: damage
tracks displacement independent of correctness, so **there is no regime
in which redirecting opacity could help.** A correct link and a random
permutation of equal magnitude are indistinguishable, and both are
destructive.

**The mechanism claim (INFERENCE, not measurement):** LRV3's return is
identical in pose, colour and texture and is observed by 48 training
view-frames, so the recipient rows are wrong about nothing. Headroom is
therefore a question about **observation supply**, not about which tensor
is carried. LRV4 — an observation-starved variant with a 1-frame return —
is being built to TEST that claim, framed as a mechanism test rather than
a rescue.

**The falsification test was BUILT, RUN, and returned an INVALID
INSTRUMENT rather than an answer**
([[lrv4-starved-fixture-result-2026-08-23]], experiments 247/248). LRV4 —
LRV3 with a one-frame return, held-out return supply cut to exactly one
third — trained a healthy substrate (28.393 dB against LRV3's 28.59), and
its integrity check passed. But the screen found **ONE recipient row**,
so `row_sets_sufficient` is false and every recipient-side statistic is a
one-pair statistic. **The mechanism claim is UNTESTED — neither branch of
the frozen rule fires.**

**The near-miss is the durable part.** From that single pair the DC
headroom ratio reads **4.995** — above the frozen 2.0 floor and higher
than anything LRV3 produced. Read without its pair count it would have
been reported as a spectacular confirmation of the claim the fixture
exists to test. It was caught only by the sufficiency flag and by `pairs`
being carried next to every ratio. **A ratio without its n is not a
measurement.** No threshold was changed after seeing the null; the pure
diagnostic that would separate a threshold artifact from a substantive
finding is specified for a new frozen spec.

## 3. The development scene contains almost no usable events

Verified by ground-truth-only curation of all 300 `cut_roasted_beef`
cam00 frames ([[crb300-event-mask-curation-2026-08-23]]): **frames 0-299
contain essentially ONE large, clean occlude-and-return event on DYNAMIC
content** — a blade laid across the beef pile, occluded 158-187, revealed
190-209. Every other high-confidence event is **static background
revealed by the cook's body**. Reveal onsets are heavily back-loaded
(0-49: 2; 100-149: 10; 250-299: 143), and **segment 100-149 contains
none**.

This is the N3V analogue of the DiVa-360 supply problem, and it bears
directly on the open real-data event-supply question.

**Decision: the 300-frame endpoint is CLASS-SPLIT** —
`dynamic_content` primary, `static_background` a separate diagnostic —
because B1 relocates capacity into dynamic-mask regions and has no
mechanism to act on background. Pooling them would let a background
result masquerade as an event result.

**Correction recorded, retracting no number:** the frozen 0-49 dev masks
are **not confirmed** by ground truth. B's and C's `[34,39]` appears to
label the occluder-PRESENT window, and all three boxes score mostly
static pixels. The ladder's +0.077/+0.345 stands as measured; its reading
as an event-region effect is weakened.

## 4. The 300-frame B0-R vs B1 comparison is FROZEN and DEFERRED

Decided on arithmetic before any lane result was read
([[crb300-b0r-b1-spec-2026-08-23]]): four cells at 12k cost ≈21 of the
24 slot-hour ceiling and would displace the paper-blocking lanes.
**A 6k variant would have fit and was REJECTED** — both arms peak near
12k, so 6k measures a pre-peak transient rather than the endpoint in
question. **B0-C cannot substitute for B0-R**: it carries no reserved
parity and trained on all units.

The spec, the packet schedule and the 300-frame masks are frozen and
launch-ready, with an independent ground-truth mask re-verification
recorded as a precondition.

## 5. Flow as a B1 BIRTH prior: implemented, verified, screen RUNNING

Flow-gated SITE selection was **rejected on inspection** — birth sites
are already multiplied by the dynamic mask, so it would have been largely
redundant. The one-variable experiment with real information content is
flow-derived **velocity initialization**, since relocated rows are born
motionless.

**Verified empirically that the SEA-RAFT assets are FORWARD flow**
(forward warp error 0.0080 vs backward 0.0247, 47.5% better than no
warp). Nothing in the code could check this, and a backward asset would
have silently reversed every velocity in the correct arm while leaving
the control untouched.

Preflight ([[b1f-preflight-result-2026-08-23]], experiment 234) passed
both frozen mechanism conditions: 98.6% of sites got valid flow, and the
basis reproduced the requested displacement in full (ratio 1.0). Six
cells launched (237-242) with a wrong-flow camera-swap control, all on
one pool including a fresh plain-B1 comparator.

**RESULT: the prior is REJECTED**
([[b1f-flow-screen-result-2026-08-23]]). The frozen attribution rule
decides it: **B1-F did NOT beat B1-X** (paired mean −0.0952 on the event
union), so the result is unattributable regardless of the plain-B1
comparison. **Everything is noise** — every delta flips sign across seeds
and all magnitudes sit far inside the measured 0.341 dB event-union seed
spread. This is the **second** time in this project that camera-swapped
flow matched or beat correct flow, and the control was mandatory because
of that precedent. It **closes the last live zero-acquisition prior
experiment** with a terminal negative, which is exactly what justified
spending the cells on a pre-registered likely null.

**Measured en route: the first genuinely-different-seed spread this
project has** — 0.635 dB global, 0.341 dB event union. **No effect below
~0.64 dB is resolvable by two seeds at the 50-frame protocol**, which
bounds every future comparison in this family.

## 6. Two instrument defects found, both affecting how past results read

* **`--seed` never reached the trainer**
  ([[seed-threading-defect-2026-08-23]]). Every recorded ladder cell
  trained at `main.py`'s default 6666; the `_s0`/`_s1` suffixes
  distinguish run identities, not seeds. **No number is retracted**, but
  the "measured B0 SEED spread ±0.28 dB" is a run-to-run reproducibility
  spread at a fixed seed. It also exposes an unresolved tension with the
  recorded 3.3e-4 dB fixed-seed reproducibility — a ~800× gap between
  configurations. Actionable floor: **at this protocol with densification
  active, run-to-run variation is ≈0.27 dB and no smaller effect is
  resolvable by two runs.** Repaired going forward by passing
  `--extra-arg=--seed` explicitly.
* **An execution-closure gap**
  ([[block-2026-08-23-live-state-and-budget]] §9): `scripts/` is outside
  the closure set apart from `submit_apollo.py`, so a dirty entrypoint
  script does not block a submission that uses it. Not repaired this
  block; every entrypoint executed here was committed first.

## 7. Method-level negatives that outlive this block

From a fresh-context adversarial review that confirmed every number and
returned STANDS WITH QUALIFICATIONS:

* **A scale-free ratio screen is insufficient.** It cannot distinguish
  "1.92× of 0.03 logits" from "1.92× of 3 logits". Any future screen
  needs an absolute-magnitude floor.
* **A screen needs a non-degeneracy precondition.** `_t` passed the
  frozen rule while being **deletion, not transfer**: copying a donor's
  temporal centre makes the recipient satisfy the donor's membership
  predicate and leave the scored window. Decidable from the row-set
  definitions with no measurement at all.
* **A placebo does not transfer across payloads.** The same-identity
  no-op is a real edit for any non-appearance tensor, and it edits donor
  rows whose support ends before the scored frames, so it cannot
  attribute harm. **The discriminating control — a within-recipient
  permutation — was NOT run**, so "the payload carries nothing" and "any
  opacity reshuffle costs ~1 dB" remain equally consistent. It is
  specified and being implemented.
* **A guard justified by a synthetic measurement must have its scale
  checked against real data.** The flow magnitude guard was withdrawn on
  a "~328 vs ~1.6" figure; real N3V coefficients are 0.0518, so the
  original guard would probably never have bound. The withdrawal stays
  correct in principle and its stated urgency was overstated
  ([[b1f-preflight-result-2026-08-23]] §3).

## 8. Cost — MEASURED, and the accounting distinction matters

**Total ≈ 18.9 GPU slot-hours against the 24 slot-hour ceiling. Under
budget.**

**The distinction that makes that number honest:** summing Determined's
`endTime − startTime` across all 26 experiments gives **28.77 h**, which
would be OVER the ceiling — but that clock starts when an experiment is
CREATED, not when it is allocated a slot, so it charges QUEUE time as if
it were occupancy. A queued experiment holds no GPU. The `duration` field
that would give allocation time is `null` on this master, so slot
occupancy was derived from the training logs' own elapsed counters:

| cells | measured occupancy |
|---|---:|
| 6 flow training cells at 6,000 iters (237-240, 254, 255) | 2.47-2.53 h each, **14.99 h** |
| 2 crashed B1-X arms (241/242, died at iter 1000) | 0.328 + 0.327 = **0.66 h** |
| A-est ×2 + LRV4 substrate (245/246/247) | 0.871 + 0.872 + 0.701 = **2.44 h** |
| flow preflight (234) | **0.41 h** |
| 9 analysis/eval cells (233, 236, 243, 244, 248-253, 256-258) | **≈ 0.42 h** |

Two operational facts worth carrying:

* **Training cells cost 2.5 h, not the 1.9 h projected — a 32%
  underestimate.** Three cells sharing one hopper node contend for
  bandwidth; the preflight, running with less company, measured
  1.03-1.22 s/it against the screen's ~1.5 s/it. **Project per-cell cost
  from a contended node, not from a solo preflight.**
* **The roster defect cost ≈ 5.6 slot-hours** — 0.66 h burnt on the two
  crashed arms plus 4.95 h to rerun them. That is the price of a guard
  that degraded silently rather than at wiring time, and it is the
  concrete argument for the fail-closed repair.

The block's headline results were nearly free by comparison: the payload
screen, the opacity falsification, the permutation control and the LRV4
screen together cost **under a minute of GPU each**.
