# SPEC (FROZEN, v4) — C-SHUFFLE and the mask-noise sweep (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Frozen before either is run and
before any degraded score is read.** Extends
[[lrv3-closed-form-membership-vote-v3-2026-08-25]]; changes nothing in it.

Same narrowness discipline as v3: this authorizes **two measurements** on the
already-measured instrument, **no training**, and **no new gate**.

## 0. Why these two, now

v3's closed-form vote measured **P 0.9659 / R 0.9839** with zero parameters on
the binding cloud, breaking the structural 0.8088 recall cap (cells 420/429
recovered at R 0.9907). Two things follow immediately:

1. **The attribution is open.** The score is not yet known to come from the
   mask *content*. C-SHUFFLE closes that.
2. **The bottleneck has moved.** v3 used ORACLE ray-traced masks. Since
   lifting 2D masks to per-primitive 3D membership is now shown *not* to be the
   limiting step, the live question is **how good masks must be before lifting
   degrades** — which the same instrument can answer directly.

## 1. INPUTS MEASURED FIRST — the lesson from v1/v2

Per v3 §0's recorded rule (*measure the inputs a rule depends on before
freezing the rule*), the object's per-camera pixel scale was measured **before**
choosing any noise magnitude:

| quantity | measured |
|---|---|
| per-camera mask area, present frame | **16 - 8,201 px** |
| effective radius `sqrt(A/pi)` | **2.3 px (cam14) - 51.1 px (cam01)** |
| area retained at erosion k=1 | 96-97% on the 14 healthy cameras |
| area retained at erosion k=8 | **60-75%** on those 14 |
| cameras ANNIHILATED at k=1 | **1** (`cam14`, 16 px) |
| cameras ANNIHILATED at k=8 | **2** (`cam14`, `cam13` at 316 px) |

**The magnitudes in §3 are chosen from this table**, so the sweep spans "barely
perceptible" to "the two weakest cameras are destroyed" — and the annihilation
points are known in advance rather than discovered as a surprise.

## 2. C-SHUFFLE — the attribution control

**Operation, frozen:** permute the mask assignment across the **16 training
cameras**, holding the frame index fixed, under a declared seed
(`seed = 0`). Camera *i* receives camera *perm(i)*'s mask for the same frame.
The permutation must be a derangement (no camera keeps its own mask) and this
is asserted.

Everything else — the cloud, the operating point (`e_min = 0`, `tau = 0.50`),
the accumulation domain (present frames only), the scoring oracle — is
**identical to v3**.

**Reading rule, frozen:** precision must fall **below 0.30**. Chance precision
on this cloud is **0.071**.

**An EMPTY shuffled selection is a PASS**, declared here and inherited from
v3 §6: on a 20-camera surround ring a camera handed another camera's mask has
near-zero row weight in that image region, so an empty selection is the correct
fail-closed behaviour and not an undefined score. The report must state which
of the two outcomes occurred.

**What a FAILURE would mean, stated before the run:** if the shuffled vote also
scores well, then v3's 0.9659 was **not** produced by reading mask content, and
v3's result must be withdrawn pending a mechanism explanation. That is the
whole point of running it.

## 3. THE MASK-NOISE SWEEP — descriptive, not gated

Four noise families, each applied to the **training-view masks only**, with the
instrument otherwise unchanged. **This is a sensitivity curve, not a gate**;
no pass/fail threshold attaches to any single point.

| family | magnitudes | grounded in |
|---|---|---|
| **erosion** | k = 1, 2, 4, 8 px | §1: 3%-40% area loss; annihilates `cam14` at k=1, `cam13` at k=8 |
| **dilation** | k = 1, 2, 4, 8 px | symmetric counterpart |
| **missing cameras** | drop 1, 2, 4, 8 of 16 | see the warning below |
| **identity switch** | relabel 5%, 10%, 25%, 50% of object pixels to the nearest non-object class | spans "annoying" to "half the object mislabelled" |

**WARNING, recorded because it is easy to get wrong:** the 960 buffers contain
only **32 distinct images — exactly 2 per camera** (event-present and
event-absent), because the scene is static apart from the object's boolean
presence. **Therefore "missing frames" is not a meaningful axis on this
fixture — dropping frames within a camera drops duplicates.** The axis is
**missing CAMERAS**, and it must be reported under that name. A "missing-frame"
robustness claim from this fixture would be vacuous.

**Frozen reading rule.** Report per-row precision and recall at every point,
against the clean v3 reference (**0.9659 / 0.9839**). Report the
**DEGRADATION POINT** for each family: the smallest magnitude at which the
standing gate (P >= 0.80 **and** R >= 0.90) no longer holds. If a family never
crosses within its swept range, report that plainly — "no crossing within the
swept range" — and do **not** extend the range to find one.

## 4. PRECONDITIONS

* **N1 — the noise actually bit.** At every magnitude, assert the perturbed
  mask set differs measurably from the clean one, and report the realized
  per-camera area change. A noise level that changed nothing must FAIL, not
  silently produce the clean score.
* **N2 — annihilation is reported, not hidden.** At each erosion level, report
  how many cameras were reduced to an empty mask. Known in advance: 1 at k=1,
  2 at k=8.
* **N3 — the derangement holds** for C-SHUFFLE (no camera keeps its own mask).
* **N4 — v3's preconditions still hold** at every point: fingerprint matches,
  cameras disjoint from `test_cameras`, flow leaf bound on every view,
  topology invariant.
* **P10/P11 tolerances.** v3's run failed `P10_mask_partition_consistent`
  (measured rel. deviation **1.0693e-06** against a **1e-06** tolerance) and
  `P11_backward_repeatable` (measured **1.5259e-05 = 2^-16**, against a
  **bitwise** requirement). **Both are float32 artifacts of CUDA `atomicAdd`
  order non-determinism, not mechanism failures**, and the discrepancy is ~2e-7
  relative to the median `w_total` of **67.26**. They are **re-declared here on
  PLATFORM grounds, not on outcome grounds**: `P10` tolerance **1e-05**, `P11`
  tolerance **1e-04 absolute** rather than bitwise. The original values and the
  reason for the change are recorded so the relaxation is auditable and is
  never mistaken for a response to an unfavourable score.

## 5. Permitted and forbidden

**Permitted.** To report the shuffle outcome and the four sensitivity curves
with their degradation points. To conclude that mask quality above a measured
level does not limit the lifting step, if that is what the curves show.

**Forbidden.** To extend a sweep range in order to find a crossing. To report
a "missing-frame" robustness result from this fixture (§3). To describe any
result here as performance under *real* masks — every point still uses
**oracle ray-traced masks, degraded synthetically**, which is not the same as
masks from a real segmenter with real failure modes. To claim anything
transfers off LRV3. To move v3's operating point, gate, or accumulation
domain.

---

## IMPLEMENTATION CRITIQUE (2026-08-25, append-only) — including a factual error in §1

Implemented as frozen; disagreements reported rather than acted on. All
accepted.

### C1 (FACTUAL ERROR IN THIS SPEC, ACCEPTED) — §1's retention ranges do not reproduce

| §1 claimed | measured over the 14 healthy cameras |
|---|---|
| k=1 retention **96-97%** | **93.6% - 96.5%** (cam10 0.9358, cam11 0.9366) |
| k=8 retention **60-75%** | **53.5% - 73.4%** (cam10 0.5355, cam11 0.5393) |

**Cause, and it is the third instance of the same mistake in this block:** the
ranges were read off the LARGE cameras and generalized to all sixteen. The
same error produced v1's "every training camera contributing 4,603-8,201 px"
(true range 16 - 8,201) and is recorded there too.

**The load-bearing part of §1 is exactly right** — the annihilation counts, 1
camera at k=1 and 2 at k=8, reproduce precisely, and nothing in the sweep
depends on the retention ranges, which are descriptive framing only.

**Carry as method, again:** a range quoted over a population must be computed
over the population, not read off its head. Three occurrences in one block.

### C2 (ACCEPTED) — §3 never said whether "missing cameras" drops the VIEW or the MASK

Implemented at **mask level**: the camera's segmenter returns no object, and
its object pixels take the nearest non-object class. Rationale accepted: §3
applies every family "to the training-view masks only, with the instrument
otherwise unchanged", and dropping the view would change `w_total` and make
the point **incomparable to the reference it is read against**.

**But this is STRONGER than absence and must be reported as such:** it does not
merely withhold evidence, it actively supplies *background* evidence in the
object's image region. That is a real segmenter failure mode; it is **not**
"camera unavailable". The report says so.

### C3-C4 (ACCEPTED) — undeclared seeds and structuring element

§3 wrote "(declared seed)" twice and declared none; seed **0** is now declared,
matching §2's, with nested drop sets and nested switch draws. §3 declared no
structuring element; **4-connectivity iterated k times** (the L1 ball) is used
— and the only evidence it is the intended convention is that §1's
annihilation counts fall out exactly.

### C5 (ACCEPTED) — §3 is internally in tension

§3 says "a sensitivity curve, not a gate; no pass/fail threshold attaches to
any single point", then defines the degradation point **by** the standing gate.
Resolved descriptively: `clears_standing_gate` is reported per point and **no
sweep point can fail the run**; only N1-N4 can.

### C6 (ACCEPTED) — §2's two outcomes are different predicates

"Precision below 0.30" and "an empty selection is a PASS" are not the same
test: an empty selection has **undefined** precision, not low precision.
Reported as three distinct outcomes: `EMPTY_SELECTION_PASS`, `PASS`, `FAIL`.

### C7 (ACCEPTED) — C-SHUFFLE must swap the WHOLE identity buffer

Permuting only class 100 would leave pixels doubly labelled or unlabelled and
break P10's partition identity, which the instrument would then read as a
**mechanism failure**. Whole-buffer swap is forced and declared.

### C8 — N2 counts at the undeclared magnitudes, for the record

k=2 → 1 camera annihilated (cam14); k=4 → 1 (cam13 still holds 35 px).
Reported, not gated, because §4 declared nothing there.

### C9 — a guard added beyond the spec, flagged as such

For a PARTIAL run the degradation point reports *"incomplete sweep: only N of
4 frozen magnitudes measured"* rather than the literal no-crossing string,
because **a partial range cannot support the claim that the range contains no
crossing**. The full sweep emits the verbatim literal, unaffected.

### C10 — identity-switch is Bernoulli, not a quota

Per-pixel Bernoulli, so on cam14 (16 px) the realized relabel counts are
0/0/1/5 at 5/10/25/50% rather than 1/2/4/8. Nested and reported per camera.

### COST, and what remains unexecuted

One render pass serves **every** point — `w_total` is mask-independent and all
five perturbations move only the supervision buffer. Backwards are deduplicated
by class-mask content digest: **44.9 distinct class masks per view** against a
naive 108, giving ~**24,255** flow backwards for the 18-point sweep versus
57,552 undeduplicated and 3,696 for v3. So **~6.6x v3's backward cost at 1x its
forward cost**; budgeted **1.5 slot-hours**.

**The entire torch/CUDA limb is unexecuted.** 244 pytest cases and 263
self-test checks cover the pure-numpy components against the real
`train_identity/` buffers, but the first Apollo run is **the first execution of
the changed GPU path**.
