# SPEC (FROZEN, v3) — the zero-parameter closed-form membership vote, and NOTHING else (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Frozen before the instrument is run
on the binding cloud and before any per-row score is read.**

## 0. Why this page is deliberately SHORT

[[frozen-scene-membership-spec-2026-08-25]] (v1) and
[[frozen-scene-membership-spec-v2-2026-08-25]] (v2) both returned **MATERIAL
DEFECT**. Both are preserved with their reviews. The diagnosis of an
independent review is that each tried to freeze the whole phase — vote plus
training plus three controls plus two gates — in a single sitting, and that
**every defect found was a fact about the data or the code that was
measurable in minutes beforehand.**

The project's carried rule is *"every frozen rule needs a frozen precondition
asserting the mechanism it reads was actually exercised."* Its missing sibling,
recorded here: **measure the inputs a rule depends on BEFORE freezing the
rule.** A spec frozen on unmeasured inputs is not protecting against post-hoc
tuning; it is guaranteeing a rewrite.

**This page therefore authorizes exactly ONE measurement.** It authorizes no
training, no joint refinement, no episodic gating, and only one control.

## 1. THE MEASUREMENT

One forward pass and a small number of backward passes on the frozen binding
cloud. **No optimizer. No learned parameter. Topology frozen.**

The membership channel composites as `M_c = sum_i f_ic * alpha_i * T_i`, where
`f` is the per-row `flow_2d` input — dynamic-only, no background term, VJP
pinned against `tests/ref_impls/flow_compositing_reference.py`. Binding
`flow_2d` to a per-row leaf of ones gives, for any upstream `g(pixel)`:

```
dL/df_i0 = sum over pixels of  g(pixel) * alpha_i * T_i
```

so `g = 1` yields `w_total_i` and `g = mask_k` yields `w_in_mask_i(k)`.

**The vote:** `score_i(k) = w_in_mask_i(k)`; assign `argmax_k`.

**Reported at BOTH class structures** (accepted findings N-1/N-9 of v2's
review): multi-class over the identity buffer's own classes, **and** the
`K = 2` structure a trained head would be limited to by the CUDA carrier. They
are different instruments and must not be compared across structures.

## 2. THE BINDING CLOUD — pinned by HASH, not by path

```
runs/elgs/20260820T002949Z_lrv3_a0_prime_0_b7952b0/chkpnt6000.pth
133,049,003 bytes   (verified present 2026-08-25)
cloud_fingerprint  460c2736534c4d6d914e83571fd65115884e367fe723308e0fe298861cf3837f
n_rows             149,794
```

Both recorded in `configs/lrv3/estimated_program_v2.json`. **A differing
fingerprint REFUSES.** The in-sphere count of **10,650** is an
**expectation measured on experiment 184's PLY, not an identity established on
this checkpoint**; the run records what it actually measures and a mismatch is
a fail-closed stop, not a silently different denominator.

## 3. THE OPERATING POINT — unchanged from v2, and oracle-blind

* `e_min = 0` **strict**: eligible iff `w_total_i > 0`; otherwise **abstain**.
* `tau = 0.50`: assigned iff `max_k w_in_mask_i(k) >= 0.50 * w_total_i`;
  otherwise **abstain**.

Both are declared judgments using only per-row quantities, so no oracle and no
held-out view enters them. The frozen `tau` and `e_min` grids are reported as
**CEILING information only, never as the score**.

## 4. THE ACCUMULATION DOMAIN — the one genuinely open choice, frozen here

v2 never stated over which `(camera, frame)` pairs the sums run, and the choice
is decisive (finding N-2).

**FROZEN: present frames only** — the 33 supervised frames
(`episode_1 [0,29]` + `return [57,59]`) x the 16 training cameras.

**Reason, stated before the run:** the binding cloud is the **ungated**
A0-prime substrate, which renders event rows *through the gap*. Including gap
frames would accumulate background weight over 27 gap frames against 33
present frames and could flip the argmax **for a reason about substrate
ghosting, not about membership.**

The gap-included variant is reported as a **labelled sensitivity**, never as
the score.

## 5. THE GATE — all three limbs, unchanged

1. **per-row precision >= 0.80**
2. **per-row recall >= 0.90**
3. **non-degeneracy** — must not reduce to the oracle sphere test; must not
   consume held-out cameras 2/7/12/17.

Chance precision on this cloud is **0.071** (10,650 of 149,794).

**Limb 3, operationalized** (v2 finding N-7, previously restored in words
only): this phase **is** oracle-supervised end to end — oracle ray-traced 2D
masks scored against the oracle's own 3D sphere test. Therefore **every result
from this page is labelled an ORACLE-MASK CEILING**, and it may not be
reported as performance under real or estimated masks. That labelling is the
operationalization; there is no threshold attached to it.

## 6. THE ONE CONTROL

**C-SHUFFLE only.** The shuffle operation is defined here as: **permute the
mask assignment across TRAINING CAMERAS**, holding frames fixed, with a
declared seed. Precision must fall **below 0.30**.

**An EMPTY shuffled selection is a PASS**, declared here (v2 finding N-5): on
a 20-camera surround ring a camera handed another camera's mask has near-zero
row weight in that region, so an empty selection is the correct fail-closed
behaviour, not an undefined score.

**Deferred, with reasons already measured.** *C-REMOVE* is **arithmetically
forced** to recall 0 for a vote (removing the event class makes its score 0 for
every row, so the argmax can never select it) and cannot return an unfavourable
result — v2 finding N-4. *C-STATIC* **cannot discriminate its two hypotheses**
at the default sphere, which carries 2.58x the event's pixel supply and 373x
its worst camera's, over 60 frames against 33 — v2 finding N-3. Neither is run
here; both need their own spec.

## 7. PRECONDITIONS

* fingerprint matches; **REFUSE** otherwise.
* camera set disjoint from `test_cameras` (2, 7, 12, 17); an empty or absent
  test roster **REFUSES** rather than passing vacuously.
* `enable_soft_routing = False` **REFUSES**.
* **the flow leaf is bound on every view** — `flow_2d` is NOT an argument to
  `render()`; it is constructed inside, and `enable_rendered_flow: false` in
  the binding config, so an unbound run would silently return nothing
  (v2 finding N-10).
* the render ran: nonzero rows carry nonzero `w_total`.
* **per-camera** mask supply reported for all 16 cameras. Measured and known:
  range **16 - 8,201 px**, **8 of 16 below 4,603**, worst `cam14 = 16 px`.
* the 960 buffers contain **32 distinct images** (2 per camera) — the effective
  supervision is **16 distinct masks, not 528 observations**.
* topology invariant across the pass.

## 8. Cost

**Well under 0.5 slot-hour**, one `hopper` cell, exploratory tier.
528 renders (16 cameras x 33 frames) at 400x300 over 149,794 rows, topology
frozen, no optimizer. Against 24 slot-hours available and **0 spent on science
this block**.

## 9. Permitted and forbidden

**Permitted.** To report per-row precision/recall at the frozen operating
point, at both class structures, with the ceiling curves and the per-cell
breakdown including cells **420 and 429** (the 2,036 rows = 19.12% of the
object that produced the structural 0.8088 recall cap). To report a negative
as a construction-level result.

**Forbidden.** To move any threshold after reading a score. To report a sweep
point as the score. To compare the two class structures against each other. To
describe the masks as anything but ORACLE ray-traced, or the result as anything
but an **oracle-mask ceiling**. To claim mechanism novelty — per-primitive
time-independent identity alpha-composited under 2D supervision **is Spacetime
Gaussian Grouping's row of the complementary-halves table**; the unoccupied
content is the **per-primitive precision/recall MEASUREMENT**, which no method
in this literature reports. To authorize training from this page. To claim
anything transfers off LRV3.
