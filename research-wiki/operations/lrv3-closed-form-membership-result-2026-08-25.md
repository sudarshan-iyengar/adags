# RESULT — per-primitive episodic membership is recoverable in CLOSED FORM, with zero learned parameters (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. Records the outcome of
[[lrv3-closed-form-membership-vote-v3-2026-08-25]] (the measurement) and
[[lrv3-mask-noise-and-shuffle-v4-2026-08-25]] (its control and sensitivity
sweep). Both specs were frozen before their runs and are preserved with their
implementation critiques.

## 1. THE RESULT

On the frozen LRV3 binding cloud, against exact per-row 3D oracle membership:

| instrument | precision | recall | parameters |
|---|---:|---:|---:|
| voxel-cell spatial partition (the recorded failure) | 0.0446 | 0.1786 | 0 |
| A2+B, the best prior instrument (oracle-derived) | 0.9375 | **0.8088** | 0 |
| **the closed-form vote** | **0.9659** | **0.9839** | **0** |
| standing gate | >= 0.80 | >= 0.90 | — |

`tp 10,479 / fp 370 / fn 171` against **10,650** true positives — a count that
matches the recorded expectation exactly, confirming the binding cloud is the
one the record describes.

**The prior best was capped at recall 0.8088, and that cap was STRUCTURAL.**
Cells 420 and 429 — **2,036 rows, 19.12% of the object** — carried
`abstain_reason: no_interior_gap`, **zero agreeing cameras, and no boundary
estimate at all**. Nothing downstream could manufacture one.

**Those exact cells now return at recall 0.9907, precision 0.9753**
(`tp 2,017 / fp 51 / fn 19`; cell 420 alone 0.9316/0.9811, cell 429
0.9938/0.9945).

**So the cap was never a property of the object. It was a property of the
estimator.** A recall mechanism keyed on *rendered contribution* reaches rows
that spatial partitioning provably could not. That was the stated bet of this
lane and it holds.

## 2. WHAT THE INSTRUMENT IS

The membership channel composites as `M_c = sum_i f_ic * alpha_i * T_i`, with
`f` the per-row `flow_2d` input — dynamic-only, no background term, VJP pinned
against an independent oracle. Binding `flow_2d` to a per-row leaf of ones
gives, for any upstream `g(pixel)`:

```
dL/df_i0  =  sum over pixels of  g(pixel) * alpha_i * T_i
```

so `g = 1` yields `w_total_i` and `g = mask_k` yields `w_in_mask_i(k)`.

**The vote:** `score_i(k) = w_in_mask_i(k)`; assign `argmax_k`. Eligible iff
`w_total_i > 0`; assigned iff the winner holds `>= tau = 0.50` of that row's own
weight; otherwise **abstain**.

Each primitive votes for the class it painted the most of. **There is nothing
to learn** — two accumulations and an argmax. Backprop is used only as an
efficient way to compute the sums.

**Abstention behaves as specified:** 25,703 rows abstained — **25,575 on zero
evidence** (fail-closed) and 128 below `tau` — with **0 ties**.

## 3. THE ATTRIBUTION CONTROL — C-SHUFFLE

Masks permuted across the 16 training cameras under seed 0, frame index fixed,
**derangement asserted (0 fixed points)**, everything else identical.

| | precision | recall |
|---|---:|---:|
| clean | 0.9659 | 0.9839 |
| **shuffled** | **0.0149** | **0.0015** |

`tp 16 / fp 1,061 / fn 10,634`. Precision collapses **65x**, and lands **below
chance** (0.071) — **4.8x worse than random**. That is the expected direction:
a camera handed a *different viewpoint's* mask does not merely lose signal, it
acquires actively wrong signal.

**Verdict: the 0.9659 is produced by reading mask CONTENT.** The v4 spec
recorded in advance that a high shuffled score would require withdrawing the
v3 result; it did not occur.

## 4. THE SENSITIVITY SWEEP

Four families on the training-view masks, instrument otherwise unchanged.
**Descriptive, not gated.**

| family | magnitude | precision | recall | gate |
|---|---:|---:|---:|---|
| clean | — | 0.9659 | 0.9839 | PASS |
| dilation | 1 / 2 / 4 / 8 px | 0.9620 / 0.9576 / 0.9530 / 0.9390 | **0.9842 at every k** | PASS |
| erosion | 1 | 0.9681 | 0.9787 | PASS |
| erosion | 2 | 0.9698 | 0.9626 | PASS |
| **erosion** | **4** | 0.9697 | **0.8995** | **FAIL** |
| erosion | 8 | 0.9708 | 0.7249 | FAIL |
| identity-switch | 0.05 / 0.10 / 0.25 | 0.9666 / 0.9669 / 0.9703 | 0.9836 / 0.9833 / 0.9809 | PASS |
| **identity-switch** | **0.50** | 0.9696 | **0.3445** | **FAIL** |
| missing-cameras | 1 / 2 / 4 | 0.9669 / 0.9669 / 0.9702 | 0.9722 / 0.9715 / 0.9402 | PASS |
| **missing-cameras** | **8 of 16** | 0.9784 | **0.6923** | **FAIL** |

**Degradation points:** erosion **k = 4**; identity-switch **0.50**;
missing-cameras **8 of 16**; dilation **"no crossing within the swept range"**.

### 4.1 The shape is the finding

**Precision is essentially immune; every failure is a RECALL failure.** Across
all 17 perturbed points precision stays in a **0.939 - 0.978** band and never
approaches the 0.80 floor.

The mechanism is direct: degrading a mask removes evidence for *object* blobs,
which then lose their vote and abstain — but it rarely persuades a *background*
blob that it is an object.

**Erosion hurts; dilation does not.** Dilation's recall is **pin-flat at
0.9842** from k=1 to k=8 — marginally above clean — while precision decays
gently. Growing a mask admits a few background blobs but never starves a real
one. Erosion starves them: 0.9787 -> 0.9626 -> 0.8995 -> 0.7249.

**Actionable consequence for the real-data lane: OVER-segmenting is safer than
under-segmenting.**

**The tolerance is wide.** The gate survives losing **a quarter of the
cameras** (4 of 16, recall 0.9402), **a quarter of object pixels mislabelled**
(0.9809), and **2 px of boundary erosion**. Only half the cameras gone, or half
the pixels mislabelled, breaks it.

## 5. PRECONDITIONS AND REPRODUCTION

**v4 reproduced v3 BIT-FOR-BIT** on a separate submission: precision, recall,
`tp`, `fp`, `fn` and `n_truth_positive` identical to the last recorded digit.

**Every precondition passed in v4** (`precondition_failures: []`), including
P1-P15 and N1-N4. Annihilation counts came out **exactly as predicted before
the run**: 1 camera emptied at erosion k=1 (`cam14`, 16 px), 2 at k=8
(`cam14`, `cam13` at 316 px).

**v3's two precondition failures and their disposition.** v3 failed
`P10_mask_partition_consistent` (measured **1.0693e-06** against a **1e-06**
tolerance) and `P11_backward_repeatable` (measured **1.5259e-05 = 2^-16**
against a **bitwise** requirement). Both are float32 artifacts of CUDA
`atomicAdd` order non-determinism — **~2e-7 relative to the median `w_total`
of 67.26**, and the ceiling is bit-identical between `e_min = 0` and `1e-6`.
They were re-declared in v4 on **PLATFORM grounds, not outcome grounds**
(P10 -> 1e-05, P11 -> 1e-04 absolute), with the original values, the
measurements and the reason recorded so the relaxation is auditable. **A
bitwise-equality requirement on a CUDA atomics path was a specification error
about the platform.**

**Cost:** 528 forward passes serve every point — `w_total` is mask-independent
and all five perturbations move only the supervision buffer. Backwards dedupe
by class-mask content digest: **24,255** flow backwards against 57,552
undeduplicated. **~2 slot-hours total across v3 and v4**, against a 24-hour
block ceiling, with **no training run**.

## 6. THE EVIDENCE BOUNDARY — what this is NOT

* **ORACLE-MASK CEILING.** Every point uses oracle ray-traced masks — in the
  sweep, oracle masks *degraded synthetically*. **A real segmenter fails in
  structured ways** — whole-object misses, temporal flicker, confusing the
  object with a shadow — **not by uniform erosion.** No number here is
  performance under real masks.
* **Oracle-supervised end to end.** Oracle 2D masks scored against the oracle's
  own 3D sphere test. The non-degeneracy limb is operationalized by *labelling*
  this an oracle-mask ceiling, not by a threshold.
* **Nothing transfers off LRV3**, whose absence is a clean ray-trace removal
  that real data will not supply.
* **The supervision `n` is 16 distinct masks, not 528 observations** — the 960
  buffers hold only 32 distinct images, 2 per camera, because the scene is
  static apart from the object's boolean presence.
* **A "missing-frame" robustness claim is VACUOUS on this fixture** and is
  forbidden; the axis measured is missing **cameras**.
* **Missing-cameras is implemented at MASK level**, which is *stronger* than
  absence: it supplies **background** evidence in the object's own image region
  rather than withholding evidence. A real segmenter failure mode, but not
  "camera unavailable".

## 7. WHAT THIS CHANGES ABOUT THE PLAN

**The learned Frozen-Scene Membership Learning phase is NOT AUTHORIZED, and
that is a result rather than a restriction.** The frozen ordering in v2 §6
requires the zero-parameter instrument to run first, on the AGENTS.md
principle *"if a simpler representation creates the same capability, prefer
it."* It cleared the gate. Training would spend hours of GPU reproducing what
a single pass already achieves.

**The bottleneck has MOVED.** Lifting 2D masks to per-primitive 3D membership
is measured *not* to be the limiting step. The live question is now **mask
acquisition on real footage**, where no oracle exists.

**The two halves of the mechanism now both have evidence on the fixture:**
episode *timing* was already exact (0 frames of error, zero false
activations), and episode *membership* is now 0.9659 / 0.9839. Membership was
the blocker.

## 8. PERMITTED AND FORBIDDEN

**Permitted.** To state that per-primitive episodic membership is recoverable
**in closed form, with zero learned parameters and explicit abstention**, from
alpha-composited training-view identity evidence, at P/R 0.9659/0.9839 on the
binding cloud — against 0.0446/0.1786 for the spatial-partition instrument
this project previously used, and breaking a structural 0.8088 recall cap. To
report the shuffle collapse as attribution. To report the four degradation
points. To state that over-segmenting is safer than under-segmenting.

**Forbidden.** To describe any of it as performance under real or estimated
masks. To claim a missing-frame result. To claim transfer off LRV3. To claim
**mechanism novelty for the membership channel** — per-primitive
time-independent identity, alpha-composited under 2D supervision, **is
Spacetime Gaussian Grouping's row of the complementary-halves table**. **The
unoccupied content is the per-primitive precision/recall MEASUREMENT, which no
method in this literature reports, and that is the only novelty these results
may carry.** To move v3's operating point, gate or accumulation domain
retrospectively. To read any sweep point other than the frozen operating point
as the score.

## 9. PROVENANCE

| item | value |
|---|---|
| binding cloud | `runs/elgs/20260820T002949Z_lrv3_a0_prime_0_b7952b0/chkpnt6000.pth`, 133,049,003 bytes |
| cloud fingerprint | `460c2736534c4d6d914e83571fd65115884e367fe723308e0fe298861cf3837f`, 149,794 rows |
| accumulation domain | present frames only — 33 frames x 16 training cameras |
| held-out cameras | 2 / 7 / 12 / 17, untouched |
| operating point | `e_min = 0` strict, `tau = 0.50`, frozen ahead of both runs |
| v3 spec commit | `4a0cf61` |
| v4 spec + implementation commit | `e950a04` |
| v4 task | `a291442a-5acb-45e5-9bea-21f5f679fa60`, `hopper`, 1 slot |
| v3 artifact | `runs/elgs/lrv3_vote_v3/membership_vote_v3.json`, 157,562 B |
| v4 artifact | `runs/elgs/membership_v4/membership_vote_v4_sweep.json`, 299,219 B |

**NOT RECORDED:** the v3 run's Determined task ID was not captured at
submission (it ran without `--detach`), so v3 is identified by its commit,
artifact and fingerprint rather than by a task handle. Both runs were `det cmd
run` cells and therefore carry **no experiment number**; the experiment ledger
remains at 294.
