# SPEC (FROZEN) — Frozen-Scene Membership Learning on LRV3 (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Frozen BEFORE any membership
parameter exists, before any training cell is submitted, and before any
per-row score is read.** Written now precisely so that no threshold,
carrier, or reading rule can be chosen after seeing which one flatters the
result.

Authority: the 2026-08-25 block directive, Stage 1. Inherits the binding
constraints of [[lrv3-membership-diagnostic-2026-08-23]],
[[lrv3-membership-gate-spec-2026-08-23]],
[[lrv3-membership-candidates-result-2026-08-23]],
[[lrv3-fixture-hazards-2026-08-23]], [[nonoracle-timing-t2-result-2026-08-23]],
[[membership-occupancy-and-decision-2026-08-23c]] and
[[sa4d-read-2026-08-24]].

## 0. What this phase is, and what it is NOT

It is a **scientific isolator**: geometry, appearance, motion, opacity and
temporal support stay frozen while membership alone learns from
training-view 2D masks. It is also the **initialization** for a later joint
phase.

**It is NOT the paper method, and it is NOT a claim that learned real-scene
membership works.** It is a phase-gated attempt to find out whether a
per-primitive membership field, supervised only through the alpha-compositing
path, can clear a bar that **every spatial-partition instrument tried so far
has failed** — see §1.

## 1. Why this is worth running at all — the record, stated honestly

Every membership instrument scored on LRV3 to date has failed the frozen
gate, and the record is specific about **which limb** failed:

| instrument | cloud | precision | recall | verdict |
|---|---|---:|---:|---|
| 2 voxel cells (what actually ran) | fresh 50k | 0.0446 | 0.1786 | the failure |
| A1 single-pass | fresh 50k | 0.0667 | 0.7143 | FAIL both |
| A2 transitive | fresh 50k | 0.0667 | 0.8810 | FAIL both |
| **B (`row_ids`)** | **trained** | **0.9688** | 0.3298 | FAIL recall |
| **A2 + B** | **trained** | **0.9375** | **0.8088** | FAIL recall |
| all 8 cells + B *(oracle-derived)* | trained | 0.9400 | 1.0000 | not an instrument |

Two facts decide the design:

1. **Precision is a CLOUD-BINDING problem and it is already solved.** Bound
   to the trained cloud, precision reaches 0.9688. Bound to the fresh
   seeding cloud, the *same cells* are 95.5% background. This phase therefore
   binds to a **trained, frozen, fingerprinted cloud** and to nothing else.
2. **Recall is an ESTIMATOR-SENSITIVITY problem and it is unsolved.** The
   A2+B ceiling of 0.8088 is **structural**: two object cells (420, 429)
   carry `abstain_reason: no_interior_gap`, **zero** agreeing cameras, and
   **no boundary estimate at all**. Nothing downstream can manufacture one.

**The scientific bet of this phase, stated as a bet:** a per-row field
supervised by 2D masks across 16 training cameras x 33 supervised frames has
a *different* recall mechanism from spatial partitioning. It can reach any row
that carries rendered contribution, regardless of whether that row's
neighbourhood produces an absence signal. Whether that is enough to cross
0.90 is exactly what is being measured.

**This is not a prediction that it will pass.** Per
[[membership-occupancy-and-decision-2026-08-23c]] §4 the forbidden claim
"non-oracle membership is impossible" still STANDS, and so does its converse:
nothing here licenses assuming it is possible.

## 2. THE BINDING CLOUD — named, hashed, and singular

The phase binds to **exactly one** cloud:

```
/apollo/users/sri/proj_adags/runs/elgs/
    20260820T002949Z_lrv3_a0_prime_0_b7952b0/chkpnt6000.pth
```

the A0-prime (ungated) LRV3 substrate, **149,794 rows**, 133,049,003 bytes.

* Its `cloud_fingerprint` (sha256 over the float32 xyz bytes,
  `elgs/trainer_hooks.py:701-704`) is computed and recorded **before**
  anything else, and every artifact of this phase carries it.
* **Any run whose loaded fingerprint differs from the recorded one is
  REFUSED, not adapted.** This is the single guard that stops the 0.9688 →
  0.0446 collapse recorded in §1.
* The oracle is the sphere test on **this cloud's** xyz:
  `(xyz - centre).norm(dim=1) <= radius`, with
  `centre = (0.70, 0.10, 0.35)`, `radius = 0.20` read from
  `data/synthetic/lrv3/event_spec.json`, never hardcoded. The recorded
  in-sphere count on this cloud is **10,650**; a run measuring a different
  count has loaded a different cloud and must refuse.

## 3. SUPERVISION — training views only, and they now exist

**A gap in the fixture was found and closed before this spec was written.**
`data/synthetic/lrv3/gt_identity/` contains **240** buffers = 4 held-out
cameras x 60 frames, and the generator comments the restriction explicitly
("held-out views only: reveal-mask source"). **LRV3 shipped no training-view
masks at all**, so the phase as directed could not have run.

`scripts/emit_training_identity.py` now emits **960** buffers = the 16
**training** cameras x 60 frames into a **separate** directory
`data/synthetic/lrv3/train_identity/`.

* Separate directory, deliberately: `gt_identity/` keeps its held-out-only
  meaning and its existing leakage guard (`scripts/estimate_episodes.py:212`)
  un-weakened.
* The emission is **additive**: `identity` was already computed for every
  camera and used for `visible_px`; only the `np.save` was gated. No RGB, no
  `points3d.ply`, no transforms, and no existing byte changed.
* Its precondition is a byte-identity re-render of held-out buffers, run
  **before** any emission and stamped into the manifest. It passed on 6
  pairs spanning both the event-present and event-absent branches.
* **A real defect it caught:** the generator's `render()` reads *mutable
  module globals* whose import-time defaults are **LRV1's** —
  `GROUND_HALF_EXTENT` 3.0 against LRV3's 1.3, and
  `DEFAULT_FIRST_RETURN_FRAME` 54 against LRV3's 57. A naive import would
  have emitted 960 plausible-looking buffers wrong about the ground plane
  everywhere and wrong about frames 54-56, silently.

**These masks are ORACLE ray-traced.** They are not curated, not estimated,
and not noisy. Every artifact says so. Mask-noise sensitivity is a **separate
later ablation** and may not be claimed from this phase.

**Measured supply** (16 training cameras, per frame): **71,625** event-object
pixels in episode 1 and at the return, **exactly 0** across the entire gap,
every training camera contributing 4,603-8,201 px. Supervision is abundant
and the absence is clean.

**Held-out cameras 2, 7, 12, 17 are untouched by every step of this phase** —
supervision, calibration, abstention, thresholds, and model selection.

## 4. WHAT LEARNS, AND WHAT DOES NOT

**Learns — exactly one new per-row tensor:**
`_membership_logits`, shape `(N, K)`, an `nn.Parameter`, one Adam group with
its own LR.

For LRV3, `K = 2`: `{event_object, other}`. **Abstention is NOT a learned
class** — see §5. Adding an "unknown" logit would let the optimizer discover
that abstaining everywhere minimizes a mask loss, which is a fail-open
disguised as calibration.

**Frozen — everything else**, asserted at runtime, not assumed:
`_xyz`, `_features_dc`, `_features_rest`, `_scaling`, `_rotation`,
`_opacity`, `_t`, `_scaling_t`, `_rotation_r`, `_route_logit`, `_motion_*`,
every scaffold tensor, and `logit_a`/`logit_b`.

**Verification that they are frozen is by MEASUREMENT, not by
`requires_grad`:** `non_pointer_state_hash`-style hashing of every frozen
tensor before and after, asserted equal. `requires_grad_(False)` is also set,
but the hash is the check that is reported, because
[[lrv3-fixture-hazards-2026-08-23]] §4 records that the existing hash had
blind spots and an invariant is only as good as its coverage. **The hash
used here must enumerate its covered tensors explicitly and assert that the
enumeration equals the model's full per-row tensor list**, so a newly added
tensor cannot silently escape it.

**Topology is FROZEN for this phase.** No densification, no cloning, no
splitting, no pruning. `N` is invariant at 149,794 and this is asserted every
iteration. Rationale: membership propagation through topology change is the
recorded amplification hazard (a 4.5%-precision seed became 4,317/4,576 gated
rows) and it belongs to the *joint* phase, under its own audit.

## 5. THE MEMBERSHIP CHANNEL, AND THE ABSTENTION RULE

### 5.1 Rendering

The membership vector is alpha-composited **through the actual render path**,
in the same pass as RGB, so that the weight a row receives is exactly the
weight it has in the image:
`M(pixel) = sum_i m_i * alpha_i * T_i`.

**Carrier decision is DEFERRED to a measurement, and the measurement is a
precondition.** Two carriers exist and neither is free:

* `flow_2d` — 2 channels, same pass, VJP repaired and pinned against an
  independent oracle (`tests/ref_impls/flow_compositing_reference.py`), and
  **no background term**, so residual transmittance is already unassigned
  mass. `K = 2` fits it exactly.
* `override_color` — 3 channels, but it *replaces* RGB and so costs a second
  pass; zero callers today.

**Both feed the DYNAMIC branch only.** `forward.cu:754` guards on
`collected_id[j] < P`; the static twin still consumes shared transmittance
`T` but writes zero to the channel. A gated pixel would then read as
"unknown" for the wrong reason.

**PRECONDITION C1 (about the setup, never the score):** measure the static
twin's share of accumulated alpha over the supervised views on this exact
cloud. LRV3 trains at route logit 4.0 (`p_dyn ~ 0.982`), so the expected
share is ~1.8%, **but it is measured, not assumed**. If the measured share
exceeds **5%** of accumulated alpha on in-sphere rows, the 2-channel
dynamic-only carrier is declared INVALID for this fixture and the phase stops
for a carrier redesign. **That number is a declared judgment**, chosen before
measurement, and labelled as one.

### 5.2 Loss

Cross-entropy between the composited membership map and the training-view
oracle mask, over supervised pixels only. `lambda_membership` is the only new
loss weight. The RGB loss is **absent** in this phase — nothing photometric
is trainable, so including it would compute a gradient with nowhere to go.

### 5.3 Abstention — a post-hoc calibrated decision, not a learned class

Per row, after training:

* **evidence mass** `e_i` = accumulated `alpha_i * T_i` over all supervised
  (camera, frame) pairs. A *measured* quantity, not a parameter.
* **posterior margin** `g_i` = softmax margin between the top two classes.

A row is **assigned** iff `e_i >= e_min` AND `g_i >= g_min`; otherwise it
**abstains** and takes `family_id = -1`, the existing ungated path
(`elgs/trainer_hooks.py:688-689`), which keeps the ordinary temporal marginal
bit-for-bit.

**`e_min` and `g_min` are NOT frozen as numbers here, and that is
deliberate.** Freezing an uncalibrated threshold is precisely the failure
[[membership-occupancy-and-decision-2026-08-23c]] §3 records across this
literature ("every shipped decision rule ... is an uncalibrated threshold
that fails either silently closed ... or catastrophically open"). Instead:

**FROZEN RULE R1.** The phase reports the **full precision/recall curve over
a declared sweep** of `(e_min, g_min)`, computed on the binding cloud against
the oracle. The gate of §6 is evaluated at the **single operating point
selected by a rule fixed here**: the point maximizing recall subject to
precision >= 0.80. If no swept point reaches precision 0.80, the phase FAILS
— it does not lower the precision floor.

**FROZEN RULE R2 — fail closed, always.** If the supervised evidence is empty
or unsupported, the selection must be EMPTY, never the whole cloud. The
inverted-mask pathology named in the record (`if count_nonzero(mask) == 0:
mask = ~mask`) is forbidden, and a test asserts its absence by neutering the
evidence and requiring an empty selection.

## 6. THE GATE — inherited unweakened

Evaluated on the **exact binding cloud of §2**, against the sphere oracle:

* **per-row precision >= 0.80**
* **per-row recall >= 0.90**
* **held-out cameras 2/7/12/17 untouched**
* **no fail-open selection when evidence is empty or unsupported**

Rendered 2D mask IoU is **supplementary and cannot substitute** for these.
This is the same gate frozen at `2789fef` and it is **not re-derived, not
re-argued, and not moved**. Its recall floor remains a declared judgment
with no intermediate-recall measurement behind it; that limitation is
inherited along with the number.

**If either floor is missed the phase STOPS before joint refinement**, and
the failure is reported broken down by visibility, segment, support width,
opacity/contribution, and boundary region.

## 7. PRECONDITIONS — all frozen, all about the SETUP

The block's own standing rule is that a frozen reading rule without a frozen
precondition is how an instrument delivers a clean null it never earned.
Each of these must PASS before any score is read, and each FAILS LOUDLY.

* **P0 — SUPERVISABILITY CEILING, and it runs FIRST, before any training.**
  For the membership channel, `dL/dm_i` is proportional to
  `sum_pixels alpha_i * T_i`. For the RGB path, `d(colour)/d(c_i)` is the
  **same** compositing weight. So backpropagating `image.sum()` into
  `_features_dc` yields, per row, a quantity proportional to exactly the
  weight the membership channel would see — **not a proxy, the same weight**.
  Measure it over the 16 training cameras and report
  `achievable_recall_ceiling = |in_sphere AND supervisable| / |in_sphere|`.
  **If that ceiling is below 0.90 the gate is unreachable on this substrate
  BY CONSTRUCTION**, and the phase stops with that as its result, having
  spent no training compute. This is a one-sided decisive test: a row with
  zero accumulated weight can never receive membership gradient.
* **P1 — the mechanism was exercised.** `_membership_logits` receives nonzero
  gradient and its values change measurably from initialization.
* **P2 — the frozen tensors did not move.** Enumerated hash equal before and
  after, with the enumeration asserted complete (§4).
* **P3 — topology invariant.** `N == 149,794` at every iteration.
* **P4 — correct masks beat shuffled masks.** A shuffled-mask negative control
  is run and must score materially worse. Frozen before results.
* **P5 — empty/unsupported masks fail closed** (rule R2), tested by neutering.
* **P6 — removing the positive signal breaks the gate.** Deleting the event
  object's mask supervision must make the membership gate FAIL, not degrade
  gracefully.
* **P7 — the object carries rendered contribution** in the supervised cameras
  and segments. Already measured: 71,625 px/frame, 0 in the gap.
* **C1 — the static-twin share** (§5.1), bound 5%.

## 8. Cost, and what is NOT spent

All of §7's P0 and C1 are **CPU/GPU-light diagnostics that precede
training**, consistent with the standing rule that an instrument is scored on
the exact cloud to which it would bind, **before** reconstruction training,
and that **no retraining is authorized for an instrument that fails its
gate**.

Training in this phase optimizes **one tensor of shape (149794, 2)** with
everything else frozen and topology fixed. It is far cheaper than a substrate
run and must be costed and reported in slot-hours like any other cell, under
the block's 24 slot-hour ceiling and the 6,000-iteration default.

**No joint refinement, no episodic gating, and no real-scene cell is
authorized by this spec.** Each has its own gate and its own page.

## 9. Permitted and forbidden

**Permitted.** To report the P0 ceiling as a result in its own right,
including a negative one. To report per-row precision/recall on the binding
cloud with the operating point selected by R1. To report that the phase
failed and why, broken down as §6 requires.

**Forbidden.** To move any floor in §6 after a score is read. To evaluate the
gate on any cloud other than §2's. To describe the training-view masks as
anything other than ORACLE ray-traced. To read held-out cameras 2/7/12/17 for
any purpose in this phase. To substitute rendered 2D mask IoU for per-row
precision/recall. To claim mask-noise robustness from this phase. To claim
anything transfers off LRV3, whose absence is a clean ray-trace removal that
real data will not supply. To authorize joint refinement because a rendered
mask looks good.

---

## CORRECTION (2026-08-25, append-only) — P0 is one-sided in the PASS direction ONLY, and my original wording had the direction wrong

Nothing above is rewritten. §7 P0 as written says the supervisability
ceiling is "a one-sided decisive test". **That is correct for a PASS and
WRONG for a FAIL**, and the difference is exactly the direction that a STOP
decision depends on. Found by an independent implementation review, then
verified by the primary against the CUDA source.

### The mechanism

The rasterizer clamps SH-evaluated RGB at zero and RECORDS the clamp so the
backward pass can kill it:

```
forward.cu:67-70    clamped[3*idx + c] = (result.c < 0);
                    return glm::max(result, 0.0f);
backward.cu:32-34   dL_dRGB.x *= clamped[3*idx + 0] ? 0 : 1;   (and .y, .z)
backward.cu:159-161 the same three lines on the 4D limb
```

So a row whose evaluated colour is negative in **all three** channels has
`dL_d(_features_dc) == 0` even when its compositing weight `sum alpha_i T_i`
is strictly positive. The instrument would read it as unsupervisable when it
is in fact supervisable.

### The corrected logical status

| measured ceiling | true ceiling | is the conclusion sound? |
|---|---|---|
| **>= 0.90** | >= measured, so **>= 0.90** | **YES — a PASS is safe** |
| **< 0.90** | could be higher | **NO — a FAIL is NOT safe on its own** |

The bias is **PESSIMISTIC**. That is the safe direction for admitting the
phase and the UNSAFE direction for stopping it, because the stop rule fires
on the low side.

### The binding repair

**P0 FAIL is not actionable until the clamp exposure is checked.** The
diagnostic emits a `clamp_exposure` block counting rows whose DC-order
colour `SH_C0 * dc + 0.5` is `<= 0` in all three channels. Reading rule,
frozen here:

* `clamp_exposure == 0` -> the caveat is **void at DC order** and a measured
  ceiling below 0.90 may be acted on as a genuine construction-level stop.
* `clamp_exposure > 0` -> a ceiling below 0.90 is **INDETERMINATE**. It may
  not trigger the stop until the exposed rows are re-measured through a path
  that does not pass the clamp.

**The DC-order screen is itself an approximation** and is labelled as one:
above degree 0 the higher SH terms contribute at the actual view direction,
so a row can clamp at a particular view while its DC term is positive. The
screen is therefore a NECESSARY-condition check, not a sufficient one, and a
nonzero count is a reason to stop and measure rather than a measurement.

### A second correction, to P7 as I specified it

My §7 P7 asked the diagnostic to "assert the per-view nonzero count is not
identical to the total for every view". **That check is wrong for this
fixture and would have spuriously refused a sound run:** LRV3 is a 20-camera
surround ring and every training view can legitimately supervise every row,
in which case per-view count == total is the CORRECT observation, not an
accumulation artefact.

Replaced by a **sentinel repeat**, which tests the thing I actually meant:
after the last view, view 0 is rendered again and its nonzero count must
reproduce its FIRST-pass count. Under leaked accumulation it would instead
report the run total. Cost is one render in 529. The report records
`sentinel.informative` so a match is not over-read when view 0 already lights
every row. My original check survives as a second limb, narrowed to the exact
accumulation signature (non-decreasing AND rising AND ending at the total).

### Two fail-closed refusals added beyond the spec, and why they are right

1. **`enable_soft_routing=False` is REFUSED.** The gradient identity relies
   on both render branches reading `_features_dc`; with soft routing off the
   static branch reads `get_static_features` instead, so its share of the
   weight silently vanishes from the measurement. This interacts directly
   with precondition C1 (§5.1) and makes the static-twin question
   unmeasurable rather than merely small.
2. **An empty or absent `test_cameras` roster is REFUSED**, rather than
   passing a vacuously-true disjointness test. This is the same class of
   defect as the recorded empty-roster hazard, where a guard degraded
   silently to "protects nothing".

### One operational trap, verified

`_merge_config` is a **YAML-over-argparse** merge
(`scripts/falsify_b2_edit.py:1075-1089`): it calls
`setattr(args, key, host[key])` for every config key, unconditionally, AFTER
argparse. So `--source_path` passed on the command line is silently
overwritten by the YAML value. The Apollo invocation must NOT pass it;
`configs/lrv3/a0_local_control.yaml` already carries the correct path, and
`model_path` is absent from that YAML so the CLI value survives.

---

## REVIEW OUTCOME (2026-08-25, append-only) — VERDICT **MATERIAL DEFECT**, accepted in full

An independent fresh-context adversarial review returned **MATERIAL DEFECT**
against this spec. Every finding below was **re-verified by the primary
against source** before acceptance. Nothing above is rewritten.

**The spine survived**: the isolator framing, binding to a trained
fingerprinted cloud, running a supervisability precondition before spending
compute, refusing abstention as a learned class, and §1's table (every figure
confirmed exact). What follows is what did not survive.

### R-F1 (MATERIAL, ACCEPTED) — P0's PROBE IS REFUTED, and it fails in the VACUITY direction

§7 P0 claims backpropagating `image.sum()` into `_features_dc` measures the
membership channel's weight — *"not a proxy, the same weight"*. **The theory is
right and the probe is wrong.**

Verified by the primary:

* `gaussian_renderer/__init__.py:305` `shs = pc.get_features` and `:356`
  `sh_static = pc.get_features` — **the same tensor** feeds both branches.
* `enable_soft_routing: true` in `configs/lrv3/a0_local_control.yaml:84`, the
  binding config.
* `gaussian_renderer/diff_gaussian_rasterization.py:249-269` returns **both**
  `grad_sh` and `grad_sh_static`; autograd **SUMS** them into `_features_dc`.
* `opacity_static = base_opacity * static_probability`
  (`gaussian_renderer/__init__.py:350`) carries **no temporal marginal**, and
  the static twin renders at the **undeformed** `pc.get_xyz` (`:349`).

So `_features_dc.grad = SH_C0 * (dynamic weight + static weight)`.

**Why this is fatal rather than untidy.** The membership carrier is
**dynamic-only** (`forward.cu:753-762`, `Flow[]` accumulates inside
`if (collected_id[j] < P)`). A row culled by the temporal marginal
(`forward.cu:434`) has **zero dynamic weight and nonzero static weight**.
P0 would call that row *supervisable*. **P0 as written cannot fire for
exactly the substrate condition it exists to detect**, and it biases the
ceiling **UPWARD**, letting the phase proceed past a stop it should have
triggered. C1 does not protect it: C1 bounds a magnitude, P0 asks whether a
quantity is exactly zero, and 1.8% is not zero.

**ACCEPTED FIX, and it is simpler than what it replaces:** probe the
**actual carrier**. Bind `flow_2d` to a per-row leaf of ones and backprop
`out_flow.sum()`. That *is* the channel, so **no identity argument is needed
at all**.

**This also SUPERSEDES the clamp correction appended earlier on this page.**
That append recorded a *pessimistic* bias from the RGB clamp
(`forward.cu:67-70`, `backward.cu:32-34`) — real, and real in the opposite
direction from R-F1. Both errors are artefacts of probing `_features_dc`, and
**both vanish under the flow-carrier probe**: `flow_2d` is a direct per-row
input with no SH evaluation and therefore no clamp, and it is dynamic-only so
there is no static term. One fix, both defects. The clamp append stands as
history; the `clamp_exposure` block becomes unnecessary rather than wrong.

*Pre-existing defect noted in passing, NOT fixed here:* `geom.clamped` is
allocated `P*3` (`rasterizer_impl.cu:180`) and passed to **both** the dynamic
and static preprocess, so with soft routing (`P_static == P`) the static pass
overwrites every dynamic clamp flag. Recorded; out of scope.

### R-F5 (MATERIAL, ACCEPTED) — §3's supply figure is WRONG, and the supervision `n` was overstated 33x

§3 says *"every training camera contributing 4,603-8,201 px"*. **That was
generalized from the first six cameras, which happen to be the high ones.**
Recomputed by the primary over all 960 buffers:

| | §3 claimed | MEASURED |
|---|---|---|
| per-camera range | 4,603 - 8,201 | **16 - 8,201** |
| cameras below 4,603 | 0 | **8 of 16** |
| worst camera | — | **cam14 = 16 px** (0.013% of a 400x300 frame) |
| next worst | — | cam13 = 316 px |
| total per frame | 71,625 | **71,625 — EXACT, confirmed** |

**And the `n` is not what §1 and §3 imply.** The 960 buffers hash to **32
distinct images — exactly 2 per camera** (event-present, event-absent). The
scene is static apart from the event object's boolean presence: `render()`
takes no time argument and the static spheres, lighting and ground are frozen
constants. So *"16 training cameras x 33 supervised frames"* is **16 distinct
masks**, not 528 observations — and two of those sixteen are effectively
empty.

**This is the record's own "a ratio without its n is not a measurement" in a
new costume, and it is load-bearing**: P7 cites the pooled 71,625 as proof
that the object carries contribution.

**ACCEPTED FIX:** the range is corrected to **16 - 8,201**; the distinct-mask
count is recorded; and **P7 becomes a PER-CAMERA precondition with a declared
floor**, not a pooled total. A camera supplying 16 px supplies no usable
membership evidence, and that must be declared before training rather than
discovered after.

### R-F2 / R-F3 (MATERIAL, ACCEPTED) — the gate lost a limb, and R1 makes the survivor unfalsifiable

* §6 says the gate is inherited *"not moved"*, but the `2789fef` gate has
  **three** limbs and §6 lists the third as "no fail-open". The actual third
  limb — ***"it must not reduce to the oracle sphere test"*** — is **absent**.
  **RESTORED verbatim.**
* R1 selects the operating point **by scoring against the oracle**. The record
  already names that construction: *"that source is oracle-supervised ... so it
  is a ceiling, not an instrument"*. So R1 as written produces a **CEILING**.
* By construction the R1 point satisfies `precision >= 0.80`, so **§6's
  precision limb cannot fail** once R1 succeeds, and the reported recall is an
  `argmax` over a sweep — optimistically biased.
* §5.3 promises *"a declared sweep"* and **declares no grid** — no range, no
  resolution, no stopping rule. §0 claims no reading rule can be chosen after
  the fact, but **the sweep grid IS the threshold choice** and it was left open.

**ACCEPTED FIX:** freeze `(e_min, g_min)` as explicit numeric lists in the
spec; select the operating point by an **oracle-blind** rule (a declared
quantile of the measured evidence distribution and a declared margin); report
the curve and the value at that frozen point, **never the argmax**. Any
oracle-selected number is labelled a CEILING and may not be reported as an
instrument score.

### R-F4 (MATERIAL, ACCEPTED) — C1 is unmeasurable as described and its premise is false

* *"LRV3 trains at route logit 4.0"* is **FALSE**. Verified:
  `configs/lrv3/a0_local_control.yaml:85` sets `route_logit_init: 4.0` — an
  **initialization** — and `:86` sets `route_lr: -1.0`, which falls back to
  `feature_lr: 0.0025` (`scene/gaussian_model.py:1572`). `_route_logit` is an
  `nn.Parameter` trained for 6,000 iterations. **The trained static share is
  unknown and is not 1.8% by construction.**
* C1 is not obtainable by any route the spec names: there is no per-row
  accumulated-weight output, `render_3d`/`render_4d` composite with
  **branch-local** transmittance (the wrong weights), and `_features_dc.grad`
  returns the sum of both branches (R-F1).
* A **pooled** 5% cannot exclude a subpopulation of in-sphere rows at 50%, and
  that subpopulation is the one that matters.

**ACCEPTED FIX:** C1 becomes a **quantile** bound over in-sphere rows, not a
pooled mean; and the flow-carrier probe of R-F1 makes P0 immune to the
contamination rather than merely bounding it.

### R-F6 (MATERIAL, ACCEPTED) — the zero-parameter baseline is missing, so a PASS would be unattributable

The membership gradient is exactly `alpha_i * T_i` (`backward.cu:1424`
`const float weight = alpha * T;`). On this fixture the background is static
and the masks are oracle front-most, so a **closed-form, zero-parameter**
instrument

```
score_i(k) = sum over supervised pixels of  alpha_i * T_i * 1[mask == k]
assign argmax_k
```

needs **one forward+backward and no optimizer** — it falls straight out of
P0's own machinery by backpropagating `(image * mask).sum()`.

**If that clears 0.80 / 0.90, the parameter, the Adam group, the 6,000
iterations and the slot-hours have no role**, and AGENTS.md's *"If a simpler
representation creates the same capability, prefer it"* applies directly.
**The spec authorized training without ever asking whether training was
needed.**

**ACCEPTED FIX:** the zero-parameter vote becomes a **frozen precondition**
computed in the same pass, with precision/recall reported. Training is
authorized only if the learned field must beat it, and that margin is frozen
before training.

### R-F7 (MATERIAL, ACCEPTED) — the control for the record's own predicted failure mode is absent

The record predicts precisely how this design fails here: a row outside its
temporal support renders at ~0 opacity and its composited identity feature
receives **vanishing gradient**, so *"the rows with the shortest `_scaling_t`
— exactly the event rows a gate must catch — are the least supervised"*
(verified: `gaussian_renderer/__init__.py:267-268`, `forward.cu:433-435`).

The fixture ships **three static spheres with their own ids in the same
buffers**, supervised at every frame with **no** temporal-marginal
attenuation. Running the identical instrument on one static sphere costs the
same pass and is the **only** control that separates *"per-row membership is
learnable on this substrate"* from *"EVENT membership is learnable despite
temporal-support gradient starvation"*. **ACCEPTED: frozen now, both
reported.**

### R-F8 (MATERIAL, ACCEPTED) — P4 and P6 are about the SCORE, not the setup

§7's header says all preconditions are about the setup. **P4 (shuffled-mask
control) and P6 (signal removal) are comparisons of the very quantity the gate
reads** and cannot be evaluated before a score. Their criteria — *"materially
worse"*, *"measurably"*, *"degrade gracefully"* — are **unquantified**, the
exact defect the block's standing finding names.

**ACCEPTED FIX:** reclassified as post-hoc **CONTROLS**, not preconditions,
with numeric pass criteria frozen before results.

### R-F9 (MATERIAL, ACCEPTED) — P0's predicate has no magnitude floor

`supervisable` is exact-nonzero, so a row at 1e-9 counts as supervisable and
will not move under any authorized LR. Meanwhile §5.3's abstention applies an
evidence floor to the **same** quantity, so the ceiling and the reachable set
are on different scales.

**ACCEPTED FIX:** report P0 as a **curve**, `achievable_recall_ceiling(e_min)`,
over the same frozen `e_min` grid the reading rule uses. Free, and it binds the
precondition to the reading rule.

### R-F10 (MATERIAL, ACCEPTED) — the completeness predicate is wrong, and the phase's own output is not persisted

* §4 asserts the hash enumeration equals *"the model's full per-row tensor
  list"*. **Not every render-affecting trainable tensor is per-row**:
  `_motion_lora_basis` is **shared across rows**, has its own optimizer group,
  and feeds the compositing weights. Non-tensor render state
  (`active_sh_degree`, `time_duration`, `enable_soft_routing`, ...) escapes any
  tensor hash entirely.
* **`_membership_logits` is not in `capture()`**, so a checkpoint written by
  this phase would **silently lose the one tensor the phase exists to
  produce** — and §0 calls this phase the initialization for the joint phase.
  **This is a verbatim repeat of the recorded `_packet_ids` defect.**

**ACCEPTED FIX:** completeness = *every `nn.Parameter` and every
render-consumed buffer*; `_membership_logits` added to `capture()`/`restore()`
with its presence asserted as a precondition.

### R-F11 (MATERIAL, ACCEPTED) — I introduced 960 unguarded oracle buffers and reported only the reassuring half

§3 says `gt_identity/` keeps *"its existing leakage guard un-weakened"*. True —
**and the contract is weakened anyway.** Verified:
`scripts/estimate_episodes.py:209-218` keys `is_forbidden_path` on the
substrings `gt_identity`, `event_spec.json`, and `oracle_*.json`.
**`train_identity` matches NONE of them**, and a repo-wide grep finds **no
other consumer of the name**. This phase has introduced 960 oracle buffers
that the runtime anti-leakage guard does not block, **for every future
oracle-blind estimator on this fixture**.

**ACCEPTED FIX:** `train_identity` added to `is_forbidden_path`, and that is a
precondition of this phase rather than a footnote.

### R-F12 / R-F13 / R-F14 (MINOR, ACCEPTED)

* **The loss is under-specified in the way that decides whether the carrier
  argument matters.** The composited map does **not** sum to 1: the missing
  mass is static-twin contribution **plus residual transmittance**, and
  residual transmittance dominates by an order of magnitude at background
  pixels. Whether CE applies to `M`, `softmax(M)` or `M/(M_0+M_1)` changes the
  gradient structure completely. **The spec bounded the smaller unassigned-mass
  source and left the larger unbounded.** FIX: write the loss as an equation,
  including normalization and the treatment of unassigned mass.
* **NOVELTY, and it must go in Forbidden.** This mechanism — per-primitive
  time-independent identity, alpha-composited, absolute-label 2D CE, geometry
  frozen, zero densification — **is Spacetime Gaussian Grouping's row of the
  complementary-halves table exactly**, and the two-stage freeze is the one
  SA4D explicitly recommends. Abstention is not a differentiator while R1
  calibrates it against the oracle. **What is unoccupied is the MEASUREMENT:
  no method in this literature reports per-primitive precision/recall.**
  **ADDED TO FORBIDDEN: may not claim mechanism novelty for the membership
  channel.**
* *"Held-out cameras are untouched by every step"* is **overstated**: §3's own
  byte-identity self-test renders held-out poses and reads `gt_identity/`.
  Benign — no information reaches the model — but the claim is qualified
  rather than absolute.
* **`K` is capped at 2 by the CUDA carrier**, not chosen for LRV3.
* `elgs/trainer_hooks.py:688-689` was cited for the ungated path; those lines
  are a **docstring**. The assignment is `family_ids.fill_(-1)` at `:929`/`:967`.

### R-F15 (NIT, ACCEPTED) — the 10,650 provenance

That figure was measured on experiment 184's `point_cloud.ply`, not on
`chkpnt6000.pth`. The fingerprint guard makes a mismatch **fail closed**, so
the risk is a spurious stop rather than a wrong number — but §2 states the
identity as established when it is inferred.

### THE REVIEWER'S FREE ADDITION, ACCEPTED

**Require P0's ceiling broken down by the eight object-overlapping cells.**
`achievable_recall_ceiling` restricted to cells **420 and 429** — the 2,036
rows (19.12% of the object) that produced the structural 0.8088 cap — is
**the single most informative number this phase can produce**, and it costs
nothing. It directly tests the residual risk in §1's bet: that those rows are
*interior*, which is also the most plausible cause of `no_interior_gap`, in
which case both instruments fail for a related reason.

### STATUS

**This spec is NOT cleared to run.** The accepted fixes above must be written
into a revised frozen specification, and the revision re-reviewed, before any
cell is submitted. P0 in particular must be re-implemented against the flow
carrier before its number means anything.

---

## SUPERSEDED (2026-08-25) — see v2

This page is **superseded by**
[[frozen-scene-membership-spec-v2-2026-08-25]], which incorporates every
accepted finding above. This page is preserved unchanged, together with the
MATERIAL DEFECT review and the corrections, because the reasoning of v2 is
not readable without it.

**Do not execute anything from this page.** v1 was never cleared to run.
