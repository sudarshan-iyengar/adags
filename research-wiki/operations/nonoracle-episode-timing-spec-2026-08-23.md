# FROZEN — non-oracle episode-boundary estimation on LRV3, phase T1
# (2026-08-23)

Status: **FROZEN before any cell output.** EXPLORATORY,
`evidence_bearing: false`. This addresses the blocker the supervisor
brief names the single largest scientific risk: every positive episodic
presence result to date uses AUTHORED boundaries.

## 1. The binding requirement, and why it sets the whole design

The measured mistiming control is the constraint:

```
correct gate  +1.05 dB  >  no gate  0  >  2-frames-early  −2.39 dB  >>  maximally wrong  −17.16 dB
```

A 2-frame boundary error is **worse than not gating at all**. So an
inferred-timing mechanism has NEGATIVE value unless it is frame-accurate
or abstains. This is not a preference; it is a measured ordering, and it
makes "can the estimator place boundaries to frame accuracy?" the only
question worth asking first.

## 2. Scope: T1 measures, it does not retrain

**Phase T1 (this specification):** estimate boundaries from training
views on an already-trained checkpoint, then score the estimate against
the authored ground truth. **No retraining, no schema change, no
`seed_families` change.**

**Phase T2 (NOT authorized here):** a program-schema v2 carrying
per-group membership and per-group gaps, plus a computed-program branch
in seeding, plus A0 / A-oracle / A-est training cells. T2 is justified
ONLY if T1 passes.

The sequencing is the point: T1 can kill the lane for a few minutes of
GPU, whereas retraining first would spend hours to discover the same
thing. It also refuses the failure mode where a not-quite-accurate
estimator is retrained and its damage attributed to the representation.

## 3. Anti-leakage contract — enforced in code, not by convention

**Allowed estimator inputs:** the trained checkpoint, TRAIN cameras only
(`0,1,3,4,5,6,8,9,10,11,13,14,15,16,18,19`), training-view GT RGB, and
renderer-derived signals.

**Forbidden as estimator input:** `scene.getTestCameras()` (cameras
`2,7,12,17`), anything under `gt_identity/`, `event_spec.json`'s
`presence_frames` / `event_object`, and **both the `region` AND the
`gaps`** of any `configs/lrv3/oracle_*.json`.

The subtlest hazard is recorded explicitly because it would fake a
success: the oracle `region` is a sphere at centre `[0.7, 0.1, 0.35]`
radius `0.2` — **identical to the event object's true geometry in
`event_spec.json`**, and the same geometry the evaluation masks derive
from. An estimator that computes boundaries but INHERITS that region for
membership is still oracle-supervised and its apparent success would be
an artifact. **Membership must be derived from training views.**

Enforcement: assert the process never calls `getTestCameras`; assert
every consumed camera resolves to a train-split id; assert no
`gt_identity` path is opened during estimation; assert
`event_candidate_manifest` and `event_boundary_support_manifest` are
empty. Structurally, ESTIMATION runs first and writes its program;
SCORING loads ground truth only afterwards. Reordering the two would
invalidate the result, and that is stated in the module.

## 4. The mechanism

**The total opacity gate is itself the ablation operator.** True
per-primitive contribution mass does not exist and cannot be extracted
without new CUDA work — verified: the forward rasterizer performs zero
`atomicAdd` and writes nothing back per-Gaussian. But the existing
`_elgs_presence_override` machinery already renders counterfactuals
without mutating live state, and the paired-render L1 comparison is
already implemented and tested for the consolidation pass.

For candidate group `G` and training view `(c, t)`:
`E_G(t)` = mean over sampled train cameras of
`L1(G ablated) − L1(G present)`, restricted to `G`'s screen footprint.
This is footprint-integrated rather than point-sampled, and frame-exact
by construction. The unablated render is SHARED across all groups at a
view, so cost is `1 + n_groups` renders per view, not `2 · n_groups`.

**Membership** comes from a fixed spatial partition of the trained cloud
— a grid over the cloud's own bounding box, or the existing
epsilon-graph connected components — never from the oracle sphere.
Sub-threshold groups are discarded.

**Decision rule, frozen, not tuned:**
* contrast: `G` is presence-varying iff
  `(max E_G − min E_G) ≥ 4 × MAD(E_G)`;
* boundaries: the frames where `E_G` crosses `(max + min)/2`, with
  hysteresis;
* **ABSTAIN** — leave the group ungated at `family_id = -1` — if the
  contrast test fails, or fewer than 3 sampled train cameras agree on
  the crossing frame within ±1, or the implied interval would violate
  `floor_len` / `floor_gap`. An inadmissible interval is never repaired.

**Abstention needs no new mechanism.** `family_id = -1` already means
ungated, and `gated_row_mask` already admits only families with
`K > 1`, so an abstaining group keeps the ordinary temporal marginal
bit-for-bit. This is the single most favourable pre-existing fact for
this lane.

**Boundary inset.** The emitted program insets boundaries by `w`
(= 2 frames on LRV3), exactly as `oracle_correct.json` does, so the
smoothstep ramp never lands on a truly-present frame. Encoded in the
emitter so it cannot be forgotten; the ramp costs ~14 dB on a frame it
lands on.

## 5. Gate, frozen before output

T1 PASSES — and only then is T2 justified — iff:

1. for the group(s) corresponding to the event object,
   **|onset error| ≤ 1 frame AND |offset error| ≤ 1 frame**;
2. **no ACCEPTED (non-abstaining) group has a boundary error ≥ 2
   frames**;
3. the anti-leakage assertions all held.

Condition 2 is not a stylistic choice: 2 frames is the measured point at
which gating becomes worse than not gating, so an accepted boundary at
that error is disqualifying by prior measurement rather than by
preference.

**If T1 fails, the negative is preserved and NO retraining is spent.**
The failure must not be converted into an unregistered soft heuristic
after the fact — the abstention path already exists precisely so that
uncertainty has a legitimate home.

## 6. Reported quantities

Onset error; offset error; the accepted-boundary error distribution;
abstention rate; false-activation count (groups in fact static that were
gated); number of groups overlapping the event object; the opportunity
denominator; render counts and measured cost; and the anti-leakage
assertions checked.

## 7. Ground truth used ONLY for scoring

Authored boundaries: episode 1 frames **0-29**, gap **30-56**, episode 2
**57-59**. Identity buffers exist for held-out cameras only (240 files,
cams 2/7/12/17), which makes `gt_identity/` a natural tripwire: any read
of it during estimation is automatically a held-out read.

**Hazard carried in from [[lrv3-fixture-hazards-2026-08-23]] §3:** the
`configs/lrv3/{a0,a1,a2}.yaml` headers state the gap as 30-53 and the
return as 54-59, which is the LRV2 timing and is **3 frames wrong**.
Anyone building an estimator or a gate from that prose would be inside
the known-harmful regime before starting. The executable artifacts are
correct; the configs must not be edited, because a YAML comment change
alters the config content hash and breaks comparator identity.
