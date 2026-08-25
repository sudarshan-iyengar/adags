# SPEC (FROZEN, v2) — Frozen-Scene Membership on LRV3, rebuilt after a MATERIAL DEFECT review (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Frozen before any membership
parameter exists, before any cell is submitted, and before any per-row score
is read.**

**SUPERSEDES** [[frozen-scene-membership-spec-2026-08-25]] (v1). v1 is
preserved unchanged, together with the full review that returned **MATERIAL
DEFECT** against it and the primary's verification of every finding. Read v1's
review section for *why* each rule below is what it is; this page does not
re-argue it.

## 0. What changed from v1, in one table

| v1 | v2 |
|---|---|
| probe `_features_dc.grad` | **probe the actual carrier, `flow_2d`** |
| ceiling as a scalar at `w > 0` | **ceiling as a CURVE over a frozen grid** |
| gate = 4 bullets | **gate = the original THREE limbs, non-degeneracy restored** |
| operating point = argmax recall s.t. precision >= 0.80, scored on the oracle | **operating point frozen ORACLE-BLIND; any oracle-selected point is a CEILING** |
| sweep "declared" but no grid | **grids written out numerically below** |
| C1 = pooled 5% of alpha | **C1 = a QUANTILE distribution, no threshold** |
| training assumed necessary | **zero-parameter baseline runs FIRST; training must beat it** |
| no static-sphere control | **static-sphere control frozen** |
| P4/P6 called preconditions | **reclassified as CONTROLS with numeric criteria** |
| supply "4,603-8,201 px/camera" | **16 - 8,201; 8 of 16 below 4,603; 32 distinct masks** |
| `train_identity` unguarded | **added to `is_forbidden_path`** |
| `_membership_logits` not persisted | **must enter `capture()`/`restore()`** |

## 1. The binding cloud — UNCHANGED from v1 §2

```
/apollo/users/sri/proj_adags/runs/elgs/
    20260820T002949Z_lrv3_a0_prime_0_b7952b0/chkpnt6000.pth
```
149,794 rows. Fingerprinted via `cloud_fingerprint`
(`elgs/trainer_hooks.py:701-704`); a differing fingerprint **REFUSES**, never
adapts. Oracle = the sphere test on this cloud's xyz, `centre` and `radius`
read from `event_spec.json`.

**Provenance narrowed (v1 R-F15):** the recorded **10,650** in-sphere count was
measured on experiment 184's `point_cloud.ply`, *not* on `chkpnt6000.pth`. It
is therefore an **expectation, not an established identity**. The run records
the count it actually measures. A mismatch is a fail-closed stop, not a
silently different denominator.

## 2. Supervision — corrected, and the `n` stated honestly

Masks: `data/synthetic/lrv3/train_identity/camNN_fFFF.npy`, int16 front-most
identity, **training cameras only**, emitted additively by
`scripts/emit_training_identity.py` with a byte-identity precondition against
the frozen held-out buffers. **ORACLE ray-traced** — never curated, never
estimated, never noisy.

**MEASURED, and v1's figure was wrong:**

| quantity | value |
|---|---|
| total event px per frame, 16 training cameras | **71,625** (v1 correct) |
| per-camera range | **16 - 8,201** (v1 said 4,603-8,201) |
| cameras below 4,603 px | **8 of 16** |
| worst / next worst | **cam14 = 16 px**, cam13 = 316 px |
| **distinct images among the 960 buffers** | **32 — exactly 2 per camera** |
| gap frames | **0 px, every camera** |

**THE EFFECTIVE `n` IS 16 DISTINCT MASKS, NOT 528 OBSERVATIONS.** The scene is
static apart from the event object's boolean presence, so every present-frame
buffer of a camera is byte-identical to every other. v1's "16 cameras x 33
frames" overstated the supervision by 33x. **Every artifact of this phase
states the 16, never the 528.**

**Consequence, frozen:** the supply precondition is **PER-CAMERA**, reported
for all 16, never pooled. A camera contributing 16 px supplies no usable
membership evidence and is declared so before training.

**Leakage.** Held-out cameras 2/7/12/17 are untouched by supervision,
calibration, thresholds and model selection. *Qualified, per v1 R-F14:* the
emitter's own byte-identity precondition does render held-out poses and read
`gt_identity/` — no information reaches the model, but the claim is not
absolute. `train_identity` is now in `is_forbidden_path`
(`scripts/estimate_episodes.py`), closing the guard gap v1 opened.

## 3. THE INSTRUMENT — one forward pass, five answers, zero parameters

The membership channel composites as `M_c = sum_i f_ic * alpha_i * T_i`, where
`f` is the per-row `flow_2d` input: **2 channels, dynamic-only, no background
term**, VJP repaired and pinned against an independent oracle
(`tests/ref_impls/flow_compositing_reference.py`).

Bind `flow_2d` to a per-row **leaf of ones**. Then for any upstream `g(pixel)`:

```
dL/df_i0  =  sum over pixels of  g(pixel) * alpha_i * T_i
```

so one forward and several backwards give, per row:

* `g = 1`        -> `w_total_i`
* `g = mask_k`   -> `w_in_mask_i(k)`

**`flow_2d` has no SH evaluation (hence no clamp) and is dynamic-only (hence
no static contamination). This IS the channel — no identity argument is
required.** Both of v1's opposing errors vanish under this one change.

The instrument reports:

1. **Ceiling as a CURVE**, `achievable_recall_ceiling(e_min)`, over the frozen
   grid of §4, plus the strict `w_total > 0` point and the `w_total`
   distribution over in-sphere rows.
2. **The ZERO-PARAMETER VOTE**: `score_i(k) = w_in_mask_i(k)`, assign
   `argmax_k`, scored per-row against the oracle.
3. **PER-CELL breakdown**, and specifically **cells 420 and 429** — the 2,036
   rows (19.12% of the object) that produced the structural 0.8088 cap.
4. **The STATIC-SPHERE CONTROL** (§6).
5. **C1**, the static-twin share, as a **quantile distribution** over in-sphere
   rows, measured with the static branch bound to a separate leaf. No
   threshold is applied.

## 4. THE READING RULE — frozen numerically, and ORACLE-BLIND

v1's R1 selected the operating point by scoring against the oracle, which the
record already names as producing *"a ceiling, not an instrument"*. Replaced.

**FROZEN OPERATING POINT, chosen before any score and using no oracle
information:**

* `e_min = 0` **strict**: a row is eligible iff `w_total_i > 0`. A row that
  received no rendered evidence **abstains** (`family_id = -1`).
* `tau = 0.50`: an eligible row is **assigned** to `argmax_k w_in_mask_i(k)`
  iff the winning class holds **at least half** of that row's own total
  weight, `max_k w_in_mask_i(k) >= tau * w_total_i`; otherwise it **abstains**.

Both are **declared judgments**, labelled as such. `tau` is scale-free and
uses only per-row quantities, so no oracle and no held-out view can enter it.

**REPORTED ALONGSIDE, as supplementary CEILING information and never as the
instrument's score:**

* `tau` grid: **{0.00, 0.25, 0.50, 0.6667, 0.75, 0.90}**
* `e_min` grid: absolute **{0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0}** and quantiles of
  `w_total` **over ALL rows** (never over in-sphere rows, which would leak the
  oracle) at **q = {0.50, 0.75, 0.90, 0.95, 0.99}**

**Any point on those grids other than the frozen `(0, 0.50)` is a CEILING and
is labelled one.** The gate is evaluated at the frozen point only.

**FAIL CLOSED (unchanged from v1 R2):** empty or unsupported evidence yields
an EMPTY selection, never the whole cloud. The inverted-mask pathology
(`if count_nonzero(mask)==0: mask = ~mask`) is forbidden and its absence is
asserted by a neuter test.

## 5. THE GATE — all THREE limbs of `2789fef`, restored

Evaluated on the binding cloud of §1, at the frozen operating point of §4:

1. **per-row precision >= 0.80**
2. **per-row recall >= 0.90**
3. **NON-DEGENERACY** — *"it must not reduce to the oracle sphere test and
   must not consume held-out cameras (2, 7, 12, 17)"*. **This limb was
   dropped in v1 and is restored verbatim.**

Rendered 2D mask IoU is supplementary and cannot substitute. No floor moves
after a score is read.

## 6. ORDERING — the zero-parameter baseline decides whether training happens

**This is the single most consequential change from v1, and it is an ordering,
not a threshold.**

```
STEP 1  the zero-parameter instrument of §3   (one pass, no optimizer)
          |
          +-- clears the §5 gate  -> TRAINING IS NOT AUTHORIZED.
          |                          The capability exists without it.
          |                          Report and stop.
          |
          +-- fails, and the §4 ceiling curve shows the gate is
          |   unreachable at every e_min  -> STOP. Construction-level
          |                                  negative, no compute spent.
          |
          +-- fails, but the ceiling admits >= 0.90  -> training MAY be
                                                        authorized (§8)
```

AGENTS.md: *"If a simpler representation creates the same capability, prefer
it."* v1 authorized training without ever asking whether training was needed.

**FROZEN MARGIN.** If training runs, it must beat the zero-parameter vote by
at least **+0.05 recall at equal-or-better precision**, at the frozen
operating point. Below that the parameter is not carrying its weight and the
result is reported as *no improvement over closed form*. Declared judgment,
frozen now.

## 7. CONTROLS — with numeric criteria, frozen before results (v1 R-F8)

v1 called these preconditions; they are comparisons of the scored quantity and
are therefore **post-hoc controls**. Their criteria are numbers, not adjectives.

Chance precision on this cloud is about **0.071** (10,650 of 149,794 rows).

* **C-SHUFFLE — shuffled-mask negative control.** Precision must fall
  **below 0.30**, and the correct-mask precision must exceed it by **>= 0.40
  absolute**. Anything less means the instrument is not reading the masks.
* **C-REMOVE — signal removal.** With the event object's mask supervision
  deleted, recall must fall **below 0.10**. Graceful degradation is a failure.
* **C-STATIC — the static-sphere control (v1 R-F7), and it is the one that
  makes a failure attributable.** The fixture ships three static spheres with
  their own ids in the same buffers, supervised at every frame with **no
  temporal-marginal attenuation**. Run the identical instrument on one.
  * static passes **and** event fails -> the failure is **temporal-support
    gradient starvation**, exactly as the record predicts (*"the rows with the
    shortest `_scaling_t` — exactly the event rows a gate must catch — are the
    least supervised"*).
  * both fail -> per-row membership is not recoverable on this substrate at
    all, and the event result says nothing specific about episodes.
  * **Without this control a failure is unattributable and a pass is
    unattributed.**

## 8. IF TRAINING IS AUTHORIZED — what learns, and what must be persisted

Only reachable via §6. Then:

* **Learns:** `_membership_logits`, `(N, K)`, one Adam group.
  **`K = 2` is a CARRIER LIMIT, not a scene choice** — `flow_2d` hard-codes 2
  channels in CUDA; `override_color` would allow 3. Say so wherever `K`
  appears.
* **Abstention is NOT a learned class** (v1, retained): the optimizer would
  otherwise discover that abstaining everywhere minimizes a mask loss.
* **`_membership_logits` MUST enter `capture()` and `restore()`**, with its
  presence asserted as a precondition. v1 omitted this, which is **a verbatim
  repeat of the recorded `_packet_ids` defect** — a checkpoint would have
  silently lost the one tensor the phase exists to produce, and §0 of v1 calls
  this phase the initialization for the joint phase.
* **Frozen-tensor completeness is `every nn.Parameter and every
  render-consumed buffer`**, NOT "every per-row tensor". `_motion_lora_basis`
  is shared across rows, has its own optimizer group, and feeds the
  compositing weights; non-tensor render state (`active_sh_degree`,
  `time_duration`, `enable_soft_routing`) escapes any tensor hash and is
  recorded separately.
* **Topology frozen**, `N` invariant, asserted every iteration.
* **THE LOSS MUST BE WRITTEN AS AN EQUATION**, including normalization and the
  treatment of unassigned mass. `M_0 + M_1` does **not** sum to 1: the missing
  mass is static-twin contribution **plus residual transmittance**, and
  residual transmittance dominates by an order of magnitude at background
  pixels. Whether CE applies to `M`, `softmax(M)` or `M/(M_0+M_1)` changes the
  gradient structure completely. **v1 bounded the smaller unassigned-mass
  source and left the larger unbounded.** No training cell may be submitted
  until that equation is written and reviewed.

## 9. PRECONDITIONS — about the SETUP only

* fingerprint recorded; mismatch REFUSES.
* camera set disjoint from `test_cameras`; an **empty or absent** test roster
  REFUSES rather than passing a vacuous disjointness check.
* `enable_soft_routing=False` REFUSES — C1 becomes unmeasurable, because with
  soft routing off the static branch reads `get_static_features` and its share
  silently leaves the measurement.
* the render ran: nonzero rows carry nonzero `w_total`.
* **per-camera** mask supply reported for all 16 cameras (§2).
* the 32-distinct-buffer fact verified and reported.
* topology invariant across the pass.

## 10. Permitted and forbidden

**Permitted.** To report the ceiling curve, the zero-parameter vote, the
per-cell breakdown, the static control and the C1 quantiles as results in
their own right — including negative ones. To stop at §6 with no compute
spent.

**Forbidden.** To move any floor in §5 after a score is read. To report an
oracle-selected sweep point as the instrument's score. To evaluate on any
cloud but §1's. To describe the masks as anything but ORACLE ray-traced. To
cite 528 observations. To read held-out cameras 2/7/12/17. To substitute
rendered 2D IoU for per-row precision/recall. To claim mask-noise robustness.
To claim anything transfers off LRV3. **To claim MECHANISM NOVELTY for the
membership channel** — per-primitive time-independent identity, alpha-
composited, absolute-label 2D CE, geometry frozen, zero densification **is
Spacetime Gaussian Grouping's row of the complementary-halves table exactly**,
and the two-stage freeze is the one SA4D explicitly recommends. **What is
unoccupied here is the MEASUREMENT — no method in this literature reports
per-primitive precision and recall — and that is the only novelty this phase
may claim.**

---

## RE-REVIEW OUTCOME (2026-08-25, append-only) — VERDICT **MATERIAL DEFECT** again; the CARRIER repair holds, the RULES around it do not

A second independent fresh-context review, scoped to *"did the fixes fix it,
and did they break anything new"*, returned **MATERIAL DEFECT**. Verified by
the primary before acceptance.

### WHAT SURVIVED — the flow-carrier identity is EXACT, confirmed against CUDA

`forward.cu:760` accumulates `Flow[ch] += flows[...] * alpha * T` strictly
inside `if (collected_id[j] < P)` (`:753`), written unmodified at `:797` with
**no background term**. Backward: `const float weight = alpha * T` (`:1427`)
then `Register_dL_dflows[ch] += weight * dL_dchannel_flow` (`:1462`) then
`atomicAdd` only under `if (gaussian_idx < P)` (`:1496-1508`). Static twins
write nothing and their gradient register is discarded. **No SH evaluation on
the flow path, hence no clamp.** `flow_2d` reaches the kernel unmodified and
`grad_flows` returns in the matching argument position.

**So the identity is exact, and it was also demonstrated empirically:** on a
numpy autograd stub, 10 deliberately culled rows read `w_total` **exactly
0.0** while their SH gradient was nonzero — **v1's probe would have called all
ten supervisable.** The refutation reproduces.

Section 2's corrected numbers reproduced exactly, including that v1's wrong
range came precisely from cameras 0,1,3,4,5,6.

### N-1 (MATERIAL, ACCEPTED) — the SPEC's class count is wrong, not the rule

Section 4 froze `tau = 0.50` while section 8 says `K = 2`. The arithmetic is
decisive: the identity buffers **partition** the image, so the class weights
sum to `w_total` identically, and **the max of two non-negative numbers
summing to S is always at least S/2** — at `K = 2` no row can ever abstain
below tau, and three grid points collapse.

**The instrument does not have this defect; the SPEC does.** The
zero-parameter vote runs over the identity buffer's actual classes
`{-1, 0, 1, 2, 3, 100}`, not two, and the tau rule demonstrably bites —
verified by the primary: sweep **3/3/1/1/1/0**, scale-free under a 1e9
rescale, non-increasing in tau.

**CORRECTED HERE:** the zero-parameter vote is **MULTI-CLASS over the identity
buffer's classes**, the partition identity is stated, and `tau` bites only
when weight splits across **three or more** classes. `K = 2` describes the
*trained* head's carrier limit and nothing else.

### N-9 (MATERIAL, ACCEPTED) — and it is the real consequence of N-1

The two arms compared are **not the same instrument**: the vote is multi-class
and can abstain; a trained `K = 2` head cannot abstain at `tau = 0.50` at all.
The trained arm is therefore systematically favoured on recall by class
structure alone, and the training authorization is staked on a recall
comparison between them.

**ACCEPTED FIX:** run the vote at **both** class structures and compare the
trained `K = 2` head against the **`K = 2` vote**.

### N-4 (MATERIAL, ACCEPTED) — C-REMOVE cannot return an unfavourable result

Delete the event class and the event score is 0 for every row, so the argmax
never selects it: **recall is exactly 0 by arithmetic**, in both readings of
"remove". The zero-parameter arm runs first and a pass is terminal, so the
only control certain to execute is the one determined in advance. **This is
the block's own flagged pathology reproduced inside the document written to
prevent it.**

**ACCEPTED FIX:** C-REMOVE becomes a **training-arm-only** control, or is
replaced by a removal that is not arithmetically forced — delete the event
mask from a declared *subset* of cameras and report recall as a function of
how many remain.

### N-3 (MATERIAL, ACCEPTED) — C-STATIC cannot discriminate its two hypotheses

Measured at frame 000 over the 16 training cameras:

| target | radius | total px | per-camera min | frames |
|---|---:|---:|---:|---:|
| event (id 100) | 0.20 | **71,625** | **16** | 33 |
| static 0 (id 1) | 0.35 | **184,890** | **5,962** | 60 |
| static 1 (id 2) | 0.28 | 143,786 | 5,717 | 60 |
| static 2 (id 3) | 0.25 | 76,106 | **0** | 60 |

The default is sphere **0** — **2.58x** the event's supply, whose *worst*
camera carries **373x** the event's worst and more than the event's *best*,
larger in 3D, and supervised on **60 frames against 33**. "Static passes,
event fails" is consistent with four uncontrolled differences besides
temporal support, so the inference is not licensed and the sentence making
the control load-bearing is not delivered.

**ACCEPTED FIX:** name the sphere; choose by matched supply (**static 2** at
76,106 vs 71,625 is the only near-match, and even it has a zero-supply
camera); restrict the static arm to the **same 33 frames**; print both
per-camera supply tables side by side; and pre-declare a supply ratio beyond
which the comparison is **unattributable** rather than read as evidence.

### N-6 (MATERIAL, ACCEPTED) — the +0.05 margin is invented, and worse than unanchored

A search of the whole of `research-wiki/operations/` found **no membership
replicate or variance figure anywhere**. The only spread on this fixture is
**PSNR** (0.09-0.17 dB), which cannot be converted into a recall margin. The
zero-parameter arm is deterministic; the trained arm's seed-to-seed membership
spread **has never been measured on any fixture in this project**.

**It is worse than the recall floor it sits beside.** 0.90 is an unanchored
judgment *labelled as one*; +0.05 is an **increment on an unknown baseline**,
so the effective bar is **unknowable at freeze time** — at a vote recall of
0.89 it demands 0.94, stricter than the gate; at 0.30 it demands 0.35, far
below it.

**ACCEPTED FIX:** state plainly that no membership replicate spread has ever
been measured; replace the increment with a bar commensurate with the gate
("training must clear the gate the vote missed"), **or** make measuring the
same-seed membership spread a precondition of authorizing training.

### N-2, N-5, N-7, N-8, N-10, N-11, N-13 (ACCEPTED)

* **N-2** — the (camera, frame) accumulation domain is never stated, and the
  choice is decisive: the binding cloud is the **ungated** A0-prime substrate,
  which renders event rows *through the gap*. Include gap frames and every
  event row accrues background weight over 27x16 frames against 33x16, and the
  argmax flips for a reason about substrate ghosting, not membership. The
  blanket ban on citing "528" also bans naming the render count. **FIX:**
  state the domain and its cardinality; narrow the ban to "may not cite 528 as
  the supervision n".
* **N-5** — C-SHUFFLE's shuffle **operation is undefined** (across cameras,
  across frames, and within-mask give different nulls); precision is
  **undefined on an empty assignment**, which is the expected outcome on a
  20-camera ring and is also the correct fail-closed behaviour; and the
  0.40 limb is **coupled to the score it controls**. **FIX:** define the
  shuffle; declare an empty shuffled selection a PASS; drop the coupled limb.
* **N-7** — the non-degeneracy limb was restored **verbatim but never
  operationalized**, and this phase is oracle-supervised end to end (oracle 2D
  masks scored against the oracle's own 3D sphere test). It will otherwise be
  adjudicated after the score is read. **v2 also dropped v1's scope framing**
  and the Forbidden item *"to authorize joint refinement because a rendered
  mask looks good"* — both re-imported.
* **N-8** — **v1's "the mechanism was exercised" precondition was deleted**,
  in the same edit that cites the discipline requiring it. Also dropped: the
  sentinel-repeat accumulation check, the verified `_merge_config`
  YAML-over-argparse trap, and v1's slot-hour accounting. All restored.
* **N-10** — **`flow_2d` is not an argument to `render()`.** It is built
  inside (`gaussian_renderer/__init__.py:311-317`) and
  `enable_rendered_flow: false` in the binding config, so the default path
  yields a **non-leaf zeros tensor** and the probe would return nothing,
  silently. The binding requires a call-boundary substitution the spec never
  named. *(The implementation does this correctly and adds a
  `flow_leaf_bound_every_view` precondition; the spec must require it.)*
  Related: the row-dropping prefilter at `:319-345` is gated on
  `compute_cov3D_python`, **False** in the binding config, so no row is
  dropped and no index remap is needed — correct here, but a config-dependent
  property the spec did not pin.
* **N-11** — **there is no static flow channel** (no `flows_static` anywhere in
  the CUDA), so C1's separate leaf is necessarily an **SH** leaf — exactly the
  clamp-exposed path this page declares superseded. The claim of a clamp-free
  C1 measurement is therefore wrong, and `clamp_exposure` is **still
  required, for C1 only**.
* **N-13** — `MANIFEST.train_identity.json`'s `leakage_note` still states the
  guard does NOT block that directory. True when emitted, **false since the
  repair**. A durable artifact now contradicts the code.

### STATUS

**v2 is NOT cleared to run**, on the rules. **The instrument is in better
shape than the specification**: verified at 162 tests / 173 self-test checks,
it implements the frozen grids exactly, pins the cloud by the recorded
fingerprint `460c2736...3837f` (149,794 rows, from
`configs/lrv3/estimated_program_v2.json`) rather than by path, and reproduces
the v1 refutation empirically.

**Judgment recorded:** two successive MATERIAL DEFECT verdicts on rules
written in one sitting is itself the finding. A v3 must be written
deliberately, not iterated the same evening, and the LRV3 lane is therefore
**paused rather than pushed** for this block.
