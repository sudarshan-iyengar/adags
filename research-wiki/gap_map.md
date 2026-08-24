# Gap Map

## Instrument-Validity Update — 2026-08-24

**This block's most transferable result is not about a gap. It is about
how this project's instruments fail**
([[operations/block-2026-08-24-handover]] §14).

Three separate instruments returned a favourable-looking result that
**could not have returned an unfavourable one**. Two were caught
automatically by a pre-declared precondition; one was caught only because
an invariant looked wrong.

**The precondition that worked is worth stating exactly, because its form
is what made it work.** V3 — *"the accepted component holds at least one
cell whose object volume comes ONLY from each named arm"* — is a statement
about the **accepted set**, evaluated **before any score**. It therefore
cannot leak the outcome, and it fires whether or not anyone is paying
attention. The instrument that lacked such a clause produced a clean null
that would have been reported as a finding.

**Added to the standing method rules:** *freezing a reading rule is not
enough. Every frozen rule needs a frozen precondition asserting that the
mechanism it reads was actually exercised.* This joins LRV4's *"a ratio
without its n is not a measurement"* and the 2026-08-20 rule that *"every
edit experiment needs a control that separates MAGNITUDE from
CORRECTNESS"* — three instances of the same underlying failure, caught
three different ways, and only the pre-declared ones were caught cheaply.

## G13 Membership Update — 2026-08-24: the hull question is unanswerable BY THIS ROUTE, and T1 is the reason

[[operations/lrv3-membership-candidates-result-2026-08-23]] §7 recorded
axis-aligned **hull completion** as reaching recall 1.0 at precision 0.94,
refused as post-hoc, with the stated weakness that it assumes the object is
**cell-convex**. LRV5-NCX was built to test exactly that weakness: an
L-shaped object whose axis-aligned voxel hull contains a deliberately empty
cell, with a **persistent, always-visible cross of thin walls standing in
the notch** so that filling the concavity would demonstrably suppress a
visible object for 27 frames.

**Both predeclared orientations are INVALID, complementarily**
([[operations/nonconvex-hull-o1-result-2026-08-24]]):

| | cells only from arm A | cells only from arm B |
|---|---:|---:|
| O1 | **0** | 3 |
| O2 | 3 | **0** |

**The accepted component never spans both arms in either orientation**, so
the hull operator's per-component bounding box never reaches the notch.

**The cause is T1's selectivity, and it is a genuine tension in the
method.** T1 gates **2 of 452** and **2 of 511** groups. The same extreme
selectivity that made its boundary inference **exact, with zero false
activations** — the property that made T1 the block-2 positive — produces
an accepted component far too small to span an object whose arms are
separated by a notch. **A high-precision, low-recall estimator cannot
exercise an operator that acts on the SHAPE of the accepted set.**

**Consequence for G13:** hull completion is neither refuted nor supported,
and remains a recorded-but-unadopted candidate. What is newly known is that
**testing it requires forcing a spanning accepted component by
construction**, not hoping T1 produces one — and every route to that
either changes the grid the LRV3 result was measured on, or trades away
T1's zero-false-activation property. Three routes are costed on the result
page; none is adopted.

**Not changed:** the structural recall cap of 0.8088 measured on LRV3, and
the finding that precision is a cloud-binding problem while recall is an
estimator-sensitivity problem.

## Measurement-Channel Update — 2026-08-24: the pre-registered low-variance endpoint FAILS on fresh data

The 2026-08-23 replicate-floor result made the same-code floor the binding
constraint on every N3V comparison. This block's variance study
([[operations/n3v-variance-study-spec-2026-08-24]]) pre-registered a
within-run contrast (`union − complement`) as a **prediction to be
tested**, on the reasoning that a run-level shift common to both regions
should cancel in their difference.

**The prediction is REFUTED at both sample sizes.** Historical n=3:
contrast sd 0.1745 against union 0.2622, a 33.4% reduction. Fresh n=3:
0.2730 against 0.1251, **2.18x worse**. Fresh **n=6**: **0.1913 against
0.1847 — still worse**.

**The mechanism is the correlation the spec never computed.** The contrast
beats the union iff `rho > s_c / (2 s_u)`. Measured:

| cohort | rho(union, complement) | threshold | outcome |
|---|---:|---:|---|
| historical n=3 | **+0.7732** | 0.4876 | contrast wins |
| fresh n=3 | **−0.6610** | 0.6938 | contrast loses |
| **fresh n=6** | **+0.3867** | 0.4292 | **contrast loses** |

The spec's stated mechanism — *"a shift common to union and complement
cancels in their difference"* — predicts rho near +1. It was never
computed, and **at n=3 the Fisher-z standard error for a correlation is
1/sqrt(n−3) = 1/0, undefined**: the co-primary was selected on a quantity
estimated where it has zero degrees of freedom.

**Registering it as a prediction rather than adopting it is what saved
this.** Adopting on the historical three would have moved every future N3V
comparison onto an endpoint worse at both sample sizes.

**What the study DID achieve, and the cost fact that closes the lane.**
sigma(union) is now **0.1847 dB, 95% CI [0.1153, 0.4530], ratio 3.93x**
against n=3's 12.07x — estimable at last. And the n=6 spread of **0.4913**
corroborates the 2026-08-23 replicate floor of **0.4945** to **0.7%**
across disjoint cohorts at different commits. **But under the spec's own
binding rule that cost uses the UPPER confidence limit, a two-arm
comparison at delta\* = 0.30 needs 37 replicates/arm = 181 slot-h, 7.5x
the block ceiling.** The study estimated sigma well enough to show the
comparison it was built to enable is **unaffordable**.

**A fresh adversarial review returned MATERIAL DEFECT and it was
accepted** — including that the spec's own exclusion rule, applied
mechanically, makes the study **inconclusive at n=6**; that delta\*'s
"external grounding" is a category error (the event union is 0.3377% of
pixel-times, so even infinite union PSNR moves whole-frame by 0.0202 dB
against a 0.38 dB anchor); and that a `git worktree` submission would have
avoided the archive deviation entirely.

**Two of the four variance levers were closed by arithmetic on data
already recorded, before any new cell ran:** pooling more pixels is
near-falsified (296x the pixel-times reduces the spread **2.8%**, not the
~17x sampling noise would give, so the variance is a **global run-level
shift**), and more replicates is unaffordable (**13/arm = 63.5 slot-h** at
δ* = 0.30 dB under the corrected Guenther small-sample correction —
the 14/arm figure first recorded used a `ceil+2` correction that an
adversarial review showed overshoots).

**A second-order observation that bounds how much any n=3 result should be
trusted:** the PRIMARY endpoint's sd differs **2.1x** between two n=3
cohorts of the identical protocol. That is not an anomaly — it is what a
sigma CI spanning 12.07x predicts, and it is the study's own premise made
visible.

## Dataset Update — 2026-08-24: ImViD is ADMITTED for reconstruction and TRAINS; event supply is unchanged

[[operations/dataset-admission-matrix-2026-08-18]]'s **NOT ADMITTED for
event supply** verdict is **unchanged** — ImViD still ships no masks and no
identity ground truth, so there is no tracker-independent instrument.

What changed is everything upstream of that. The calibration passed a
reprojection gate at **1.215 / 1.162 / 1.214 px at native** against a 2 px
native gate with 35/35 cameras, the sparse union is confirmed reusable
without re-triangulation, a converter closed the loader gap, and **ImViD
trained end to end for the first time** with a held-out evaluation.

**A hazard worth carrying to any new dataset:** the loader route that
**fails closed** on an unsupported camera model is harmless; the Blender
route reads intrinsics with **no camera-model field and no distortion
field** and would train distorted images silently, wrong by a median of
14.7 px. The check everyone was watching was the safe one.

**A binding scale fact:** the full Opera take is **15,215 frames**, so the
full split is 532,525 training units against N3V-50f's 950 — a **560x**
exposure gap. No authorized schedule can train it; a frozen event-selected
tranche is mandatory, not preferable.

**Acquisition is rate-limited per-IP at ~62 GiB**
([[operations/imvid-acquisition-quota-2026-08-24]]), diagnosed by a probe
from a different host still being served. Five escalating backoff attempts
across 4.75 h were all refused, indicating a long-horizon cap.

## Novelty Update — 2026-08-24: Spacetime Gaussian Grouping does NOT occupy, and one premise is corrected

[[operations/spacetime-gaussian-grouping-read-2026-08-24]]. Read in full
from the CC-BY journal version. Identity is **time-independent**, no
suppression is implemented in any form, and nothing is inferred. **The
standing forbidden-claim entry is DISCHARGED.**

**One premise corrected append-only:** "none of the seven operates on a 4D
representation with per-primitive temporal support" was true of those seven
and false in general. SGG is per-primitive identity on exactly the
`(mu^tau, s^tau)` substrate family. Every downstream consequence survives,
and the per-primitive-metrics finding is **strengthened** — the one method
sharing our substrate also reports only rendered-2D mIoU, with "PSNR"
occurring zero times in its full text.

**One genuinely new mechanism, and it matters for G13:** SGG's supervision
is **absolute-label cross-entropy**, not a within-frame contrastive term.
So unlike the contrastive family it **does** have a cross-episode
supervisory path — the thing §3 of the occupancy check identified as
structurally absent. It is bounded by the authors naming
disappearance-reappearance as exactly where their pipeline fails.

**SA4D now replaces SGG as the highest-value remaining read**, because it
advertises a *time-varying* identity field, which is nearer the
time-windowed cell than SGG is.

## Membership-Measurement Update — 2026-08-23 (block 2)

**G13's representation limb now has BOTH necessary conditions measured,
and they separate cleanly**
([[operations/lrv3-membership-diagnostic-2026-08-23]], zero GPU-hours).

Timing was established recoverable EXACTLY. Membership is now measured at
the moment it binds and it is **precision 0.0446, recall 0.1786** on the
fresh seeding cloud — 336 rows gated, 15 correct, against 84 in-sphere.

**The apparent contradiction with the 96.9% substrate precision is
resolved, and the resolution is the durable part.** Membership is
estimated on a TRAINED cloud where densification has concentrated 10,650
rows onto the object, then re-applied through an absolute world-space
voxel grid to a FRESH 50,000-row cloud where the same cells are 95.5%
background. **An instrument validated on the cloud it was estimated from
can be an order of magnitude worse on the cloud it is applied to, and
nothing in the pipeline notices.** That generalizes past this fixture to
any membership transferred across a densification boundary.

**The binding constraint is a HARD CAP, not a tuning problem.** Only 15
of the fresh cloud's 84 in-sphere rows lie inside the two gated cells at
all. The other 69 are in the six cells the estimator abstained on, so no
geometric refinement can reach them. Measured: a 64³ occupancy grid
sourced from the gated cells improves precision **6×** (0.0446 → 0.2679)
and moves recall **not at all**. Every downstream repair therefore lands
in PARTIAL membership, the regime already measured as worse than both
alternatives (24.67 << 27.14 < 28.19).

**Fixing recall makes over-gating worse**: perfect group-level recall
caps precision at 0.0571, because the sphere occupies **6.6%** of the
volume of the 8 cells that cover it. Under-recall and over-gating are one
geometric defect, not two repairable limbs.

**Consequence for the route:** authored membership becomes a quantified
declared limitation rather than a pending repair
([[operations/paper-path-decision-2026-08-23b]]). T1's boundary output is
untouched and remains exact. **Boundary inference is solved on this
fixture; membership inference by spatial partition is refuted.**

## Measurement-Channel Update — 2026-08-23 (block 2)

**A finding that bounds every comparison in the N3V ladder family, and it
is not about any mechanism** ([[operations/b1f-flow-postmortem-2026-08-23]]).

Two arms differing by two config lines execute identical numerical code
until iteration 1000. In the pre-intervention window 501-999 they already
separate by **0.089-0.236 relative RMS training loss** — statistically
indistinguishable from their post-intervention separation. At
`densify_from_iter: 500` the divergence jumps **1400×**: a threshold
comparison flips, a clone/split decision differs, and the trajectories
part.

**So the substrate is chaotic at this protocol and densification is the
amplifier.** The recorded 0.635 dB global / 0.341 dB event-union seed
spread is a LOWER BOUND on the resolution floor, because same-seed
same-code runs separate comparably. **Any 6k/50-frame comparison must
state a same-code replicate floor, not only a seed spread.** No recorded
number is retracted; what changes is what two runs can be said to
resolve.

This makes the same-code replicate floor the **single next measurement**
for the whole N3V utility lane, upstream of every mechanism question.

Two premises are also corrected. The camera-swapped flow control is
magnitude-matched but only **~84% decorrelated** (13 of 14 swaps read the
physically adjacent camera; β ≈ 0.16 of the correct velocity survives
against 0.005-0.011 content-free) — the rejection stands, gate 3's FAIL
is not decisive. And on `cut_roasted_beef` the birth-gating "dynamic
mask" is a **top-15% photometric-residual quantile** for every training
camera, because real masks exist only for held-out `cam00` — so G8's
"flow-gated site selection is redundant here" premise is false on this
scene.

## Starvation-Fixture Update — 2026-08-23 (block 2)

**G14's observation-supply question is not answered, and the instrument
built to answer it is now known to be unable to**
([[operations/lrv4-lo-distribution-result-2026-08-23]], experiments
259/260, interpretation frozen in advance).

The pre-identified `lo`-distribution diagnostic ran and selected reading
(b): LRV4's one-recipient null is **not** a threshold artefact. `P2` is 8
rows against LRV3's 3,925, and support widths are **27× larger and
DISJOINT** from LRV3's recipients. 277 rows sit in the object region at
the return, but with median support width **25.6 s — the whole
sequence**.

**A one-frame return does not produce under-trained localized rows; it
produces no localized rows at all.** No floor rescues it. The
observation-supply mechanism claim stays UNTESTED, and a future fixture
must starve VIEWS while preserving a temporally localized return.

**Method carried forward:** the diagnostic's decisive column was support
WIDTH, frozen as load-bearing before any output. On `lo` alone the 7
below-floor rows would have read as a threshold artefact and the fixture
would have been re-run at a lower floor.

Updated 2026-08-08 (night) after Loop 2 ([[operations/elgs-method]]).
Earlier updates remain binding history.

## Timing-Inference Update — 2026-08-23

**G13's representation limb crosses the threshold that mattered most:
episode boundaries are recoverable from TRAINING VIEWS ALONE, to exact
frame accuracy** ([[operations/nonoracle-episode-timing-result-2026-08-23]],
Determined experiment 235). Every positive episodic-presence result to
date used AUTHORED boundaries, and the measured mistiming control put a
2-frame error at −2.39 dB — *below* not gating at all. So "can inference
be frame-accurate?" was the binding question, not "does gating help".

Measured: 417 candidate groups from a voxel grid over the trained
cloud's own bounding box; **2 gated, 415 abstained (99.52%)**; both
gated groups on the event object with 4-of-4 camera agreement; **onset
and offset both exactly right (30 and 57, 0 frames error)**; **zero false
activations**; 0.188 slot-hours. All 60 frames evaluated, so no boundary
was interpolated.

**The load-bearing design fact is what was NOT inherited.** The oracle
`region` is the event object's TRUE geometry, so an estimator that infers
only the gaps while inheriting that sphere is still oracle-supervised and
would have produced a fake success. Membership came from the cloud's own
bounds. Held-out cameras — the only ones carrying identity buffers —
were never touched, and the artifact records every anti-leakage check.

**Abstention needed no new mechanism**: `family_id = -1` already means
ungated, so an abstaining group keeps the ordinary temporal marginal
bit-for-bit. A mechanism whose errors are worse than inaction gets a
first-class way to decline, and it used it 99.5% of the time.

**Limits, recorded as limits:** recall is 2 of 8 event-overlapping
groups — a high-precision, low-recall instrument.

**PHASE T2 THEN RAN AND FAILED, and the failure is instructive**
([[operations/nonoracle-timing-t2-result-2026-08-23]]). Retraining with
the inferred program costs **−2.469 dB** on `event_return` against not
gating, matching the 2-frame mistiming arm's −2.386 dB — **while the
timing is exact to ~1e-14**. So the gate fires at the right times, helps
during presence (`event_episode1` 30.20 beats the ungated 29.60), renders
the absence (`ghost_gap` 23.88 near the oracle's 22.83) — and destroys
the return. **Fully gated 28.19 > NOT gated 27.14 >> PARTIALLY gated
24.67.**

**This adds a second necessary condition to G13's representation limb.**
The project had established that the mechanism needs frame-accurate
TIMING. It now has that timing can be inferred exactly and still leave
the mechanism HARMFUL, because per-row MEMBERSHIP is an independent
precision requirement of comparable severity. Imprecise membership, like
imprecise timing, has negative value. What is refuted is voxel-cell
membership at 8³, not non-oracle gating. And LRV3's absence is
genuine removal from the ray-trace, not occlusion, so the ablation signal
is a clean step that real data will not supply.

## Consolidation-Payload Update — 2026-08-23

**G14's identity-conserving-promotion limb gets a decisive negative, and
it is about the FIXTURE rather than the payload**
([[operations/payload-headroom-result-2026-08-23]], experiments 233/236).

With proposal ambiguity fully removed by an oracle-correct link, **no
transferable per-row quantity has material headroom on LRV3** —
appearance, opacity, temporal support width, position, extent and
orientation all sit at or below the same-identity floor. Position
DISCRIMINATES identity well (10.5-35.4) while having no headroom
(1.09), which is the clean signature of an instrument that detects
identity when identity is detectable and still finds nothing to recover.
The oracle-correct opacity edit was then measured **actively harmful
(−1.19 dB)**, not merely neutral.

**The generalization:** the 2026-08-20 DC falsification was not evidence
that appearance is the wrong payload — it was evidence about this
fixture. LRV3's returning surface is identical in pose, colour and
texture and is observed by 48 training view-frames, so the recipient rows
are wrong about nothing, and **headroom is a question about OBSERVATION
SUPPLY rather than about which tensor is carried.** That is an inference,
and it is falsifiable; an observation-starved fixture is what tests it.

**Per the frozen rule, the recommendation on the record is the
representation-only pivot.**

**New negative knowledge that generalizes past this experiment.** A
fresh-context adversarial review confirmed every number and returned
STANDS WITH QUALIFICATIONS; three of its findings are method-level:

* **A scale-free ratio screen is not sufficient.** "1.92× of 0.03
  logits" and "1.92× of 3 logits" are indistinguishable to it. The rule
  needs an absolute-magnitude floor.
* **A screen needs a non-degeneracy precondition.** `_t` passed the rule
  and is not a payload at all: copying a donor's temporal centre makes
  the recipient satisfy the DONOR's membership predicate, removing the
  row from the window the metric evaluates. That is deletion, not
  transfer — decidable from the row-set definitions with no measurement.
* **A placebo does not transfer across payloads.** The same-identity
  no-op is a genuine placebo for appearance under a nearest-appearance
  map and a REAL edit for any other tensor. Worse, it edits donor rows
  whose support ends before the scored frames, so it cannot attribute
  harm. The discriminating control is a within-recipient permutation —
  it WAS run and it OVERTURNED the attribution: destroying identity costs
  −0.97 dB at the same displacement and half the edit volume, so the
  −1.19 dB says nothing about identity. Damage tracks DISPLACEMENT, not
  incorrectness, which strengthens the payload negative: there is no
  regime in which redirecting opacity could help.

## Audit-Instrument Update — 2026-08-18 (block 4)

**G13's absence limb does not get its decision this block, and the reason is
worth more than the decision would have been.**

The two-stage M1-A0b audit — the only instrument that could convert the
DiVa-360 absence question from open to decided — reached revision 3 with its
sequence universe reconciled and its applicable-camera set adopted, and was
then **BLOCKED at round 2** by a fresh-context statistical review
([[operations/elgs-audit-prereg-2026-08-18]] REVIEW ROUND 2). **Stage 1 was
not executed; no candidate frame was rendered, displayed or transmitted.**

The reconciliation and the camera decision both stand, and are new durable
structure:

* the sequence universe is **20 coverage-population sequences → 18 eligible →
  12 with `N_s > 0` → `E_select` = 10**, and the simultaneous-bound divisor is
  **m = 10**. The "16 more eligible", the 10, and the 11-vs-12 were three
  different sets, not a contradiction;
* the applicable-camera set is **D3**, the sealed per-candidate frustum rule,
  adopted by the user. The estimand narrows to *unobservability across the
  cameras the frozen candidate generator geometrically considered applicable,
  followed by same-identity reappearance* — **not** physical absence, **not** a
  rig-wide set, **not** generator-independent supply.

**The new negative knowledge is about inheritance, and it generalizes past this
audit.** Two of the four blocking findings are cases of a preregistration
inheriting a sample and an instrument from an earlier frozen document, carrying
the sample's *identity* forward, and silently dropping the *design facts* that
came with it:

* **the sampling design was not carried.** The frozen 73-window sample is a
  round-robin over strata, 3 rounds, without replacement. Per-sequence
  Clopper-Pearson on `(k_s, n_s)` assumes equal within-sequence inclusion
  probabilities, which requires `3·N_s/n_s` to be an integer. **Verified: it is
  not, for five sequences including `pour_tea` and `tea` — two of the four that
  decide the outcome.** So the decision-bearing bound is not the confidence
  bound the kill rule treats it as, while the design-consistent
  Horvitz-Thompson estimator sits in the same file with no decision weight;
* **a disclosed bias was not carried.** The parent document discloses that
  audit frames are drawn over a window that *includes bridged frames, at which
  the identity is associated by construction of the flicker rule* — so a
  bridged window can never be A3-positive whatever the scene contains.
  Deflated A3 → **kill easier**, the wrongful-kill direction. **Verified: the
  word appeared zero times in either audit document.**

**The lesson to carry into every future instrument:** citing a frozen artifact
by name inherits its identity, not its assumptions. A preregistration that
reuses a sample must restate that sample's design, and must re-import every
disclosed bias, inside its own frozen text — or its estimator is being applied
to a design it has never seen.

Two further verified facts that bound what any repair can claim:

* **the finite-population omission is not a rounding matter.** `U_s` routinely
  exceeds the logically attainable maximum `k + (N − n)` — for `tea` at every
  `k`. Capping it moves the kill from **2,580 to 3,141 of 7,000**, a 21.7%
  shift, and `tea`'s published "6 of 9" threshold is unattainable in fact;
* **stage-1 decidability was never general.** It exists only because `scissor`
  and `poker` are excluded, on a frozen sensitivity reading sitting 0.003 and
  0.015 below a threshold. Verified in git that the exclusion is **not**
  post-hoc: the freeze commit predates every coverage figure.

**Not changed:** G-OA's FAIL is not reopened; no floor, threshold or estimand
moved; `scissor` and `poker` remain `indeterminate`; and the 0-of-597
corroboration figure stands exactly as it did. What changed is that the route's
*decision* is now blocked on statistics rather than on compute, and the block
is repairable.

## Dataset-Reachability Update — 2026-08-18 (block 4)

**ImViD's full release is reachable after all, and the previous block's
contrary finding is corrected rather than rewritten**
([[operations/dataset-admission-matrix-2026-08-18]] block-4 append). It is
world-readable with no credential: 325 files, 1.181 TB, verified by the primary
and enumerated two independent ways. But only **7 of 16 published takes** are
there (47.2% of the published bytes), `scene3_classroom` is missing a camera,
and a 122 GiB `moving_rig` folder is unaccounted for in the README's own total.
**This changes cost and reachability, not admissibility** — ImViD remains NOT
ADMITTED for event supply.

**Google Immersive's 46-videos-vs-45-calibrations gap is resolved against the
sealed artifacts**: the uncalibrated file is `camera_0046`, and `camera_0001` —
STG's held-out test view — is present and calibrated. So it is a per-scene
calibration failure whose index **moves between scenes**, and any visibility
ledger built by enumerating `camera_*.mp4` will mis-index differently on
different scenes. The dome's event-supply exclusion now carries a number: the
whole 46-camera rig spans ~50° of parallax at 1 m against a surround rig's
180°, so the multi-view corroboration the absence instrument needs is not
merely hard there — **it is undefined**.

## Coverage-Instrument Update — 2026-08-18

**G13's absence limb: the coverage floor that eliminated the two
richest sequences was an artifact of the falsified visibility gate.**
Retracts nothing; it changes which sequences the absence question can
still be asked about, and it sharpens a coupling this map already
recorded.

The 2026-08-14 entry above notes that `track_coverage_upper_bound` and the
absence limb share the `v >= 0.5` constant, so part of their
anticorrelation is an instrument identity. M-2 then showed `v` is BINARY,
which makes every threshold in `(0, 1]` equivalent and the
"lower-the-threshold" repair vacuous. The repair the evidence supports is
a component-membership gate, and its effect on coverage has now been
measured ([[operations/elgs-coverage-bounding-pair-result]], Determined
experiment 154, all five contract checks exact):

| sequence | frozen `v >= 0.5` | anchor-agreeing | any-report (upper) |
|---|---:|---:|---:|
| `scissor` | 0.441 | **0.852** | 0.916 |
| `poker` | 0.382 | **0.796** | 0.834 |
| `put_candy` | 0.507 | 0.727 | 0.861 |
| `pour_tea` | 0.591 | 0.710 | 0.795 |

So **the 0.5 coverage floor did not eliminate scissor and poker because
their foreground was untracked** — it eliminated them because a per-point
self-occlusion flag was being read as an association signal. Both clear the
floor by more than they previously fell short of it.

**They are nevertheless classed `indeterminate`**, because the frozen
convention-dependence rule demotes on a sensitivity crossing and the
transposed-anchor reading lands at 0.497 and 0.485. The rule was frozen
before the reducer existed and was not changed after the numbers; the
demotion is conservative and the transposed variant is diagnosed on the
page as a null rather than a rival convention (it admits 4.5-11x fewer
reports).

**The consequence is structural and it is new knowledge about the route,
not about the scenes.** Excluding scissor and poker removes 452 of the 597
candidate windows, and that exclusion is the only reason the amended
audit's stage-1 kill rule is decidable at all: with scissor, poker and
pour_tea all admitted, no possible audit outcome can fire the kill, because
`U_s = N_s * UB_s` scales with the candidate count while the bound is
floored by the sample size ([[operations/elgs-audit-prereg-2026-08-18]]
section 4). **A larger candidate population makes the kill harder.** That
is a preregistered tripwire now, and it means any future rule that admits
those two must re-specify stage 1 first.

**Nothing here reopens G-OA's FAIL, changes a floor, or admits any sequence
to evidence use.** Admission requires a fresh preregistration under a
corrected instrument.

**One further open item, recorded because it bounds every A3 and A_S count
the audit could produce:** the audit's presence decoys bound the
instrument's false-positive side only, and nothing estimates its miss rate
on genuine full-view absence, because no DiVa-360 window is known to be
absent. That is the specific hole
[[operations/kubric-testbed-scope-2026-08-18]] scopes, and it is not
closable on real data.

## Renderer Update — 2026-08-18

**The admitted image is reproducible run-to-run; the old one was not.**
Three runs of the repaired kernel at fixed seed agree to **3.3e-4 dB** of
held-out PSNR, against **0.10 dB** (same metric) and **0.36 dB**
(training-log metric) between two old-image runs
([[operations/renderer-integrity-admission-2026-08-18]] Appendix C). This
resolves that page's open question in the direction of the repair, and it
bounds the `atomicAdd` explanation empirically: if float-summation order
were the dominant source here, the repaired image would vary similarly.

Consequence for every lane: single-run-per-arm comparisons **on the
admitted image** no longer inherit the 0.36 dB penalty in that
configuration. The figure does not transfer to 15k DiVa-360 runs, and the
matched presence spec still measures its own spread.

## Measurement-Closure Update — 2026-08-14

**G13's ABSENCE limb loses its measured supply; its OCCLUSION limb keeps
it.** A frozen, four-times-reviewed diagnostic over all 597 corrected
DiVa-360 tranche-1 true-absence windows returned **status_2 (material
defect), UNANIMOUS across 144 sensitivity readings**
([[operations/elgs-absence-diagnostic-result]]):

- **ZERO of 597** scored true-absence windows are corroborated as genuine
  full-multiview disappearance — zero pooled and zero in every one of the
  twelve sequences.
- **96.6%** are windows where an eligible foreground component sustained
  multi-view-consistent occupancy of the instrument's own frozen anchor
  while the tracker's report failed to qualify: **87.6%** of the evidence
  is a per-point visibility flag below 0.5, **12.2%** is cameras in the
  applicable set that were never queried and can never associate. The
  tracker never LOST the point (C2 = 0 everywhere).
- Because `track_coverage_upper_bound` uses the SAME `v >= 0.5` threshold,
  coverage and absence are coupled through one constant — part of the
  measured coverage/absence anticorrelation (r = -0.765) is an instrument
  identity, not a scene fact. Occlusion, which requires association in
  >= 2 cameras (tracking WORKING), is barely coupled (r = -0.178).

**New negative knowledge.** A tracker visibility flag is a per-point
self-occlusion signal, not an existence signal: on a surround rig a surface
point can be self-occluded in every queried camera while the object is
plainly present. Any future presence/absence instrument must separate
"unobserved" from "absent" by evidence that does not reduce to the tracker's
own confidence, and must not let its applicable-camera set include cameras
it never queried.

**Not concluded:** that the objects were physically present. C2/C3 cannot
separate "component still there, untracked" from "identity left and the
manipulating hand covers the vacated site". Only the frozen M1-A0b audit
can, and its 73-window stratified sample has been emitted but NOT run.

G14/CC4 is unchanged by this: it already had zero measured supply
([[operations/elgs-substrate-remeasurement-result]]). G9's
uncertainty/abstention need is sharpened — the instrument had no abstention
class at all until this diagnostic introduced one.

## Loop-2 Update - 2026-08-08 (night)

Under user-relaxed constraints (external priors; any public dataset;
per-scene fixed), three verified sweeps ([[operations/loop2-sweep-2026-08]])
found: (i) tracker visibility states have NEVER been consumed as
representation-level presence/identity (MoSca/SoM loss masks only);
(ii) no existence inference with an observation model exists in
differentiable rendering (CIF nearest, segmentation-scoped); (iii) the
occlusion-order+memory conjunction remains unoccupied even on
surround/ego rigs (ST-NeRF verified to lack persistent hidden state;
no GS successor); (iv) non-rigid permanence through occlusion on
surround capture and egocentric hidden-state dynamic GS are open;
(v) DiVa-360 is the event-dense benchmark with no GS baselines.
EL-GS occupies (i)+(ii) with the LGS substrate — G13/G14 now have a
candidate occupant with a calibrated 8.0 conditional novelty; the
remaining gate is formal (v8 write-out + one fresh adversarial round).
G9 (uncertainty/occlusion confidence) is directly addressed by the
censored-evidence ontology (conditional claims, no calibration).
New negative knowledge: track-state⇒existence naive mappings are
invalid measurement semantics; e-process validity claims for adaptive
structural acceptance are unsupportable; per-segment bridge selection
creates chimera evidence (all ledgered in
[[operations/elgs-review-history]]).

## Post-Representation-Run Update - 2026-08-08 (evening)

Five verified representation-level sweeps + nine deep-dives + five
fresh-context adversarial rounds + a calibrated novelty check
([[operations/repr-sweep-2026-08]], [[operations/lgs-novelty-record]])
tightened the map:

- **G13 (visibility events)**: the representation-level boundary is now
  precise: per-primitive multi-interval/reactivating presence, latched
  presence, per-primitive changepoints, and exact compact-support
  absence are VERIFIED UNOCCUPIED across all eight dynamic-GS families;
  discrete lifecycle exists only in streaming methods and always births
  NEW rows. [[operations/lgs-method]] occupies this slice (selected
  candidate, 6.5/10 novelty, awaiting user decision). Near misses to
  cite: Ex4DGS (single flat-top), CTRL-GS (scene-global segments),
  TOM-GS (presence-only single bump), AD-GS/TRiGS (single window),
  CLOTH-HUGS (order without memory), CIF (occupancy × semantic
  identity), PersistGS/4DPM/MoPe (pose/log-odds permanence), CubifyGS
  (frozen assets, discrete maintenance), TSA (2D slot activation).
- **G14 (identity-conserving promotion)**: reactivation-with-own-content
  is confirmed unoccupied AND mechanistically opposite to the entire
  relocation/respawn family (3DGS-MCMC donor-clone overwrite verified
  at code level; FreeTimeGS++ 2605.03337 ablates that family). LGS's
  lineage tying + reactivation is the candidate occupant; its
  irreversible-fragmentation limitation (no merge) is recorded.
- **G5 (capacity)**: counterfactual trial-render structural acceptance
  is unoccupied (closest: L2D2-GS 2606.29374, offline policy reward);
  it enters LGS as supporting machinery only.
- **New negative knowledge** ([[operations/rejected-representations-2026-08]]):
  occlusion-order layer stacks are rig-hostile here (P03); ratio-based
  description-cost economics is undefined/gameable as a principle; soft
  content assignment over a candidate library has no coherent geometry
  under migration and contradicts strict scalar caps; same-thread refine
  approval again failed to predict fresh-context survival.

## Post-Method-Discovery Update - 2026-08-08

Five verified literature sweeps + nine paper deep-dives + four
fresh-context adversarial reviews substantially tightened the map:

- **G5 (capacity allocation)**: [[operations/star-gs-v9-method]] is the
  preserved training-side candidate on this axis (deficit-carved,
  budget-neutral spacetime birth) — NOT the approved lead direction; the
  next phase is representation-level discovery. Its test plan is
  preserved at [[operations/star-gs-v9-experiment-plan]], review record
  at [[operations/star-gs-v9-review-history]], sweep evidence at
  [[operations/sota-sweep-2026-08]]. Occupied neighbors verified at mechanism level:
  CEC-4DGS ([[papers/kang2025_cec_4dgs]]) = error-driven time-local 4D
  birth at single-view rendered depth (unbudgeted); FreeTimeGS = periodic
  budget-neutral relocation to existing high-score regions; SharpTimeGS
  stage-2 = fixed-count error/motion densification; TAD-GS +
  [[papers/cho2026_4d_scaffold_gs]] = presence-weighted statistics. The
  residual open slice: depth-free multiview deficit localization +
  audited budget accounting + causal/event validation.
- **G13 (visibility events)**: WildRayZer (CVPR 2026 Highlight) occupies
  learned transient-mask gradient gating; [[papers/mazur2026_4dpm]]
  (CVPR 2026 Oral) occupies primitive permanence via motion extrapolation
  (monocular, rigid). Optimizer-level "protection" approaches were
  examined and rejected this run ([[operations/rejected-approaches-2026-08]]).
- **G7 (evaluation)**: externally corroborated — ViDAR
  ([[papers/nazarczuk2025_vidar]]) quantifies co-visibility-mask static
  bias (mean 26% dynamic pixels) and establishes -D dynamic-mask metrics;
  TAD-GS's M-PSNR is precedent. The field has no standard temporal
  metric (tOF/tPSNR borrowed ad hoc) — adopt, don't invent.
- **New negative knowledge** (review-derived, recorded in the rejected
  ledger): per-primitive optimizer-timescale interventions are causally
  unidentified pre-experiment and collide with sparse/selective-Adam
  tooling; residual images carry no cross-view correspondence signal;
  time-shift permutation nulls lack exchangeability for nonstationary
  video; static-scene densification theory now has three distinct
  accounts (SteepGS saddle points, GDAGS direction coherence,
  Structure-Aware aliasing) — cite, don't re-derive.

## Post-Stage-1 Update - 2026-07-29

Stage 1 of CSVL-VPL was executed on 2026-07-26 and returned three no-gos
([[operations/phase9-csvl-vpl-stage1-result]],
[[operations/phase9-csvl-vpl-stage1b-result]],
[[operations/phase9-csvl-vpl-stage1c-result]]). This falsifies the 2026-07-25
statement below that "the first unresolved gap is uncertainty-bearing temporal
surface association and abstention": the sealed P03 evidence layer contains
zero front/rear cross-order candidates in all 19 scanned windows, so there is
nothing for a temporal association to associate. The binding constraint is the
evidence representation itself — multilayer bin occupancy requires multi-camera
co-support of two depth layers in one bin, which a frontal rig almost never
produces (93.4% of 3.07M bins rejected for insufficient camera co-support).

The approved direction is [[operations/phase9-csvl-vpl-v2-direction]]:
primitive-centric reprojection visibility (E1/E1-int/E2) replacing the P03 bin
route, a from-scratch lifecycle that never touches rendered opacity, a restored
oracle-capacity attribution lane, and an evidence-opportunity census (Phase 0)
gating all further evidence investment. G9 and G13 remain the target gaps; the
first unresolved question is now empirical opportunity abundance in the
primitive-centric representation, not association design.

Additional negative evidence recorded for G13: the Stage-1 association scored
camera-swapped flow above valid flow; flow was non-causal for its output; and
the R034 synthetic fixture (AUC 1.0) predicted nothing about real admission
(R035 accepted 0/72). Fixture passage must never again be a Go criterion.

## Post-B01 Update - 2026-07-25

The corrected 256-slot B01 continuation is an operator-stability control, not
mechanism evidence: global PSNR improved by only `+0.048315468 dB`, dynamic-mask
PSNR by `+0.011161912 dB`, and static PSNR by `+0.055157407 dB`. It used an
event-blind target rule and therefore did not test calibrated visibility-guided
allocation.

The selected direction is [[operations/phase9-post-b01-csvl-vpl-direction]]:
CSVL-VPL, a calibrated surface visibility ledger coupled to a surface-owned
primitive lifecycle. This refines G5, G9, G13, and G14 together. Fixed-count
reassignment remains a matched-count control and optimizer-safe transaction
substrate; generic extra capacity is mandatory as the capacity control.

The sealed P03 artifact supplies calibrated multilayer opportunity evidence but
does not propagate persistent surface identity. The first unresolved gap is
therefore uncertainty-bearing temporal surface association and abstention, not
another capacity intervention.

Implementation-level novelty pressure is stronger than the earlier paper-only
map: temporal-visibility densification, opacity modulation, proxy-guided growth,
multi-bank promotion, and layered representations all have close precedents.
The narrower open hypothesis is calibrated non-rigid front/rear surface identity
plus abstaining evidence and controlled surface-owned lifecycle changes.

## Tournament Update - 2026-06-30

The selected ADAGS direction is [[ideas/self-calibrated-prior-reliability-field]]. It addresses G1/G2/G3/G7/G8/G9/G11 by making reliability the mechanism that decides where masks, flow, tracks, static exclusion, and detail priors may act. [[ideas/boundary-aware-static-anchor-negative-space]] is the deterministic baseline/static-leakage safeguard. [[ideas/adags-failure-atlas-mechanism-screen]] is the backup and required reporting frame.

New blocking gaps:

- Reliability must be calibrated as an error/usefulness predictor, not just a mask recipe.
- Evaluation masks must be independent enough to avoid circular validation.
- Reliability must retain hard dynamic-core pixels; easy-pixel selection invalidates the method.
- Wins must be shown against LoRA route0 under matched realized budget and static-quality preservation.

## Problem-First Redo Update - 2026-06-30

The problem-first redo deliberately treats ADAGS as prototype infrastructure, not the method boundary. It demotes reliability-gated priors to the safe ADAGS fallback and elevates representation-level questions that still remain after Multi4D, RiGS, SharpTimeGS, AdaGaR, MAPo, PaMoSplat, MoE-GS, USplat4D, Ground4D, MoSca, and Prior-Enhanced GS.

New high-upside candidate directions:

- [[ideas/event-causal-visibility-gaussians]]: visibility events for occlusion, disocclusion, birth, split, merge, and retirement.
- [[ideas/identity-conserving-detail-carriers]]: parented transient detail carriers that preserve identity while recovering high-frequency motion detail.
- [[ideas/frequency-adaptive-temporal-support]]: temporal support bandwidth tied to dynamic frequency/detail and uncertainty.
- [[ideas/counterfactual-prior-usefulness-routing]]: route priors by estimated downstream usefulness, not confidence alone.

New blocking gaps:

- G13: occlusion/disocclusion are still often modeled as smooth deformation or implicit lifespan effects rather than causal visibility events.
- G14: dynamic detail can be recovered by transient capacity, but identity-preserving promotion/demotion rules remain underdeveloped.
- G15: prior confidence is not the same as prior usefulness; the field lacks counterfactual tests for when masks, tracks, flow, depth, or geometry priors should be trusted.

## G1 - Dynamic-Region Sharpness Needs A Direct Objective

Global PSNR can hide the failure ADAGS cares about: food, hands, and heads remain smeared even when full-image metrics improve. Recent papers make this gap explicit: MAPo targets blurred high-dynamic regions, SharpTimeGS targets sharp and stable temporal visibility, PaMoSplat targets substantial intricate motions, AdaGaR makes high-frequency dynamic detail explicit, and Multi4D frames the tradeoff between oversmoothed deformation fields and overparameterized 4D primitives.

Status: open
Priority: high
Literature pressure: [[papers/jiao2026_mapo]], [[papers/liao2026_sharptimegs]], [[papers/deng2026_pamosplat]], [[papers/chan2026_adagar]], [[papers/wang2026_multi4d]], [[papers/jiang2024_motiongs]]
Related ideas: [[ideas/dynamic-mask-static-exclusion]], [[ideas/rendered-flow-gated-supervision]], [[ideas/dynamic-region-diagnostic-benchmark]]

## G2 - Static/Dynamic Leakage Is A Representation And Evaluation Problem

Static/dynamic separation is no longer novel by itself. SWinGS has static/dynamic weighting, SplatFlow decomposes static background and dynamic objects, 4DGS-SLAM classifies static and dynamic Gaussian sets, SharpTimeGS uses temporal lifespan to balance long-lived static and short-lived dynamic regions, Hybrid 3D-4DGS uses distinct static/dynamic representation capacity, and RiGS explicitly separates coherent rigid transformations from residual deformation. ADAGS needs a more precise claim around reducing static-branch ghosting under reversible routing.

Status: open
Priority: high
Literature pressure: [[papers/shaw2024_swings]], [[papers/sun2025_splatflow]], [[papers/li2025_4dgs_slam]], [[papers/liao2026_sharptimegs]], [[papers/oh2025_hybrid_3d_4dgs]], [[papers/wu2026_rigs]], [[papers/wang2026_flow4dgs_slam]]
Related ideas: [[ideas/dynamic-mask-static-exclusion]], [[ideas/static-anchor-negative-space]]

## G3 - Long-Range Tracks And Depth Priors Are Becoming Table Stakes

MoSca, Shape of Motion, and Prior-Enhanced GS all use long-range tracks and/or depth/foundation priors. ADAGS already has a track-flow hook, but current configs leave `lambda_track_flow: 0.0`. A publishable method needs either to activate and improve this path or explain why a lighter alternative works.

Status: open
Priority: high
Literature pressure: [[papers/lei2025_mosca]], [[papers/wang2025_shape_of_motion]], [[papers/shih2025_prior_enhanced_gs]]
Related ideas: [[ideas/track-prior-scaffold-motion]], [[ideas/rendered-flow-gated-supervision]]

## G4 - Scaffold Residual Motion Is Crowded By MoSca And Prior-Enhanced GS

Plain "motion scaffolds for dynamic Gaussian reconstruction" is occupied territory. MoSca uses 4D motion scaffolds, and Prior-Enhanced GS adds scaffold-projection loss tying motion nodes to tracks. ADAGS should not pitch scaffold residual motion alone; the gap is a lighter reversible LoRA plus scaffold variant with diagnostic proof, or a training-only prior/flow-gated version.

Status: open
Priority: high
Literature pressure: [[papers/lei2025_mosca]], [[papers/shih2025_prior_enhanced_gs]]
Related ideas: [[ideas/track-prior-scaffold-motion]], [[ideas/rendered-flow-gated-supervision]]

## G5 - Capacity Allocation Must Be Matched And Dynamic-Aware

HiCoM, SharpTimeGS, SpeeDe3DGS, MAPo, Hybrid 3D-4DGS, Disentangled4DGS, and Multi4D all treat dynamic capacity, temporal pruning, grouping, partitioning, or representation allocation as central. ADAGS fixed-budget screens are useful, but realized point counts, dynamic-region point density, and high-frequency/detail retention must be audited before claiming allocation gains.

Status: open
Priority: medium-high
Literature pressure: [[papers/gao2024_hicom]], [[papers/liao2026_sharptimegs]], [[papers/tu2026_speede3dgs]], [[papers/jiao2026_mapo]], [[papers/oh2025_hybrid_3d_4dgs]], [[papers/feng2025_disentangled4dgs]], [[papers/wang2026_multi4d]]
Related ideas: [[ideas/motion-aware-densification-budget]]

## G6 - Single Global Motion Models Are A Known Weakness

MAPo partitions high-dynamic Gaussians, MoE-GS routes to specialized experts, PaMoSplat uses part-aware motion, HiCoM uses hierarchical coherent motion, RiGS separates rigid transforms from residual deformations, MotionScale scales motion/geometry reconstruction, Multi4D uses multi-level competitive allocation, and the SE(3) B-spline paper models continuous motion with explicit bases. ADAGS LoRA route0 is stable, but the novelty gap is specialized motion without losing stability.

Status: open
Priority: high
Literature pressure: [[papers/jiao2026_mapo]], [[papers/jin2026_moegs]], [[papers/deng2026_pamosplat]], [[papers/gao2024_hicom]], [[papers/wu2026_rigs]], [[papers/zhou2026_motionscale]], [[papers/wang2026_multi4d]], [[papers/zhang2026_continuous_motion]]
Related ideas: [[ideas/part-aware-reversible-routing]], [[ideas/track-prior-scaffold-motion]]

## G7 - A Benchmark/Diagnostic Claim Is Necessary

MonoDyGauBench argues monocular dynamic Gaussian results are brittle and scene-dependent, and it standardizes apples-to-apples comparisons. D4RT also raises the speed/generalization baseline outside per-scene optimization, while MotionScale and Mono4DGS-HDR remind that geometry, motion, exposure, and photometric artifacts can be mixed together. ADAGS should report dynamic-mask PSNR, static ghost score, track-flow error, edge/sharpness proxies, realized point count, and qualitative panels, not just global PSNR.

Status: open
Priority: high
Literature pressure: [[papers/liang2025_monodygaubench]], [[papers/zhang2025_d4rt]], [[papers/zhou2026_motionscale]], [[papers/liu2025_mono4dgs_hdr]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]]

## Renderer note (2026-08-18) — read before interpreting any flow or gradient gap

G6 and G8 below are unchanged as SCIENTIFIC gaps: making the rendered-flow
gradient live does not gate it for reliability, and the track-flow hook is
still inert at `lambda_track_flow: 0.0`.

Two engineering facts now bound how any gradient-based result in this
repository may be read ([[operations/renderer-integrity-admission-2026-08-18]],
[[operations/rasterizer-backward-two-defects-2026-08-17]]):

* until 2026-08-18 the ACTIVE backward render kernel gated itself on
  UNINITIALISED device memory, so its behaviour depended on allocator
  history rather than on the scene. No prior run's gradients are known
  reproducible. This does NOT establish that any recorded result is
  wrong, and the reproducibility bound measuring the old image's own
  spread is recorded on the admission page.
* rendered-flow supervision was non-functional before 2026-08-18 — the
  VJP lived in a kernel that was never launched. It is now live and
  correctly routed. That closes an INSTRUMENT blocker, not a gap: the
  flow VJP's numerical correctness is still unestablished, and per the
  2026-08-18 decision memo flow supervision is recommended to stay
  shelved because no EL-GS claim has a flow term and the primary dataset
  has no flow.

Nothing below is retracted on this basis. The historical flow-lane
readings cited in G8 used the track-flow path, not rendered flow, so they
are not affected by the rendered-flow repair.

## G8 - Flow Supervision Needs Reliability Gating

MotionGS, PaMoSplat, SplatFlow, and Flow4DGS-SLAM all support explicit flow/motion guidance in some form, but ADAGS W&B suggests naive flow lanes underperform while render-gated flow looks more plausible. The gap is not "add flow"; it is robustly gating flow to reliable dynamic cores and boundaries.

Status: open
Priority: high
Literature pressure: [[papers/jiang2024_motiongs]], [[papers/deng2026_pamosplat]], [[papers/sun2025_splatflow]], [[papers/wang2026_flow4dgs_slam]]
Related ideas: [[ideas/rendered-flow-gated-supervision]]

## G9 - Uncertainty And Occlusion Confidence Are Underused In ADAGS

USplat4D shows uncertainty can improve monocular 4D reconstruction and motion tracking. ADAGS masks, residuals, tracks, and flow losses currently lack a principled confidence model for occlusion, disocclusion, and mask noise.

Status: open
Priority: medium
Literature pressure: [[papers/guo2026_usplat4d]]
Related ideas: [[ideas/rendered-flow-gated-supervision]], [[ideas/track-prior-scaffold-motion]]

## G10 - Practical N3V Cooking-Scene Niche Is Still Available

Many recent methods target autonomous driving, compression, SLAM, HDR/low-light, sparse multi-view capture, or general monocular reconstruction. ADAGS can still own a narrower claim if it demonstrates fast cooking-scene motion improvement under fixed budgets and training-only priors.

Status: open
Priority: medium
Literature pressure: [[papers/sun2025_splatflow]], [[papers/song2025_coda4dgs]], [[papers/kumar2026_l2dgs]], [[papers/aerodgs2026]], [[papers/zhou2026_4c4d]], [[papers/liu2025_mono4dgs_hdr]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]], [[ideas/dynamic-mask-static-exclusion]]

## G11 - Representation Frequency Is A New Sharpness Axis

AdaGaR, Multi4D, and frequency-oriented dynamic reconstruction framing make dynamic blur a representation-frequency problem, not just a missing loss or bad mask. ADAGS currently logs dynamic edge magnitude, but it does not yet claim or evaluate frequency/detail preservation directly.

Status: open
Priority: high
Literature pressure: [[papers/chan2026_adagar]], [[papers/wang2026_multi4d]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]], [[ideas/motion-aware-densification-budget]]

## G12 - Feedforward 4D Models Raise The Baseline

D4RT-style feedforward 4D reconstruction changes the comparison space for speed and generalization. ADAGS can still be valuable as a per-scene, fixed-budget, diagnostic-driven method, but should avoid broad "fast 4D reconstruction" claims unless compared against amortized 4D baselines.

Status: open
Priority: medium-high
Literature pressure: [[papers/zhang2025_d4rt]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]]

## G13 - Visibility Events Are Not Smooth Deformation

Many dynamic GS methods improve motion smoothness, lifespan, partitioning, or transient capacity, but occlusion and disocclusion are event-like changes. Treating them as smooth deformation can create boundary blur, ghost trails, and flicker.

Status: open
Priority: high
Literature pressure: [[papers/wang2026_multi4d]], [[papers/wu2026_rigs]], [[papers/liao2026_sharptimegs]], [[papers/zhao2026_ground4d]]
Related ideas: [[ideas/event-causal-visibility-gaussians]]

Negative evidence: R017 actual opacity gating, R025 non-oracle candidate-local refinement, and R027 non-oracle boundary-gated micro-densification all failed the frozen R009 event-crop gate. R027 produced only small directional gains over route0 (`+0.0569 dB` PSNR, `-0.0000903` L1) and recovered less than 1% of the oracle crop upper bound. R028 posthoc audit found the R026 boundary support essentially missed the frozen crops. R029 route0 continuation worsened route0, so R027's tiny positive movement was not generic continuation. R030 oracle-support micro-densification also failed with mean PSNR `29.9021`, mean L1 `0.0158770`, and `0/5` route0 PSNR+L1 wins. This preserves the visibility-event gap but rejects support-only continuation of the current posthoc micro-densification recipe.

## G14 - Detail Needs Identity-Conserving Promotion Rules

Persistent primitives preserve correspondence but can oversmooth high-frequency detail. Transient capacity can sharpen detail but risks fragmenting identity. The missing piece is a promotion/demotion rule that says when detail should remain attached, become new geometry, or retire.

Status: open
Priority: high
Literature pressure: [[papers/wang2026_multi4d]], [[papers/wu2026_rigs]], [[papers/chan2026_adagar]], [[papers/jin2026_moegs]]
Related ideas: [[ideas/identity-conserving-detail-carriers]], [[ideas/frequency-adaptive-temporal-support]]

## G15 - Prior Usefulness Needs Counterfactual Calibration

Masks, tracks, flow, depth, and geometry priors can be confident but wrong near occlusions, boundaries, static/dynamic leakage, or out-of-distribution geometry. A routing field should estimate whether trusting a prior improves future reconstruction, not merely whether the prior appears confident.

Status: open
Priority: high
Literature pressure: [[papers/guo2026_usplat4d]], [[papers/shih2025_prior_enhanced_gs]], [[papers/zhao2026_ground4d]], [[papers/sun2025_splatflow]]
Related ideas: [[ideas/counterfactual-prior-usefulness-routing]], [[ideas/self-calibrated-prior-reliability-field]]

## Oracle-Headroom Update — 2026-08-19

**G13's representation limb got its instrument this block, and two of the three
things that went wrong were matching failures of a kind worth naming.**

The K=1 four-cell experiment is CLOSED and was correctly closed: `softmax` over
a one-element vector is identically `[1.0]` with gradient exactly `0.0`, so that
cell measured the cost of deleting the temporal marginal rather than the cost of
a presence representation. `K` turned out to be a hardcoded literal at the
seeding call with **no configuration surface at all**, and the only runtime path
to `K >= 2` was a heuristic FISSION. `elgs_oracle_episodes`
([[lrv1-oracle-headroom-spec-2026-08-19]]) supplies a fixed program instead,
leaving the preregistered seeding granularity and the frozen prereg untouched.

**Established: a non-degenerate `K=2` EL-GS cell trains stably.** The
admitted-image preflight crossed iteration 600 — where densification first
fires and where the `elgs_a` optimizer-group defect used to crash both the
clone and the prune path — and the cloud went 50,000 -> 43,977 -> 74,590, so
both paths ran repeatedly against a non-per-point parameter group without
raising. That was the single largest implementation risk and it is now retired.

**NOT established: any headroom.** No interpretable A1-vs-A0 number exists yet.

### The three lessons, in decreasing order of how far they generalize

**1. A dose-matched control is not automatically a RAMP-matched one.** The
wrong-time control was matched on episode count, `dim(a)`, every episode
duration, the gap duration and the total present duration — and a test asserted
each of those. None of them constrains where the *smoothstep edge band* falls.
Placing episode 2's onset at the midpoint between the last absent and the first
present frame — the obvious choice — put presence at `0.15625` on the first
evaluated frame and `0.84375` on the second, while the control sat at `1.0` on
all six. **The decisive metric was biased against the hypothesis by 7-21 dB by
construction.** Caught by a fresh-context review, primary-verified, repaired
before any output was read.

The generalization: **matching is only as good as the enumeration of what was
matched.** When a comparison is declared "matched on X, Y, Z", the question to
ask is not whether X, Y and Z are equal but what *else* differs — and
especially whether anything the metric window touches differs. Here the metric
window was six frames long and the unmatched quantity lived inside it.

**2. If the initialization does not cover the visible surface, held-out numbers
measure floaters rather than the method.** The first synthetic fixture put a
ground plane out to 3.0 against an initialization cloud spanning `[-1.3, 1.3]`,
leaving **13.94% of every image** on surface with no primitive anywhere near
it. Densification clones and splits *existing* primitives; it cannot create
them from nothing. The optimizer filled that band with floaters that satisfy
the 16 training views and fail from the 4 held-out ones, and the control came
out at **34.23 dB on training views against 19.31 dB held out**.

The generalization: **a large train/held-out gap on a scene that should be easy
is an initialization-coverage symptom before it is anything else**, and it is
cheap to test directly — ray-cast the scene and measure the fraction of visible
surface outside the init bounding volume. On any authored testbed this should
be checked before the first training cell, not after.

**3. The event-admission gate did its job, and that is the point of having
one.** LRV1 failed it on the reconstructibility floor (`event_episode1`
23.289 dB against 25.0), so no representation verdict follows from any A1
number on that scene — and the A1 cell was cancelled 45 minutes into a 2.4 hour
run rather than spend the compute on an uninterpretable number. Both thresholds
were fixed before any output existed, in response to a review finding that the
two gate items had been written as the *same predicate*. Neither was moved
afterwards.

### A protocol correction that outlives this block

[[stg-n3v-protocol-parity-2026-08-19]] records, primary-verified in source,
that this repository's recorded `cut_roasted_beef` figure of 34.366 dB was
measured at **676x507** — `n3v2blender.py` halves the native frames on disk and
`ModelParams.resolution: 2` then halves them again — against the 1352x1014 at
which Spacetime Gaussians publishes 33.52. The "inside the competitive band"
reading in [[sota-sweep-2026-08]] is **not supported**. Separately,
`utils/image_utils.py` is called with an unbatched `(3, H, W)` at both ADAGS
eval sites, so the reported PSNR is the mean of three per-channel PSNRs rather
than the pooled PSNR; the bias is `10*log10(AM/GM)`, always non-negative, and
measured at 0.268 dB on a real run.

### What is still PAUSED, and what would restart it

DiVa claim-grade instrumentation remains paused. Nothing this block produced
evidence of representational headroom, and the argument for pausing was that
evidence-acquisition investment should follow a demonstration that the
representation can use the evidence. **That argument is unchanged.** What would
restart it is a valid, gate-passing A1-vs-A2 result showing episode-timing-
specific headroom — which is exactly what the LRV2 matrix is for.

### DECISIVE RESULT (2026-08-19) — the representation was handed perfect evidence and got worse

On the admitted LRV3 event, with capacity matched to within 34 primitives at
the cap, frozen oracle boundaries, structural rounds off and verified
checkpoint/evaluator provenance
([[operations/lrv1-oracle-headroom-spec-2026-08-19]] RESULT PART 5):

```
D1 = A1(correct oracle) - A0(temporal) =  -5.2316 dB   on event_return
D2 = A2(wrong-time)     - A0           = -17.1619 dB
A1 - A2                                = +11.9303 dB
```

**Correct fixed `K=2` episode structure did not improve event reconstruction —
it cost 5.23 dB.** An order of magnitude past the 0.5 dB decision floor, in the
wrong direction, so this is a result rather than a near-miss.

**G13's representation limb therefore has its first real answer, and it is
negative — but the deficit is not where the hypothesis predicted.** On the same
object surface during 30 frames of *continuous presence*, where the correct
oracle's presence is a constant 1.0 and the episodic machinery is doing
nothing, A1 is still **3.16 dB** behind; globally it is **0.48 dB** behind. Most
of the event-region deficit is a **fixed cost of the representation swap**, not
a failure at the return. The two mechanisms were named in review *before* the
cells ran: EL-GS replaces the temporal marginal for **every** primitive (A1 gave
up a learnable temporal lobe on 149,000 primitives to gain episodic presence on
~780), and the oracle is a **voxel-cell** oracle ~8x the object's volume, so it
gates background off with the object — visible as `ghost_gap` losing 6.56 dB.

**What is cleanly established: episode timing matters enormously inside the
representation.** A2 sits at 11.70 / 11.76 / 11.81 dB per return frame against
the scene's independently computed floor of **9.7487 dB** — it barely
reconstructs the return at all. Its mechanism is legible: 27 frames of true
absence outvoted 3 of presence and drove its object primitives transparent, and
its `ghost_gap` of 28.66 dB (against A0's 28.97) confirms it fixed the gap by
destroying the object. That is the +11.93 dB.

**Consequence for the route.** DiVa claim-grade evidence acquisition stays
**PAUSED**, and this strengthens rather than merely maintains the case: the
representation has now been handed *perfect* evidence — exact boundaries, on an
admitted event, at matched capacity — and was worse than not having it.
Acquiring imperfect evidence for it is not the next thing to fund.

**The next experiment is specific, cheap and pre-identified**: keep the
per-primitive temporal marginal for non-oracle families so the swap is local to
the primitives that need it; make the oracle per-primitive rather than
per-voxel-cell; and add the small-mistiming control that separates "timing
precision matters" from "a hard gate matters". Only after those does a negative
become a statement about the representation rather than about its wiring.

## Localized-Presence Update — 2026-08-20

**G13's representation limb has its first POSITIVE, and the 2026-08-19
negative is now fully attributed to wiring.** The pre-identified follow-up
above was run as a corrected cell
([[operations/lrv3-local-presence-corrected-cell-2026-08-20]], experiments
184/185, rules frozen before output): localized presence (non-oracle rows
keep the temporal marginal), per-primitive oracle membership (~84 rows, 8
families), a TOTAL opacity gate (the static twin — which the 2026-08-19
code audit found bypassing presence entirely — is now gated too), and
routing pins off.

```
event_return   A1-LOCAL − A0′ = +1.0496 dB   (floor 0.5, matched capacity,
                                              A1-LOCAL 1,126 primitives FEWER)
event_episode1                 = +1.24 dB
ordinary_all                   = −0.39 dB    (within the 0.5 bound)
first return frame             = +2.0 dB
vs the 2026-08-19 global A1    = +6.15 dB at the return
```

**The per-frame ghost diagnostic showed the total gate rendering EXACT
absence — infinite PSNR, zero error — on 21 of 27 gap frames**, something
a temporal marginal cannot do (the control leaks tail energy at 38–48 dB
everywhere). The pooled ghost_gap deficit is entirely the two DESIGNED
smoothstep ramp frames at presence 0.5. Corrected en route (append-only):
the old A1's ghost deficit was attributed to the voxel-cell oracle (M3);
removing that mechanism barely moved ghost_gap (22.41 → 22.83), so the
ramps were always the dominant cause.

Also corrected this block, and it touches every route0 lane ever run:
**`route_logit_init` in YAML never controlled a fresh run** — route logits
materialized from the constructor default 4.0 before the YAML was read
(repaired 2026-08-20; every historical cell trained from p_dyn ≈ 0.982).
And **both ADAGS eval call sites channel-split PSNR** (repaired; pooled
now, +tests). Under the published pooled+clamped convention the substrate
reads **33.5050 dB vs STG's published 33.52** on cut_roasted_beef frames
0-49 at 1352x1014 — parity at 6k vs 25k iterations
([[operations/stg-n3v-protocol-parity-2026-08-19]] Appendix C).

**The small-mistiming control DECIDED (experiment 191/198): a 2-frame
timing error is worse than no gate at all** — correct gate +1.05 > no
gate 0 > 2-frames-early −2.39 >> maximally-wrong −17.16 at the return.
Timing PRECISION, not gate existence, is what matters; real-data hard
gating therefore requires frame-accurate boundaries. CCR's consolidation
never gates support and is structurally immune.

**The CCR ladder round 1 is TERMINAL
([[operations/ccr-ladder-round1-results-2026-08-20]]):** the
observation-born relocation operator (B1) is globally NEUTRAL over two
paired seeds (+0.011 dB mean, per-seed ±0.28 = the B0 seed spread) while
improving the frozen event-ray union on BOTH seeds (+0.077/+0.345;
region A +0.09 on both); the certified consolidation pass admitted ZERO
edges on both seeds (funnel: proposals exist, ~half the screen survivors
starve on confirmation slots for ~6-frame packet supports, the rest fail
the deliberately strict 16-unit certificate) — so B2-DC ≡ B1 and the
finding is "no certified opportunity on this segment", with the two
bottlenecks named for a round-2 spec. G5's matched-capacity discipline
held throughout (600k cap binding in every arm, counts within 74 rows).

**Still open:**
the −0.39 dB ordinary-region cost is real and unattributed; everything
about real-data event supply is exactly as open as before — the fixture
is authored. The method lane that inherits this datum is CCR
([[operations/ccr-method-2026-08-20]]): observation-born packets plus
reconstruction-certified post-training appearance consolidation, frozen
after a 3-round hostile external review, with the B0/B1 ladder cells
running on the STG-matched protocol.

## G13 Membership Correction — 2026-08-24 (later): the PROJECTION ROUTE to the hull question is RETIRED

Append-only correction to the G13 section above, which recorded the hull
question as *"unanswerable BY THIS ROUTE"*. That is confirmed and now has a
second, independent reason. The first was **T1**, which cannot supply a
spanning component. The second is the **instrument**: even with components
supplied by construction, the shell-supply projection cannot decide the
operator — in absolutes or in deltas
([[operations/nonconvex-hull-projection-limits-2026-08-24]], ZERO GPU).

**Hull completion remains neither refuted nor supported.**

**(1) ABSOLUTES ARE UNUSABLE, calibrated against data already on the record.**
The proxy scores `object_shell / (object_shell + filler_shell)`, assuming every
gated row is object or filler. O1's base rule gated **10,374 rows at precision
0.5868** of which **1,152 are filler**, so 6,087 are object and **3,135 —
30.2% — are NEITHER**. Fed those exact counts the proxy's accounting returns
**0.8409** against a measured **0.5868**: **+0.2541, about 25x the ~0.01
discretization error** two drafts quoted as their uncertainty. The denominator
structurally cannot hold background, so **the bias is one-sided — always high.**
O2 fails in kind: `filler_shell[occupied] == 0` exactly, assigning its base set
literally zero false positives where the substrate measured 0.6768.

**(2) DELTAS ARE UNUSABLE TOO, and this is the sharper finding.** Spec §9 reads
the verdict only from the delta, so the delta was the obvious fallback.
**(a)** Enumerating index bounding boxes and evaluating each at its maximal
in-bbox mask makes `delta <= 0` a **THEOREM**: every added cell then lies
outside the occupied set, and object shell there is **exactly zero**. The
favourable branch is unreachable by construction. **An earlier draft reported
156/156 negative deltas from that family and read it as a result.**
**(b)** Without that restriction, the two available accountings — surface-area
-weighted shell counts, and volume-weighted row counts that DO include
background — **disagree by an order of magnitude** on how often H1 helps (O1
size 5-7: **0.0-2.7%** shell-positive against **27.8-33.1%** row-positive).
They do not corroborate; they diverge. **No sign statement is available**, and
the earlier claim *"H1 never improves precision on this fixture"* is **false**
(best +0.0350 shell, +0.1608 row). **The excluded regime was the realistic
one** — T1 accepted 4 cells of 452 groups, so real components are small and
sparse, exactly where the sign is contested.

**(3) SCOPE the drafts omitted:** every decomposition is the **fresh 50k
seeding grid**, and the completed runs realized a different one — experiment
274's accepted cells sit at **`j = 5`** while the fresh grid places the object
at `j` in {3,4}. Cell counts do not transfer to a trained substrate.

**(4) CARRY AS METHOD — four notes, and they cost four drafts.**
*A proxy needs a CALIBRATION, not a stability estimate.* The ±0.01 quoted
through two drafts is **discretization** error and is defensible as such; it
says nothing about whether the proxy scores the right **population**, and the
population error was **25x larger and one-sided**. **A tight stability figure
attached to a badly biased instrument is worse than no figure**, because it
invites the comparison that had to be withdrawn.
*A bias direction must be computed, not intuited* — "the missing rows are false
positives, so counting them makes it worse" is natural, was written into a
draft, and is **backwards**: a population common to both sides of a difference
compresses it.
*An enumeration can be vacuous exactly as a reading rule can.* The block's
standing rule was **recited in §9 of the very draft whose §2 violated it**.
Writing the rule into the same document did not prevent violating it; the
precondition has to be **computed** — here `object_shell[~occ] == 0`, one line.
*When several instruments agree, check whether they COULD disagree.* Two
accountings agreeing 156/156 looked like corroboration and was one theorem
twice; where they genuinely could disagree, they did, by an order of magnitude.

## Measurement-Channel Correction — 2026-08-24 (later): the PAIRED DESIGN is retired for the mechanism comparison

[[operations/n3v-paired-design-packet-2026-08-24]] left the paired design as
the only remaining lever after independent arms were priced at 37
replicates/arm = 181 slot-h. It is now retired for a B1-vs-B0 comparison at
**ZERO GPU cost**
([[operations/n3v-paired-design-retirement-2026-08-24]]).

**AN INTERLOCK, WITH NO THRESHOLD CHOSEN.** Cost per pair is
`2.4443 x (2 - k/6000)` slot-h, so seven pairs within the 24 slot-h ceiling
requires **`k >= 3584`**. Packet birth fires at
`{1000, 1500, 2000, 2500, 3000, 3500, 4000}`, so after `k = 3584` **only the
4000 birth remains**. **Any branch point affordable at the ceiling leaves at
most 1 of 7 births post-branch**, and a smaller prefix is strictly more
expensive (k=900 costs 31.7 slot-h for seven pairs against 24.0 at k=3584).
An earlier version argued this from two chosen bars — "≥5 of 7 births", ">50%
prefix" — which invited the objection that the bars were picked; the cost model
gives the same conclusion with no chosen number.

**Conditional on the frozen schedule.** `packet_birth_from_iter` / `until_iter`
/ `interval` are config fields; a schedule confined to 3500-4000 would change
this. The retirement is of option 1 under the frozen `ladder_b1_crb.yaml`
schedule, and any schedule change needs its own spec — including why the new
schedule is not chosen to rescue the design.

**THREE DURABLE CODE FACTS, newly verified and reusable.**

* **A localized, time-windowed opacity control ALREADY EXISTS and needs no code
  change.** `_apply_visibility_event_gate` selects rows by a **2D `crop_xyxy`
  screen box**, restricts them to an inclusive **frame window**, and multiplies
  **activated** opacity by a tunable `opacity_attenuation`. A **tracked**
  manifest exists and **tracked** configs already drive it with all seven keys;
  `ladder_b1_crb.yaml` simply does not set them.
* **Branching at `k` silently discards iteration `k`'s densification round,
  optimizer step, and packet birth**, because `scene.save` precedes all three
  and resume begins at `k + 1`. **Branching at a birth iteration destroys one
  of seven mechanism firings with no error and no log line.** Arbitrary-`k`
  branching is otherwise available: `scene.save` writes `chkpnt<k>.pth`
  whenever `k` is in `--save_iterations`.
* **The 600k HARD cap provably never fired.** All five point-removing sites are
  reachable only through `densify_and_prune`, called only inside the gate the
  cap closes, so once it closes the count can never fall; finals of
  599,396-599,470 are proof. `max_total_points` is not a second field — it is
  the parameter name of the same `densify_until_num_points` knob.

**CARRY AS METHOD: prefer an INTERLOCK to a THRESHOLD.** Where a conclusion can
be derived from two quantities that are functions of the same free parameter
pulling in opposite directions, it should be — no bar has to be defended.

**AND: a negative existence claim needs a SEARCH, not checks on the first
candidate.** An earlier draft declared a localized control impossible on three
findings that were each **verified and true** — region A is a 2D bbox with no
3D referent, `_opacity` is a pre-activation logit whose halving *brightens*
most rows, and opacity is time-invariant. All three are true of `_opacity`, the
wrong object. **Three true facts composed into a false conclusion because the
question was scoped to one mechanism**, while the capability sat in the render
path with tracked configs already using it.
