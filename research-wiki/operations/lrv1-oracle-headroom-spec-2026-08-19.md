# FROZEN EXPLORATORY SPEC — does a non-degenerate episodic representation have reconstruction headroom? (2026-08-19)

Status: **FROZEN before any cell produced an output.** EXPLORATORY throughout,
`evidence_bearing: false` on every cell. This is not a preregistration and it
does not license a claim; it fixes the decision rules before the numbers exist
so that a disappointing result cannot be re-read afterwards.

Reads [[renderer-integrity-admission-2026-08-18]] (the admitted image and its
run-to-run spread), [[elgs-matched-triple-frozen-spec-2026-08-18]] (the K=1
experiment this replaces), [[elgs-v8-formal-spec]] (the episodic
parameterization), [[dataset-admission-matrix-2026-08-18]].

## 1. The question, and why the previous experiment could not ask it

The matched 4-cell presence experiment was **BLOCKED at round 2** on a
structural finding: its cells seed every family with `K=1` and the latched
spanning rule, so `dim(a) = 2K + 1 - n_lat = 1`, and `softmax` over a
one-element vector is identically `[1.0]` with gradient exactly `0.0`. That
cell measured the cost of **deleting the temporal marginal**, not the cost or
benefit of a presence representation. It is closed and is not rerun.

The question this spec asks instead:

> Given the CORRECT episode structure, does EL-GS's existing episodic
> representation reconstruct a returning surface better than the ADAGS temporal
> substrate — and is any gain attributable to the episode TIMING rather than to
> having two episodes at all?

It deliberately bypasses learned evidence heads, DiVa census uncertainty,
proposal quality, acceptance policy, and claim-grade event-supply measurement.
Those are all downstream of the representation having headroom at all.

**A negative result here is informative and cheap. A positive result is the
first thing that would justify further evidence-instrument investment.**

## 2. Why K >= 2 needed code, and exactly how much

`K` was a hardcoded literal `1` at `elgs/trainer_hooks.py::seed_families`, with
no configuration surface anywhere on the `elgs_*` option set, and the only
runtime path to `K=2` was an accepted FISSION whose gap is chosen by a
mid-plateau heuristic — neither fixed nor externally supplied.

Landed at `d389a4d`, three edits across two files:

* `elgs_oracle_episodes` (a `str`, because `ParamGroup` renders every `bool` as
  `store_true` and so a bool could never be set False from the CLI, and neither
  `int` nor `float` can carry boundaries);
* the option plumbed onto `ElgsTrainerState`;
* a branch in `seed_families` that hands a cell's intersecting families the
  supplied program.

**Seeding granularity is unchanged** — still one family per nonempty voxel
cell, so the preregistered `max_families` cap binds exactly as before. Only the
interval PROGRAM differs. The frozen prereg is not edited. The oracle file
gives the INTERIOR absence gaps only; the outer endpoints are always the
latched `(-w_m, T + w_m)`, so the Omega-sum identity holds by construction and
`elgs.intervals.inverse` performs every floor and admissibility check —
an absence shorter than the preregistered `floor_gap` fails closed at setup
rather than at render.

### 2.1 A runtime blocker had to be repaired first, and it was not optional

`ElgsRuntime.presence_multiplier` runs on **every render call** and expanded
the per-family column to rows with a per-row Python list comprehension and a
`torch.stack` of one scalar per row. Measured on the local GPU, forward +
backward:

| rows | before | after |
|---:|---:|---:|
| 50,000 | 1.0356 s | 0.5088 s |
| 150,000 | 2.0514 s | 0.5046 s |
| 300,000 | 4.1303 s | 0.4742 s |

Before the repair the cost was **linear in row count** — 104 minutes of pure
overhead for a 6000-iteration cell at only 50k rows, and ~7 hours at 300k,
which alone exceeded the budget of the experiment that needs it. After, it is
flat, because the residual is the O(families) Python loop rather than O(rows).

The repair is a `searchsorted` gather over the compact per-family column
(`torch.unique` returns sorted values, which is what makes the mapping exact).
Tests pin that it is a pure speedup: values bit-identical to the old per-row
stack across six timestamps including unassigned `-1` rows, and the gradient on
the a-logits bit-identical. **The residual 0.5 s was measured, not chased.**

## 3. The scene — LRV1, and why it is synthetic

`scripts/build_synthetic_reveal_scene.py`, deterministic analytic ray-trace.
20-camera surround ring, 60 frames at 6 fps (t in [0, 9.8333] s), 400x300,
16 train / 4 held-out cameras (`cam02/07/12/17`), 50,000 uniform init points.

One object — a textured sphere at a fixed pose — is present over frames
**0-29**, **genuinely absent** over **30-53** (removed from the world, not
occluded), and present again over **54-59** at the same pose with the same
appearance.

**Why not real data.** The event this question needs is disappearance and
same-identity return. The project's own absence diagnostic
([[elgs-absence-diagnostic-result]]) found **0 of 597** DiVa-360 true-absence
candidates corroborated as genuine full-multiview disappearance, pooled and in
every sequence. So the one admitted event-supply dataset has no verified
instance of the event class, and the directive's own instruction — do not infer
an event from tracker visibility alone — forbids manufacturing one. An
authored scene is the only route on which the oracle is actually an oracle:
presence intervals, per-pixel front-most identity, and the newly-revealed
pixel-times are exact rather than estimated.

**What that costs.** A positive result here supports the mechanism and says
nothing about whether real data can supply the event. That limit is stated in
the outcome table and is not negotiable afterwards.

**The event is a genuine disappearance, not an occlusion.** This matters: an
occluded object still exists, the renderer handles it by depth ordering, and no
presence representation is required. A single-lobe temporal marginal cannot
represent two disjoint presence intervals, which is exactly the gap the
episodic representation claims to fill.

## 4. The cells — frozen

| cell | substrate | episodes | structural rounds |
|---|---|---|---|
| **A0** | ADAGS temporal (`elgs_enable: false`) | n/a — temporal marginal | n/a |
| **A1** | EL-GS `K=2` | **CORRECT** oracle | **OFF** |
| **A2** | EL-GS `K=2` | **WRONG-TIME**, dose-matched | **OFF** |

Structural rounds are OFF in all cells. They mix the representation with
proposal, acceptance and capacity allocation; they belong in a later experiment
only if correct episode structure first demonstrates value.

`configs/lrv1/{a0,a1,a2}.yaml` are byte-identical outside the block marked
`DIFFERS`, and A1 differs from A2 in exactly two lines: a header comment and
the oracle-episode path. A0 carries `elgs_reserved_parity: true` so it drops
the same ~25% of training units the EL-GS path reserves; without it the control
would train on a third more data than the cells it is compared to.

`elgs_a_lr: 0.0` **freezes** the supplied boundaries. They are an oracle, not
something to be learned, and freezing also removes the possibility of A2
drifting toward the correct timing and quietly destroying its own role.

### 4.1 The wrong-time control, and why a reflection

A2's episodes are A1's **reflected about the window midpoint**. A reflection is
the only transform that exactly preserves the episode count, every episode
duration, the gap duration and the total present duration — so A2 is
dose-matched by construction rather than by inspection:

| | episodes (s) | durations | gap | present frames |
|---|---|---|---|---|
| A1 | `[-0.3333, 4.9167]`, `[8.9167, 10.1667]` | 5.25, 1.25 | 4.0 | 34 of 60 |
| A2 | `[-0.3333, 0.9167]`, `[5.0833, 10.1667]` | 1.25, 5.25 | 4.0 | 34 of 60 |

Both sum to Omega = 10.5 s; both episode lengths exceed the preregistered floor
of 0.8333 s. Measured carrier agreement with ground-truth presence:
**A1 58 of 60 frames, A2 12 of 60**. (A1's two "disagreements" are the
smoothstep edge bands at frames 29 and 54 — the designed `w = 2*dt` transition,
not an error.)

**A2 is present at the true return frames 54-59.** It is not crippled at the
event itself: it can render the return, it simply believes the object was there
throughout the second half and absent while it was actually visible. That is
what makes it a control on TIMING rather than on capacity.

**Recorded limitation, before any result.** A reflected schedule is wrong in
two ways at once — it loses true-presence observations AND must suppress
ghosting where it wrongly believes the object present. So a large A1-minus-A2
gap would establish that timing matters without isolating which of the two
mechanisms carries it.

### 4.2 Two confounds, named now

* **The oracle is spatial as well as temporal.** A1 and A2 both know WHICH
  primitives are the object (the region test at seeding); only A0 does not. So
  A0-vs-A1 conflates the spatial oracle with the timing. **A1-vs-A2 is the
  clean comparison** and is the one the timing conclusion rests on.
* **Routing pins.** Rows of `K > 1` families get their route-logit gradient
  zeroed (`apply_elgs_routing_pins`). At `K=1` nothing is pinned. So A1 and A2
  are pinned identically and A0 is not — another reason the decisive comparison
  is A1-vs-A2.

## 5. The event metric — frozen before any output

The decisive metric is over exactly the pixel-times where the ground-truth
renderer says the event object is the **front-most** surface in a **held-out**
view, on the **return frames (54-59)** only. Because the object is genuinely
absent for the whole preceding gap, every one of those pixel-times is newly
revealed by construction.

The mask is renderer ground truth: correspondence-backed to the same surface by
exact identity, defined at return time, independent of every compared method's
prediction, and frozen when the scene was built. **A broad box, foreground
aperture or ROI could not identify those pixels and cannot carry this
conclusion.**

Measured supply: **113,868 held-out event pixel-times at return** (per view:
cam02 51,420, cam07 23,094, cam12 11,046, cam17 28,308 — every held-out view
sees the returned surface), against 569,340 over episode 1 and **0** during the
gap.

Reported per cell by `scripts/eval_lrv1_event.py`:

| region | what it answers |
|---|---|
| `whole_frame` PSNR / SSIM / LPIPS | ordinary reconstruction quality |
| **`event_return`** | **the decisive metric** |
| `event_episode1` | the in-scene reconstructibility oracle (gate item 6) |
| `ghost_gap` | energy inside the vacated footprint where GT has no object |
| `ordinary_return`, `ordinary_all` | so a gain that costs ordinary quality is visible |

PSNR pools over channels and pixels (the standard batched form), reported both
pooled over the region — primary — and as a mean of per-frame PSNRs.

The evaluator **restores the EL-GS state explicitly** and refuses to emit a
number if a cell declares `elgs_enable` and the runtime is not live.
`scripts/eval_diva360_heldout.py` never calls `setup_elgs`, so it would have
scored these cells through the temporal marginal — a presence semantics they
were never trained under.

## 6. Event-admission gate — checked in this order

A negative A1 is interpretable only if all of these hold. Items 1, 2, 4, 5 and 7
are settled by construction or already measured; **item 3 and item 6 are
checked on A0's output, before A1's is inspected.**

| # | requirement | status |
|---|---|---|
| 1 | enough pixel-times for a measurable metric | **MET** — 113,868 |
| 2 | returned surface visible in >= 1 held-out view | **MET** — all 4 |
| 3 | the matched ADAGS control shows a relevant error there | **checked on A0 first** |
| 4 | correct episode boundaries identifiable | **MET** — authored |
| 5 | training stable and adequately optimized | preflight + loss curve |
| 6 | region reconstructible in principle | **A0's `event_episode1` vs `event_return`** |
| 7 | not dominated by calibration/sync/preprocessing failure | **MET** — authored; camera convention verified by reprojecting the object centre through the loader's own transform onto ground-truth object pixels, 24 of 24 held-out return view-frames |

If the gate fails, the scene is **unsuitable or inconclusive** and no
representation verdict follows.

## 7. Decision rules — fixed here

Let `D1 = event_return PSNR(A1) - event_return PSNR(A0)` and
`D2 = event_return PSNR(A2) - event_return PSNR(A0)`.

| reading | conclusion |
|---|---|
| `D1` clearly positive, `D2` not | **episode-timing-specific headroom** — the strongest available basis for further evidence work |
| `D1` and `D2` similarly positive | gain is **not** attributable to correct timing; two episodes of capacity suffice |
| neither positive | **no demonstrated episodic headroom on this admitted event** — pause DiVa evidence investment |
| A1 unstable / crashes | implementation or optimization blocker, **not** a scientific negative |
| `D1` positive but `ordinary_all` materially worse | a trade, not a win; report both |

**"Clearly positive" is not given a numeric threshold here, and that is
deliberate rather than an omission.** No same-arm spread has ever been measured
on this scene, and importing one would be exactly the over-reading
[[renderer-integrity-admission-2026-08-18]] Appendix C exists to prevent. What
IS on record: on the admitted image, three runs of an N3V smoke at fixed seed
agreed to **0.00033 dB** of held-out PSNR, where the superseded image's two runs
disagreed by 0.10-0.36 dB. That is a prior expectation that the spread here
will be small, not a licence to treat any difference as real. **If the result
is close, the honest report is "close, and no spread was measured", and the
next action is a replicate — not a claim.**

## 8. Cost and provenance

Admitted image
`sudarshaniyengar/adags@sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683`,
pool `dgx` for every LRV1 cell (recorded, not silent: lane B runs on `hopper`,
and the two lanes are never compared to each other). Measured rate from the
preflight: **1.32 s/iteration** at batch 2, so ~2.2 GPU-h per EL-GS cell at
6000 iterations and less for A0, which does not pay the presence cost.

Per cell: O_EXCL claim, content-hashed config, digest-pinned image, O_APPEND
ledger line, `evidence_bearing: false`.

## 9. What this experiment cannot establish

* **That real data can supply this event.** LRV1 is authored. A positive result
  supports the mechanism and leaves event supply exactly as unresolved as it
  was.
* **That the evidence mechanism works.** No tracks, no evidence heads, no
  proposals, no acceptance. This is the representation and nothing else.
* **Anything about capacity allocation.** Structural rounds are off.
* **A SOTA placement.** LRV1 is a five-primitive authored scene.
* **Transfer.** One scene, one event, one seed per cell.

## 10. Termination

The experiment ends when A0, A1 and A2 are terminal, the gate is applied to
A0's output, and the section 7 rule is applied. A replicate is authorized only
under the "close" branch. No further cells are authorized by this page under
any outcome.

---

## RESULT PART 1 (2026-08-19, append-only) — the preflight, and what it cleared

Nothing above is rewritten. This section records the admitted-image preflight
and the launch-time verifications; the cell results follow in a later part.

### 11.1 Experiment 165 — the K>=2 preflight PASSED its decisive check

`lrv1_preflight_a1` r0, pool `dgx`, commit `5874fd1`, image
`sha256:70a28e3d...`, config `configs/lrv1/preflight_a1.yaml` (byte-identical
to `a1.yaml` except `iterations: 800`).

**Seeding, on Apollo, identical to the local setup preflight:**

```
{"elgs_seeding": {"families": 512, "iteration": 0, "oracle_K": 2,
  "oracle_episodes": "configs/lrv1/oracle_correct.json",
  "oracle_families": 8, "oracle_rows": 84, "rows": 50000}}
{"elgs_setup": {"evidence": false, "families": 512,
  "frame_dt": 0.16666666666666607, "restored": false,
  "schedule": "full", "time_span": 9.833333333333334}}
```

**The decisive observation is at iteration 600.** Under this capacity policy
densification first fires there, and that is exactly where the `elgs_a`
optimizer-group defect used to crash — `AssertionError: Group elgs_a has more
than one param` on the clone path, `IndexError: mask [N] vs indexed tensor [1]`
on the prune path. The run crossed it and continued:

| iteration | points |
|---:|---:|
| 500-600 | 50,000 |
| 610-630 | **43,977** |

The count **fell**, which means the prune path ran too, not merely the clone
path — a net removal of ~6,000 rows of the uniform random initialization while
the `elgs_a` group held 512 per-family tensors, 8 of them at `K=2` with
`dim(a) = 3`. **Both per-point optimizer paths executed against a
non-per-point group and neither raised.** That is the whole reason this
preflight existed.

Also established by the run: training is stable (train PSNR 10.74 at iteration
0 rising through 21.30 by 580) and the measured rate is **~1.30-1.43 s/it at
batch 2**, against A0's **3.85 it/s** — the EL-GS path is ~5.4x slower per
iteration, which is the per-family Python loop inside `presence_multiplier`
that section 2.1 measured and deliberately did not chase.

### 11.2 Reserved-unit parity — verified per run, not assumed

The matched-comparison requirement is that the temporal control must not train
on more data than the EL-GS cells. Verified from A0's own log line:

```
{"elgs_reserved_parity": {"reserved_units": 240, "training_units_after": 720}}
```

240 of 960 training units reserved, 720 trained on. The EL-GS path reaches the
same 240 through a different function (`filter_elgs_reserved` rather than the
parity shim, so it prints no such line); `tests/test_elgs_reserved_parity.py`
asserts the two paths drop **identical indices**, and the local setup preflight
reported `reserved_indices` of exactly 240 for both A1 and A2. So the control
and the cells train on the same 720 units.

### 11.3 A third confound, named before any number was read

Section 4.2 named two. There is a third, and it is inherent to the mechanism
rather than a defect:

**Presence gating changes which primitives accumulate densification gradient.**
A row whose family is absent at time `t` renders with exactly zero opacity, so
it contributes no gradient and no `xyz_gradient_accum`. Object rows are
therefore suppressed for 24 of 60 frames in A1 and for a different 24 frames in
A2. The densification POLICY and the point CAP are identical across cells, but
the realized allocation need not be — and indeed the preflight ended iteration
630 at 43,977 points where A0 was at ~104,000 by iteration 1000.

**Consequence, stated now:** "capacity matched" here means matched policy and
matched cap, **not** matched realized primitive count. Final primitive counts
are reported per cell and must be read alongside every delta. A1-vs-A2 is
affected far less than A0-vs-A1, because both carry the same 24-frame
suppression dose — which is a further reason the timing conclusion rests on
A1-vs-A2.

### 11.4 Launch record

| exp | cell | retry | pool | commit | note |
|---|---|---|---|---|---|
| 165 | `lrv1_preflight_a1` | r0 | dgx | `5874fd1` | preflight |
| **167** | `lrv1_a0_temporal_control` | **r0** | dgx | `43d9d46` | **ERROR — my mistake**: passed `--checkpoint_iterations`, which `main.py` does not define, so argparse rejected it in ~18 s. Claim index consumed, never reusable. |
| 168 | `lrv1_a0_temporal_control` | r1 | dgx | `43d9d46` | A0 |
| **166** | `b1_stg_matched_crb` | **r0** | hopper | `2ad0074` | **CANCELED before allocation.** All three `hopper` H100 slots were held by unrelated long-running Commands since 16:13Z and the trial was never given an agent. Zero compute consumed. |
| 169 | `b1_stg_matched_crb` | r1 | **dgx** | `43d9d46` | B1, moved to `dgx` |
| 170 | `lrv1_a1_oracle_correct` | r0 | dgx | `d892e28` | A1 |

**Pool switch recorded, not silent:** B1 moved from `hopper` to `dgx` because
`hopper` was saturated by other users. It is compared to a PUBLISHED number and
to no local cell, so nothing in this block requires it to share hardware with
anything; if a local STG reproduction is ever run it must be on `dgx` too.
`dgx` has **6** slots, three permanently occupied by unrelated Commands, so the
effective budget for this block is three concurrent cells.

---

## REVIEW ROUND 1 (2026-08-19, append-only) — verdict **DEFECTIVE**, one blocking finding, repaired before any output was read

A fresh-context adversarial reviewer with no project context was asked one
question — does the frozen matrix answer the question it states, or is there a
defect that would make the result uninterpretable whichever way it comes out.
It returned **DEFECTIVE**. Recorded append-only; nothing above is rewritten.

**Timing matters for how this is read: no cell output had been inspected when
the repairs below were made.** A0 and B1 were mid-run and A1 was cancelled; no
`lrv1_event_eval.json` existed. The outcome rules were still being fixed while
no number existed, which is the only condition under which changing them is
legitimate.

### B1 — BLOCKING, PRIMARY-VERIFIED, and it lands on the decisive metric

**Presence does not step, it ramps.** `elgs/presence.py` composes two clamped
cubic smoothsteps over the half-width `w = 2 * frame_dt = 0.333 s`, so presence
rises over the two frames after an episode start and falls over the two before
its end. The frozen `oracle_correct.json` placed episode 2's onset at the
midpoint between the last absent frame (53) and the first present frame (54) —
the obvious choice — which puts that ramp **on frames 54 and 55, two of the six
frames the whole comparison turns on**.

Primary-verified by running the production path (`load_oracle_episodes` →
`forward` → `episode_presence`) on the two frozen files:

| | presence on return frames 54-59 | mean |
|---|---|---:|
| **A1 (correct, as frozen)** | `[0.15625, 0.84375, 1, 1, 1, 1]` | **0.8333** |
| **A2 (wrong, as frozen)** | `[1, 1, 1, 1, 1, 1]` | **1.0000** |

So A1 rendered the returned object at **15.6% opacity on the first evaluated
frame** while its dose-matched control rendered it fully. The reviewer's
composite calculation puts A1's achievable `event_return psnr_pooled` at a
**ceiling near 19 dB regardless of how good the representation is**, an
artefact of 7-21 dB against A1 depending on how well both cells reconstruct.
Every row of the section-7 table was reachable for reasons unrelated to
headroom.

**Why the existing checks did not catch it.** The dose-match test asserts
episode count, `dim(a)` and the sorted duration multiset. **None of those
constrains where a ramp falls.** Section 4.1 above even notes the frame-54 edge
band — and treats it as a carrier-agreement footnote, never connecting it to
section 5's "return frames (54-59) only". The transferable lesson:
**a duration-matched control is not automatically a ramp-matched one.**

**Repair (landed `d0a7b3e`).** Inset each boundary by `w` from the last/first
present frame, so the presence **plateau** rather than the raw interval matches
ground truth:

| | gap (s) | episodes (s) | durations | gap len | total present |
|---|---|---|---|---:|---:|
| A1 | `[5.1666667, 8.6666667]` | `[-0.3333, 5.1667]`, `[8.6667, 10.1667]` | 5.5, 1.5 | 3.5 | 7.0 |
| A2 | `[1.1666667, 4.6666667]` | `[-0.3333, 1.1667]`, `[4.6667, 10.1667]` | 1.5, 5.5 | 3.5 | 7.0 |

Verified after repair: **A1 is presence 1.0 on all 36 ground-truth-present
frames**, with its two ramps landing on frames 30 and 53 — inside the absence
gap, where they cost only the secondary `ghost_gap` diagnostic. **A2 remains
1.0 across the true return frames**, so it can still render the return and the
comparison stays fair at the event itself. Duration multiset, gap and total
present time are identical between the two, and every episode and the gap clear
the preregistered floor of 0.8333 s.

Three tests pin it (`tests/test_elgs_oracle_episodes.py::RampPlacementTests`),
including an **anti-vacuity** test that rebuilds the old construction and
asserts the check fails on it at exactly `0.15625` / `0.84375`.

**Experiment 170 was cancelled** — it was running the defective configuration.
A1 relaunched as **171** at retry 1.

### B2 — MATERIAL, accepted and disclosed rather than repaired

The reviewer's second blocking finding: A2 does not hold "two episodes" fixed
in the sense the question needs. EL-GS presence *replaces* the temporal
marginal, and oracle families are hard-zeroed outside their episodes, so A1's
object rows get 36 frames of mutually consistent supervision while A2's get 24
frames of "be invisible" against 6 of "be the object". `D1 > D2` could
therefore follow from **supervision consistency alone** — "a correct hard gate
beats a maximally wrong one" — with no representational headroom over the
substrate. The middle row of the section-7 table is close to unreachable
because a reflection *maximises* mistiming.

**This is accepted.** The A1-vs-A2 conclusion must be stated as *a correct hard
presence gate beats a maximally wrong one*, which is weaker than *episode
timing is what matters*. The proper fix is a **small-mistiming control** — the
same `K=2` program with the gap translated by a few frames — which keeps
supervision largely consistent and isolates timing precision from gate
existence. That is one further ~2.6 GPU-h cell.

### Further findings accepted as DISCLOSURE

* **M2 — the largest A0-vs-A1 confound was not the one section 4.2 named.**
  Under `elgs_enable`, `marginal_t` is replaced by `get_elgs_presence` for
  **every** primitive, not only oracle ones. Non-oracle families are `K=1`
  spanning with presence identically 1.0 at all 60 frames, so A1/A2 have **no
  learnable per-primitive temporal lobe anywhere outside the ~780 oracle rows**,
  while A0 has one everywhere. That is a global representation change, strictly
  larger than the spatial-oracle and routing-pin confounds. It also means the
  dominant gradient path into `_t` is dead under EL-GS, so
  `densify_grad_t_threshold` fires differently. **A0-vs-A1 is a weaker
  comparison than section 4.2 implied; A1-vs-A2 is unaffected.**
* **M3 — the oracle region is a voxel-cell oracle roughly 8x the object's
  volume**, not "which primitives are the object": a cell is oracle if *any* of
  its points lies in the sphere, so ~780 seeded rows carry the program of which
  ~95 are actually inside the object. Harmless here (no static sphere and no
  ground intersects that box) but section 4.2 overstated the oracle's precision.
* **M4 — capacity is matched by policy and cap, not in effect** — already
  recorded independently as RESULT PART 1 section 11.3, and the reviewer adds
  the mechanism on A2's side: 24 frames of "be invisible" against 6 of "be
  visible" drives object rows toward `thresh_opa_prune`, plausibly pruning the
  very primitives that would render the return. Final primitive counts are
  reported per cell and are to be read as a confound indicator, not a neutral
  fact.
* **M9 — under-training is not direction-neutral.** ~16.7 epochs at 720 units
  and batch 2. A0 must *discover* the timing and reallocate capacity; A1 is
  handed it. So under-training favours A1, i.e. favours the hypothesis.

### Repaired in the same pass

* **M1** — three artifacts disagreed about what A2 was. `event_spec.json` no
  longer duplicates the episode programs at all; it states the ground-truth
  presence FRAMES and points at the config files. Scene regenerated; images and
  `points3d.ply` bit-identical.
* **M8** — the evaluator checked the runtime was live but not that it came from
  the checkpoint. Without `elgs_state`, `setup_elgs` silently re-seeds over the
  TRAINED cloud and every reported family count still looks plausible. Now
  checked.
* **M5** — gate items 3 and 6 were the same predicate. Item 6 is now an absolute
  floor (episode-1 >= 25 dB), item 3 a minimum deficit (>= 1 dB).
* **M7** — the reducer implemented "clearly positive" as bare `d1 > 0`. It now
  returns INDETERMINATE below a 0.5 dB floor inherited from the matched-triple
  spec's `max(0.5 dB, S)`.
* **M6** — the gate could not be obtained without also handing the reducer A1.
  `--a1` is now optional.

### What the reviewer verified as CORRECT (not to be re-checked)

Config matching (A1-vs-A2 is genuinely a one-variable change); the dose-match
arithmetic as far as durations go; the carrier-agreement figures 58/60 and
12/60 exactly; seeding granularity genuinely unchanged; `elgs_a_lr: 0.0`
genuinely freezes the boundaries (`update_learning_rate` touches only the
`"xyz"` group); the `presence_multiplier` repair exactly equivalent, with tests
that would catch a real regression in the expansion; `elgs_reserved_parity`
genuinely matched, with the reservation rule frame-uniform so there is **no
return-frame bias**; determinism across cells; the scene is what this page
describes; the event is a genuine disappearance with no occluder; held-out
cameras genuinely held out; the event mask method-independent and its
missing-mask guard fail-closed; and the supply figures 113,868 / 569,340 / 0
exact.

**One item the reviewer could not reproduce:** the camera-convention check
asserted in gate item 7 ("24 of 24"). It was run by the primary as a scratch
script and its output recorded, but no tracked artifact carries it. That is a
fair criticism of the record, not of the fact.
