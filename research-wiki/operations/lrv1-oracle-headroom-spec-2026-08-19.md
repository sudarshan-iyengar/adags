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

---

## ANCHORS (2026-08-19, append-only) — the scale `event_return` is read on

Computed from the scene's own renderer, **independent of every cell**, and
recorded here **before any cell output was inspected** so that the scale cannot
be chosen after seeing a number. Reproducible from
`scripts/build_synthetic_reveal_scene.py::render` alone.

| anchor | pooled `event_return` PSNR |
|---|---:|
| **FLOOR** — a cell that reconstructs the background behind the object perfectly and the returned object not at all, i.e. renders the return frames as if it never came back | **10.1235 dB** |
| per held-out view at the floor | cam02 9.39, cam07 10.62, cam12 15.02, cam17 10.05 |
| **PRACTICAL CEILING** — correct everywhere but off by one 8-bit quantisation step | **58.92 dB** |

So a cell scoring near 10 dB on this metric has not reconstructed the returned
surface at all; the useful dynamic range is roughly 10 to 40 dB. cam12's higher
floor is the view where the returned object subtends fewest pixels and is
partly occluded by a static sphere, which is also why it carries only 10% of
the pooled weight against cam02's 45%.

**A note on the ramps that survive the B1 repair**, so the secondary
diagnostics are not misread. Neither repaired program ramps on a decisive
return frame, but the residual half-presence frames land differently:

* **A1** ramps at frames **30 and 53** — both inside the absence gap, so A1
  pays for them in `ghost_gap` and nowhere else;
* **A2** ramps at frames **6 and 29** — both ground-truth-present, so A2 pays
  for them in `event_episode1`.

`ghost_gap` is therefore **not** directly comparable between A1 and A0, and
`event_episode1` is not directly comparable between A2 and A0. `event_return`,
the decisive metric, is unaffected in both.

---

## RESULT PART 2 (2026-08-19, append-only) — LRV1's gate FAILED, and the cause is the fixture, not the representation

Nothing above is rewritten. **The gate thresholds applied here were fixed
before any cell output existed** (REVIEW ROUND 1, finding M5) and are not
moved now that a number exists.

### 12.1 A0 — the matched ADAGS temporal control (experiment 168)

`lrv1_a0_temporal_control` r1, `dgx`, commit `43d9d46`, image
`sha256:70a28e3d...`, 6000 iterations, terminal COMPLETED, 149,602 primitives
(the 150,000 cap binds), checkpoint at iteration 6000. Scored by experiment
173 (`lrv1_eval_a0` r1).

| region | pooled PSNR | mean-per-frame | pixel-times |
|---|---:|---:|---:|
| `event_return` | **21.9952** | 23.4587 | 113,868 |
| `event_episode1` | **23.2890** | 24.9772 | 569,340 |
| `ghost_gap` | 14.1091 | 15.4887 | 455,472 |
| `ordinary_return` | 19.4833 | 19.7400 | 2,766,132 |
| `ordinary_all` | 19.2501 | 19.5142 | 28,116,792 |
| `whole_frame` | 19.3105 | 19.5722 | SSIM 0.69694, LPIPS-Alex 0.23476 |

Per return frame: 54 → 18.33, 55 → 22.72, 56 → 24.29, 57 → 25.27, 58 → 25.55,
59 → 24.60 dB. The first returned frame is the worst by ~6 dB, which is the
shape the hypothesis predicts.

### 12.2 The evaluator agrees with `main.py`, so the measurement is sound

This matters before anything is read into the numbers. `main.py`'s own held-out
evaluation of the same run reports `best_val/psnr` **19.5785** and
`best_val/ssim` **0.6969425**; this evaluator reports whole-frame pooled PSNR
**19.3105** and SSIM **0.6969427**.

**SSIM agrees to six decimal places** — two independent implementations
rendering the same 240 held-out views and reaching the same number. The
0.268 dB PSNR difference is the expected sign and magnitude of the
channel-pooling defect recorded in [[stg-n3v-protocol-parity-2026-08-19]]:
`main.py` averages three per-channel PSNRs where this evaluator pools over
channels and pixels, and the bias is `10*log10(AM/GM)` over the per-channel
MSEs, always non-negative.

### 12.3 THE GATE: FAILED on item 6

| item | rule (fixed before any output) | measured | verdict |
|---|---|---:|---|
| 3 — control errs at the return | `event_episode1 - event_return >= 1.0 dB` | **1.2939 dB** | **PASS** |
| 6 — region reconstructible in principle | `event_episode1 >= 25.0 dB` | **23.2890 dB** | **FAIL** |

**GATE FAILED.** Under section 6 the consequence is fixed and not negotiable:
**LRV1 is UNSUITABLE / INCONCLUSIVE for this question, and no representation
verdict follows from any A1 number on it.**

Experiment 171 (LRV1 A1, correct oracle, repaired boundaries) was ~45 minutes
into a ~2.4 hour run when the gate was applied. It was **cancelled**, because
its remaining hour of compute could not buy an interpretable result and the
slot was needed for a scene that can. That is not cancelling valid work to make
a report terminal — it is stopping work whose interpretive basis a measurement
had already removed.

### 12.4 Why it failed, measured rather than guessed

A0 reached **34.23 dB on training views and 19.31 dB held out** — a 15 dB gap
on a five-primitive analytic scene, which is not a plausible generalization gap
and pointed at the fixture.

**The initialization cloud does not cover the visible scene.** The cloud is
uniform in `[-1.3, 1.3]^3`; the ground plane reaches to `GROUND_HALF_EXTENT =
3.0`. Measured over all 20 cameras by ray-casting the scene's own geometry:

| ground half extent | pixels showing surface OUTSIDE the init cube | background |
|---:|---:|---:|
| **3.0 (LRV1)** | **13.94%** | 32.90% |
| 1.3 | **0.00%** | 46.84% |

Densification **clones and splits existing primitives**; it cannot create them
where none are nearby. So ~14% of every image is surface the optimizer can only
fit with floaters — and a floater that reproduces a training view exactly is
wrong from a held-out view 18 degrees away. That is the mechanism of the 15 dB
gap, and it also explains the ordering in the table above: `event_episode1`
(23.29) and `event_return` (22.00) are *better* than `ordinary_all` (19.25),
because the event object sits near the centre of the scene where the init cloud
is dense, while the badly-reconstructed ground is everywhere else.

**This is a defect in the fixture, authored by the primary, not a property of
the representation and not a property of the event.**

### 12.5 What LRV1 does and does not establish

**Establishes:**

* the whole measurement chain works end to end on the admitted image —
  fixture, loader, trainer, checkpoint, evaluator, reducer;
* the evaluator agrees with an independent implementation to six decimals on
  SSIM;
* a real, quantified failure mode for authored testbeds: **if the
  initialization does not cover the visible surface, held-out numbers measure
  floaters rather than the method.** That is worth more than the cell would
  have been.

**Does NOT establish:** anything about whether the episodic representation has
headroom. A1 was cancelled and would have been uninterpretable anyway.

### 12.6 LRV2 — the same event, on a scene that is actually initialized

`scripts/build_synthetic_reveal_scene.py --scene-id LRV2
--ground-half-extent 1.3`. **One constant changes.** Cameras, frames, the event
object and its timing, the oracle-episode files, the training budget and the
capacity policy are all unchanged, and the held-out event supply is **identical
at 113,868 return pixel-times**. Uncovered visible surface measures **0.00%**.

Cells `configs/lrv2/{a0,a1,a2}.yaml` differ from their LRV1 counterparts in
`source_path` and comments only, and reuse `configs/lrv1/oracle_*.json`
unchanged — the presence programs depend only on frame timing and the frozen
prereg constants, neither of which moved.

**Every rule in sections 5, 6 and 7 above, and the anchors, carry over
unchanged and are NOT re-derived for LRV2.** The gate floor stays at 25.0 dB
and the deficit minimum at 1.0 dB. Applying the same fixed rules to a repaired
fixture is the point; loosening them because the first scene failed them would
not be.

---

## RESULT PART 3 (2026-08-19, append-only) — LRV2: the fixture repair worked, and the event turned out too easy

Nothing above is rewritten. Thresholds unchanged from the values fixed before
any cell output existed.

### 13.1 The repair worked, and the numbers say so unambiguously

A0 on LRV2 (experiment 174, 6000 iterations, 149,825 primitives, scored by
experiment 176), against A0 on LRV1:

| region | LRV1 | **LRV2** | change |
|---|---:|---:|---:|
| `whole_frame` | 19.3105 | **28.0303** | **+8.72** |
| whole-frame SSIM | 0.69694 | **0.91846** | **+0.222** |
| `ordinary_all` | 19.2501 | **27.9998** | +8.75 |
| `event_episode1` | 23.2890 | **29.6922** | +6.40 |
| `event_return` | 21.9952 | **28.7314** | +6.74 |
| `ghost_gap` | 14.1091 | **28.7899** | +14.68 |
| training-view PSNR | 34.23 | **40.98** | +6.75 |

**One constant changed** — the ground plane half extent, 3.0 to 1.3 — and the
train/held-out gap collapsed from 15 dB to ~13 dB in absolute terms while
held-out quality rose by nearly 9 dB. The diagnosis in §12.4 is confirmed by
the repair.

**Whole-frame numbers are NOT comparable across the two scenes** (background
rises from 32.9% to 46.8% of pixels, and background is trivially correct), but
the masked regions are: `event_episode1` and `event_return` are computed over
the *same* object pixels in both, and both rose by 6-7 dB. The single most
telling row is `ghost_gap`, up 14.68 dB.

### 13.2 THE GATE: FAILED again, and this time on the other item

| item | rule (unchanged) | LRV1 | **LRV2** | verdict |
|---|---|---:|---:|---|
| 6 — reconstructible in principle | `event_episode1 >= 25.0 dB` | 23.2890 FAIL | **29.6922** | **PASS** |
| 3 — control errs at the return | `deficit >= 1.0 dB` | 1.2939 PASS | **0.9608** | **FAIL** |

**GATE FAILED, by 0.0392 dB on item 3.** The threshold is not moved. Missing a
pre-registered floor by a hair is precisely the situation that floor exists
for, and loosening it after seeing the number would retroactively void every
other freeze on this page.

### 13.3 What the failure MEANS, and it is the most useful thing this block has produced

This is not the same kind of failure as LRV1's. LRV1 failed because the fixture
was broken. **LRV2 fails because the ADAGS temporal substrate solves this
event.**

* the control reconstructs the returned surface at **28.73 dB**, only **0.96 dB**
  below its performance on the same surface during a 30-frame continuous
  presence;
* it is **not** cheating by keeping the object alive through the absence:
  `ghost_gap` is **28.79 dB**, essentially the same as its ordinary-region
  quality, so it genuinely removes the object and genuinely brings it back;
* per-frame, the first returned frame is the weakest, which is the shape the
  hypothesis predicts — but the whole effect is under a decibel.

**So on this event there is at most ~1 dB of headroom for ANY representation to
capture, episodic or otherwise.** That bound is independent of EL-GS and is the
reason the gate refuses the scene: a testbed where the control barely errs
cannot discriminate between representations, whatever the representation does.

This is a real negative about the *event*, and it is worth more than an
uninterpretable positive would have been. A 4D-Gaussian temporal substrate,
given 16 training cameras, a 150k primitive budget and 6000 iterations, handles
a 24-frame disappearance and a 6-frame same-pose return with about a decibel of
residual error.

### 13.4 A1 on LRV2 (experiment 175) — what it can and cannot say

It was already 40 minutes into a 2.4 hour run when the gate was applied, and it
was **allowed to finish** rather than cancelled, because unlike LRV1's A1 it can
still produce something honest: **a bound**. With the control erring by only
0.96 dB at the return, the most a correct-oracle episodic representation could
recover here is that decibel, which sits at the edge of the 0.5 dB resolution
floor with no measured same-arm spread. So A1's number is reported as a bound
on the available gain, explicitly **not** as a representation verdict.

### 13.5 LRV3 — the event gets harder along its one honest axis

The only pre-registered requirement LRV2 fails is that the control must err.
Making the event harder to satisfy an admission criterion that was fixed in
advance is not tuning the test until the hypothesis wins — it is what the
criterion is for.

**The RETURN shortens from 6 frames to 3** (57-59), so the model sees the
returned surface in 48 training views instead of 96, and the absence lengthens
to 27 frames. Nothing else changes.

The admissibility bound is tight and is now **enforced in the generator**:
episode 2's duration is `10.5 - first_return_frame/6` and must strictly exceed
`floor_len = 0.8333 s`, so the first return frame cannot exceed **57**. A
2-frame return sits exactly ON the floor and is refused rather than silently
accepted.

Held-out event supply halves to **56,934** return pixel-times — still ample.
The wrong-time control remains exactly dose-matched: durations `{5.5, 1.0}`
both, gap `4.0` both, sum = `Omega` = 10.5, presence 1.0 across the true return
frames. **Every gate and outcome rule carries over unchanged.**

**If LRV3's control also errs by less than 1.0 dB, that is the finding**: the
temporal substrate handles this whole event class at this budget, and the
useful next move is a different event class or a tighter capacity budget — not
a third adjustment of the same knob.

---

## RESULT PART 4 (2026-08-19, append-only) — LRV3: the gate PASSES, and the decisive cells are running

Thresholds unchanged throughout. Only the event's difficulty axis moved, and
only because a pre-registered admission criterion demanded it.

### 14.1 The gate, on the same rules, across all three fixtures

A0 on LRV3 = experiment 177 (6000 iterations, 149,834 primitives), scored by
experiment 178.

| | LRV1 | LRV2 | **LRV3** |
|---|---:|---:|---:|
| return length (frames) | 6 | 6 | **3** |
| `event_episode1` | 23.2890 | 29.6922 | **29.7727** |
| `event_return` | 21.9952 | 28.7314 | **27.2763** |
| **deficit** | 1.2939 | 0.9608 | **2.4963** |
| `ghost_gap` | 14.1091 | 28.7899 | 28.9700 |
| `ordinary_all` | 19.2501 | 27.9998 | 28.2591 |
| `whole_frame` | 19.3105 | 28.0303 | 28.2822 |
| whole-frame SSIM | 0.69694 | 0.91846 | **0.91960** |
| item 6 (`episode1 >= 25.0`) | **FAIL** | PASS | **PASS** |
| item 3 (`deficit >= 1.0`) | PASS | **FAIL** | **PASS** |
| **GATE** | FAILED | FAILED | **PASSED** |

**The three fixtures separate the two failure modes cleanly**, which is the
useful structure here:

* **LRV1** failed item 6 — the scene was broken, and the control could not
  reconstruct the surface even under continuous presence;
* **LRV2** fixed that (item 6 up 6.4 dB) and then failed item 3 — the control
  reconstructed the *return* nearly as well as continuous presence, so there
  was nothing to discriminate;
* **LRV3** halves the return and item 3 clears comfortably, while item 6 is
  **unchanged at 29.77 vs 29.69** — exactly what should happen, since episode 1
  was not touched. That invariance is the check that the difficulty knob moved
  what it was supposed to move and nothing else.

**Halving the returned surface's training observations (96 view-frames to 48)
raised the control's return deficit from 0.96 dB to 2.50 dB** — a 2.6x
increase from a 2x reduction in observations. That relationship is itself a
recorded fact about how much the temporal substrate depends on return-window
supply.

### 14.2 What is now admitted, and what it licenses

LRV3 satisfies every item of the section-6 admission gate. The event covers
56,934 held-out return pixel-times, the returned surface is visible in all four
held-out views, the correct boundaries are exact by construction, the region is
demonstrably reconstructible (29.77 dB under continuous presence), the control
demonstrably errs at the return (2.50 dB below that), and the fixture's
geometry is pinned by tests.

**So a negative A1 result on LRV3 IS interpretable** as "no demonstrated
episodic headroom on an admitted event", and a positive one is interpretable
subject to the A1-vs-A2 timing check and the confounds recorded in REVIEW
ROUND 1.

### 14.3 Cells running

| exp | cell | retry | what it decides |
|---|---|---|---|
| **179** | `lrv3_a1_oracle_correct` | r0 | D1 — does correct fixed `K=2` episode structure beat the temporal substrate at the return |
| **180** | `lrv3_a2_wrong_time` | r0 | D2 — is any gain attributable to the TIMING, or merely to having a hard presence gate |

Both `dgx`, commit `66953f5`, admitted image, 6000 iterations, `elgs_a_lr: 0.0`
(boundaries frozen), structural rounds off.

**Experiment 175 (LRV2 A1) was cancelled** to free the slot. Its scene's gate
had failed, so it could only bound the available gain on an inconclusive
fixture, whereas LRV3's A1/A2 pair is the decisive comparison. That trade is
recorded rather than silent: **no LRV2 A1 number exists**, and the ~1 dB bound
implied by LRV2's control deficit stands as the only statement about that
scene.

### 14.4 The honest position on three fixtures

Three iterations of a synthetic testbed is more than intended, and the reason
each one happened is on the record: LRV1's was a defect I introduced
(initialization not covering the visible surface), LRV2's was the event being
too easy for the control, and only the second of those is a *scientific*
finding rather than a repair. **The difficulty knob was moved exactly once, in
one direction, to satisfy a criterion fixed before any cell ran** — and item 6's
invariance across that move (29.69 -> 29.77) is the evidence that it was not a
search over knobs until something passed.

**What would have made this unnecessary:** ray-casting the scene to measure
initialization coverage before the first training cell, and running the control
alone first to measure its return deficit before committing to the full matrix.
Both are cheap. Both are now the recommended order for any authored testbed on
this project.

---

## RESULT PART 5 (2026-08-19, append-only) — THE DECISIVE RESULT: no demonstrated episodic headroom, and the deficit is not where the hypothesis said it would be

Every threshold and rule below was fixed before any cell output existed and
none was moved. LRV3's gate PASSED, so this result **is** interpretable.

### 15.1 The three cells

A0 = experiment 177, A1 = 179, A2 = 180; scored by 178, 183, 182. All `dgx`,
commit `66953f5`, admitted image `sha256:70a28e3d...`, 6000 iterations,
checkpoint at iteration 6000 in all three, 240 held-out views scored in all
three.

| region (pooled PSNR) | **A0** temporal | **A1** correct oracle | **A2** wrong-time |
|---|---:|---:|---:|
| **`event_return`** | **27.2763** | **22.0448** | **10.1144** |
| `event_episode1` | 29.7727 | 26.6149 | 12.7728 |
| `ghost_gap` | 28.9700 | 22.4121 | 28.6610 |
| `ordinary_return` | 28.1841 | 27.9140 | 27.8820 |
| `ordinary_all` | 28.2591 | 27.7771 | 27.7830 |
| `whole_frame` | 28.2822 | 27.7275 | 25.4266 |
| whole-frame SSIM | 0.9196 | 0.9006 | 0.8893 |
| primitives | 149,834 | 149,868 | 149,862 |

**Capacity is matched in realized count, not merely in policy** — all three land
within 34 primitives of each other at the 150,000 cap. That retires the
capacity confound (REVIEW ROUND 1, M4) for this comparison specifically.

Both EL-GS cells report `families: 512`, `K_histogram {1: 504, 2: 8}`,
`a_lr: 0.0`, `rounds_enabled: false`, `runtime_live: true`, and the expected
oracle file. The representation was live, non-degenerate and frozen, as
intended.

### 15.2 The frozen outcome rule, applied

```
D1 = A1 - A0 =  -5.2316 dB
D2 = A2 - A0 = -17.1619 dB
A1 - A2      = +11.9303 dB
```

**SECTION 7 READING: A1 falls below A0 by 5.2316 dB — NO demonstrated episodic
headroom on this admitted event.**

This is not a near-miss and it is not inside the resolution floor. It is an
order of magnitude larger than the 0.5 dB floor, in the wrong direction.

### 15.3 The deficit is NOT specific to the event, and that is the finding

The single most informative row is `event_episode1` — the same object surface
during 30 frames of **continuous presence**, where the correct oracle's presence
is a constant 1.0 and the episodic machinery should be doing nothing at all:

| region | A1 - A0 |
|---|---:|
| `ordinary_all` (non-event pixels everywhere) | **-0.48** |
| `event_episode1` (object, continuously present) | **-3.16** |
| `event_return` (object, at return) | **-5.23** |
| `ghost_gap` (object footprint during absence) | **-6.56** |

**Turning on EL-GS costs ~0.5 dB globally and ~3.2 dB on the object region even
where presence is constant.** So most of the event-region deficit is a fixed
cost of the representation swap, not a failure to reconstruct the return. Only
about 2 dB of the 5.23 is additional at the return.

Two mechanisms, both named in REVIEW ROUND 1 **before** these cells ran, account
for this:

* **M2 — EL-GS replaces the temporal marginal for EVERY primitive**, not only
  oracle ones. Non-oracle families are `K=1` spanning with presence identically
  1.0, so A1 and A2 have **no learnable per-primitive temporal lobe anywhere**
  outside the ~780 oracle rows, while A0 has one on all 149,834. That is the
  most plausible source of the -0.48 dB global and much of the -3.16 dB.
* **M3 — the oracle is a VOXEL-CELL oracle roughly 8x the object's volume.** A
  cell is oracle if *any* of its points lies in the sphere, so the hard gate
  switches off background primitives near the object as well. `ghost_gap` is
  the direct evidence: A1 loses **6.56 dB** there, in exactly the region where
  its oracle families are held at presence zero and therefore render nothing —
  including the background those cells were also responsible for.

**So the experiment answers its question, and the answer is negative — but a
substantial part of the measured deficit is attributable to how the oracle was
wired rather than to the episodic representation being unable to help.** Both
causes were disclosed in advance; neither is an excuse invented afterwards.

### 15.4 What IS cleanly established: timing matters enormously, within EL-GS

`A1 - A2 = +11.9303 dB` on the decisive metric, and A2's per-frame return
profile is flat at **11.70 / 11.76 / 11.81 dB** against the scene's
independently computed **floor of 9.7487 dB** — the score of a cell that never
reconstructs the returned surface at all. **A2 barely reconstructs the return.**

The mechanism is legible. A2's wrong schedule holds its object families
*present* through the true absence, so 27 frames of "there is nothing here"
outvote 3 frames of "there is an object here", and the optimizer drives those
primitives transparent. Its `ghost_gap` of **28.66 dB** — nearly A0's 28.97 —
confirms it: A2 fixed the gap by destroying the object, and paid for it at the
return.

So **within the episodic representation, correct episode timing is worth ~12 dB
at the event.** That is a real and large effect. It is also, as REVIEW ROUND 1's
finding B2 warned, the weaker of the two claims available: it shows *a correct
hard presence gate beats a maximally wrong one*, not *timing precision is what
matters*. A small-mistiming control would separate those and was not run.

### 15.5 The honest summary

1. **The representation executes exactly as specified.** Non-degenerate `K=2`,
   frozen oracle boundaries, structural rounds off, capacity matched to 34
   primitives, checkpoint and evaluator provenance verified. Nothing here is an
   implementation failure.
2. **Correct episode structure did not improve event reconstruction. It made it
   5.23 dB worse.**
3. **The deficit is dominated by a fixed cost of the swap, not by the event.**
   -0.48 dB globally, -3.16 dB on the object under constant presence.
4. **Episode timing matters enormously inside the representation** (+11.93 dB
   over the wrong-time control), which is why the failure is interesting rather
   than merely disappointing: the machinery clearly *does* something, and what
   it does is not yet worth what it costs.
5. **Two named, pre-disclosed wiring choices plausibly account for most of the
   cost**, and both are fixable.

### 15.6 What this licenses, and what it forbids

**Forbidden by this result:** any claim that EL-GS's episodic presence improves
reconstruction. On the only admitted event this project has, it does not.

**Not established by this result:** that episodic presence *cannot* help. The
two confounds in §15.3 are not decorative — they are the difference between
"the representation is wrong" and "the representation was wired to pay for
things it did not need". Specifically, A1 was made to give up the temporal
marginal on 149,000 primitives that had no event to model, in exchange for
episodic presence on ~780.

**The next experiment is now specific rather than exploratory**, and it is
cheap:

* **keep the per-primitive temporal marginal for non-oracle families** so the
  swap is local to the primitives that need it. If A1's `-0.48 dB` global and
  much of the `-3.16 dB` episode-1 cost disappear, the representation swap is
  not intrinsically expensive and the event-specific question can finally be
  asked cleanly;
* **make the oracle per-primitive rather than per-voxel-cell**, so background
  near the object is not gated off with it. `ghost_gap` is the diagnostic that
  says whether this mattered;
* **add the small-mistiming control** that REVIEW ROUND 1 asked for, so
  "timing matters" can be separated from "a hard gate matters".

**DiVa claim-grade evidence work remains PAUSED and this result strengthens the
case for keeping it paused.** The argument for pausing was that
evidence-acquisition investment should follow a demonstration that the
representation can use the evidence. The representation has now been handed
*perfect* evidence — exact episode boundaries, on an admitted event, with
matched capacity — and was 5.23 dB worse than not having it. Acquiring
imperfect evidence for it is not the next thing to fund.

### 15.7 A refinement to §15.3's attribution, from the numbers themselves

The decomposition constrains the mechanisms better than §15.3 stated, and the
constraint is worth writing down because it points the follow-up experiment at
the right thing.

`ordinary_all` loses only **0.48 dB**. That region has **no routing pins** and
**no oracle gating**, and it also lost its per-primitive temporal marginal. The
scene is static outside the event object, so the marginal was doing little
there — and 0.48 dB is what losing it costs where it was not needed.

`event_episode1` loses **3.16 dB** on a surface that is continuously present,
where the oracle's presence is a flat 1.0 and the hard gate is therefore
inactive. So roughly **2.7 dB is specific to the object region and is not
explained by the marginal loss alone.** The remaining named candidates are:

* **routing pins.** `apply_elgs_routing_pins` zeroes the route-logit gradient
  for rows of `K > 1` families — i.e. exactly the ~780 oracle rows and nothing
  else. Their dynamic/static route mixture is frozen at `route_logit_init: 0.0`
  for the whole run, while A0's is free to learn. This is a per-row handicap
  applied to precisely the rows the experiment cares about, and it is active
  during episode 1 when the gate is not.
* **the marginal loss mattering more where there IS temporal structure.** The
  object is the only thing in the scene with a temporal signature, so it is the
  one place a temporal marginal earns its keep.

These two are separable and the separation is cheap: **run one cell with
`K >= 2` oracle episodes and routing pins disabled.** If the ~2.7 dB
object-specific gap largely closes, the pin is the cost and the episodic
representation is cheaper than this experiment made it look. If it does not,
the cost is the marginal and the follow-up in §15.6 item 1 is the right fix.

`ghost_gap`'s **-6.56 dB** remains attributed to the voxel-cell oracle (M3):
it is the only region where the hard gate is active, and the gate switches off
background primitives that share the oracle cells with the object.

**None of this changes the headline.** A1 is 5.23 dB behind A0 on the decisive
metric and the negative stands as recorded. What it changes is which follow-up
is worth running first, and it makes that follow-up a one-cell question rather
than a redesign.
