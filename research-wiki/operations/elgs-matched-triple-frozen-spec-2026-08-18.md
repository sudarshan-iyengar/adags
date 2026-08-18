# FROZEN SPEC — the matched 4-cell presence experiment (NOT LAUNCHED)

Date: 2026-08-18. Status: **FROZEN, NOT SIGNED, NOT LAUNCHED.** Every
decision rule below is fixed before any cell runs. Launch requires (a) a
fresh-context sign-off, (b) the section 6 blocker closed, and (c) explicit
user compute authorization for ~25 GPU-hours.

Supersedes nothing. Reads [[elgs-matched-triple-readiness-2026-08-18]] (the
readiness audit this spec answers), [[renderer-integrity-admission-2026-08-18]]
(the admitted image and Appendix C's spread), [[exp123-mechanism-audit]],
[[diva360-visual-hull-initialization]], [[elgs-evidence-head-freeze-packet-2026-08-18]].

## 1. The question, and why four cells rather than three

The EL-GS presence representation — latched multi-episode presence replacing
the temporal marginal — has never trained past a 220–600-iteration smoke.
Every 6k–15k photometric number on record (experiments 84, 101, 104, 123 and
the sweeps) ran `elgs_enable: False`. So the representation the method
actually renders with is **photometrically unmeasured**, and no M2 substrate
claim can rest on it until that changes.

The readiness audit specified three cells. This spec runs **four**, adding a
seed replicate of the control, because the decision rule needs a measured
same-arm spread and importing one would be exactly the over-reading
[[renderer-integrity-admission-2026-08-18]] Appendix C was written to
prevent.

| cell | substrate | seed | structural rounds | photometric rounds |
|---|---|---|---|---|
| **T-1** | temporal-Gaussian (control) | 0 | n/a | n/a |
| **T-1'** | temporal-Gaussian (control replicate) | 1 | n/a | n/a |
| **T-2** | EL-GS K=1 latched presence | 0 | **OFF** | OFF |
| **T-3** | EL-GS K=1 latched presence | 0 | OFF | **ON** |

T-2 isolates the representation: presence in the forward and backward, no
structural search. T-3 adds the photometric-acceptance rounds. T-2 and T-3
**do not need frozen evidence heads** — they run with the evidence machinery
inactive (no `elgs_tracks_dir`), which is why the head-freeze blocker
([[elgs-evidence-head-freeze-packet-2026-08-18]]) does not gate this
experiment. **Stated explicitly because a later reader will assume the heads
were frozen and they were not.**

## 2. Matched across all four cells — no exceptions

| held identical | value |
|---|---|
| scene | DiVa-360 `scissor`, the 30-FPS materialization `scissor_screen_w0_561_s4_30fps` |
| split | the official shipped 35 train / 6 held-out (`0,16,17,33,43,44`) |
| window | 141 frames, `time_duration [0, 4.7]` |
| resolution | `resolution: 1` (1160x550), the official evaluation size |
| background | black, GT alpha-composited (the official convention) |
| initialization | the frozen visual hull ([[diva360-visual-hull-initialization]]) |
| image | `sudarshaniyengar/adags@sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683` — the admitted image, **all four cells, no mixing** |
| commit | one commit for all four, recorded |
| pool | one pool for all four, recorded; H100 and V100 runtimes are never compared as though hardware were controlled |
| iterations | 15,000, **from scratch** — no resume from any checkpoint |
| capacity policy | experiment 123's: `densify_from_iter` 500, `densify_until_iter` 6,000, `densification_interval` 100, `densify_grad_threshold` 2e-4, `densify_until_num_points` 600,000, `opacity_reset_interval` 30,000, `thresh_opa_prune` 0.005, `percent_dense` 0.01 |
| LR schedule | experiment 123's exactly, including `position_lr_max_steps: 6_000` unrescaled |
| losses | `lambda_dssim` 0.2, `lambda_dynamic_roi` 0.5, `lambda_static_exclusion` 0.02, all others as experiment 123 |
| motion | LoRA rank 8, 32 anchors, `motion_track_dt` 1/30 |
| **reserved-unit filter** | **applied to ALL FOUR** via `elgs_reserved_parity: True` on T-1/T-1' and the EL-GS path's own filter on T-2/T-3 |
| evaluation | official-convention DiVa-360 metrics on the 6 held-out cameras, PLUS matched training-view evaluation, PLUS the mechanism audit |

**Differing by design, and only:** `elgs_enable` (T-1/T-1' False, T-2/T-3
True), the seed (T-1' = 1, all others 0), and the photometric-round switch
between T-2 and T-3.

### The reserved-unit parity requirement, and why it is now satisfiable

The readiness audit's blocker was that an EL-GS cell trains on ~75% of the
training units while a bare control trains on 100%, so the control would get
a third more data on a benchmark whose open question is held-out
generalization. That is repaired: `elgs_reserved_parity` builds only the
stratified reservation and drops the same indices, with one shared
implementation of the `(frame_order + camera_order) % 4 == 0` rule used by
both paths (`elgs/trainer_hooks.py: build_reserved_pool`,
`reserved_indices_for_parity`; `tests/test_elgs_reserved_parity.py` asserts
both paths drop identical indices). **The flag is mandatory on T-1 and T-1'
and its effect must be verified in each run's log line
(`elgs_reserved_parity.reserved_units`) before any comparison is read.**

### The scalar-budget cap

`setup_elgs` sets `bundle.search_cost.scalar_budget` from the live row count
at setup times the row cap times 1.5. It bounds the structural search's
bookkeeping, not the model, and with structural rounds OFF on T-2 and T-3 it
is **non-binding by construction**. No control-side equivalent is therefore
required. This is recorded rather than left implicit because the readiness
audit listed "make the cap non-binding or apply it to the control
identically" as an open item, and the resolution is the first branch.

## 3. Decision rule — FIXED HERE, before any number exists

Let `D = held-out official PSNR(T-2 or T-3) - held-out official PSNR(T-1)`
and let `S = |PSNR(T-1) - PSNR(T-1')|` be the measured same-arm spread.

**Presence is "comparable" iff `|D| <= max(0.5 dB, S)`.**

| reading | consequence |
|---|---|
| comparable or better | the EL-GS substrate is photometrically viable; M2 substrate establishment proceeds on it |
| worse, **with** the same pathologies as the temporal substrate (temporal-support collapse, endpoint degradation) | a **representation decision goes to the user** before any evidence run |
| worse, **without** those pathologies | capacity and schedule work moves onto the presence substrate rather than the temporal one |

`S` is measured in this experiment and is not replaced by Appendix C's
0.00033 dB figure, which was measured at 600 iterations on N3V at 50k points
and does not transfer to 15k on DiVa-360 at 400k+. What Appendix C changes is
the **prior expectation** that `S` will be small on the admitted image — and
if `S` comes out large, that is itself a reportable finding about this
configuration.

**No dev-split ranking.** This is a single decision among four named cells,
not a search, so the dev-split discipline that caught the −0.17 dB false win
on the sweeps does not apply. Saying so in advance stops it being
reintroduced post hoc.

**Pathology definitions, fixed now** so "same pathologies" is not decided
after the fact: *temporal-support collapse* = the mechanism audit's
per-primitive temporal support distribution shifting toward the window
endpoints relative to T-1 by more than its own inter-cell spread;
*endpoint degradation* = held-out PSNR on the first and last 10 frames
falling more than the mid-window PSNR does, relative to T-1.

## 4. Cost

Four cells at 15,000 iterations from scratch. Experiment 123's own 9,000
resumed iterations are the only direct measurement available, so the estimate
is stated as an estimate: **~5–7 GPU-hours per cell, ~20–28 GPU-hours
total**, one pool. A preflight measurement of the first 500 iterations of one
cell is REQUIRED before the remaining three are submitted, and the projection
in the submission manifest must be revised to the measured rate rather than
left at the estimate. Appendix C recorded a projection that understated
actual by 2.4x because it counted training time only; that mistake is not
repeated by inheriting its projection method.

## 5. Provenance requirements

Per-cell: O_EXCL claim, content-hashed config, digest-pinned image, O_APPEND
ledger line, `evidence_bearing: false`, exploratory. All four cells' claim
indices, experiment ids, pool, commit, and measured wall-time recorded before
any comparison is read. Scheduler completion is not scientific completion:
each cell's `summary.json` is hashed and its recorded commit, config hash and
image digest checked before the number enters the comparison.

## 6. THE REMAINING BLOCKER — there is no way to turn structural rounds off

**Verified in code, and it blocks T-2 and T-3 as specified.**

Structural rounds fire from the frozen schedule:
`elgs/trainer_hooks.py:770` runs a round whenever
`state.runtime.is_round_boundary(iteration)`, and the boundaries come from
`configs/elgs/prereg_structural_v1.json`'s `schedule.full.round_iterations =
[3000, 4500, 6000]`. **There is no configuration switch that disables them.**
With `elgs_enable: True` and no `elgs_tracks_dir`, the evidence machinery
stays inactive but `_propose_smoke_candidates` still proposes and the rounds
still execute — so a 15,000-iteration cell would run SMOKE-TIER structural
search machinery inside what is supposed to be a clean representation
comparison.

Consequences:

* **T-2 ("rounds OFF") cannot be run today at all.**
* **T-3 ("photometric rounds ON") would run the smoke proposer**, which is
  declared smoke-tier supporting machinery and is not what "photometric
  rounds" is meant to denote.

**Minimal repair, specified but deliberately NOT implemented here:** a config
flag — `elgs_rounds_enabled`, default `True` so no existing lane changes —
that makes `is_round_boundary` return `False` for every iteration when unset,
leaving seeding, the presence representation, the reserved-unit reservation
and the refit path untouched. The frozen prereg's `round_iterations` is not
edited; only whether the trainer consults it.

It is not implemented in this block for the same reason the reserved-unit
switch was not improvised at launch: **it changes what a cell means**, so it
must land against a signed spec rather than ahead of one. This spec is that
document, and the flag should land with its sign-off.

Second-order note: the prereg's `refit_until` is 10,000 and
`run_post_refit_classification` only fires when `state.committed_decisions` is
non-empty. With rounds disabled there are no committed decisions, so the
post-refit path is inert by construction and needs no separate switch. That
was checked rather than assumed.

## 7. What this experiment cannot establish

* **That EL-GS's evidence mechanism works.** All four cells run with the
  evidence machinery inactive. This measures the presence REPRESENTATION's
  photometric cost, nothing else.
* **A SOTA placement.** Scissor at 141 frames is not comparable to published
  scissor at 1,125 frames ([[diva360-protocol-parity-audit]]); only internal
  deltas are readable.
* **Transfer.** One sequence, one scene, one split.
* **That a comparable result licenses M2.** M2's preconditions also include a
  supply gate or R3, frozen heads, and one reproduced baseline, none of which
  this experiment touches.

## 8. Termination

The experiment ends when four cells are terminal, `S` is measured, `D` is
computed for T-2 and T-3, and the section 3 rule is applied. No further cells
are authorised by this page under any outcome, and a disappointing result
licenses no follow-up run without a new spec.

---

## REVIEW ROUND 1 (2026-08-18) — VERDICT: **BLOCKED**, and the blocker is a THIRD one this spec did not know about

A fresh-context adversarial reviewer with no prior project context read this
spec and the implementation and returned **BLOCKED**. Recorded append-only.
**Section 6's rounds blocker is not the binding one.**

### BLOCKING — the `elgs_a` optimizer parameter group crashes ordinary densification

**Verified independently in source by the primary agent before recording.**

`elgs/trainer_hooks.py:684` registers the a-logits as an optimizer parameter
group named `elgs_a` whose `params` is `list(state.runtime.logit_parameters()
.values())` — **one tensor per family**, so its length is the family count, not
1. `scene/gaussian_model.py`'s `cat_tensors_to_optimizer` (`:1654-1664`) asserts

```
assert len(group["params"]) == 1, f"Group {name} has more than one param"
```

and its skip-list is `("gate_mlp", "gate_params", "motion_lora_basis",
"motion_scaffold_coeff", "motion_scaffold_basis")` — **`elgs_a` is not in it.**
`_prune_optimizer` (`:1545-1580`) has the same skip-list and indexes
`group["params"][0][mask]`, which is equally wrong for a per-family tensor.

And the group exists from the start: `setup_elgs` calls `seed_families(...,
iteration=0)` at `:311`, which calls `_refresh_logit_param_group` at `:650`.
The M0 smoke reports **131 families** on a 50k-point cloud, so
`len(group["params"])` is 131 at iteration 0.

Under **this spec's own capacity policy** — `densify_from_iter` 500,
`densification_interval` 100, from scratch — the first densification fires at
**iteration 600** and hits that assert. **T-2 and T-3 as specified crash at
iteration 600.**

**Why it has never been seen:** no configuration in the repository has ever
combined `elgs_enable: True` with active densification. Every EL-GS config sets
`densify_until_iter` below `densify_from_iter` (`smoke_elgs.yaml`: 400 vs the
default 500), and every densifying config sets `elgs_enable: False`. No test
covers it either — `tests/test_elgs_trainer_setup.py` stubs the Gaussian model
and never calls the real `densify_and_prune`.

**Section 6's proposed `elgs_rounds_enabled` repair does not touch this.**
Disabling rounds stops the structural search; it does not remove the `elgs_a`
group, which is created at seeding.

This is the most valuable finding of the block: the spec would have consumed a
GPU-slot claim, crashed 600 iterations in, and the failure would have looked
like an infrastructure problem rather than a design one.

### MATERIAL — and the second one is a trap this spec set for itself

1. **The mandatory 500-iteration preflight stops one densification interval
   short of the crash.** Densification begins at 500 and the first interval
   boundary is 600, so a 500-iteration preflight would complete cleanly and
   **falsely appear to validate the configuration**. Any preflight must run past
   the first densification boundary — at least 700 iterations.
2. **"Temporal-support collapse" is degenerate for T-2/T-3 as specified.** With
   K = 1 latched full-span families and rounds off, per-primitive temporal
   support is constant by construction, so the pathology cannot be observed on
   the cells it is meant to discriminate. The definition needs replacing with
   one that is measurable under the actual cell configuration, or the branch it
   feeds needs restating.
3. **The reserved-unit parity test's decisive assertion is weaker than this
   spec claims.** `tests/test_elgs_reserved_parity.py`'s
   `test_both_paths_drop_identical_indices` compares the control path against a
   hand-built state carrying the shared pool, and pins the link to `setup_elgs`
   by **source-text pattern matching** rather than by executing `setup_elgs`.
   The reviewer traced the mechanism manually and found it correct, so the claim
   is true — but the test does not prove it end to end, and this spec should say
   so rather than cite the test as if it did.
4. **The decision rule's formula is symmetric while its outcome table is
   one-sided.** `|D| <= max(0.5 dB, S)` treats a large POSITIVE `D` (presence
   much better) as "not comparable", yet the table's first row is "comparable
   **or better**". A large positive `D` is therefore technically undefined.
   Needs a one-sided union: comparable iff `D >= -max(0.5 dB, S)`.
5. **`S` is a single-replicate estimate with no ceiling.** One pair gives one
   difference; an unlucky T-1/T-1' pair with a large `|D|` would inflate `S` and
   could launder a real regression as "comparable". Needs a stated ceiling above
   which `S` is treated as evidence of an unstable configuration rather than as
   a tolerance.

MINOR: the endpoint-degradation definition is operational as written, and the
cost estimate's transparency is fine once the blocking finding is closed.

### Status

**BLOCKED. Not repaired in this block.** Unlike the audit preregistration, this
spec's blocker is a code defect in `scene/gaussian_model.py`, and repairing it
means changing the optimizer's densification path for every EL-GS run — which
is a larger and more consequential change than a flag, and one that should not
be improvised at the end of a block against an unsigned spec.

**The launch preconditions are now three, not two:**

1. the `elgs_a` densify/prune defect repaired **and tested against real
   `densify_and_prune`**, not a stub;
2. the `elgs_rounds_enabled` flag landed (section 6);
3. the MATERIAL findings 1–5 addressed in a revised spec, and a second review
   round.

Only then does user compute authorization become meaningful. **Do not launch
the triple.**

---

## REVISION 2 (2026-08-18) — the five MATERIAL findings repaired; still NOT LAUNCHED

Append-only. Nothing above is rewritten; where a rule changes, the superseded
rule is named and left standing. Status stays **FROZEN, NOT SIGNED, NOT
LAUNCHED**, and the launch preconditions are unchanged in number.

### R2.1 (finding 1) The preflight must cross the crash, not stop before it

Section 4's "first 500 iterations" preflight is **SUPERSEDED**. Densification
begins at 500 and its first interval boundary is 600, so a 500-iteration
preflight terminates cleanly one interval short of the `elgs_a` defect and
**falsely validates the configuration**. That is the trap the spec set for
itself.

**The preflight is now ≥ 700 iterations and must satisfy all five:**

| requirement | check |
|---|---|
| crosses the first densification boundary | iteration reached ≥ 700 |
| a real densify/prune actually fired | the row count changed at least once at an interval boundary, read from the run log — not merely "no exception" |
| rounds are disabled where specified | T-2's log shows zero structural rounds and zero smoke proposals |
| reserved-unit parity holds | `elgs_reserved_parity.reserved_units` present and equal on the control and EL-GS paths |
| optimizer-group integrity | the `elgs_a` group still holds one tensor per live family after densify/prune, and its Adam moments were not silently reset |

A preflight that raises, or that reaches 700 without a single densify/prune
event, does **not** license the remaining cells. Its measured rate — not the
section 4 estimate — becomes the submission manifest's projection.

### R2.2 (finding 2) Temporal-support collapse is RETIRED for these cells, and may not be reported as a result

With K = 1 latched full-span families and structural rounds off, per-primitive
temporal support is **constant by construction**. The distribution is a point
mass, so the pathology cannot be observed on the two cells it exists to
discriminate. Section 3's `temporal-support collapse` definition is therefore
**RETIRED for T-2 and T-3**.

**Binding prohibition:** if this quantity is computed on T-2 or T-3 anyway, it
is a structural artifact of K = 1 and **must not be reported as a result,
cited as evidence of stability, or read as the absence of a pathology.** A
degenerate metric returning "no collapse" is not a finding.

It remains defined and measurable on **T-1 and T-1'**, where the temporal
marginal is live, and is reported there as a control-side descriptive only.

**The pathology set for the section 3 branch becomes two measurable items:**

1. **Endpoint degradation** — unchanged and already operational: held-out PSNR
   on the first and last 10 frames falling more than the mid-window PSNR does,
   relative to T-1.
2. **Per-frame held-out PSNR profile divergence** — NEW, defined here before
   any number exists. Compute per-frame held-out PSNR over the 141-frame
   window for every cell. The profile diverges iff the mean absolute per-frame
   difference between the EL-GS cell and T-1 exceeds the mean absolute
   per-frame difference between T-1 and T-1'. That is a same-arm-calibrated
   comparison, it is measurable in both substrates, and it needs no structural
   variation to be non-degenerate.

Item 2 replaces item 1's lost discriminating power rather than adding a third
criterion; the branch still asks one question — *does the EL-GS cell fail in
the same shape the temporal substrate does?*

### R2.3 (finding 3) The parity claim is downgraded to what the test actually proves

Section 2 cites `tests/test_elgs_reserved_parity.py` as if it demonstrated
end-to-end that both paths drop identical indices. **It does not.**
`test_both_paths_drop_identical_indices` compares the control path against a
hand-built state and pins the link to `setup_elgs` by **source-text pattern
matching**, not by executing `setup_elgs`. Round 1's reviewer traced the
mechanism by hand and found it correct, so the *claim* is true — but the test
does not establish it, and this spec must not cite it as though it did.

**Launch precondition, added:** replace the source-text assertion with an
**executable mechanism test** that calls the real `setup_elgs` and observes
the reserved indices it actually installs, asserting equality with the control
path's. Until that test exists and passes, the parity requirement rests on a
manual trace, and any comparison read from these cells must carry that
qualification.

### R2.4 (finding 4) The decision rule is made directionally consistent

Section 3's `|D| <= max(0.5 dB, S)` is **SUPERSEDED**. It is symmetric while
its own outcome table's first row reads "comparable **or better**", which
leaves a large positive `D` formally undefined.

**The rule is now one-sided, matching the table:**

> Let `tol = max(0.5 dB, min(S, S_max))`. Presence is **comparable or better**
> iff `D >= -tol`. It is **worse** iff `D < -tol`, and only then does the
> pathology branch decide.

| reading | consequence |
|---|---|
| `D >= -tol` | comparable or better — the EL-GS substrate is photometrically viable; M2 substrate establishment proceeds on it |
| `D < -tol`, **with** a pathology from R2.2 | a representation decision goes to the user before any evidence run |
| `D < -tol`, **without** those pathologies | capacity and schedule work moves onto the presence substrate rather than the temporal one |

**One safeguard added, because the cells differ only in the presence
representation.** If `D > +1.0 dB`, the result is **not** read as a
representation win until a matched-configuration audit confirms the four cells
were in fact matched. A gain of that size from swapping the presence
representation alone is more likely a configuration mismatch than a finding,
and the cheapest time to notice is before the claim is written.

### R2.5 (finding 5) `S` gets a ceiling, and the justification is stated

`S` is a **single-replicate** estimate: one pair, one difference. With no
ceiling, an unlucky T-1/T-1' pair inflates the tolerance and could launder a
real regression as "comparable". That is the failure the replicate was added
to prevent, so leaving it uncapped defeats the fourth cell's purpose.

**`S_max = 1.0 dB`, frozen here.** The justification, so the number is not
arbitrary:

* it is **2× the decision rule's own 0.5 dB floor**, so `S` can at most double
  the comparison band and never turn it into a rubber band;
* it is roughly **10× the largest same-arm spread ever measured in this
  project on this metric** — the old renderer image's 0.10446 dB
  ([[renderer-integrity-admission-2026-08-18]] Appendix C) — and ~3,000× the
  admitted image's 0.00033 dB, so it is generous against every measurement on
  record rather than tuned to a hoped-for outcome.

**If `S > S_max`, the comparison is VOID.** The configuration is unstable at
15k on this scene, no comparability verdict may be recorded in either
direction, and `S` itself is reported as the finding and returned to the user.
A void here is a real outcome, not a failure to obtain one.

Note that `tol` uses `min(S, S_max)`: `S` above the ceiling voids the
experiment rather than widening the band, so the two rules cannot be played
against each other.

### R2.6 What revision 2 does NOT change

The four cells and their matching table; the scene, split, window, resolution,
background, initialization, image digest, iteration count and capacity policy;
the reserved-unit parity requirement itself; the 0.5 dB floor; the provenance
requirements; section 7's limits; and section 8's termination rule. The
`elgs_a` defect and the `elgs_rounds_enabled` flag remain launch preconditions
1 and 2 and are **not** closed by this revision.

### R2.7 Launch preconditions after revision 2

1. the `elgs_a` densify/prune defect repaired and tested against the **real**
   `densify_and_prune`, not a stub;
2. `elgs_rounds_enabled` landed, default `True`;
3. the executable reserved-parity mechanism test of R2.3 landed;
4. a ≥ 700-iteration preflight meeting all five R2.1 checks;
5. one fresh-context re-review of this revision;
6. explicit user compute authorization for ~20–28 GPU-hours.

**Do not launch the triple.** Revision 2 repairs the specification; it does
not authorize the experiment.

---

## REVIEW ROUND 2 (2026-08-18) — VERDICT: **BLOCKED**. The experiment cannot answer the question it poses

Append-only. A fresh-context adversarial reviewer read revision 2 and returned
**BLOCKED** with four blocking findings. **Do not launch. The integration
preflight is NOT authorized either**, because three of its five checks are
themselves among the defects.

### B1 — T-2 has no presence representation to measure. PRIMARY-VERIFIED.

This is the finding that matters, and R2.2 walked right past it: R2.2 correctly
identified the K=1 degeneracy **in the metric** and stopped there. It is a
property of **the cell**.

The frozen `prereg_structural_v1.json` `family_seeding` fixes *"each nonempty
voxel cell is one family with a K=1 spanning program, latch pattern (1,1),
`dim(a)=1`"*. And `elgs/intervals.py:189` computes
`sigma = torch.softmax(state.a, dim=0)`.

**Softmax over a one-element vector is identically `[1.0]` with gradient
exactly `0.0`.** Verified directly in the project environment:

```
softmax over dim(a)=1 -> [1.0]
gradient             -> [0.0]
```

The reviewer ran the full realization at the spec's own `time_duration [0,4.7]`
/ 141 frames / `dt = 1/30` and found presence **exactly 1.0 at every frame**,
the realization **invariant across a ±50 logit range**, and
`d(sum presence)/da = 0`.

So on T-2 the `elgs_a` optimizer group is a group of parameters **that can
never move**, and latching, multi-episode structure and exact-zero absence —
the three things that make this representation the method — are all **inert**.

**What T-2 actually measures is the cost of DELETING the temporal-Gaussian
marginal** (presence *replaces* `get_marginal_t` in the opacity product). That
is a legitimate ablation. It is not the stated question, and section 3's "the
EL-GS substrate is photometrically viable" inference is **not licensed** by a
cell in which the representation reduces to the identity.

**A sharp irony worth recording:** the `elgs_a` optimizer repair landed this
block is correct and necessary — it prevents a real crash and is required the
moment `K >= 2` — but in the exact configuration this spec calls for, the
parameter group it protects is provably inert.

### B2 — the cells are not capacity-matched in the sense the question requires

[[exp123-mechanism-audit]] measured the control substrate on this exact scene:
median temporal scale 0.0073 s, **6.2% of primitives active at a typical
frame**. Under B1, T-2/T-3 have **100%** active at every frame — a **~16×**
difference in effective per-frame capacity, introduced by construction and
uncontrolled. Section 2 matches the *stored* budget and calls that matched; the
project's standing G5 gate is explicit that this is insufficient. Worse,
exp123's own reading predicts the confound's **direction**: T-2 plausibly
*gains*, and that gain would be read as representation viability.

### B3 — `D` as specified is not producible for T-2/T-3

`scripts/eval_diva360_heldout.py` restores a checkpoint but never calls
`setup_elgs`, and `GaussianModel.restore` only *stashes* the EL-GS payload. So
`elgs_runtime is None`, `elgs_active` is False, and the renderer takes the
`get_marginal_t` branch — **T-2/T-3 would be scored through the untrained
temporal marginal**, parameters that never gated opacity during their training.
`scripts/audit_mechanism.py` has the identical defect, which also invalidates
its `active_points` statistic on those cells.

### B4 — three of the five preflight checks are vacuous or impossible

* **check 3 is vacuous at ≥700 iterations** — round boundaries are
  `[3000, 4500, 6000]`, so no round can fire at ≤700 whether the flag is set or
  not, and `rounds_enabled` is not in the setup log line at all. *This repeats
  round 1's own mistake: a horizon chosen short of the thing it claims to test.*
* **check 4 is impossible** — `reserved_indices_for_parity` returns `None`
  whenever `elgs_enable` is True, so `elgs_reserved_parity.reserved_units` can
  never appear on the EL-GS path and nothing equivalent exists there.
* **check 5 is not observable** from any emitted artifact — `moment_reset_log`
  goes only into the binary checkpoint.

### Material findings

Ten in total. The load-bearing ones: **M4 — `S_max` is calibrated against the
wrong quantity.** Every Appendix C figure is a *same-seed* replicate, while `S`
is a *different-seed* spread this project has twice recorded as never measured.
The arithmetic is fine; the comparison is not. **M2 — the replacement pathology
is near-tautological**, so the "worse without pathologies" row is effectively
unreachable. **M3 — per-frame held-out PSNR is emitted by nothing**, so R2.2's
replacement metric is not implementable against the specified artifacts.
**M6 — the outcome table still has no kill branch**, leaving a clean path from
a capacity-confounded result to a representation claim. **M8 — no config file
exists for any of the four cells**, and `elgs_smoke_schedule` is unspecified
though it decides whether any round fires at all.

### What round 2 verified as correctly repaired

R2.1's arithmetic (first densify/prune genuinely at iteration 600, and 700 does
cross it; checks 1 and 2 are satisfiable); **R2.3's downgrade is honest and if
anything understates**; R2.4's one-sided rule is directionally consistent with
its table; R2.2's degeneracy claim is true for T-2 and *stronger* than stated;
section 1's evidence-inactivity claim; section 2's scalar-budget argument; and
that densified rows inherit family ids correctly.

**Launch preconditions 1 and 2 are closed** by commit `c4ff0d4`, which the
reviewer independently confirmed — with the note that the implementation
short-circuits the caller and the proposer rather than making
`is_round_boundary` return `False` as section 6 specified. R2.7's list is stale
in that respect.

**A provenance correction, since the reviewer misread it:** commit `c4ff0d4`
was made by the **primary**, after inspecting the diff, reverting the repair to
confirm the tests catch the defect, and receiving an independent code review of
**APPROVE**. No subagent staged, committed or pushed anything in this block;
both reviewers were read-only and both correctly remained so.

### Status

**BLOCKED at round 2, and the block is deeper than revision 2's.** B1 says the
experiment as specified cannot answer its question at all, and no wording
repair reaches that. Either the cells are re-specified so presence has degrees
of freedom (`K >= 2`, or an unlatched pattern giving `dim(a) = 2`), or the
question is restated as what the cells actually measure — and B2 then requires
an added capacity control either way. **This returns to the user.**
