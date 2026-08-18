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
