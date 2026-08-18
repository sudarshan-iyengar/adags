# Matched presence-substrate triple — readiness and blockers (2026-08-18)

Preparation only. **Nothing here authorises a launch**: the triple waits
on the M-2 instrument verdict and the user's event-supply route decision,
which together name the sequence. This page states what is ready, and
names the one implementation blocker that would silently invalidate the
comparison if it were launched today.

Reads [[elgs-experiment-plan]], [[renderer-integrity-admission-2026-08-18]],
[[elgs-m2-oncomponent-split-design]], [[elgs-evidence-head-freeze-packet-2026-08-18]].

## 1. The experiment

The EL-GS presence representation — latched presence replacing the
temporal marginal — has **never been trained beyond 220–600-iteration
smokes**. Every 6k–15k photometric number on record (experiments 84, 104,
123 and the sweeps) is the plain ADAGS temporal-Gaussian + LoRA
substrate, with `elgs_enable: False` throughout. So the representation the
method actually renders with is photometrically unmeasured.

Three cells, everything matched but the substrate:

| cell | substrate | structural rounds | photometric rounds |
|---|---|---|---|
| T-1 | temporal-Gaussian (control) | n/a | n/a |
| T-2 | EL-GS K=1 presence | OFF | OFF |
| T-3 | EL-GS K=1 presence | OFF | ON |

## 2. THE BLOCKER — a 33% training-data confound

**Verified in code, and it invalidates a bare comparison to experiment
123.**

`elgs/trainer_hooks.py:198-226` reserves a stratified ~25% of training
units at iteration 0: within each timestamp group, every unit with
`(frame_order + camera_order) % 4 == 0` is reserved for the §7
confirmation measure, rotating across cameras and spreading over time.
`main.py:1161` then calls `filter_elgs_reserved(training_dataset,
elgs_trainer_state)`, and `elgs/trainer_hooks.py:1092-1107` drops exactly
those indices from the refit dataset.

`filter_elgs_reserved` **returns the dataset unchanged when `state is
None`** (`:1099-1100`), and that state is built only when `elgs_enable`
is set. Therefore:

* an EL-GS cell trains on **~75%** of the training units;
* a temporal-substrate cell trains on **100%**.

The control would see **one third more training data than the cells it is
being compared against.** On a benchmark whose whole open question is
held-out generalization — experiment 123 measured train-view 26.5 against
held-out 23.2 — a 33% data advantage to the control is not a nuisance
term. It would plausibly produce the "EL-GS presence is worse" result all
by itself, and that result would be an artefact.

**There is currently NO configuration switch that applies the reservation
without enabling EL-GS.** This is the blocker.

**Minimal repair, recommended, NOT implemented here:** a flag that builds
only the stratified reservation and applies `filter_elgs_reserved`,
without constructing any other EL-GS state — the reservation arithmetic
at `:207-225` depends on nothing but the camera list, so it is separable.
The control then trains on the SAME 75% of the SAME units. This is a
small surgical change, but it changes what a control means, so it should
land with the triple's frozen spec and not be improvised at launch.

Rejected alternative: disclose the confound and compare anyway. A
comparison whose leading confound is known and fixable is not a matched
comparison.

## 3. What IS ready

* **The renderer.** Admitted image
  `sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683`,
  commit `d21f1e9`, verified on V100
  ([[renderer-integrity-admission-2026-08-18]]). All three cells must use
  this one image; mixing images across the triple would reintroduce
  exactly the gradient-provenance problem the repair closed.
* **Initialization.** The visual-hull init is frozen and reusable, worth
  +1.08 dB official.
* **Evaluation.** Official-convention DiVa-360 metrics were read from
  `brown-ivl/DiVa360` rather than inferred, and the four earlier protocol
  mismatches are fixed. Matched training-view evaluation and the
  mechanism audit both exist as procedures.
* **Provenance.** O_EXCL claims, content-hashed config, digest-pinned
  image, append-only ledger — all in place.

## 4. What is NOT ready

| item | status |
|---|---|
| 25% reserved-unit filter for the control | **BLOCKER**, §2 — no switch exists |
| the sequence | undecided; waits on M-2 + the user's route decision |
| evidence heads | cannot be frozen ([[elgs-evidence-head-freeze-packet-2026-08-18]]); T-2/T-3 as specified run with structural rounds OFF, so they do NOT need frozen heads — but this must be stated in the frozen spec, because a later reader will assume they were frozen |
| `se` citability | blocked by the confirmation-slot time collapse |
| ported baseline | none exists on DiVa-360; a 4DGS or STG smoke is a separate M2 precondition |

## 5. Matching requirements, for the frozen spec

Identical across all three: sequence and split; admitted image digest;
commit; hull initialization; capacity policy and point ceiling; seed;
hardware class (one pool, recorded); iteration count and schedule;
official-convention evaluation plus matched training-view evaluation;
**and the 25% reserved-unit filter, applied to all three or to none.**

Differing by design, and only: `elgs_enable`, and the photometric-round
switch between T-2 and T-3.

**No dev-split ranking is needed.** This is a single decision among three
named cells, not a search, so the dev-split discipline that caught the
−0.17 dB false win on the sweeps does not apply here — and saying so in
advance stops it being reintroduced as a post-hoc filter.

## 6. Decision rule, to be fixed BEFORE launch

Per the standing plan, and stated here so it is not written after the
numbers:

| reading | consequence |
|---|---|
| presence comparable or better on held-out at matched capacity | the EL-GS substrate is photometrically viable; M2 substrate establishment proceeds on it |
| worse, with the SAME pathologies as the temporal substrate (temporal-support collapse, endpoint degradation) | a representation decision goes to the user before any evidence run |
| worse, but the pathologies are GONE | capacity and schedule work proceeds on the presence substrate rather than on the temporal one |

## 7. Cost

~15–20 GPU-hours for the triple at 15k iterations, on one hardware class.
Not authorised by this page.
