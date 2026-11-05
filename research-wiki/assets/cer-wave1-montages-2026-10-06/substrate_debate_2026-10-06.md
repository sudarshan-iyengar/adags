# Substrate debate, 2026-10-06 evening (Claude Opus 5.5 vs Fable 5.1, two rounds)

Question (principal): based on the Multi4D+CER montages and bank renders, what is the verdict on the ARS substrate? Should we
build on Multi4D instead of Ex4DGS, rectify what is happening there and find novelty?

Status: a recommendation to the principal, not a decision. Exploratory reasoning over recorded evidence; no new evidence-bearing run.

## Evidence added during the debate (descriptive)

- **Render-time check.** Ghost-region PSNR, frames 63/75/87 pooled: the full render against the persistent banks alone (the
  transient bank removed at render time only, not retrained).
  - Multi4D+CER:
    - bottle: 33.4→36.5, 30.9→33.4, 30.8→35.7;
    - carton: 34.3→31.9, 34.3→30.9, 34.0→33.1;
    - mug: 34.7→35.5, 35.2→35.5, 34.8→35.2.
  - Base Multi4D, persistent banks alone: 12.6–13.9 dB. That is the object itself.
  - STG gap window: 43.2–47.6 dB.
  - The script is inline in the session. Inputs are `agent-control/ars/cer/m4d_banks/renders` and the scene's ghost masks and
    cam00 frames.
- **`event_spec.json`.** The target is stationary in both same-pose and moved-return scenes.
  - Moved return: frames 0–59 at [6.25, 7.8, 11.6], absent 60–89, then frames 90–299 at [4.0, 7.8, 11.3].
  - The scenes differ in how long the original spot is occupied: 270 of 300 frames for same-pose and pass-through, 60 of 300
    for moved return.
- **Multi4D configuration.** `iterations` (20,000) and `position_lr_max_steps` (20,000) are separate fields;
  `densify_until_iter` is 16,000. `train.py` uses `iterations` only for the final evaluation and save.
- **Wave 2 was never rendered.**

## Round 1, Claude (summary)

1. Switch the primary substrate to Multi4D. Its pre-gap, return, whole-frame and static numbers match STG. Ex4DGS is 9–11 dB
   lower on the target, an inherited deficit that CER does not touch.
2. Base Multi4D represents disappearance as occlusion: a transient occluder painted over a persistent object. CER is the
   minimal fix.
3. The leftover transient patch is not the main deficit. The newly uncovered background is.
4. On wave 1 the return is fully observed, so memory cannot be rewarded. Show memory where it matters, for example with
   starved returns.
5. Make Multi4D+CER the headline arm; spend at most one bounded experiment, on a starved return.

## Round 1, Fable (summary)

- **Agrees:** Multi4D unambiguously.
- **Corrections:**
  - The memory cohort lives in the static background bank (bg ≈ 2,000–2,500 rows, fg ≈ 0), so the claim must be narrowed.
  - The return advantage over STG belongs to Multi4D's persistence, not to CER (34.69 against 34.73).
- **The gap deficit is a schedule artefact of the port.**
  - CER starts at 16k with densification frozen, so the uncovered background gets 4k iterations and no new rows.
  - It sits at 33–36.5 dB, against 44.4 dB for ordinary static background in the same model.
- **The central question.** Whether an exact-memory model can match temporal opacity in the gap is the paper's central
  question, not engineering.
- **Priority.** Starved returns are a wave-2/CVPR item, not a nine-day item.
- **Proposed probe:** three bottle same-pose CER runs with fine iterations 20k→30k, read on persistent-alone gap PSNR.
- **One factual error:** "on moved returns the object is dynamic". It is not.

## Round 2, Claude (summary)

- **Concedes:** the return advantage; starved returns out for this window (wave 2 was never rendered); the gap is the central
  question.
- **Corrects:**
  - The mechanism is the duty cycle of the spot, not object motion.
  - The STG return drop (1.5–1.7 dB below its own pre-gap) is at least as plausibly an edge effect of STG's 50–99 segment
    (n = 1). Do not claim it before the one-model STG run reads.
- **Checks the probe knob:**
  - It is clean before 16k.
  - After 20k the position learning rate stays at its floor.
  - The scorer refuses non-standard schedules.

## Round 2, Fable (summary) and consensus

- **Duty-cycle version accepted**, with a prediction already consistent with the table:
  - On moved return the new spot is occupied 210 of 300 frames.
  - Base Multi4D's return there is 1.0–2.6 dB below STG, where on same-pose it is above STG.
  - Claude adds: its pre-gap target at the old spot, occupied 60 of 300 frames, is 28.5 / 33.2 / 29.3 dB, against STG's
    34.8 / 38.7 / 35.0.
- **STG return drop:** withdrawn as evidence.
- **The probe rule must be asymmetric:**
  - **≥ 40 dB persistent-alone in 2 of 3 runs** means the deficit was iterations. A confirmatory longer schedule follows, with a
    matched longer base.
  - **Otherwise** the deficit is unresolved between capacity and the learning-rate floor. The early-gate port (labels carried
    through clone/split and the foreground-to-transient lift) is the only route that addresses both, or the item leaves the
    workshop paper.
  - Do not also raise `position_lr_max_steps`.
  - Read the full-render gap as well.
  - Anchor: STG's bottle gap is 44.85 dB.
- **Schedule:** write the port code at zero GPU while the probe trains.
- **Consensus:**
  - (a) Substrate: Multi4D headline, Ex4DGS+CER as replication.
  - (b) Claim:
    1. the duty-cycle mechanism;
    2. an exact memory gate removes the object from memory for the gap without harming the return, on two substrates;
    3. the return advantage belongs to Multi4D;
    4. benchmark rule 2 fails on both substrates in the gap, by 9–13 dB against STG, stated plainly with the decomposition.
  - (c) Next action: the three-run probe with the amended rule, the principal's call.
