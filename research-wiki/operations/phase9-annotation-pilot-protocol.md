# Phase 9 Annotation Pilot Protocol — prepared, NOT run

Date: 2026-07-30
Status: pilot materials prepared under workstream A. Execution is blocked
until the user resolves the four sign-off decisions in
[[operations/phase9-annotation-contract-draft]]. No labels exist; no packet
has been generated; nothing here has been shown to any annotator.
Parent: [[operations/phase9-csvl-vpl-v2-direction]];
[[operations/phase0-census2-result]] (the reference-set motivation).

## Purpose

Before the full >= 24-track annotation effort, test whether two human
annotators can consistently identify, from RGB alone: (1) the foreground
occluder region, (2) the hidden/rear surface region, (3) the occlusion onset
frame, and (4) the reveal frame — and measure agreement, ambiguity, and
non-evaluable rates, so the full rubric is validated (or revised) on ~10% of
the cost.

## Design

- Scene: `cut_roasted_beef` (development tier) only.
- Six 24-frame pilot windows, both annotators label all six (full overlap —
  a small pilot buys maximal agreement statistics, unlike the 20% overlap of
  the final set).
- Pilot windows, selected label-free by a deterministic rule and avoiding the
  historical R009 frame ranges (95-110, 140-155):

| Window | Frames | Camera |
|---|---|---|
| P1 | 40-63 | cam01 |
| P2 | 64-87 | cam05 |
| P3 | 115-138 | cam08 |
| P4 | 160-183 | cam13 |
| P5 | 205-228 | cam16 |
| P6 | 250-273 | cam20 |

  Cameras rotate across rig positions among the 17 evidence-healthy training
  cameras. `cam12`/`cam19` are avoided (their P01 evidence coverage is
  structurally deficient, and keeping pilot windows comparable to E1 evidence
  requires healthy cameras); `cam00` is the held-out test camera and is never
  opened; `cam04` is not a training camera in this scene.
- **Census-assisted supplement (decision 2, final set only — NOT the pilot):**
  the census-v2 certified-event activity outside R009 clusters at frames
  188-190 and 251-259, suggesting flagged supplement windows 180-210 and
  245-275 if the user opts into census-assisted selection after the forensic
  audit. Recorded here for the decision; carries the disclosed selection-bias
  caveat and the census's known noise domination.

## Pilot label schema (subset of the contract schema)

Per window, per annotator: occluder polygon at three keyframes; hidden/rear
surface polygon at last-fully-visible and first-fully-revealed frames; onset
frame; reveal frame; per-frame state of the tracked region
(`visible | partially_occluded | occluded | uncertain`); ordering assertion
(`occluder_in_front | uncertain_ordering`); per-field confidence
(`high | medium | low`); window-level `non_evaluable` escape with reason
(no occlusion event present / motion blur / ambiguous surfaces). `uncertain`
and `non_evaluable` are first-class outcomes, not failures.

## Metrics reported after the pilot

- onset and reveal frame deltas between annotators (median, max);
- occluder and hidden-surface polygon IoU at native resolution;
- per-frame state agreement (Cohen's kappa);
- ordering agreement fraction;
- fraction of frames marked `uncertain`; fraction of windows `non_evaluable`;
- free-text friction notes per field (what was hard to decide and why).

Pre-declared pilot success guide (advisory, not a gate): median onset/reveal
delta <= 2 frames, polygon IoU >= 0.5, state kappa >= 0.6 on evaluable
windows. Outcomes below this trigger rubric revision and a second pilot
round before full annotation, not a silent lowering of the bar.

## Quarantine rules

- Pilot labels live under
  `$WORK/proj_adags/runs/phase9-depth-visibility-capacity/annotation-pilot/`
  (outside the repository), hashed at creation.
- Pilot labels are never merged into the final locked reference. Pilot
  windows may re-enter the final set only if the final selection rule
  independently re-draws them, and any such overlap is documented.
- Pilot labels are evaluation-development material only: they may refine the
  rubric wording; they may not tune any method threshold.

## Execution procedure (for after sign-off; documented, not run)

1. Build the packet with the existing tooling
   (`depth_visibility/cvat_annotation.py: generate_cvat_annotation_templates`),
   pointed at the six pilot windows; packets contain training-camera RGB
   frames and frame indices only — no census output, no depth, no renders.
2. Each annotator labels independently in CVAT; no discussion until both
   submit.
3. Import with the fail-closed validators in `depth_visibility/annotation.py`;
   compute the metrics above; write the pilot result page with agreement
   tables and friction notes.
4. User reviews pilot outcomes and authorizes (or revises rubric before)
   full annotation.
