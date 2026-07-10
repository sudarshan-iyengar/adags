# Visibility-Gate Method Tracker

Date: 2026-07-10

## Purpose

This tracker owns the next mechanism-changing hide/reveal attempt after R030 and R031-R033. The goal is not to find a nicer support mask. The goal is to test whether a Gaussian state can remain persistent while its visibility is gated through a temporary occlusion/reveal event.

## Evidence Boundary

- R013/R015: image-level oracle upper bound only; GT crop pixels were blended into final outputs.
- R017: oracle-support runtime opacity attenuation failed.
- R025: non-oracle candidate-local posthoc refinement failed despite meaningful R020 overlap.
- R027: non-oracle boundary micro-densification had a tiny directional gain but failed the predeclared gate.
- R030: oracle crop support plus Gaussian-only posthoc micro-densification still failed.
- R031-R033: DA3 support sidecars are reusable, but current hard support masks localize frozen windows weakly.

Therefore the next method must alter representation/training dynamics: persistent state plus visibility gating/admission, integrated during training, compared against a matched smooth control.

## Candidate M3: Training-Loop Persistent Visibility Gate

### Hypothesis

For local event candidates, an `H_event` model that keeps position/motion/appearance state available while multiplying opacity by a time-dependent visibility gate `v_i(t)` can better handle temporary occlusion/reveal than ordinary visible smooth transport, provided the event is accepted only when it beats `H_smooth` on frozen training-time criteria.

### Non-Oracle Cues Allowed

- Route0 dynamic/static disagreement over full training frames.
- Dynamic-mask interior/boundary sidecars.
- Flow validity/magnitude boundaries where sidecars already exist.
- Route0 temporal flicker over full frames.
- DA3 depth/confidence sidecars as soft reliability cues only if frozen before scoring.
- Ordinary training-image photometric losses, used to optimize/evaluate hypotheses on training observations.

### Prohibited Inputs

- Frozen R009 crop coordinates as method support.
- GT crop pixels copied into final renders.
- Thresholds selected by looking at frozen-window overlap or frozen-window metrics.
- Any posthoc change to the real method after seeing R037 frozen-window results, except to declare FAIL and start a new predeclared candidate.

## Matched-Budget Design

`H_smooth`:
- Same route0 checkpoint/backbone, data, iterations, and point budget.
- No event visibility gate.
- Receives the same candidate support only for budget accounting or matched local capacity allocation if needed.

`H_event`:
- Same route0 backbone.
- Adds event-local visibility parameters/gates that multiply opacity before rasterization.
- Keeps position/motion/appearance persistent through hidden intervals.
- Uses hysteresis so accepted event state cannot flicker every frame.
- Allocates/reinitializes local capacity inside the original training loop, not as a short posthoc patch.

## Frozen Admission Rule

Before real-window scoring, freeze:
- candidate-field generation command and parameters;
- local photometric/temporal consistency score;
- event-vs-smooth acceptance margin;
- hysteresis enter/exit margins;
- point/capacity cap;
- iteration budget;
- static-exclusion safeguard.

## Run Plan

| Run | Question | Required Output | Gate |
| --- | --- | --- | --- |
| R034 | Does `H_event` separate synthetic hide/reveal from smooth motion under matched budget? | synthetic metrics, accepted-event log, identity/reconnection audit | PASS before real pilot |
| R035 | Can the frozen non-oracle candidate field be generated on real source scenes? | candidate manifest, validation, guardrail metadata | PASS before real pilot |
| R036 | What is the matched real smooth-control result? | checkpoints, eval renders, frozen-window metrics | COMPLETE before comparing R037 |
| R037 | Does visibility gating improve the frozen real windows without leakage? | checkpoints, eval renders, accepted-event stats, frozen-window metrics | PASS/FAIL by strict gate |
| R038 | Are the claims supported and the protocol clean? | result-to-claim and experiment-audit notes | Required before any positive claim |

## PASS / FAIL

PASS requires a non-oracle Gaussian-rendered method that improves the frozen windows and recovers a meaningful fraction of the oracle upper bound according to `refine-logs/EVENT_CROP_METHOD_TRACKER.md`.

FAIL is acceptable and scientifically useful if R034/R037 are valid and do not improve over baselines. In that case, preserve the logs, summarize the failure, and start a new method tracker for the next predeclared mechanism.

BLOCKED only applies after three serious attempts at the same implementation or infrastructure blocker.
