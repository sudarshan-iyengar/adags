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

## 2026-07-10T02:32+02:00 Update

### R034 synthetic fixture: PASS

Local run:

```bash
python scripts/run_hide_reveal_poc.py synthetic --out-dir refine-logs/hide_reveal_poc/r034_visibility_gate_synthetic --seeds 0 1 2 3 4 5 --clips-per-type 8
```

Held-out metrics in `refine-logs/hide_reveal_poc/r034_visibility_gate_synthetic/synthetic_summary.json`:

- candidate recall: `1.0`
- accepted precision / recall: `1.0` / `1.0`
- false event rate on normal controls: `0.0`
- margin AUC: `1.0`
- identity reconnection accuracy: `1.0`
- `proceed_to_real_windows=true`

Interpretation: the synthetic fixture is not a real-result claim, but it passes the predeclared sanity gate for trying one real matched pilot.

### R035 proxy admission: FAIL

HPC command:

```bash
python scripts/run_hide_reveal_poc.py admit-visibility-events --candidate-manifest refine-logs/hide_reveal_poc/r020_high_recall_motion_supported_nonoracle_candidates/nonoracle_candidate_manifest.json --out-dir refine-logs/hide_reveal_poc/r035_visibility_event_admission --admission-margin 0.0005 --lambda-temporal 0.25 --lambda-budget 0.0001 --opacity-attenuation 0.85 --dynamic-probability-min 0.55 --event-beta 1.0
```

Output summary:

- candidates scored: `72`
- accepted: `0`
- validation: `ok=true`
- mean smooth score: `0.0183901`
- mean event-proxy score: `0.219373`
- mean delta score: `+0.200982`
- minimum delta score: `+0.118458`

Interpretation: the cheap image-space hypothesis "attenuate the dynamic contribution now and score the crop" is not defensible on the real training observations. This failure should not be rescued by weakening the frozen margin.

### R036/R037 revised real pilot

Because R035 rejected every candidate, the full pilot is revised before any R036/R037 frozen-window scoring:

- `H_smooth`: use the same fixed R020 high-recall candidate field for local ROI/capacity pressure, but no event opacity gate.
- `H_event`: use the same fixed R020 high-recall candidate field plus the visibility-event opacity gate during the original 6000-iteration training loop.
- No frozen crop labels or frozen-window metrics are used to select the R020 field or tune thresholds.
- PASS/FAIL remains the predeclared frozen-window gate against route0, residual, matched-lifespan, and oracle-gap recovery.

This is a new method attempt, not a relaxation of R035.

## 2026-07-10T06:12+02:00 Update

### R036/R037 training: COMPLETE

The matched full-training pilot ran through the original 6000-iteration loop rather than a 200/400-step posthoc patch.

R036 `H_smooth`:

- manifest: `refine-logs/visibility_event_smooth_train_jobs_20260710_024626.tsv`
- jobs: `49042444`, `49042445`, `49042446`
- status: all `COMPLETED`, exit `0:0`
- checkpoints: all three `chkpnt6000.pth` observed under `$WORK/proj_adags/runs/visibility_event_smooth_control_6000/`

R037 `H_event`:

- manifest: `refine-logs/visibility_event_train_train_jobs_20260710_024651.tsv`
- jobs: `49042510`, `49042512`, `49042514`
- status: all `COMPLETED`, exit `0:0`
- checkpoints: all three `chkpnt6000.pth` observed under `$WORK/proj_adags/runs/visibility_event_train_6000/`

### Eval state

R036 smooth eval was submitted with `EVAL_TIME=01:30:00`:

- jobs: `49045923`, `49045924`, `49045925`
- manifest: `refine-logs/visibility_event_smooth_eval_jobs_20260710_060900.tsv`

R037 event eval has not been submitted. SSH to Leonardo now fails because the loaded certificate expired at `2026-07-10T06:08:38`.

This is an infrastructure BLOCKED state, not a PASS/FAIL result. The method verdict must wait for:

1. R036 eval completion check.
2. R037 eval submission and completion.
3. One frozen-window scoring run against route0, residual/uncertainty, matched-lifespan, R036, and R037.
4. R038 result-to-claim / experiment-audit review.

No frozen-window metrics have been inspected for R036/R037 yet, and no threshold has been changed after training.

## Resume / Scoring Guardrail

When SSH access is renewed, R037 event eval must be submitted before any frozen-window verdict is made. The final R036/R037 scoring manifest must be built by adding `visibility_event_smooth_control` and `visibility_event_train` to `refine-logs/hide_reveal_poc/r029_r030_disambiguation_manifest.json`, because that manifest already contains the strict comparison systems:

- `route0`
- `residual_uncertainty`
- `matched_lifespan`
- `hide_reveal` oracle upper bound
- prior controls `event_boundary_micro_densify`, `oracle_crop_support_micro_densify`, and `route0_continue_6400`

Do not score from `hide_reveal_real_windows.json` alone, since that would only include `route0` and would weaken the predeclared gate.

## Blocked Audit

As of `2026-07-10T06:18:12+02:00`, the same Leonardo SSH credential blocker has repeated across three consecutive goal turns. The event-gated method has not received a PASS or FAIL verdict because R037 eval and frozen-window scoring require renewed HPC access.

Blocked items:

- Check whether R036 smooth eval jobs `49045923`, `49045924`, `49045925` completed.
- Submit R037 event eval from `refine-logs/visibility_event_train_train_jobs_20260710_024651.tsv`.
- Build and validate `r036_r037_visibility_event_manifest.json`.
- Run final frozen-window scoring and R038 result-to-claim / experiment-audit.

## 2026-07-10T07:55+02:00 Final Verdict

SSH access was renewed and the blocked items above were completed.

R036 smooth eval:

- jobs: `49045923`, `49045924`, `49045925`
- status: all `COMPLETED`, exit `0:0`
- eval folders: complete, 300 frames each under `test/ours_6000/{renders,gt,static,dynamic}`

R037 event eval:

- jobs: `49051779`, `49051782`, `49051783`
- status: all `COMPLETED`, exit `0:0`
- eval folders: complete, 300 frames each under `test/ours_6000/{renders,gt,static,dynamic}`

Final strict scoring artifacts:

- manifest: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.json`
- validation: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_manifest.validation.json`, `ok=true`
- metrics: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_real_eval/`
- logs: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_logs/`
- crop strips: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/crop_strips/`
- decision memo: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_decision_memo.md`
- result-to-claim: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/result_to_claim.md`
- integrity audit: `refine-logs/hide_reveal_poc/r036_r037_visibility_event_summary/experiment_audit.md`

Gate result for R037 `visibility_event_train`:

- mean PSNR `30.1089` versus route0 `30.5021`
- mean L1/proxy-LPIPS `0.0157600` versus route0 `0.0148316`
- route0 PSNR+L1 wins `0/5`
- strict all-baseline PSNR+L1 wins `0/5`
- static no-worse versus route0 `1/5`
- mean oracle PSNR-gap recovery `-0.0391`

Final verdict: FAIL. The current fixed non-oracle training-loop opacity-gate mechanism does not improve frozen hide/reveal windows and does not recover the oracle event-crop upper bound.

R038 independent reviewer verdict: `claim_supported=no`, `confidence=high`, `integrity_status=warn`. The warnings are missing learned LPIPS, missing identity/gate statistics, and limited five-window scope; no fake GT, self-normalized scoring, or phantom results were found.
