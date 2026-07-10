# R038 Experiment Audit: R036/R037 Visibility-Event Pilot

Date: 2026-07-10

Auditor: independent Codex subagent, read-only

## Overall Verdict

`WARN`

The experiment is valid enough to reject the intended positive claim. The warnings limit broader positive or general claims, but they do not undermine the negative result.

## Checks

### A. Ground Truth Provenance: PASS With Caveat

Actual smooth/event metrics compare rendered frames to dataset `gt_dir` crops in the frozen-window evaluator. The `hide_reveal` row uses GT crop compositing, but it is explicitly labeled as an oracle upper bound and is not treated as a trained Gaussian method.

### B. Score Normalization: PASS

PSNR, L1/proxy-LPIPS, flicker, and static ghost are computed directly from rendered outputs and references. No evidence of self-normalization by prediction statistics was found.

### C. Result File Existence And Number Consistency: PASS After Tracker Update

The result files exist locally:

- `real_event_window_report.md`
- `real_event_window_summary.json`
- `real_event_window_metrics.csv`
- `r036_r037_visibility_event_manifest.json`
- `r036_r037_visibility_event_manifest.validation.json`

The validation file reports `ok=true`, `n_windows=5`, and zero errors/warnings. Report/JSON/CSV numbers agree. Earlier tracker rows described R037 as blocked, but this close-out updates them to the completed FAIL state.

### D. Metrics Actually Called: WARN

PSNR, L1/proxy-LPIPS, flicker, and static ghost are called and written. Learned LPIPS was not computed, and confident-track identity switches were not inferred.

### E. Scope Assessment: WARN

Scope is five frozen windows from three scenes, one run per condition. This is enough for the predeclared pilot verdict and negative method decision, but not enough for a broad general claim.

### F. Evaluation Type: WARN

The actual R036/R037 rows are checkpoint-rendered real-GT evaluations. The `hide_reveal` row is an oracle GT-crop composite upper bound. Mixed row types are acceptable only while clearly separated.

## Claim Impact

- Positive claim that R037 improves hide/reveal windows: unsupported.
- Negative claim that this fixed opacity-gate variant failed the strict gate: supported.
- Broad claim that visibility-event modeling is impossible: unsupported.

## Action Items

- Preserve R037 as a failed method attempt.
- Do not retune on the five frozen windows.
- If continuing this line, define a new predeclared mechanism and include gate-activation statistics, learned LPIPS, and identity/reconnection evidence where feasible.
