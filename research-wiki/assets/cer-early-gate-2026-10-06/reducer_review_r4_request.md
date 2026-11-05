# Code review request: the CER reducer's early-gate extension, round 4 (focused)

Your round 3 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r3_codex.md) found overflow and
non-finite descriptive outputs, weak run-identity checks, and drift blocking confirmation. In D:\adags-cer\agent\cer_reduce.py
(diff cer_reduce_m4dearly.diff vs cer_reduce.pre_m4dearly.py):
- the descriptive comparison is now a helper, early_vs_16k(), called after every verdict; everything in it runs inside a
  broad guard (any exception -> listed problem, UNAVAILABLE); PSNRs pass _psnr_value (numeric, not bool, float conversion
  with OverflowError caught, finite, 0..1000 dB); run identities must be non-empty strings, distinct within the 16k arm and
  not equal to any confirmatory score's run; every pooled contrast and the per-layout table are checked finite
  (_all_finite) and replaced by unavailable records otherwise; status OK / PARTIAL / UNAVAILABLE reflects that;
- drift is descriptive: absent/malformed/non-finite drift -> that run's drift shares None with drift_problems; never refuses
  and never affects meets_preconditions;
- strict JSON is documented in the docstring as applying to every substrate.
Tests (test_cer_reduce.py; diff test_cer_reduce_m4dearly.diff): malformed JSON, missing window, NaN, non-numeric, 10**400,
1e308, missing / null / list run identity, wrong method, reuse of a confirmatory run identity, PARTIAL status (forced
failure of one pooled contrast), all-failed status, drift None / missing / inf / missing bank. All 14 tests pass locally.
Check the fixes, whether anything in the descriptive path can still raise or influence a verdict, and that ex4dgs and
multi4d are unchanged. Verdict line READY / READY WITH FIXES / MATERIAL DEFECT; findings with severity, file:line, fix.
