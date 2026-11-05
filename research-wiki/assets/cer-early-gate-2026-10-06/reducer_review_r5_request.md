# Code review request: the CER reducer's early-gate extension, round 5 (focused)

Your round 4 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r4_codex.md) found drift conversion
overflow and finite-but-huge drift values overflowing sums and means. In D:\adags-cer\agent\cer_reduce.py drift shares are
now accepted only as numbers (not bool) whose float conversion succeeds (OverflowError caught), finite and within [0, 1]
(they are shares of the pre-gap target mass); anything else -> that run's share None plus a drift problem; the run's
summed share is then <= 2 and every scene mean finite. Tests added in test_cer_reduce.py: 10**400, 1e308, -0.1, a string,
inf, missing field, missing bank (each: share None, problem recorded, rule 4 OK everywhere, strict JSON via run()), and
shares at the [0,1] edge in every run of a scene (finite means, strict JSON, rule 1 PASS). All 14 tests pass locally.
Compatibility note accepted and documented in the docstring: strict JSON (allow_nan=False) now applies to every substrate;
otherwise ex4dgs and multi4d outputs are unchanged. Diffs: cer_reduce_m4dearly.diff, test_cer_reduce_m4dearly.diff.
Check the fix and whether ANY input to the descriptive parts can still raise or block writing the verdicts. Verdict line
READY / READY WITH FIXES / MATERIAL DEFECT; findings with severity, file:line, fix. Terse.
