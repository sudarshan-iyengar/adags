# Code review request: the CER reducer's early-gate extension, round 2 (focused)

Your round 1 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r1_codex.md; request beside it) found:
all-failed declarations abort via the descriptive contrast; undefined drift became 0; descriptive 16k inputs could block the
confirmatory reading; non-strict JSON. Fixed in D:\adags-cer\agent\cer_reduce.py and test_cer_reduce.py (updated diffs
cer_reduce_m4dearly.diff, test_cer_reduce_m4dearly.diff against the *.pre_m4dearly.py originals):
- the 16k arm left the manifest; arm16k() loads it apart; missing or mismatched 16k scores -> comparison UNAVAILABLE with the
  problems listed, verdicts unaffected; the comparison is computed after every verdict; a rank-deficient pooled contrast is
  reported as unavailable instead of aborting;
- cer_drift is required as carried (per_bank with both banks and both share fields; missing -> refuse; non-finite ->
  refuse); a None share stays None and makes the run's summed share None;
- main() writes with allow_nan=False; the test round-trip uses allow_nan=False;
- new tests: missing and mismatched 16k scores, all nine early runs declared failures, undefined drift, missing drift fields,
  non-finite drift, missing bank. All 14 tests pass locally.
Check the fixes, anything they broke (ex4dgs and multi4d paths must be unchanged), and anything still open from round 1.
Verdict line READY / READY WITH FIXES / MATERIAL DEFECT; findings with severity, file:line, fix. Terse.
