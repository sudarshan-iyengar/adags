# Code review request: the CER reducer's early-gate extension, round 3 (focused)

Your round 2 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r2_codex.md) found: malformed /
incomplete / non-finite 16k scores still raise; the separate loader lost the distinct-run check; an all-unavailable
comparison reported OK. Fixed in D:\adags-cer\agent\cer_reduce.py (arm16k now validates every 16k score inside a guard:
unreadable JSON, wrong method/scene, missing run_checks.run, missing windows, non-finite or non-numeric PSNR, repeated run
identities -> listed problems, comparison UNAVAILABLE, values None; the pooled contrasts' failures are added to the
problems; status OK only if all four pooled contrasts exist, PARTIAL if some, UNAVAILABLE if none). New tests in
test_cer_reduce.py cover each case and the all-failed status; all 14 tests pass locally. Updated diffs:
cer_reduce_m4dearly.diff, test_cer_reduce_m4dearly.diff.
Check the fixes and whether anything in the descriptive path can still raise or change a verdict; and that ex4dgs and
multi4d paths are unchanged. Verdict line READY / READY WITH FIXES / MATERIAL DEFECT; findings with severity, file:line, fix.
