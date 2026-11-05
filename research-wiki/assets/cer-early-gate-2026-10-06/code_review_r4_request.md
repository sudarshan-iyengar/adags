# Code review request: "CER on Multi4D, early gate" (code review, round 4 — focused)

Your round 3 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\code_review_r3_codex.md) left one confirmed defect:
E8 rebuilt its witness set from rows that had passed the q=0 control and accepted >= 50%. The implementer added one test
commit, 96e92ee, on branch cer-early (D:\adags-cer\multi4d-early): the witness set per bank is now every memory row (or,
per the T3 precedent, a row labelled memory for the test) whose instrument contribution in the gap view at q=0 is >= 1e-6;
every witness must have nonzero opacity and DC-colour gradients at q=0 and exactly zero at q=1; every witness visible in the
pre-gap view (same floor) must have both nonzero there (>= 1 per bank); no fraction-based acceptance; failing rows printed.

Review `git -C D:\adags-cer\multi4d-early diff 6b75e9c..96e92ee` and the E8 code it touches (and the cross-process
comparison that consumes the witness set). Confirm the defect is fixed and that nothing else in E8 regressed. Your round-3
"suspected" (unfrozen short runs save before maintenance) and "missing" items are dispositioned as follows; say if you
disagree: the F1 forecast reads only cer_begin.json; the frozen==false path gets runtime coverage from the F1 run itself;
the multi4d_cer_early scorer method (requiring frozen==true, mode early, begin 3000, full schedule) is written at
submission time; GPU validation of 96e92ee runs now on the cluster.

Verdict line READY / READY WITH FIXES / MATERIAL DEFECT, then findings (severity, file:line, fix). Terse.
