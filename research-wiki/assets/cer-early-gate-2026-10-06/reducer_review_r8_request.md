# Code review request: the CER reducer's early-gate extension, round 8 (focused)

Your round 7 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r7_codex.md): sensitivity caught only
Refuse; boolean counters; oversized gate integers; suspected share rounding above 1. Changes in
D:\adags-cer\agent\cer_reduce.py (sha256 3d6b8a95..., diff cer_reduce_m4dearly.diff):
- the Ex4DGS sensitivity re-run catches every exception -> {"unavailable": "<type>: <message>"};
- early preconditions require begin_iter / lifts_before_begin to be non-bool ints (False / True now fail rule 4);
- gate numbers: float conversion guarded (OverflowError -> Refuse "a huge integer"); shares allowed 1e-6 of slack at both
  ends of [0, 1]; masses >= 0;
- confirmatory score files: unreadable or malformed JSON -> Refuse (not a raw exception); missing PSNR window -> Refuse;
- tests: corrupt and window-less original-base files leave the official verdict written; 10**400 mass; missing P1;
  1 + 1e-9 share accepted; unreadable confirmatory score; boolean counter and boolean begin_iter; and the test runner now
  reseeds the RNG per test (a pre-existing test had become order-dependent). 17 tests pass locally.
Compatibility evidence (cluster, Leonardo login node, this exact file, sha256 3d6b8a95...): re-running it on the real
2026-10-02 scores reproduces checks/wave1/cer_reduce_multi4d.json and cer_reduce_ex4dgs.json with 0 differences (fields
compared recursively, floats to 1e-12 relative; tool, tool_sha256, policy_record and the new
nonfinite_descriptive_replaced excluded; nothing replaced on either).
Check these changes and whether anything material remains. Verdict line READY / READY WITH FIXES / MATERIAL DEFECT;
findings with severity, file:line, fix. Terse. Reserve MATERIAL DEFECT for defects that could give a wrong verdict or
prevent writing a correct one on inputs the scorer can actually produce.
