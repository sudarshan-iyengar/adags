---
type: experiment
id: r029-r030-disambiguation-wave
status: blocked
date: 2026-07-08
related_idea: event-crop-fix
---

# R029/R030 Disambiguation Wave

Purpose: distinguish "M2 failed because the support missed the event crops" from "posthoc Gaussian refinement/densification cannot recover the oracle crop upper bound even with correct support", while also controlling for 400 extra training iterations.

R029 is a matched route0 `6000 -> 6400` continuation control with no event support and no motion-aware densification. R030 is an oracle-support diagnostic that uses the frozen R009 crop windows as support but does not composite GT crop pixels.

Status:
- PASS train: R029 jobs `48935431`, `48935450`, `48935478` completed with `ExitCode=0:0` and wrote `chkpnt6400.pth`.
- PASS train: R030 jobs `48935580`, `48935581`, `48935583` completed with `ExitCode=0:0` and wrote `chkpnt6400.pth`.
- BLOCKED eval: Slurm rejected or hung on eval submissions after training completed. `saldo -b` showed the allocation still active and not exhausted, so this is recorded as scheduler/account submission blockage, not method failure.

No metric verdict exists yet. The next scientific step is to render `test/ours_6400` for both systems and run the frozen-window scoring gate.
