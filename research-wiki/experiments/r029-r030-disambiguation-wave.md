---
type: experiment
id: r029-r030-disambiguation-wave
status: failed
date: 2026-07-08
related_idea: event-crop-fix
---

# R029/R030 Disambiguation Wave

Purpose: distinguish "M2 failed because the support missed the event crops" from "posthoc Gaussian refinement/densification cannot recover the oracle crop upper bound even with correct support", while also controlling for 400 extra training iterations.

R029 is a matched route0 `6000 -> 6400` continuation control with no event support and no motion-aware densification. R030 is an oracle-support diagnostic that uses the frozen R009 crop windows as support but does not composite GT crop pixels.

Status:
- PASS train: R029 jobs `48935431`, `48935450`, `48935478` completed with `ExitCode=0:0` and wrote `chkpnt6400.pth`.
- PASS train: R030 jobs `48935580`, `48935581`, `48935583` completed with `ExitCode=0:0` and wrote `chkpnt6400.pth`.
- PASS eval: R029 eval jobs `48969017`, `48969019`, `48969021` and R030 eval jobs `48969090`, `48969092`, `48969093` completed with `ExitCode=0:0`.
- FAIL scoring: frozen-window scoring job `48969825` showed R029 worsened route0 and R030 failed despite oracle crop support.

R029 result: mean PSNR `30.3532` vs route0 `30.5021`, mean L1 `0.0150603` vs route0 `0.0148316`, route0 PSNR+L1 wins `1/5`, static no-worse `1/5`. Generic continuation does not explain R027's small positive movement.

R030 result: mean PSNR `29.9021`, mean L1 `0.0158770`, route0 PSNR+L1 wins `0/5`, strict all-baseline wins `0/5`, static no-worse `4/5`, oracle recovery negative. Oracle support alone does not rescue the current posthoc micro-densification mechanism.

Conclusion: the likely bottleneck is posthoc optimization/capacity allocation rather than only support localization. Support-only variants of the current recipe should be deprioritized.
