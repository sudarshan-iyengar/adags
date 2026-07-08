# R029/R030 Disambiguation Decision Memo

Date: 2026-07-09

Verdict: FAIL for the current posthoc Gaussian micro-densification mechanism.

R029 matched continuation control completed and worsened route0 on mean PSNR/L1: mean PSNR `30.353246` versus route0 `30.502119`, mean L1 `0.015060306` versus route0 `0.014831561`, route0 PSNR+L1 wins `1/5`, and static no-worse `1/5`. This means R027's tiny positive movement was not simply generic 400-iteration continuation.

R030 oracle-support Gaussian-only diagnostic also completed and failed: mean PSNR `29.902108`, mean L1 `0.015877018`, route0 PSNR+L1 wins `0/5`, strict all-baseline PSNR+L1 wins `0/5`, PSNR oracle recovery `-0.053511`, and L1 oracle recovery `-0.085931`. Static ghost improved in `4/5`, but that does not rescue the event-crop objective because crop fidelity worsened.

Scientific interpretation:
- R026/R027 remains a failed non-oracle recipe.
- R028 showed R026 support missed the frozen crop regions, so R027 alone did not isolate the optimizer from the support detector.
- R030 now isolates that question: even with frozen-window crop support as an oracle diagnostic, the current posthoc micro-densification/refinement machinery does not recover the oracle crop upper bound.
- The most likely bottleneck is the posthoc optimization/training mechanism, not only support localization.

This does not prove the broader event/reveal idea impossible. It does argue against spending more compute on support-only variants of the current posthoc recipe. The next scientifically meaningful directions are training-loop integration, a different capacity/allocation mechanism, or a deliberately small hyperparameter/iteration sensitivity check only if it is framed as mechanism diagnosis rather than claim-seeking.
