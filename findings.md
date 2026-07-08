# Findings

## Research Findings

### 2026-07-07: R025 event-candidate refine does not support the non-oracle event-crop-fix claim

R025 tested M1 non-oracle residual-component local refinement on the five frozen R009 real windows. The run appears to satisfy the method-integrity side of the gate: checkpoint-backed Gaussian renders, no GT crop compositing, and no frozen event crops as test-time support.

The quantitative result is negative. R025 improved 0/5 windows versus route0 on both PSNR and L1/proxy-LPIPS, and 0/5 versus all three baselines. Mean PSNR changed by -1.5629 dB versus route0, mean L1 worsened by +0.004043, and the oracle recovery fraction was negative for both PSNR and L1. Static ghost improved slightly on average but only 2/5 windows were no worse than route0, below the 3/5 gate.

Conclusion: R025 invalidates the intended M1 claim. The oracle event-crop result remains an upper bound, not a demonstrated non-oracle Gaussian-model fix.

### 2026-07-08: R027 boundary micro-densification does not support the event-crop-fix claim

R027 tested M2 occlusion-boundary gated micro-densification on the same five frozen R009 real windows. The method-form side of the test was valid: R026b support recorded no GT residuals, no GT crop pixels, and no frozen crop labels as support; R027 produced checkpoint-backed Gaussian renders from `chkpnt6400.pth`; and the scoring job completed with `ExitCode=0:0`.

The quantitative result is negative. R027 achieved only 2/5 strict all-baseline PSNR+L1 wins, although it reached 3/5 route0 PSNR+L1 wins and 3/5 static no-worse windows. Mean PSNR improved by only +0.0569 dB versus route0, mean L1 improved by only -0.0000903, mean flicker slightly worsened, and oracle recovery was below 1% on both PSNR and L1. This misses the predeclared +0.5 dB PSNR, -0.001 L1, and 25% oracle-recovery thresholds.

Conclusion: R027 is weak directional evidence that boundary support does not catastrophically damage the scene, but it does not support the claim that non-oracle boundary-gated micro-densification recovers the event-crop failures. Treat boundary-local densification alone as insufficient under this support and budget.
