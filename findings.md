# Findings

## Research Findings

### 2026-07-07: R025 event-candidate refine does not support the non-oracle event-crop-fix claim

R025 tested M1 non-oracle residual-component local refinement on the five frozen R009 real windows. The run appears to satisfy the method-integrity side of the gate: checkpoint-backed Gaussian renders, no GT crop compositing, and no frozen event crops as test-time support.

The quantitative result is negative. R025 improved 0/5 windows versus route0 on both PSNR and L1/proxy-LPIPS, and 0/5 versus all three baselines. Mean PSNR changed by -1.5629 dB versus route0, mean L1 worsened by +0.004043, and the oracle recovery fraction was negative for both PSNR and L1. Static ghost improved slightly on average but only 2/5 windows were no worse than route0, below the 3/5 gate.

Conclusion: R025 invalidates the intended M1 claim. The oracle event-crop result remains an upper bound, not a demonstrated non-oracle Gaussian-model fix.
