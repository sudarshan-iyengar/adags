# Result-To-Claim Reviewer Response

- `claim_supported`: **no**

- `what_results_support`:
  - R025 appears to satisfy the method-form constraint: checkpoint-backed Gaussian-rendered output, not GT crop compositing.
  - It tests a relevant non-oracle path: residual/mask/flicker-derived candidate supports without frozen crop support at test time.
  - Mean static ghost is slightly lower than route0, but this does not pass the per-window gate.

- `what_results_dont_support`:
  - Quantitative gate fails decisively: `0/5` windows improve versus all three baselines on PSNR and L1.
  - `0/5` improve versus route0 on both PSNR and L1.
  - Mean PSNR is worse than route0 by `-1.56 dB`; mean L1 worsens by `+0.00404`.
  - Static ghost no-worse count is `2/5`, below required `3/5`.
  - Oracle recovery fractions are negative for both PSNR and L1.

- `missing_evidence`:
  - Learned LPIPS and confident-track ID-switch metrics remain unavailable.
  - These are not needed to reject the claim; the predeclared PSNR/L1/static gates already fail.

- `suggested_claim_revision`:
  - "R025 shows that the current M1 non-oracle residual-component local refinement can produce checkpoint-backed Gaussian renders, but does not recover the oracle event-crop fix on the five frozen real windows."

- `next_experiments_needed`:
  - Do not run paper-scale validation or ablations for M1 as a positive method.
  - Diagnose whether failure is from candidate support, local optimization, or Gaussian capacity by separating detector recall/overlap from refinement damage.
  - Next serious attempt should change mechanism, e.g. event-local densification, stronger geometry/identity support, or a validation-tuned non-oracle support proposal.

- `confidence`: **high**

Recorded the negative verdict in `research-wiki/` and `findings.md`.
