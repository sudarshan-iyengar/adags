# Parsed Verdict

- claim_supported: no
- confidence: high
- integrity_status: unavailable
- what_results_support: weak directional improvement over route0 and satisfaction of non-oracle method-form constraints.
- what_results_dont_support: the predeclared recovery gate fails decisively: strict all-baseline PSNR+L1 wins are `2/5`, mean PSNR gain is `+0.056932 dB`, mean L1 gain is `-0.00009035`, and oracle recovery is below 1% on both PSNR and L1.
- missing_evidence: true LPIPS, more windows, repeated seeds, and a separate integrity audit.
- suggested_claim_revision: report R027 as a small directional but gate-failing result, not as event-crop recovery.
- next_experiments_needed: diagnostic decomposition before any confirmatory ablation or recovery claim.
