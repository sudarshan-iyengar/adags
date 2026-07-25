# Phase 9 Slice B v13 B01 Decision

Date: 2026-07-25
Status: B01 completed; do not promote to B03 from this evidence alone
Decision: B01 passes feasibility/stability, but the reconstruction effect is too small to justify a claim-grade expansion without the registered A06 and B02 prerequisites.

## Scope

This record covers the corrected train-only Slice B B01 rerun after two implementation failures were found and fixed:

- Resume-trigger bug: checkpoint resume skipped iteration 5001, so v11 route0/capacity continuations did not execute the registered transaction.
- Donor-selection scalability bug: the capacity-only donor selector used an effectively quadratic neighbor search and stalled before the first optimization step at 562147 points.

The v13 run is bound to implementation-freeze commit `a714ba5d02e43ebe7c416ffdded80b06c8fba621` and freeze SHA `ab4e78edc777351b7cea957c35b0017cd529f593e94d4d8a96269070e09d85de`.

## Evidence

- Route0 Slurm job `50224610`: completed `0:0` in `00:14:13`.
- Capacity-only Slurm job `50224656`: completed `0:0` in `00:14:03`.
- Both runs used the sealed v11 common checkpoint `18f80a0f2c2d7f0c6f66a31c9ddcc58db451ce6358cb36467a859464ad5cdb98`.
- Both runs preserved the matched point budget: `562147` total points and `0` hard-static points.
- Route0 ledger: `transaction_count=1`, `iteration=5001`, `realized_k=0`, `reason=route0_no_op`.
- Capacity-only ledger: `transaction_count=1`, `iteration=5001`, `realized_k=256`, `candidate_limit=4096`, `inspected_candidate_count=288`.

Machine-readable metrics: `research-wiki/operations/phase9-slice-b-v13-b01-metrics.csv`.

## Metrics

| Metric | v13 route0 5250 | v13 capacity-only 5250 | Delta |
|---|---:|---:|---:|
| PSNR | 34.003838552 | 34.052154020 | +0.048315468 |
| SSIM | 0.960487562 | 0.960574518 | +0.000086956 |
| Dynamic-mask PSNR | 25.124321868 | 25.135483780 | +0.011161912 |
| Static-region PSNR | 34.343198942 | 34.398356349 | +0.055157407 |
| Static ghost score | 0.091287721 | 0.091475706 | +0.000187985 |
| Track-flow L1 | 0.039332430 | 0.039332430 | +0.000000000 |
| Dynamic edge magnitude | 0.035752095 | 0.035814334 | +0.000062239 |
| Total points | 562147.000000000 | 562147.000000000 | +0.000000000 |

## Interpretation

B01 is an admission/stability pilot. Under the registered contract, it asks whether the operator is finite, budget-preserving, and free of catastrophic early global/static harm at 5250. It does not tune donor rules and it does not establish the B03/B04 paper claim.

The corrected B01 passes feasibility/stability: optimization completed, the K=256 transaction executed, checkpoint/eval artifacts were written, and the point budget was unchanged. The measured reconstruction effect is weak: capacity-only is +0.048315 dB PSNR, +0.011162 dB dynamic-mask PSNR, +0.055157 dB static-region PSNR, and +0.000188 worse on static ghost score versus route0.

This is not enough to claim that the representation mechanism improves reconstruction. It only says the point-neutral capacity operator is now executable and not catastrophically harmful in the 5250 pilot.

## Next Step

Do not launch B03 or six-scene B04 from this evidence alone. The Slice B contract requires a genuine A06 engineering pass and B02 admission before B03. Current run-tree inspection found no completed A06/B02/B03 terminal artifacts under the Phase 9 run root.

Continue with the registered prerequisites, or explicitly revise the experiment contract if the project direction changes. Keep human labels, evaluator masks, and cam00 RGB out of non-oracle training lanes.
