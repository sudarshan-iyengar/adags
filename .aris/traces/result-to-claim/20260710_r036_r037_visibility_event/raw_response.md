# Raw Reviewer Response

`claim_supported: no`

`confidence: high`

The R036/R037 pipeline appears to have run a matched comparison: smooth and event configs share the same 6000-iteration budget and R020 candidate field, with the event config enabling `visibility_event_manifest` while smooth leaves it empty. The evaluator scored 5 frozen windows.

The real event method failed the claim. Mean `visibility_event_train` is worse than `visibility_event_smooth_control`: PSNR `30.1089` vs `30.2936`, L1 `0.015760` vs `0.014445`, flicker `0.008413` vs `0.008053`, static ghost `0.165836` vs `0.116426`. Per-window, event loses L1 to smooth on all 5 windows and only wins PSNR on 2/5, with worse ghosting on all 5.

Oracle recovery is negative. The oracle `hide_reveal` row is much better, but it is a GT-crop composite upper bound, not a trained Gaussian output.

Missing evidence: learned LPIPS was not computed; identity switches/confident tracks were not inferred; accepted-event/gate activation statistics are not attached to the final scoring report; only 5 windows and one run per scene are present.

Suggested claim revision: “A matched training-loop visibility-event gate was implemented and evaluated on five frozen hide/reveal windows, but this fixed non-oracle opacity-gating variant did not improve over the matched smooth control and did not recover the oracle crop-composite upper bound.”

Integrity audit: GT provenance PASS with caveat; score normalization PASS; result existence/number consistency WARN before tracker update; metrics actually called WARN because learned LPIPS/identity metrics were not computed; scope WARN; evaluation type WARN because actual rows and oracle rows are mixed but clearly separable. Overall integrity verdict: WARN. No evidence of fake GT, self-normalized scores, or phantom metrics.
