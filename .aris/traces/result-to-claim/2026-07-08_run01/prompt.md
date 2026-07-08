# Result-To-Claim Prompt

Intended claim: a non-oracle, checkpoint-backed Gaussian method can recover the frozen R009 event-crop occlusion/reveal failures using occlusion-boundary gated micro-densification, without GT crop compositing or using the frozen R009 crop labels as test-time support.

Experiment run: R027 `event_boundary_micro_densify` on N3V scenes `cut_roasted_beef`, `flame_steak`, and `sear_steak`. R026 support used dynamic-mask/flow/render-boundary cues and recorded `uses_gt_residual=false`, `uses_gt_crop_pixels=false`, `uses_frozen_window_labels=false`. Training resumed route0 `chkpnt6000.pth` to `chkpnt6400.pth` with point cap `625000`. Eval rendered complete `test/ours_6400` folders for all three scenes. Frozen R009 scoring job `48874592` completed with `ExitCode=0:0`.

Predeclared PASS gate: at least 3/5 frozen windows improve versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS; at least 3/5 do not worsen static ghost versus route0; mean PSNR improves over route0 by at least `+0.5 dB`; mean L1 improves by at least `-0.001`; and the method recovers at least 25% of the oracle upper bound on either mean PSNR or mean L1.

Results:
- `event_boundary_micro_densify` means: PSNR `30.559051`, L1 `0.01474122`, flicker `0.00801033`, static ghost `0.12563431`.
- route0 means: PSNR `30.502119`, L1 `0.01483156`, flicker `0.00799083`, static ghost `0.12733274`.
- matched_lifespan means: PSNR `29.818134`, L1 `0.01635463`, flicker `0.00795601`, static ghost `0.12733274`.
- residual_uncertainty means: PSNR `30.073395`, L1 `0.01657234`, flicker `0.00803902`, static ghost `0.14570154`.
- oracle derived hide_reveal means: PSNR `41.714904`, L1 `0.00266536`, flicker `0.00168586`, static ghost `0.12733274`.
- Gate counts: strict all-baseline PSNR+L1 wins `2/5`; route0 PSNR+L1 wins `3/5`; static no-worse vs route0 `3/5`.
- Mean deltas vs route0: PSNR `+0.056932 dB`, L1 `-0.00009035`, flicker `+0.00001951`, static ghost `-0.00169843`.
- Oracle recovery: PSNR `0.005077`, L1 `0.007426`.

Known caveats: only five frozen windows; LPIPS is proxy L1 because LPIPS was disabled; R009 crop labels were evaluation-only and not used by support/training; support artifact is deterministic non-oracle but the final method result is checkpoint-backed renders.
