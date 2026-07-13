# R016 Go/No-Go Memo

Generated: 2026-07-06T02:56:29+02:00

## Decision

**NO-GO for paper-scale validation now.**

The PoC is promising enough to justify one narrow next implementation step: run the actual frozen hide/reveal gate on the same five R009 real windows and produce trained Gaussian or checkpoint-backed outputs. It is not yet strong enough to launch broad baselines, paper-scale validation, or SOTA comparisons.

## Criterion Outcomes

| Criterion                                                             | Outcome                          | Evidence                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1: normalized event margin separates hide/reveal from normal motion  | PASS                             | Synthetic heldout n=40: candidate recall 1.000, margin AUC 1.000, accepted precision/recall 1.000/1.000, false event rate 0.000, mean true-event delta -0.5660, mean normal delta +0.3937.                                                                                                                                                                                             |
| C2: counterfactual identity test is more than lifespan/opacity gating | PASS on synthetic evidence       | Synthetic heldout identity reconnection is 1.000 for the full method and 0.000 for matched lifespan-only/no-identity variants, while matched lifespan can still accept patch events. This supports the intended novelty boundary on controlled labels.                                                                                                                                 |
| C3: frozen synthetic-calibrated gates improve real occlusion windows  | NOT PASSED for the actual method | The five predeclared real windows are fixed and comparable, and the upper-bound hide/reveal composite is directionally positive. However `derived_poc_metadata.json` records `is_trained_model_output=false`; the real hide/reveal row uses GT inside event crops and is not a trained Gaussian/checkpoint output. Learned LPIPS and confident-track ID switches are also unavailable. |

## Real-Window Evidence

| System | n | PSNR up | L1/proxy-LPIPS down | Flicker down | Static ghost down | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| route0 | 5 | 30.5021 | 0.01483 | 0.00799 | 0.12733 | Smooth-transport baseline. |
| matched_lifespan | 5 | 29.8181 | 0.01635 | 0.00796 | 0.12733 | Does not explain the improvement; worse PSNR/L1 than route0. |
| residual_uncertainty | 5 | 30.0734 | 0.01657 | 0.00804 | 0.14570 | Does not explain the improvement; worse PSNR/L1/static ghost than route0. |
| derived hide_reveal | 5 | 41.7149 | 0.00267 | 0.00169 | 0.12733 | Strong upper-bound sanity signal, but uses GT crop compositing rather than a trained method output. |

Qualitative strips in `refine-logs/hide_reveal_poc/r015_poc_summary/crop_strips/` confirm that the chosen event crops are visually meaningful and that the upper-bound edit targets the intended occluders. They should not be reported as final method evidence.

## What This Means

The safe claim is: **synthetic C1/C2 passed, real-window upper-bound C3 is directionally positive, but C3 remains unproven for the implemented method.**

Do not expand to paper-scale baselines yet. The next work item, if this direction continues, should be a small R017-style implementation check on the same frozen R009 windows:

- emit actual hide/reveal-rendered outputs from Gaussian state or checkpoint-backed inference, not GT crop composites;
- keep the R009 windows, frozen synthetic thresholds, and baseline outputs unchanged;
- add learned LPIPS and/or confident-track identity-switch evidence if sidecars can be produced without retuning;
- pass only if a majority of the five windows improve versus route0, matched-lifespan, and residual/uncertainty without static ghost degradation.

## Final R016 Status

R016 scientific decision: **FAIL / NO-GO for paper-scale validation**.

This is a useful failure, not a dead end: it prevents a premature broad experiment sweep and identifies the exact missing evidence needed before scaling.
