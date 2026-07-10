# R038 Result-To-Claim: R036/R037 Visibility-Event Pilot

Date: 2026-07-10

## Intended Claim

A non-oracle visibility-event Gaussian method with persistent state and a time-dependent opacity gate, trained in the original loop and compared to a matched smooth-control budget, improves the five frozen hide/reveal windows and recovers a meaningful fraction of the oracle event-crop upper bound.

## Verdict

`claim_supported: no`

`confidence: high`

The intended positive claim is not supported. The event-gated method produced complete checkpoint-backed Gaussian renders, but failed the predeclared frozen-window gate.

## Evidence

- R036 smooth control: mean PSNR `30.2936`, mean L1 `0.0144448`, strict all-baseline PSNR+L1 wins `2/5`, static no-worse `3/5`, mean oracle PSNR-gap recovery `-0.0183`.
- R037 visibility-event method: mean PSNR `30.1089`, mean L1 `0.0157600`, route0 PSNR+L1 wins `0/5`, strict all-baseline PSNR+L1 wins `0/5`, static no-worse `1/5`, mean oracle PSNR-gap recovery `-0.0391`.
- route0 baseline: mean PSNR `30.5021`, mean L1 `0.0148316`.
- oracle `hide_reveal` upper bound: mean PSNR `41.7149`, mean L1 `0.00266536`; still an image-level GT crop composite, not a trained Gaussian output.

## What The Results Support

The experiment supports a negative, mechanism-specific claim:

> A fixed non-oracle training-loop opacity gate on the R020 candidate field does not recover the hide/reveal oracle crop gap under the current ADAGS Gaussian training setup.

The result also supports that the pipeline now has a valid matched-budget evaluation harness for training-loop visibility-event variants.

## What The Results Do Not Support

- They do not support a positive visibility-event repair claim.
- They do not show that the current gate improves over the matched smooth control.
- They do not show meaningful oracle-gap recovery.
- They do not rule out all visibility-event modeling, because LPIPS, identity switches, accepted-event/gate statistics, and broader scenes/seeds are still missing.

## Suggested Claim Revision

“A matched training-loop visibility-event gate was implemented and evaluated on five frozen hide/reveal windows, but this fixed non-oracle opacity-gating variant did not improve over route0 or the matched smooth control and did not recover the oracle crop-composite upper bound.”

## Next Experiments

Do not tune R037 on the five frozen windows. A next attempt needs a new predeclared mechanism that changes the bottleneck, such as soft reliability weighting, explicit occluder/surface capacity separation, or a training-time identity/reconnection objective rather than candidate-local opacity attenuation alone.

## Reviewer Trace

Independent reviewer trace: `.aris/traces/result-to-claim/20260710_r036_r037_visibility_event/`.
