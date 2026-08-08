# EL-GS — Calibrated Novelty Record

Date: 2026-08-08. Referee: GPT-5.6-Sol, fresh thread, same scale as
STAR-GS (5.5) and LGS (6.5). Full disclosure given: v8 formal write-out
pending + one further fresh adversarial round required.

## Verdict

**8.0/10 — PROCEED WITH CAUTION — conditional** on the v8 mathematics
being written exactly and surviving the remaining fresh-context audit.
"The decisive distinction is that q must be part of a genuine
observation model, not an elaborate confidence weight. If
implementation reduces it to visibility-weighted tracker loss or
permits the model to suppress inconvenient opportunities, the score
falls toward 6.5-7.0."

Scale position: above STAR-GS 5.5 (not another training-side rule; new
temporal representation + explicit mechanism for when missing
observations may support structural change); above LGS 6.5 (adds the
previously-demanded NEW INFERENCE PRINCIPLE: renderer-defined
counterfactual observation opportunity + censored likelihood-ratio
evidence — "'nothing was detected' becomes evidence only when the scene
model says detection should have been possible"); below 8.5-9 (the
statistical machinery is inherited detection theory; search/acceptance
heuristic; tracker/bridge dependence; DiVa-360 requires ported
baselines — "a field-level inference innovation, not a foundational new
inference theory").

## Per-claim

| Claim | Novelty | Closest |
|---|---|---|
| CC1 episodic lineage representation | HIGH externally; MEDIUM increment over LGS | Ex4DGS/TOM-GS/4D-Scaffold/SharpTimeGS/CTRL-GS fragments |
| CC2 renderer-conditioned censored evidence | **HIGH** | MoSca/SoM masks; TrackerSplat init; L2D2-GS offline reward — none makes the renderer define a counterfactual observation channel deciding structure |
| CC3 tracker visibility as representation evidence | MEDIUM | new cell, new use of an existing signal; depends on CC2 |
| CC4 reactivation with own content | **HIGH** | 3DGS-MCMC donor-clone overwrite (opposite); FreeTimeGS++ respawn family; OmniRe/PersistGS other corners |
| CC5 full-system conjunction | **HIGH** | "CC1 and CC2 meet at a coherent inference boundary" |

## Ten binding viability conditions

Recorded in [[operations/elgs-method]] (formal energy; renderer
self-exoneration controls; one bridge latent; identical-search β=0;
evidence-specific causality; identity-evidence boundary; conditional
claims; CC1+CC2 necessity; benchmark risk; operation-level accuracy).

## Approved positioning (referee wording)

"EL-GS introduces an episodic lineage representation for dynamic
Gaussian splats and a renderer-conditioned censored-evidence procedure
that uses frozen multi-view track reports only when the current scene
model predicts a valid observation opportunity, enabling data-supported
selection among disappearance, reactivation, birth, fission,
truncation, and merge hypotheses under explicit bridge and permanence
assumptions."

## Hostile one-liner + rebuttal (referee wording)

"A kitchen-sink interval model that wraps tracker masks and
hand-engineered topology operations in Bayesian language while allowing
the renderer to certify its own disappearances." → "The tracker signal
is not used as a mask: it enters a bridge-marginalized likelihood ratio
whose censored case is exactly structure-invariant, while query-source
exclusion, cross-fitting, frozen/oracle-q arms, identical-search β=0,
and dose-matched null lanes directly test renderer self-exoneration.
If those safeguards are weakened or fail, the hostile critique wins."

## Program trajectory (same scale)

STAR-GS v9 (training-side): 5.5 → LGS (representation, heuristic
search): 6.5 → EL-GS (representation + inference principle): **8.0
conditional**. The user-relaxed constraints (external priors; dataset
freedom) were what unlocked the inference-principle half.
