# LGS — Novelty Record (calibrated referee verdict)

Date: 2026-08-08. Referee: GPT-5.6-Sol via Codex MCP, xhigh, fresh
thread, calibrated against the same scale that scored
[[operations/star-gs-v9-method]] at 5.5/10.

## Verdict

**6.5/10 — PROCEED WITH CAUTION.** Referee's scale statement: "6-7 means
an unoccupied, technically meaningful composition of mostly familiar
ingredients, without a new inference principle or foundational
primitive. LGS fits that band. It is stronger than STAR-GS v9's 5.5
because its central mechanism has no verified field-level occupant; it
does not reach 8 because the representation is readily describable as
constrained episode primitives plus parameter tying."

Run-mandate context: the run targeted ≥8.5. The verdict, after five
verified sweeps, dedicated gap searches, nine deep-dives, and ten
reviewer passes, is that the strongest verified-open representation
slice tops out in the 6-7 band on this scale; reaching 8+ would require
a new inference principle or foundational primitive, which this
direction does not contain. Recorded honestly; decision escalated to
the user (see final run report).

## Per-claim scores

| Claim | Novelty | Closest work |
|---|---|---|
| CC1 compact-support multi-episode latched presence (exact absence, per-primitive changepoints) | **HIGH** | Ex4DGS (one contiguous flat-top per independent row); CTRL-GS (scene-global boundaries); AD-GS/TRiGS (single window); TOM-GS (single bump, verified) |
| CC2 hard cross-episode radiance tying w/ episode-local pose/motion | MEDIUM | canonical-deformation/Dynamic3DG (persistent through ALL time, no episodes); Ex4DGS (episodes, independent content); VQ (no temporal semantics) |
| CC3 predicted-pose reactivation of OWN trained content | MEDIUM | 3DGS-MCMC (CODE-verified donor-clone overwrite — mechanistic opposite); FreeTimeGS++ 2605.03337 (ablates the donor-respawn family); ReAct-GS (name collision, active-primitive perturbation); ElasticFusion (non-GS ancestry) |
| CC4 counterfactual micro-render acceptance + Lagrangian cost | MEDIUM | L2D2-GS 2606.29374 (one tentative render as offline policy reward, no live per-decision acceptance); codec mode decision (ancestry) |
| CC5 the lineage conjunction | **HIGH** (field-specific, not foundational) | no cited system combines the four elements |

## Relative position vs STAR-GS v9 (referee's own analysis)

LGS +1: no mechanism-level occupant for CC1/CC2/CC3/CC5 (STAR-GS had
the decisive CEC-4DGS collision); CC3's nearest family performs the
opposite operation; genuinely different representational state.
STAR-GS advantages LGS lacks: a sharper inference contribution
(residual consensus localization); intrinsic budget neutrality; a close
precedent supporting feasibility (lower execution risk).

## Binding viability conditions (10, all carried into
[[operations/lgs-experiment-plan]])

1 real multi-episode use (preregistered K>1 fraction with genuine
held-out gaps/returns); 2 representation-isolating controls
(independent episode rows, tied contiguous windows, new-birth-on-
return, random/wrong grouping, untied radiance; matched separately for
stored scalars AND peak rows); 3 graph-source isolation + no identity/
association language; 4 onset and return holdouts (unseen onsets, not
observed boundaries); 5 reactivation attribution (vs new birth,
donor-copy respawn, zero-order hold, constant velocity, oracle
retrieval; report false reactivations, missed returns, pose basin);
6 counterfactual audit (frozen protocol; do sampled deltas predict
full-objective deltas; count rejected proposals + rollout cost);
7 complete resource accounting (dormant content, index, metadata,
moments, all search compute); 8 multi-operating-point Pareto survival;
9 tying must carry the gain (untied capacity-matched match ⇒ CC2/CC5
lose empirical justification); 10 no rescue by search alone (if gains
come from counterfactual acceptance without tying benefit, reposition
as structural optimization).

## Approved positioning (referee wording)

"We introduce a dynamic-GS lineage representation that hard-ties
radiance across a small ordered set of mutually exclusive
compact-support episodes while retaining episode-local geometry and
motion, enabling exact temporal absence and predictable-pose
reactivation of the lineage's own learned content; unlike prior
single-window or new-row formulations, it does not claim physical
identity or association of independently born content."

## Hostile summary + approved rebuttal

Hostile: "LGS is just K ordinary episode Gaussians sharing SH and
opacity, plus a hand-engineered wake-up heuristic and expensive
trial-render pruning." Rebuttal (referee wording): "Algebraically, LGS
is a constrained factorization — and that is why its novelty is 6.5
rather than 8 — but prior dynamic-GS state spaces do not impose its
mutually exclusive compact supports, hard cross-episode content
identity, episode-local motion origins, or own-content revival. The
reduction establishes simplicity, not prior occupancy; the controlled
activation and capacity-matched experiments must establish usefulness."

## Evidence provenance

Gap searches this run (all primary-source): trial-render acceptance —
no occupant (L2D2-GS closest, verified mechanism); dormant reactivation
with own content — no occupant, opposite-mechanism family verified at
code level (3DGS-MCMC relocate_gs); multi-interval presence —
reconfirmed unoccupied (AD-GS, TRiGS, ChronoGS checked and ruled out).
Full boundary: [[operations/repr-sweep-2026-08]].
