# Phase 8A Independent Methodology Review

Date: 2026-07-15

Status: complete review preserved verbatim with dispositions. This review is evidence for contract design, not an experiment result.

## Reviewer provenance

- Complete-review task: `/root/contract_check`.
- Runtime: independent Codex collaboration subagent; the exact backend model identifier was not exposed by the runtime.
- Context isolation: spawned with `fork_turns="none"`.
- Initial input: only the path to `agent-control/phase8-objective/objective-draft.md` and a narrow methodology-review prompt.
- Allowed actions: read-only; no web search and no file edits.
- Independence qualification: after a delayed file read, the primary agent sent a compact factual summary of the draft's proposed thresholds and controls. The findings below therefore remain an independent critique, but not a perfectly zero-context/blinded review.
- Earlier partial task: `/root/objective_reviewer` confirmed structural coverage and warned about non-executable or leakage-prone criteria, but was interrupted before returning a complete review. It is not the source of the eight findings below.

## Complete reviewer response

1. **R009 is not a clean holdout.** Its five windows already shaped R013-R037, the objective, and the proposed `3/5` threshold. Treat them as a historical benchmark, not unbiased confirmation. Create a new locked test set; use separate development events for thresholds and route selection.

2. **Gate A lacks independent ground truth.** Held-out-view depth/appearance, z-order consensus, and pseudo-labels reuse the same depth/geometry cues being evaluated, creating circular validation. Require manual visibility/order labels on a disjoint subset or synthetic/controlled ground truth, and fit depth alignment using training views only.

3. **Gate A metrics are not executable yet.** Define event units, positive/negative populations, surface correspondences, boundary tolerances, region masks, ordering pairs, ECE bins, and macro/micro aggregation. Non-abstained accuracy is gameable without a minimum coverage threshold and per-scene coverage reporting.

4. **Gate B failure would not localize the cause.** A Gate-A aggregate pass does not prove the signal is correct on Gate-B reveal surfaces. Make the correctly reprojected oracle-evidence-plus-capacity diagnostic mandatory: oracle failure indicts capacity; oracle success with predicted-evidence failure indicts evidence or transfer.

5. **The controls need a true matched factorial design.** Use route0, evidence-only, capacity-only, coupled, shuffled/misaligned evidence, and oracle evidence with the identical capacity implementation. Match parameter count, peak and integrated point budget, optimizer steps, compute, and candidate opportunity; realized final point count alone does not isolate evidence from capacity.

6. **Gate B denominators and formulas remain ambiguous.** Predeclare event/static masks, perceptual metric, flicker/ghost metrics, "catastrophic window," "scene-wide failure," and oracle-gap formula, including zero/negative denominators. Report paired scene/event effects and confidence intervals rather than only means and `3/5` counts.

7. **R031-R033 are overgeneralized.** The camera-confounded overlap audit only falsifies those implementations and selections, not single-frame depth edges generally. Likewise R030/R037 do not prove that all support-only densification or opacity attenuation "cannot" work. Narrow Section 4 and the dominant claim to the tested configurations.

8. **Threshold logic needs reconciliation.** "Within 2% or lower" permits an arbitrarily smaller, unmatched model; use a symmetric budget interval or an explicit quality-budget curve. Clarify how "no scene completely missed" coexists with Gate A passing on only `2/3`, and whether static bounds apply per window, per scene, or only in aggregate. Five clustered windows and two seeds also require scene-level uncertainty before claim-grade conclusions.

## Disposition in the versioned contract

| Finding | Disposition |
| --- | --- |
| R009 contamination | Accepted. R009 is historical continuity only; a new locked claim-grade set is required. |
| Independent reference | Accepted. Human-audited tracks and controlled geometry are approved; alignment is development-only. |
| Executable Gate A | Accepted in contract. Units, tolerances, coverage, aggregation, and two threshold tiers are specified; annotation instructions remain a method-refinement deliverable. |
| Gate B attribution | Accepted. Oracle-capacity admission precedes predicted-evidence coupling. |
| Factorial controls | Accepted conditionally within the five-lane one-scene envelope. Route0 is reused; the five new lanes cover capacity-only, oracle-capacity, evidence-only, coupled, and shuffled evidence. |
| Gate B formulas | Accepted in contract. LPIPS, paired event reporting, static masks, budget matching, and failure definitions are predeclared; exact implementation is deferred. |
| Overclaiming | Accepted. Claims are restricted to the tested R030-R033/R037 recipes and the new novelty statement is explicitly a hypothesis. |
| Threshold/budget logic | Accepted. Engineering and claim-grade tiers are separated, budgets are symmetric/integrated, and transfer scenes must pass independently. |

## Unresolved objections carried into method refinement

1. The exact event-track annotation manual, adjudication procedure, and surface-correspondence representation are not yet written.
2. Real N3V "oracle" visibility will remain an annotated/reprojected upper bound rather than metric ground truth; its construction must be audited for leakage.
3. The five one-scene lanes can admit engineering but cannot establish the dominant scientific claim.
4. Parameter count, integrated point-time, candidate opportunity, and wall-clock matching need an executable measurement protocol.
5. The new flame/sear event set must remain sealed from method development, including qualitative inspection after locking.
6. Novelty relative to VAD-GS, Proxy-GS, 4C4D, PackUV-GS, and other 2025-2026 methods remains a working hypothesis pending a complete mechanism matrix and implementation-specific comparison.
