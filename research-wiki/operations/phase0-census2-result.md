# Phase 0 Census-v2 Result — CENSUS2_NO_GO, diagnosis D3

Date: 2026-07-29
Branch: `csvl-vpl-v2-phase0`
Preregistration: [[operations/phase0-census2-preregistration]] (floors G1-G4
and diagnosis rules D1-D4 frozen at commit `67fd058`, before execution)
Smoke job (non-scientific, 40 frames, validated all four checkpoint
restores): `50793065`, COMPLETED `0:0`, 7:26
Scientific job: `50795061`, COMPLETED `0:0`, elapsed 1:09:03, MaxRSS 12.8 GiB
Output: `$WORK/proj_adags/runs/phase9-depth-visibility-capacity/`
`phase0-census2-v1/census2-v1.json` (canonical scientific SHA-256 recorded)
Outcome: **CENSUS2_NO_GO**. Diagnosis:
**D3_residual_evidence_or_rule_failure — the carrier hypothesis is
exonerated.** Phase 1 remains unauthorized.
Preserves: [[operations/phase0-census-result]] unchanged.

## Floor verdicts (applied exactly as preregistered)

| Floor | Requirement | Observed | Verdict |
|---|---|---|---|
| G1 certified abundance | >= 2,000 pairs, >= 10 end frames, >= 8 cameras | **10,798 pairs**, 225 end frames, 16 cameras | PASS |
| G2 control separation | rho >= 3.0 | **rho = 0.192** (valid 10,798 vs shuffle 56,129) | **FAIL** |
| G3 evidence validity | conflict <= 15%, map pass >= 90% | conflict **0.010%** (2/19,226); maps **100.0%** | PASS |
| G4 non-degeneracy | max camera share <= 60% | 43.2% | PASS |

## The carrier question (user-added hypothesis): answered

Every carrier probe returned the same answer — the checkpoint is not the
problem:

| Configuration | valid pairs | shuffle pairs | rho |
|---|---:|---:|---:|
| primary 6000, moving | 10,798 | 56,129 | 0.192 |
| primary 6000, **frozen positions** | 10,866 | 57,605 | 0.189 |
| maturity 3000, moving | 10,067 | 56,421 | 0.178 |
| maturity 9000, moving | 11,146 | 55,384 | 0.201 |
| independent run 20260701, moving | 10,884 | 57,371 | 0.190 |

- **Frozen vs moving is indistinguishable** (0.189 vs 0.192): removing all
  primitive motion — the entire carrier degree of freedom — changes nothing.
- **Checkpoint maturity and run identity are irrelevant**: D4 dependence
  ratio 1.13 (threshold 2.0) across four checkpoints including an
  independently trained run whose restore succeeded.
- **Stratification shows only a weak, monotone motion effect** (rho 0.232 in
  the least-moving quartile vs 0.152 in the most-moving; scale similar,
  opacity flat) — a small interaction term, far below any threshold, in the
  direction expected if primitive motion adds noise but is not the cause.
- One stratification lesson: near-surface-stability quartiles cannot
  stratify certified pairs — certification *requires* occlusion, so certified
  primitives concentrate in the lowest near-fraction quartile by
  construction. Recorded so census-v3 designers do not repeat it.

Under the preregistered rules this is **D3**: even motionless carriers with a
temporally-structured certification rule cannot separate valid evidence from
frame-shuffled evidence. The v1 F4 failure was not a carrier artifact.

## The deeper finding: the separation is inverted

Shuffled evidence certifies **5x more** reveals than valid evidence, in every
cell. Provisional mechanistic reading (consistent with both censuses'
run-length data, not independently proven): genuine occlusions in this scene
are long and ragged — hands and utensils linger for tens of frames with
flickering boundaries — so real episodes exhaust the strict grace budget or
never complete a clean two-frame reveal, while temporal shuffling
manufactures short, coherent-looking occlusion bursts (two shuffled frames
drawn from different hand-over-region moments have similar depths, passing
the coherence test) followed by clean pseudo-reveals. The rule family rewards
exactly the temporal incoherence it was designed to exclude. R009-window
overlap is again not elevated (300 certified events in the two windows vs
~1,368 expected uniform — certified events avoid the busy hand windows,
consistent with long-ragged-episode abortion).

Positive knowledge retained: the evidence substrate remains abundant and
internally consistent (G1, G3 — conflict 0.010%, 100% consensus-map validity
after the preregistered cam12/cam19 exclusion), and the entire carrier axis
is now cleanly eliminated from the hypothesis space at one job's cost.

## Binding restrictions

No floor, rule, or parameter adjustment and no re-run under this cycle's
authority. Two label-free certification designs (v1 naive transitions, v2
anchored hysteresis) have now failed their own preregistered controls in
opposite directions (v1: no separation; v2: inverted separation). Any third
label-free variant would be the same guess-and-check pattern the project's
history warns against.

## Smallest justified next action (recommendation, requires user approval)

Stop iterating blind label-free certification rules. The two failures
bracket the problem: per-primitive, per-frame certification against pixel
states cannot distinguish genuine occlusion episodes from structured noise
without either (a) object/region-level aggregation or (b) a reference set.
Recommended path, in order:

1. **Pull the annotation path forward.** The drafted
   [[operations/phase9-annotation-contract-draft]] (awaiting sign-off) gives
   ~30 human event tracks; the development-scene portion becomes the
   engineering-admission reference the contract always intended (Gate A,
   Section 7), so rule quality is *measured* against real events instead of
   inferred from proxy controls. This converts census-v3 from a third blind
   cycle into ordinary Gate A engineering with ground truth.
2. **Redesign certification at region granularity** for the next cycle:
   aggregate primitives into spatial clusters and certify cluster-level
   episodes (per-primitive boundary flicker averages out), and/or certify
   from the evidence side (a coherent nearer-surface region sweeping across)
   rather than per-primitive state sequences.
3. Only then preregister a census-v3/Gate-A-engineering evaluation with
   floors expressed against the annotated reference.

## Links

- [[operations/phase0-census2-preregistration]]
- [[operations/phase0-census-result]] (v1, preserved)
- [[operations/phase9-csvl-vpl-v2-direction]]
- [[operations/phase9-annotation-contract-draft]]
