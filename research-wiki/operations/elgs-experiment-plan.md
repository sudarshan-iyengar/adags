# EL-GS — Preregistered Experiment Plan (plan only; gated)

Date: 2026-08-08. Status: claim-driven program for testing
[[operations/elgs-method]]. GATED: implementation requires (i) the v8
formal write-out, (ii) one further fresh-context adversarial round on
it, (iii) user approval. Encodes the calibrated referee's 10 viability
conditions and all adversarial-round control demands. Annotation/audit
rules: event audits blinded to method output; annotations never used
for scene selection, tuning, activation floors, or stopping; held-out
cameras never enter tracking or observability.

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1 (representation): the episodic lineage substrate adds capability | oracle-structure probes; single-vs-multi-interval and independent-episode-appearance ablations at matched search (referee cond. 8) | B2 B4 |
| C2 (inference principle, dominant): renderer-conditioned censored evidence improves structural selection beyond supervision/search | q-source/update matrix (geometry-only / frozen / refreshed / camera-fold cross-fitted / oracle q — the decisive contrasts); β=0 identical-search; substrate+naive-loss same budget; dose-matched shifts/shuffles selectively destroying linkage gains | B3 B4 |
| C3 (system): competitive reconstruction + operation-level accuracy on event-dense data | end-to-end vs ported external baselines (compute-matched) + operation metrics (return timing, false reactivation/birth, identity retention, strata) | B2 B5 |
| Anti-claims | capacity (dual caps + ledgers), compute (iso-compute + search-cost ledger), search-alone (β=0), supervision-alone (naive-loss), self-exoneration (cross-fitted/frozen/oracle q), generic tying (grouping controls inherited from LGS protocol) | B3 B4 |

## Blocks

B0 Static checks + formal-spec unit tests (~5 GPU-h): energy/invariance
tests (censored-segment zero across ops; clone/merge invariance;
censoring equality), transition-table property tests incl.
latch-inheritance over every op and serialization load-validation,
latch/simplex round-trip tests (forward∘inverse identity across all
four latch patterns and K=1..4; dim(a) = 2K+1−n_lat; Ω-sum identity;
strict positivity of unlatched slacks; exact-floor targets rejected),
winner-lookup, SNIS/importance-weight sanity (weights ≤ 1/λ_u with
clip-inactivity; empirical bias → 0 with n vs a closed-form ν-mean;
CRN pairing determinism; per-replicate bootstrap renormalization;
cluster-degeneracy rejection — the estimator is finite-sample biased
SNIS per spec §7, no unbiasedness asserted), tracker-pipeline dry run,
metric spec freeze (incl. object-to-lineage mapping protocol; power
analysis). Details: [[operations/elgs-implementation-readiness-errata]].
B1 EVIDENCE + ACTIVATION CENSUS GATE (~25 GPU-h; kills cheaply):
DiVa-360 dev subset: track coverage (bound fraction of dynamic
content), opportunity distributions, event-class counts vs preregistered
floors (full training-view occlusion; audited true absence;
same-object return); baseline 4DGS/STG port smoke. FAIL ⇒ recorded
negative; one preregistered revision cycle.
B2 Backbone + substrate establishment (~80 GPU-h): ported external
baselines (documented tuning, compute-matched); LGS substrate
(evidence-off) lanes; oracle-structure probes (C1).
B3 Decisive inference matrix (~120 GPU-h): q-source/update matrix
(5 arms × dev scenes × 3 seeds decisive cells); β=0 identical-search;
substrate+naive-loss; dose-matched multiple non-wrapping shifts +
ID-shuffles within validity windows (effective factor mass reported);
all-misses-negative / visibility-flag-only / frustum-only /
no-detectability / mean-pose q / query-only exclusion; oracle bridge;
oracle linkage with episode-appearance-flexibility sweep. Kill rules:
refreshed-renderer q ≤ geometry-only q ⇒ CC2 dies; cross-fitted ≫
refreshed gap ⇒ self-exoneration confirmed ⇒ CC2 compromised; β=0
matches full ⇒ evidence adds nothing; shifted lanes' false-event rate
above floor ⇒ procedure unsafe.
B4 Attribution + honesty analyses (~40 GPU-h): decision-classification
distributions (data-supported / prior-pivotal / unsupported /
equivalence-class), risk-coverage, intention-to-treat incl.
screened-out candidates; lineage-removal sensitivity; ε-bound power
curve vs oracle opportunity; self-censoring and clone-invariance
audits; feasibility ledger vs worst-case tables.
B5 Freeze → transfer → stress → finals (~90-120 GPU-h): freeze
constants; held-out DiVa-360 sequences; Ego-Exo4D cooking stress tier;
HOT3D/ADT metric validation; N3V/Technicolor continuity check
(event-sparse, reported as such); failure taxonomy (tracker failures,
bridge misspecification, K-overflow, late-return misclassification).

## Milestones

| M | Gate | GPU-h |
|---|---|---|
| M0 | B0 green; specs committed | ~5 |
| M1 | B1 census floors pass | ~25 |
| M2 | B2 baselines + substrate established | ~80 |
| M3 | B3 decisive matrix verdicts | ~120 |
| M4 | B4 analyses complete | ~40 |
| M5 | finals + stress | ~90-120 |

Total ≈ 360-390 GPU-h + frozen-tracker preprocessing (ledgered
separately). All configs committed before submission; job IDs
sacct-verified; activation diagnostics mandatory.

## Sufficiency argument

C2 is carried by orthogonal legs — the q-source matrix (isolates
renderer conditioning from evidence per se), β=0 (isolates evidence
from search), naive-loss (isolates the censoring model from
supervision), shifts/shuffles (temporal/identity alignment causality),
oracle arms (headroom) — each with preregistered kill rules, so a
positive result cannot be attributed to supervision, search, capacity,
compute, or self-exoneration, and negative results are decisive and
cheap (ordered gates). C1 has its own ablation legs; C3 is carried by
external baselines plus operation-level metrics validated on pose-GT
data. Honest-failure branches are preregistered outcomes.
