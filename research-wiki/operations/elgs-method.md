# EL-GS — Evidence-Lineage Gaussian Splatting (Loop-2 selected candidate)

Date: 2026-08-08 (Loop 2, user-relaxed constraints: external priors
allowed; any public dataset; per-scene optimization fixed).
Status: **selected Loop-2 candidate at calibrated novelty 8.0/10
(PROCEED WITH CAUTION), conditional. FORMAL GATE NOT YET PASSED: the
v8 mathematical write-out (fix set below) must be completed and survive
one further fresh-context adversarial round before implementation
approval.** Nothing implemented/trained/submitted.
Provenance: Loop-2 sweeps ([[operations/loop2-sweep-2026-08]]) →
refine loop 5.7→8.2→8.9 → FIVE fresh-context adversarial rounds
(hostile novelty 4→6→7→7→8) → calibrated referee 8.0
([[operations/elgs-novelty-record]]). Review substance:
[[operations/elgs-review-history]]. Plan:
[[operations/elgs-experiment-plan]]. Predecessors: LGS
([[operations/lgs-method]], 6.5) remains the evidence-off substrate and
internal baseline; STAR-GS ([[operations/star-gs-v9-method]], 5.5)
untouched.

## Single conditional claim

EL-GS couples an episodic lineage representation with
renderer-conditioned censored track evidence for DATA-SUPPORTED
STRUCTURAL SELECTION under stated opportunity and linkage conditions.
Never claimed: physical absence (only bridge-consistent presence);
identifiability (reactivate-vs-rebirth equivalence under zero
opportunity is an owned observational-equivalence class); statistical
validity of acceptance (preregistered heuristic, empirically
stress-tested); calibration.

## Representation (substrate = Loop-1 LGS, family form)

Lineage FAMILY f (clone-descendant set): ONE presence program P_f of
≤4 ordered disjoint compact-support episodes (exact-zero absence,
latched plateaus, chain-invariant duration/gap floors, edge-band frames
excluded from evidence), episode-local pose/motion at immutable
per-episode origins, tied radiance content, winner-lookup rendering,
dual caps, presence-aware pruning, topology-inheriting clone/split, and
explicit return operations.

## Inference (the Loop-2 contribution)

Evidence: frozen multi-view point tracks (per-camera CoTracker3-class
queries from 3D surface seeds visible ≥2 training cameras; identity =
common-seed membership; robust consensus triangulation; <2 cameras ⇒
3D unknown; late births: deterministic lineage-local seeding with the
SAME delay/audit protocol; bindings audited once at 2.8k then
PERMANENT; held-out cameras never in tracking or observability).
Ontology: presence z (binary on plateaus) | renderer-derived
counterfactual observability q (bridge- and track-specific;
family-present query-source-excluded transmittance × in-frustum ×
pre-gap detectability d) | raw reports y (soft logits, bridge-indexed
residuals, miss tokens) | report reliability r (preregistered
diagnostics; distinct from d).
Energy: E({P_f},θ) = L_render + β·Σ_f Σ_clusters Σ_segments Φ + priors
(κ·episodes + duration/gap + χ·fresh-parameter cost at return births —
disclosed object-permanence + complexity prior). Cluster-segment
factors are likelihood RATIOS Φ = −log[A_B(L_z)/A_B(L_cens)] with
A_B = normalized tempered mixture over a PREREGISTERED bridge family
(per-bridge aggregation first; ONE bridge latent per decision), so
censored segments contribute EXACTLY ZERO under any structure;
L1 = r[q̃·p_vis + (1−q̃)·p_cens] + (1−r)·p_outlier, L0 = r·p_cens +
(1−r)·p_outlier (censoring equality identical); q̃ = q·d;
hypothesis-independent opportunity gate with lower-confidence-bound
min-opportunity conservatism; seed-cluster ESS tempering;
correlated-camera capping.
Structural ops (closed set with admissibility + full transition table
required by the fix set): fission, truncation, reactivation, birth,
merge; proposed by per-lineage constrained semi-Markov interval engines
inside an approximate global structural search (conflict graphs on
current∪bridge footprints; Gauss-Seidel; priority queue with
per-component confirmation); accepted by a preregistered HEURISTIC on
reserved fresh render samples (hash-partition rule from iteration 0;
candidate-targeted rays + uniform-support mixture; accept iff
ΔÊ < −k·SE); validated empirically by dose-matched shifted/shuffled
lanes (stress tests, not causal validation of natural-data
correctness). Schedule: warm-up→2.5k; seeding 2.5k; audit 2.8k; rounds
{3k,4.5k,6k} (round 3 truncation/fission only); refit→10k;
compute-matched baselines.
Decision reporting: 3-way classification (DATA-SUPPORTED /
PRIOR-PIVOTAL / UNSUPPORTED vs EQUIVALENCE-CLASS, separated);
intention-to-treat incl. screened-out candidates; risk-coverage.

## v8 FIX SET (mandatory before the final adversarial round)

1. Full write-out of A_B exactly as: ℓ_b = α_ess·Σ log L_z(y;b);
   −log A_B = −τ·log[(1/|B|)·Σ_b exp(ℓ_b/τ)] — every normalization,
   temperature, tie rule, and ESS operator fixed.
2. Likelihood-ratio factors (already adopted) proven to zero censored
   segments across ALL admissible transitions.
3. One bridge latent per decision (no per-segment bridge switching);
   fission invariance restated accordingly.
4. ε-bound B(ε) derived WITH stated density floors and ESS dependence.
5. Complete transition table (pre/post state, parameter init, evidence
   ownership, prior delta per op); REACTIVATE-vs-MERGE path dependence
   disclosed as a limitation; importance-weight spec (target measure,
   clipping, dependence-unit SE, common random numbers).

## Binding viability conditions (calibrated referee, 10)

(1) complete the formal energy; (2) prevent renderer self-exoneration
(q-source/update matrix binding: geometry-only / frozen / refreshed /
camera-fold cross-fitted / oracle); (3) one bridge latent per decision;
(4) β=0 genuinely identical-search; (5) evidence-specific causality
(shifts/shuffles selectively destroy linkage gains); (6) identity
evidence boundary respected (stratified reactivation reporting);
(7) conditional claims only; (8) CC1 AND CC2 necessity shown
(independent-episode-appearance, single-interval, naive-loss,
substrate-only, oracle-bridge, oracle-linkage arms); (9) benchmark risk
controlled (documented ported baselines; pose-GT metric validation does
real work); (10) operation-level accuracy reported (return timing,
false reactivation/birth, identity retention, gap/opportunity strata),
not just rendering quality.

## Datasets

Primary DiVa-360 (53-cam surround, 25 hand-object interaction seqs,
masks, MIT; GS baselines to be established); HOT3D/Aria Digital Twin
pose GT for metric validation only; Ego-Exo4D cooking stress tier;
N3V/Technicolor backbone continuity.

## Honest assessment

Calibrated 8.0 conditional (fall-back 6.5-7.0 if q degrades to a
confidence weight); referee ceiling: field-level inference innovation,
not foundational theory. Largest risks: renderer self-exoneration;
bridge-family misspecification; tracker reliability on non-rigid
content; search/confirmation compute; ported-baseline fairness.
