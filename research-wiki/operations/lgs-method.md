# LGS — Lineage Gaussian Splatting (selected representation-level method)

Date: 2026-08-08. Status: **selected candidate of the representation-first
discovery run; survived 5 fresh-context adversarial rounds
(SURVIVES-WITH-RISKS); awaiting user approval; nothing implemented,
trained, or submitted.** Pipeline provenance: 5 verified literature
sweeps → method convergence → 4-round refine loop (9.2 READY) → 5
fresh-context adversarial rounds with 4 full redesigns → novelty check.
Records: [[operations/lgs-review-history]], [[operations/lgs-novelty-record]],
[[operations/lgs-experiment-plan]], [[operations/repr-sweep-2026-08]],
[[operations/rejected-representations-2026-08]].
Relationship to STAR-GS: [[operations/star-gs-v9-method]] remains a
PRESERVED, independent training-side candidate; LGS imports none of its
components (necessity test: no residual carving, no parent-free birth,
no donor accounting — the trial-render evaluator and λ-cost rule were
derived from codec mode decision, not from SRC).

## Thesis

Dynamic Gaussian representations force each primitive's content to live
and die with one contiguous temporal window. LGS replaces the dynamic
primitive with a LINEAGE: tied radiance attributes (world-frame SH +
base opacity) plus an ordered set of disjoint compact-support EPISODES,
each with episode-local geometry (translation; covariance-only rotation
offset; log-scale offset) and episode-local motion (rank-8 coefficients
over the backbone's shared time basis at an immutable per-episode
origin), with EXACT-zero absence between episodes. This makes exact
absence, reactivation of tied content, and cross-episode observation
pooling expressible in the representation — capabilities verified
unoccupied in dynamic GS — with a minimal structural search whose every
decision is confirmed by sampled counterfactual micro-renders.

Scope honesty (adversarially forced, binding): the claim is "tied
radiance attributes across trajectory episodes," NOT a full canonical 4D
object and NOT physical identity; rendered-scene expressivity equals
that of independent compact-support episode rows with equality
constraints — the novelty is the primitive/lineage-level state
(persistent identity container, exact absence, reactivation semantics),
carried by results.

## State space

Lineage i: content Θ_i = {SH deg-3 (world frame), base opacity,
log-scales, canonical rotation}; episodes j = 1..K_i (K_max = 4):
- Intervals via chained invariant parameterization: b_1 in a bounded
  transform onto [−margin, T+margin] (explicit boundary states for
  exact endpoint init); len_j = 2w + softplus(û_j); gap_j = 2w +
  softplus(v̂_j); final endpoint bounded by T+margin. Invariant under
  ANY optimizer step: episode count, ordering, duration floor 2w, gap
  floor 2w. Boundaries refine continuously (disclosed as not
  event-gated; per-parameter lr bounds; drift monitored). "At floor" =
  preregistered numerical threshold (softplus never reaches it exactly).
- w FIXED for the whole run (2 frame intervals). Disclosed event
  resolution: minimum episode and gap are 4 frame intervals; shorter
  absences are out of representational scope.
- Presence π_j(t) = S((t−b_j)/w)·S((d_j−t)/w), S = clamped cubic
  smoothstep (0 below 0, 3u²−2u³ on [0,1], 1 above 1): EXACT zero in
  gaps, exact plateau inside, LATCHED (no mid-episode dip expressible);
  at most one active episode per lineage at any t; winner by interval
  lookup (unique).
- Pose gauge: first episode is the reference (q = I, ℓ = 0 constants);
  on its removal, exact render-preserving re-anchoring with full gauge
  transport equations for translation/rotation/scale/motion; transformed
  optimizer moments RESET (logged); render-invariance unit-tested.
- Motion origins τ_j: fixed at episode creation, never changed. Fission
  children inherit the parent's τ (coefficient copy exact); reactivation
  episodes get fresh τ (basis in-distribution); clones copy per episode;
  active-interval-vs-basis-domain drift monitored.
- Sharing is HARD and lineage-internal only (no library, no soft
  assignment, no merge — the rejected variants are ledgered).

## Rendering & integration

Per timestamp: winner lookup per lineage; rendered opacity
σ(o_i)·π_winner(t)·routing; position x_j + LoRA offset at (t−τ_j);
covariance from (q_can∘q_j, s_i+ℓ_j); world-frame SH everywhere. One
row per lineage per timestamp; Python-side (presence/motion already
pre-rasterizer in ADAGS); no CUDA changes. Routing element-owned;
lineages with K>1 or any gap pinned dynamic (frozen logit, logged);
static conversion only for K=1 near-full-span lineages. Initialization:
FROM SCRATCH; initial-cloud lineages K=1 spanning the full sequence
("spanning-then-carve" default route); densification-born lineages
latched-open from birth (changed temporal prior vs backbone disclosed).

## Structural search (supporting machinery, not a contribution)

ONE evaluator for structural-event ops (scope: fission, truncation,
reactivation, birth-arbitration, episode prune — backbone clone/split/
prune keep their rules; three-regime structure disclosed):
sampled COUNTERFACTUAL MICRO-RENDERS — each candidate is scored by
rendering ≤16 tile-cropped affected (camera, timestamp) pairs (selection
ALGORITHM preregistered) with the change applied, measuring the exact
sampled ΔL (sign-correct, includes compositing reveal/insertion);
additions (birth/reactivation) receive a preregistered brief inner
rollout before scoring (codec modes are compared fitted — otherwise the
evaluator is biased toward removals); accepted iff ΔL̂ + λ·ΔS < 0 under
the dual caps (λ = preregistered stored-scalar price; acceptance test
only, no optimality/MDL claims; codec-mode-decision ancestry cited).
Screening: fp32 signed 5-frame-bin accumulators (sign-correct removal
estimate) select ≤64 candidates per decision point.
Dormant retrieval (deterministic): voxel hash of dormant/truncated
lineages' predicted poses (zero-order hold + optional bounded
constant-velocity, horizon ≤ H, preregistered), rebuilt per decision
point; residual site (multi-view residual consistency at current model
depth, disclosed heuristic) → radius-r lookup → ≤4 nearest candidates →
micro-render trials; none pass → birth. Scope: reactivation reaches
returns near a predictable pose; far/transformed returns become births
(no merge; missed reactivations permanent — accounted limitation).
Schedule (exact): screening windows 400 iters from iter 1000; removal =
2-window sign agreement THEN micro-render confirmation (earliest ≈ iter
1800; ≤8 rounds); creation trials at densification cadence from iter
1000; structural cutoff iter 4800 (≥1200-iter runway); K-overflow =
reject + log (reported representational-capacity failure metric).
Clone/split of multi-episode lineages: volume-preserving opacity split
uniformly across ALL episodes; per-episode mean perturbation in each
episode's own pose frame; content child fresh-init with ZERO moments;
pose/motion copied WITH moments (the single Adam rule); episode-local
clones prohibited; atomic transactions. Episode prune: len at floor AND
micro-render confirms. Lineage prune: episodeless or lifetime-
unsupported; dormancy alone never prunes.
Budgets: dual caps (peak rendered rows ≤ 600k; total stored trainable
scalars ≤ baseline budget) + FULL ledger (episode metadata,
accumulators, moments, staging, hash) + micro-render accounting
(cumulative candidate renders, accepted/tried, rasterizer and
topology-management time, peak memory, end-to-end GPU-h).

## Claims

1. (Representation) The lineage state space is new to dynamic GS
   (verified boundary, [[operations/lgs-novelty-record]]) and makes
   exact absence, reactivation, and cross-episode pooling expressible.
2. (Capability, results-carried, preregistered branches) On N3V cooking
   scenes: improved dynamic-region and event-region reconstruction at
   matched dual budgets, or Pareto-dominant held-out quality vs TOTAL
   memory, vs backbone AND untied reduction AND grouping controls.
3. (Search, supporting) Screening + micro-render acceptance suffices to
   instantiate the state space on real scenes; activation accounting
   reported; low activation = recorded METHOD FAILURE (never an
   exemption).
Never claimed: occlusion inference; physical identity; association of
independently born content; calibrated statistics; optimality;
densification novelty; edge-shape novelty.
Tier mapping: occlusion robustness ← compositing + latch + dormancy-safe
pruning (supporting); absence/reactivation structure ← presence programs
+ events (core); pooling quality/efficiency ← hard tying (core,
causally isolated per the experiment plan).

## Binding survival conditions (from adversarial round 5)

(1) substantial legitimate multi-episode activation on several real
scenes; (2) quality-vs-total-memory advantage surviving untied +
capacity-matched random + wrong grouping with the backbone below the
frontier; (3) evidence the search discovers useful episodes beyond
annotation-selected cases. Plus: annotations fixed before training and
never used for scene selection, tuning, activation floors, or stopping.

## Honest assessment

Hostile-mode novelty trajectory 4 → 5 → 4 → 7 → 5.5; calibrated
verdict in [[operations/lgs-novelty-record]]. The kernel was conceded
unoccupied by all five fresh reviewers; the recurring discount is
"obvious constrained factorization" — the burden is decisive empirical
validation under the preregistered Pareto/grouping controls. The
largest scientific risk is mechanism-benchmark mismatch: genuine
disappear-and-return events may be too rare on N3V for claim-grade
evidence; the plan's Phase-A census gate kills cheaply if so.
