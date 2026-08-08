# LGS — Preregistered Experiment Plan (plan only; nothing scheduled)

Date: 2026-08-08. Status: complete claim-driven plan for testing
[[operations/lgs-method]] if and when implementation is approved. No
code written, no jobs submitted. Encodes ALL ten viability conditions
of [[operations/lgs-novelty-record]] and the round-5 survival
conditions of [[operations/lgs-review-history]]. Project discipline
unchanged: N3V protocol (1352x1014, cam00 held out, 300 frames), dev =
cut_roasted_beef + cook_spinach, locked = flame_steak + sear_steak,
stress = coffee_martini + flame_salmon_1; 6000 it / 600k rows; R009
historical-only; annotations evaluation-only, fixed before training,
never used for scene selection, tuning, activation floors, or stopping;
configs committed before submission; job IDs sacct-verified.

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1 (dominant, representation): hard cross-episode radiance tying over the lineage state space improves held-out reconstruction where content is absent-and-returns, beyond any capacity/compute/search/grouping explanation | (a) B3 fixed-graph tied-vs-untied wins under the frozen 2×2 protocol at adequate coverage; (b) B2 system table: LGS on a better quality-vs-total-memory Pareto frontier than backbone AND untied reduction; (c) B3 grouping controls (random/wrong) fail to reproduce the gain; (d) episode-onset holdout gains | B2 B3 B4 |
| C2 (supporting, search sufficiency): the minimal preregistered search instantiates the state space on real scenes | B1 activation census passes preregistered floors; reactivation attribution matrix shows own-content retrieval > new birth and > donor-copy respawn at matched structure | B1 B3 |
| Anti-claims to rule out | capacity (dual caps + Pareto multi-point), compute (iso-compute lane; search cost ledger), search-alone (viability cond. 10: tied benefit must persist when search is frozen), generic compression (random/wrong grouping), single-window sufficiency (tied contiguous-window control), annotation leakage (prohibitions + external masks) | B2 B3 B4 |

## Experiment blocks

### B0 — Static checks and metric freeze (MUST-RUN; ~5 GPU-h)
Unit tests: interval invariants under adversarial optimizer steps;
winner-lookup uniqueness; re-anchoring render invariance (position,
covariance, opacity, all SH degrees, multiple timestamps); atomic
transaction rollback; ledger correctness. Metric spec frozen: dynamic-
mask source, event-region metrics (annotated, evaluation-only,
output-blind), masked-LPIPS crop protocol, tOF implementation, TOTAL-
memory definition (stored scalars + episode metadata + accumulators +
optimizer moments + retrieval index), compute measure (GPU-h). Smoke:
winner-gather and micro-render overhead measured.

### B1 — ACTIVATION CENSUS GATE (MUST-RUN FIRST; ~15-20 GPU-h; cheap kill)
End-to-end LGS on both dev scenes, 1 seed each + 1 repeat on cut.
Preregistered floors (committed before running): (i) K>1 lineage count
and fraction; (ii) pixel share of multi-episode lineages inside
dynamic-mask regions ≥ floor; (iii) ≥ preregistered count of accepted
reactivations and fissions with post-hoc micro-render audit panels;
(iv) ineffective-fission and false-reactivation rates below ceilings;
(v) K-overflow and cap-rejection rates reported. FAIL ⇒ recorded
METHOD FAILURE for the recurrence half (viability cond. 1; round-5
survival cond. 1); one preregistered parameter-revision cycle allowed
(new floors committed first). Event-length census from existing project
census data recorded as design input BEFORE running.

### B2 — Main system table + Pareto (MUST-RUN; ~95 GPU-h)
Dev scenes, 3 seeds: backbone (route0) vs LGS vs UNTIED REDUCTION
(identical machinery, sharing disabled — the defined baseline) at
IDENTICAL dual caps. Plus Pareto operating points for all three
systems: λ ∈ {λ/3, λ, 3λ} × caps ∈ {450k, 600k} (1 seed each,
replication-on-signal). Iso-compute lane: backbone given LGS's measured
search compute as extra optimization. Metrics: dynamic-mask PSNR/LPIPS
(co-primary), event-region PSNR/LPIPS (annotated, supplementary), tOF,
static no-harm (−0.05 dB bound), global non-inferiority, full ledgers
(TOTAL memory, GPU-h, activation accounting). Success: LGS on a better
quality-vs-total-memory frontier than backbone AND untied (viability
cond. 8) with the backbone below the frontier. Kill: untied matches LGS
at matched budgets ⇒ CC2/CC5 lose empirical justification (cond. 9).

### B3 — Decisive isolation + attribution matrix (MUST-RUN; ~85 GPU-h)
(a) 2×2 STRUCTURE-CONTENT isolation: graphs from tied AND untied B2
runs; branch from common checkpoints; freeze graph/intervals/poses/
density/routing/ALL geometry; evaluation masks frozen from checkpoints
+ EXTERNAL masks (annotated event regions; estimated-flow motion masks,
labeled estimated); reset content (SH+opacity only) to a common init,
zero moments, shared RNG; content-only optimization in all 4 cells;
3 seeds on the decisive within-graph contrasts. Coverage report gates
interpretation; below floor ⇒ uninterpretable AND recorded activation
failure. (b) Episode-onset holdouts: contiguous onset blocks of
preregistered externally-defined recurrence windows excluded from
structure AND content learning in all arms; report quality vs
observations-available-in-episode (viability cond. 4). (c) Grouping
controls at matched capacity: random grouping, wrong grouping
(gauge-aligned, matched episode count/lifetimes) (cond. 2).
(d) Tied CONTIGUOUS-window control (single interval spanning, tied) —
separates multi-episode structure from tying. (e) Reactivation
attribution matrix on preregistered return sites: own-content retrieval
vs new birth vs donor-copy respawn (3DGS-MCMC-style) vs zero-order vs
constant-velocity vs ORACLE retrieval ceiling; report false
reactivations, missed returns, pose basin (cond. 5). Kill rules: tied ≤
untied within both graphs at adequate coverage ⇒ pooling thesis dies;
random/wrong grouping reproduces the quality-memory point ⇒ generic
compression, thesis dies.

### B4 — Simplicity, search audit, misspecification probes (MUST-RUN core, ~50 GPU-h)
Deletions: fission-off; reactivation-off; single-interval (K=1);
latch-vs-bump (appendix); compact-support vs sigmoid tails (appendix).
Search audit (cond. 6): frozen protocol; correlation of sampled
acceptance ΔL̂ with held-out/full-objective deltas; rejected-proposal
accounting; all rollout/micro-render cost in the ledger. No-rescue-by-
search (cond. 10): freeze the learned structure, disable search, verify
tied benefit persists; if gains exist only with live search and no
tying benefit ⇒ reposition as structural optimization (preregistered
honest branch). Misspecification probes: shared+episode-residual soft
sharing (material win ⇒ hard sharing reported misspecified);
object-frame SH sharing control. NICE-TO-HAVE: 12k-iteration diagnostic
with compute-matched baseline/untied; synthetic recurrence fixture with
oracle lineage GT (DIAGNOSTIC ONLY — never a Go criterion, per standing
project rule).

### B5 — Freeze, transfer, finals, failure analysis (~100-130 GPU-h)
Freeze all constants after B3/B4, before transfer. Locked pair
(flame_steak, sear_steak): single post-freeze evaluation. Six-scene
full-length finals: LGS vs backbone vs untied at published-comparable
budgets + published-number context table (FreeTimeGS, SharpTimeGS,
TAD-GS/2606.23212-VAD) with protocol pinning (LPIPS-Alex, cam13
handling disclosed) and full ledgers; 3 seeds headline systems if
budget allows. Stress tier (coffee_martini, flame_salmon_1):
preregistered-checkpoint evaluation, reveal panels, failure analysis
(false reactivations, mis-carved gaps, appearance-bias cases,
K-overflow inventory).

## Run order, cost, gates

| Milestone | Goal | Gate | GPU-h |
|---|---|---|---|
| M0 | B0 static checks, metric freeze, smokes | tests green; specs committed | ~5 |
| M1 | B1 activation census | preregistered floors pass (else cheap recorded negative) | ~15-20 |
| M2 | B2 system table + Pareto + iso-compute | LGS on better frontier than backbone AND untied | ~95 |
| M3 | B3 isolation + attribution | tied > untied within graphs; grouping controls fail to reproduce | ~85 |
| M4 | B4 deletions + search audit + probes | no-rescue and misspecification branches evaluated | ~50 |
| M5 | freeze → locked transfer → finals → stress | Gate-B-style verdicts; tables complete | ~100-130 |

Total ≈ 350-420 GPU-h (within the standard few-hundred envelope; M5
trims seeds on overrun). Every lane: config committed before
submission; activation diagnostics mandatory; cam00 never used in
construction, search, or tuning; annotations never consumed by
training, selection, or stopping.

## Why this program is sufficient

C1 is carried by three mutually reinforcing legs — causal isolation at
fixed structure (B3a), system-level Pareto dominance (B2), and
counterfeit-explanation exclusion (grouping/contiguous/iso-compute/
search-frozen controls) — each with preregistered kill branches, so a
positive result cannot be attributed to capacity, compute, search, or
generic compression, and a negative result is decisive and cheap
(ordered gates kill at M1 or M3 before finals spend). C2 is carried by
the census gate and the attribution matrix, with the oracle-retrieval
ceiling bounding what better search could add. Honest-failure branches
(activation failure, misspecification, search-only) are preregistered
outcomes, not post-hoc rescues.

## Risks

Activation rarity on N3V (B1 kills cheaply; stress scenes provide
additional event density); micro-render/search overhead eroding the
compute story (measured, ledgered, iso-compute control); appearance
bias across episodes (object-frame + soft-sharing probes); 2×2
graph-source is post-treatment (only within-graph contrasts claimed;
oracle-graph fixture is diagnostic); six-scene statistical power
(hierarchical reporting, per-scene effects, locked transfer as honesty
check).
