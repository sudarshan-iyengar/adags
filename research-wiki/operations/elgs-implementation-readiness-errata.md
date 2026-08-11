# EL-GS — Implementation-Readiness Errata (2026-08-11)

Status: targeted post-gate correction. NOT a new research gate,
method change, or novelty review — the gate result of commit
`519626d` ([[operations/elgs-review-history]], PASSED 2026-08-09)
stands. Scope: two implementation-affecting inconsistencies inside
[[operations/elgs-v8-formal-spec]], their direct dependency cones,
and nothing else. The spec itself is updated inline (revision 4);
this page is the audit record, not the source of truth — implementers
read the spec.

## Issue 1 — latch and simplex semantics: CONFIRMED, resolved

The committed v8.3 text simultaneously stated (i) a latched outer
endpoint is exact and carries no simplex coordinate (rev-2 item 1,
rev-3 A1: dimension = #unlatched outer slacks + K + (K−1)); (ii) an
unconditional interval vector a ∈ R^{2K_f+1} (§1); and (iii) BIRTH as
a terminal latch encoded by "slack_post = 0 at init" (§5) — a value a
softmax coordinate cannot take. (i) vs (ii)+(iii) is a real
contradiction: two faithful implementations would diverge on vector
dimension and on whether an exact boundary is a latch bit or a zero
coordinate.

CANONICAL RESOLUTION (now spec §1, single source of truth):

- Latch bits (ℓ_pre, ℓ_post) ∈ {0,1}² on b_1/d_K only; all four
  patterns admissible; interior latches inadmissible; toggled only by
  structural ops.
- a ∈ R^{2K+1−n_lat}, n_lat = ℓ_pre + ℓ_post, in a canonical
  (serialized) coordinate order; σ = softmax(a).
- Forward map: unlatched slack = Ω_free·σ; latched slack ≡ 0 (no
  coordinate); len/gap = floor + Ω_free·σ; Ω_free = Ω − K·floor_len −
  (K−1)·floor_gap NEVER depends on latches; the Ω-sum identity holds
  for every pattern, so d_K ≤ T+w_m identically, equality iff
  ℓ_post = 1.
- Exact-boundary rule: softmax coordinates are strictly positive, so
  exact contact exists ONLY as a latch; "zero slack coordinate" is
  unrepresentable and no op may target one.
- Inverse map: σ_i = (value_i − floor_i)/Ω_free, a_i = log σ_i −
  max_j log σ_j (deterministic gauge); targets at an exact floor are
  inadmissible; an exactly-zero outer-slack target must set the latch.
- K=0: empty program, no latches, no vector (unchanged from v8.2).
- Latch inheritance: preserved with a preserved outer endpoint;
  discarded when the outermost episode on that side is deleted;
  cleared by TRUNCATE-shorten of a latched endpoint; REACTIVATE
  outside a latched endpoint inadmissible; MERGE takes ℓ_pre/ℓ_post
  from the parents owning the extreme endpoints; BIRTH = (0,1);
  interior ops never touch latches.
- Optimizer moments: every structural op re-derives a via the inverse
  map ⇒ simplex-moment reset, logged; latch bits carry no moments.
- Serialization: persist (K, ℓ_pre, ℓ_post, a); loader validates
  dim(a) = 2K+1−n_lat. No pre-errata checkpoints exist (nothing
  implemented) ⇒ no migration path required.

Superseded-text handling: §1's unconditional R^{2K+1} was EXCISED and
replaced by the canonical definition (its supersession is recorded in
the revision-4 header); BIRTH's "slack_post = 0" is quoted-superseded
inside the rewritten §5 entry; the v8.2 phrase "Ω_free computed
excluding latched spans" is tagged [SUPERSEDED → §1] in the retained
revision header.

## Issue 2 — importance-estimator semantics: CONFIRMED, resolved

The committed v8.3 text defined R̂ = Σ_i a_i·ℓ(x_i)/Σ_i a_i and named
it self-normalized importance sampling (§7) while the §7 title said
"estimator exact" and the v8.2 header item (6) said "ORDINARY
(unnormalized) importance sampling … ν normalized ⇒ unbiased". The
formula is SNIS: a ratio estimator, not unbiased in general at finite
sample size (bias O(1/n), vanishing only in degenerate cases such as
λ_u = 1), consistent as n → ∞. "Unbiased" and "exact" were false as
applied to it.

CANONICAL RESOLUTION (now spec §7): EL-GS uses SELF-NORMALIZED
importance sampling, explicitly declared a finite-sample-biased,
strongly consistent, bounded-weight (a_i ≤ 1/λ_u, clipping provably
inactive) heuristic estimator. SNIS is retained over the unbiased
unnormalized form deliberately — the bootstrap as committed
renormalizes weights per replicate (an SNIS construction) and paired
CRN differencing cancels shared normalization noise. "Exact" survives
only where true: the non-sampled tracker/prior deltas computed in
closed form, and the exactness of the weights themselves. The
acceptance rule ΔÊ + k·SE < 0, CRN pairing, B=200 paired cluster
bootstrap, degeneracy rejection, freezing of π_D/λ_u, and the no-hash
pre-partitioned sample-slot grid are unchanged. §9 non-claims now
state the absence of estimator unbiasedness explicitly.

## Claims: what stands, what is reclassified

- UNCHANGED: the single conditional scientific claim of
  [[operations/elgs-method]]; PROP 1 (censored zero), PROP 2
  (conditional clone/split invariance), PROP 3 (merge accounting);
  the two-sided ε-bound (§6); censoring equality; the gate verdict
  (SURVIVES-WITH-RISKS, hostile novelty 8/10); calibrated novelty 8.0
  conditional ([[operations/elgs-novelty-record]]); all ten binding
  viability conditions; the experiment plan's claim map and kill
  rules. None of these ever depended on estimator unbiasedness —
  acceptance was already a non-claimed heuristic — or on a specific
  simplex encoding of exact boundaries.
- RECLASSIFIED (weakened language, no scientific content change):
  "estimator exact / unbiased" → "SNIS: consistent, finite-sample
  biased, bounded-weight; disclosed heuristic." This removes an
  overclaim; nothing strengthens.

## Implementation consequences

1. Interval state is (K, ℓ_pre, ℓ_post, a) with variable dim(a) =
   2K+1−n_lat; every structural op writes children via the §1 inverse
   map; loaders validate the dimension.
2. Exact boundaries are branch-on-latch code paths, never numeric
   zeros; no epsilon-thresholding of slacks may stand in for a latch.
3. The acceptance code implements SNIS with per-replicate bootstrap
   renormalization; no code or comment may describe the sampled
   estimate as unbiased/exact; logging should record n, ESS, and the
   paired ΔÊ.

## Required M0 (B0) tests — added to [[operations/elgs-experiment-plan]]

- Latch/simplex round-trip: forward∘inverse identity on random
  admissible states across all four latch patterns and K ∈ {1..4};
  dimension checks 2K+1−n_lat; Ω-sum identity per pattern; strict
  positivity of unlatched slacks; inadmissibility of exact-floor
  targets.
- Latch-inheritance property tests over the full op table (birth,
  fission, truncate-shorten/-delete, reactivate, merge, prune),
  including latch discard/clear cases, moment-reset logging, and
  serialization load-validate.
- SNIS sanity: weights ≤ 1/λ_u and clip-inactivity on synthetic
  fixtures; empirical bias → 0 with n (consistency check) against a
  closed-form ν-mean; CRN pairing determinism; bootstrap
  per-replicate renormalization; degeneracy (≤5 clusters) rejection
  path.

## Fresh-context review of the repair (2026-08-11)

One blind mathematical reviewer (fresh context, xhigh effort) audited
the two repaired definitions, their dependency cones, the claim-change
question, and the new-contradiction question. Verdict: **PASS — no
fatal or critical findings.** It independently confirmed: PROP 1-3 and
the §6 ε-bound do not depend on either repaired definition; the only
claim-level movement is the disclosed weakening of the estimator
language; canonical §1 is consistent with the retained rev-3 A1
header; no untagged text asserting the old semantics survives. Three
minor wording findings were adopted: (i) "biased at any finite sample
size" tightened to "not unbiased in general (bias O(1/n), vanishing in
degenerate cases such as λ_u = 1)"; (ii) the BIRTH admissibility "iff"
completed with t_birth > −w_m (slack_pre > 0); (iii) this page's
superseded-text bookkeeping corrected (§1's old text was excised, not
tagged in place).

## Affected files

- [[operations/elgs-v8-formal-spec]] — inline revision 4 (canonical
  §1, §5 BIRTH + all-ops note, §7, §9; superseded text tagged).
- [[operations/elgs-experiment-plan]] — B0 test list extended.
- [[operations/elgs-method]] — stale "hash-partition" wording aligned
  with the spec's no-hash pre-partitioned slot grid; errata pointer.
- [[operations/elgs-review-history]] — dated post-gate errata note.
- This page (new).
