# EL-GS v8.3 — Formal Specification (gate document, revision 4)

REVISION 4 (2026-08-11, implementation-readiness errata; NOT a new
gate round — the gate result of commit 519626d stands): two
implementation-affecting inconsistencies were confirmed against this
document and closed. E1 — LATCH/SIMPLEX: §1 is now the SINGLE
CANONICAL interval-state definition (admissible latch patterns,
per-pattern vector dimension 2K+1−n_lat, forward and inverse maps,
K=0, exact-boundary rule, latch inheritance across K-changing ops,
optimizer-moment resets, serialization). The unconditional
a ∈ R^{2K_f+1} of revisions ≤3 and BIRTH's "slack_post = 0 at init"
encoding are SUPERSEDED: an exact boundary is a latch bit, never a
zero simplex coordinate (softmax coordinates are strictly positive).
E2 — ESTIMATOR: the acceptance render estimator is canonically
SELF-NORMALIZED importance sampling (§7): strongly consistent,
bounded-weight, but not unbiased in general at finite sample size
(ratio-estimator bias O(1/n)); every
"ordinary/unnormalized", "unbiased", or "estimator exact" description
in earlier revision headers is superseded. No scientific claim
strengthens — acceptance was already a non-claimed preregistered
heuristic (§9). Superseded sentences below are tagged
[SUPERSEDED → §1] / [SUPERSEDED → §7] and are retained only as
revision history. Errata record and M0 test additions:
[[operations/elgs-implementation-readiness-errata]].

REVISION 3 (closes the three category-A defects of gate round 3):
A1 — LATCHED SPANS DEFINED: latch bits exist ONLY on b_1 (latched ⇒
exactly −w_m) and d_K (latched ⇒ exactly T+w_m); every other latch
pattern is inadmissible. The residual budget is ALWAYS
R = (T + 2w_m) − K·floor_len − (K−1)·floor_gap; the simplex dimension
is (#unlatched outer slacks) + K + (K−1) — a latch removes its outer
slack coordinate from the simplex (nothing else is subtracted). The
preregistered K-change tables specify latch-bit inheritance (children
of terminal ops inherit the parent's outer latch on the surviving
side; interior ops never touch latches) and optimizer-moment treatment
(re-parameterized simplex logits: moments reset, logged).
A2 — CANONICAL CLUSTER POINT: every cluster u stores, at binding, the
canonical point x_u := the seed surface point of the LOWEST-ID seed in
its connected component (deterministic, persisted). bind(u) = the
family of the nearest primitive to x_u within the preregistered
threshold (single-valued by construction); component re-formation at
MERGE recomputes x_u by the same lowest-ID rule over the merged
component.
A3 — MERGE SURVIVOR CONVENTION: the merged family RETAINS the older
parent's family ID, birth time, birth site, and lineage key; both
parents' cluster bindings are redirected to the surviving ID; the
younger parent's ID is retired and never reused. All predicates
(return-family, tie-breaks, exclusivity) operate on surviving IDs.


Date: 2026-08-09. Revision 2 after gate round 2. NEW IN v8.2:
(1) BOUNDARY LATCH BITS: exact boundary states (latched-open endpoints
d_K = T+w_m; b_1 = −w_m) are represented by DISCRETE per-endpoint latch
flags toggled only by structural ops — a latched endpoint is exact and
carries no logit; unlatched interior endpoints use the softmax simplex
over the remaining budget [SUPERSEDED → §1: Ω_free NEVER depends on
latches — a latched slack is identically zero; the phrase "Ω_free
computed excluding latched spans" is void].
"Len at floor" predicates are replaced by len ≤ floor + δ_tol
(preregistered tolerance). K=0 is a defined state (empty program:
renders nothing; prune-pending). Exact inverse maps for K-changes are
given by allocating the affected spans and re-normalizing the remaining
simplex (deterministic formulas in the op table).
(2) FAMILY-TO-EVIDENCE MAP: clusters are ASSIGNED at binding by the
single-valued map bind(u) = the family whose primitives contain u's
seed surface point at seeding time (deterministic nearest-primitive
rule with distance threshold; unassigned ⇒ cluster inactive);
U(f) := bind⁻¹(f); ℓ_b(P_f) sums ONLY over u ∈ U(f). Inheritance:
fission/truncate keep bind; MERGE unions; BIRTH starts unbound (or
late-seeded); clone/split does not alter bind (family-level).
(3) SNAPSHOT SEMANTICS (frozen functional): at each round boundary the
ENVIRONMENT (θ^{(r)}, committed programs of all OTHER families) is
frozen; for ANY candidate hypothesis (incl. new births/fissions/
merges), q is computed under this frozen environment with the
candidate family counterfactually inserted — deterministic and defined
for candidates that did not exist at snapshot time; A follows from q
by the cap rule. MERGE candidates evaluated this round use the union
anchor set to derive windows immediately (the "next round" re-derivation
applies only to post-commit bookkeeping).
(4) Transition deltas per case: FISSION(f,k,(g0,g1]): children
intervals (b_k,g0], (g1,d_k]; Δψ = ψ_dur(len_1')+ψ_dur(len_2')−
ψ_dur(len_k) + ψ_gap(g1−g0); interior TRUNCATE-delete removes two
adjacent gaps and creates one (Δψ accordingly); terminal delete removes
one; REACTIVATE inserts one episode creating one or two adjacencies by
position (both cases enumerated); MERGE gap terms recomputed on the
concatenated ordered episode sequence (formula: Σψ_gap over new
adjacencies − old adjacencies of both parents). Gauge re-anchoring
equations: q_can' = q_can ∘ q_ref; s' = s + ℓ_ref; per-episode offsets
re-expressed q_j' = q_ref⁻¹ ∘ q_j, ℓ_j' = ℓ_j − ℓ_ref, x_j re-expressed
in the new frame; world-frame SH untouched; moments of re-expressed
parameters reset (logged).
(5) PROP typing/quantifiers: Σ_{s∈S}Σ_{t∈s}; PROP 1 quantifies over
ALL eligible cameras, tracks, frames, and bridges; when fewer than
C_cap cameras are eligible, ALL eligible cameras are taken (both
streams); PROP 3 concludes "exactly one ownership cluster" (the term
'stream' = the cluster's capped report multiset, defined); α_u ∈ (0,1]
stated in the bound; the ε-aggregation bound holds AT FIXED capped
sets A (stated; cap-set changes are handled by the two-sided
Lipschitz argument applied to both streams).
(6) Acceptance estimator: [SUPERSEDED → §7: the adopted estimator is
SELF-NORMALIZED importance sampling — consistent, finite-sample
biased; "ordinary (unnormalized) ⇒ unbiased" is void] weights exact and
≤ 1/λ_u; paired (CRN) samples; bootstrap: paired cluster resampling of
(camera,frame) units with the SAME resample indices for candidate and
incumbent, B=200, SE = sd of paired replicate differences; degeneracy
(≤5 clusters) ⇒ reject; π_D and λ_u frozen before any confirmation
draw; deterministic component ordering = (min lineage ID in component);
one confirmation per component per pass.
(7) Anchor intervals formalized: maximal plateau runs whose capped
visible-report count at binding exceeds a preregistered floor; ψ_dur,
ψ_gap = preregistered quadratic barriers; DSSIM = the backbone's
implementation constants; dual-cap predicates, micro-render
confirmation, and lifetime-support given executable definitions in the
op table; classification data-term floors preregistered; operational
labels always printed with the qualifier "(fixed-path decision
decomposition, not statistical support)".
Prose: [[operations/elgs-method]].

## 1. State space (CANONICAL interval-state definition, rev 4)

Families f ∈ F with presence program P_f of K_f ≤ 4 episodes. This
section is the single source of truth for the interval state; it
consolidates the rev-2/rev-3 latch-bit and simplex fragments and
supersedes every earlier dimension or encoding statement.

LATCH BITS. Each family with K_f ≥ 1 carries exactly two latch bits
(ℓ_pre, ℓ_post) ∈ {0,1}²: ℓ_pre = 1 ⇔ b_1 = −w_m exactly;
ℓ_post = 1 ⇔ d_{K_f} = T + w_m exactly. All four patterns (0,0),
(1,0), (0,1), (1,1) are admissible; a latch anywhere else (interior
endpoints) is inadmissible. Latch bits are toggled ONLY by structural
ops (§5), never by gradient steps, and carry no optimizer state.

VECTOR DIMENSION. n_lat := ℓ_pre + ℓ_post. Raw vector
a ∈ R^{2K_f+1−n_lat}, i.e. (2 − n_lat) unlatched outer slacks + K_f
lengths + (K_f−1) gaps. Canonical coordinate order (also the
serialization order): [slack_pre iff ℓ_pre=0], len_1, gap_1, …,
gap_{K_f−1}, len_{K_f}, [slack_post iff ℓ_post=0].

FORWARD MAP. Ω = T + 2w_m; Ω_free = Ω − K_f·floor_len −
(K_f−1)·floor_gap (an op is admissible only if Ω_free > 0; Ω_free
NEVER depends on the latch pattern — a latched slack is identically
zero and simply has no coordinate). σ = softmax(a); each unlatched
slack = Ω_free·σ_(its coord); each latched slack ≡ 0; len_k =
floor_len + Ω_free·σ_(len_k); gap_k = floor_gap + Ω_free·σ_(gap_k).
b_1 = −w_m + slack_pre; endpoints follow by summation. Identity:
slack_pre + Σ_k len_k + Σ_k gap_k + slack_post = Ω for every latch
pattern (softmax sums to one over the remaining coordinates), so
d_{K_f} ≤ T + w_m holds IDENTICALLY under any optimizer step, with
equality iff ℓ_post = 1.

EXACT BOUNDARY RULE. Softmax coordinates are strictly positive, so
every unlatched slack is strictly > 0: exact boundary contact is
representable ONLY by a latch bit. A "zero slack coordinate" does not
exist in this parameterization and no op may target one.

INVERSE MAP (used by every structural op to write the child state).
Given an admissible target (latch pattern; every unlatched slack > 0
strictly; every len_k > floor_len and gap_k > floor_gap strictly):
σ_i = (value_i − floor_i)/Ω_free (floor_i = 0 for slacks), then
a_i = log σ_i − max_j log σ_j (gauge fixed so max a_i = 0;
deterministic). An op whose target would put an unlatched coordinate
exactly at its floor is inadmissible as stated; a target with an
outer slack exactly 0 must set the corresponding latch instead.

K_f = 0: defined state (empty program) — no latch bits, no vector a;
renders nothing; prune-pending (unchanged from v8.2).

LATCH INHERITANCE (canonical rule; expands rev-3 A1, consistent with
it): latches live on the family's outer endpoints. An op that
preserves an outer endpoint preserves its latch; an op that deletes
the outermost episode on a side DISCARDS that side's latch (the new
outer slack enters as an unlatched coordinate via the inverse map);
TRUNCATE-shorten applied to a latched outer endpoint CLEARS that
latch; REACTIVATE insertion outside a latched outer endpoint is
inadmissible (zero room); MERGE: ℓ_pre is taken from the parent
owning the globally earliest b_1, ℓ_post from the parent owning the
globally latest d_K; BIRTH sets (ℓ_pre, ℓ_post) = (0,1) (§5);
interior ops (interior fission/truncation/reactivation/prune) never
touch latches.

OPTIMIZER MOMENTS. Every structural op rewrites a through the inverse
map (dimension and/or values change), so the family's simplex-logit
moment state is RESET and the reset logged (rev-3 A1 unchanged).
Latch bits never carry moments.

SERIALIZATION / CHECKPOINTS. Persist per family: (K_f, ℓ_pre, ℓ_post,
a) in the canonical coordinate order; the loader MUST validate
dim(a) = 2K_f + 1 − n_lat (K_f = 0 persists as the empty program with
no latch bits). No checkpoints predate this errata (nothing has been
implemented), so no migration path exists or is needed; the dimension
check is mandatory from the first implementation.

Floors: floor_len = 2w + δ_len, floor_gap = 2w + δ_gap.
z_f(t) = 1 iff t in a plateau [b_k+w, d_k−w]; discrete edge bands
X_f = {t : |t − nearest edge| < w in frame units, strict}. Rendering,
pose/motion, gauge, pruning, caps: as [[operations/elgs-method]].

## 2. Evidence objects

Report OWNERSHIP (canonical): each track j is created from exactly one
seed s(j); each seed belongs to exactly one cluster (connected
components of the seed-overlap graph at binding); hence the map
o(j,c,t) = u(s(j)) is single-valued and the streams J(u) are pairwise
disjoint BY CONSTRUCTION. MERGE aggregation rules (deterministic):
merged clusters take r_u = min over members, d_u = min over members,
α_u recomputed from the merged n_cam and the preregistered correlation
model. Observation space Y = {miss} ⊔ ([v_min,v_max] × D_img), where
the positional coordinate is the RAW report position in the image
domain D_img (bounded). Heads (all normalized densities over Y w.r.t.
one base measure; fitted only on calibration scenes, frozen):
p_vis(y|b) = 1_miss·π_m^v + (1−1_miss)(1−π_m^v)·g_v(v)·g_pos(y_pos|b),
with g_pos = a truncated-Gaussian density over D_img centered at the
bridge-projected point (normalized over D_img for every b BY
CONSTRUCTION — no Jacobian issue);
p_cens(y) = 1_miss·π_m^c + (1−1_miss)(1−π_m^c)·h_c(v)·(1/|D_img|);
p_out analogous. FLOORS/CAPS: h_{c,o} ≥ h_floor > 0;
π_m^{c,o} ∈ [π_floor, 1−π_floor]; g_v ≤ g_cap; g_pos ≤ pos_cap;
|D_img| > 0. r_u ∈ [r_min,1]; d_u ∈ [0,1].
L1(y|b,q̃,r) = r[q̃·p_vis + (1−q̃)·p_cens] + (1−r)·p_out;
L0(y|r) = r·p_cens + (1−r)·p_out. Censoring equality at q̃=0: identical.

## 3. Observability

Windows W(f): maximal spans between consecutive anchor intervals; a
family with fewer than two anchors has NO evidence windows
(photometric-only; reported in coverage); merged families re-derive
windows from the union anchor set at the next round. Bridge family
B(W) as before (evidence-independent constructors; stop-gradient).
Sigma points: DETERMINISTIC NONNEGATIVE weights summing to one
(preregistered grid scheme; no negative-weight UT); κ_res ∈ [0,1] by
construction (clipped area ratio); q := clip(Σ ω·1_frustum·T_{−(f;j)}·
κ_res, 0, 1) — q̃ = q·d_u ∈ [0,1] always. q is a ROUND SNAPSHOT: the
round-r objective is written E^{(r)}(P, θ; q^{(r)}, A^{(r)}, H) —
candidates are scored under the snapshot (stated; refresh only at
round boundaries; the clone-invariance audit checks refresh-boundary
drift).

## 4. Objective (state energy + transaction ledger)

E^{(r)} = L_render(θ, {P_f}) + β·Σ_f Σ_W Φ_{f,W}(P_f)
 + Σ_f [ κ·K_f + Σ_k ψ_dur(len_k) + Σ_{k<K_f} ψ_gap(gap_k) ]
 + C(H), where C(H) = χ·N_returnbirth(H) + μ·N_merge(H) is an explicit
TRANSACTION LEDGER over the event history H (declared as such — the
objective is state energy PLUS ledger; acceptance always compares
E^{(r)} including the candidate's transaction increment). κ is charged
PER EPISODE via κ·K_f (state term: refunded automatically when
episodes are deleted); χ and μ are ledgered and never refunded.
L_render: the backbone loss (L1 + λ_ssim·DSSIM at training weights),
per-pixel mean over the evaluation measure ν = uniform over
(training-camera, frame) pairs × pixels; no masks.

Evidence term. Segment set S(P_f, W) = the maximal runs of frames
{t ∈ W : t ∉ X_f}, each run carrying its constant z_f(t) (definition
explicit). CAP OPERATOR (per bridge): A_{b,j,t}(P) = the C_cap cameras
of highest q̃_{b,j,c,t}, ties broken by ascending camera ID
(bridge-independent tie-break; report ELIGIBILITY is bridge-independent
by construction — all (j,c,t) with c ∈ eligible-camera mask). Both the
scored and censored streams use exactly A_{b,j,t}:
ℓ_b(P_f) = Σ_u α_u Σ_{j∈J(u)} Σ_{t∈S(P_f,W)} Σ_{c∈A_{b,j,t}}
log L_{z_f(t)}(y_{j,c,t} | b, q̃_{b,j,c,t}, r_u);
ℓ_b^cens identical with L0 in place of L_z (same A_{b,j,t}).
Φ_{f,W}(P_f) = −τ_B·log[ Σ_b e^{ℓ_b/τ_B} / Σ_b e^{ℓ_b^cens/τ_B} ]
(equivalently Λ − Λ^cens with the normalized tempered form; τ_B > 0).
TERMINOLOGY: this is a TEMPERED BRIDGE AGGREGATION (not marginal
likelihood: τ_B ≠ 1 and different bridges score different capped report
subsets — an engineered energy, disclosed).

PROP 1 (censored zero, repaired). If q̃_{b,j,c,t} = 0 for ALL b over a
segment s, then for every b: (i) the cap sets A_{b,j,t} coincide across
b on s (all q̃ equal zero ⇒ selection is by the bridge-independent
camera-ID tie-break alone); (ii) L1 = L0 on s (censoring equality).
Hence the segment adds the SAME constant s_const to every ℓ_b and every
ℓ_b^cens, and since F(x + c·1) = F(x) − c for F = −τ log-mean-exp, the
contribution cancels in Φ for any program and any transition
re-partitioning s. ∎
PROP 2 (clone/split invariance, conditional). AT FIXED ROUND SNAPSHOT
(q^{(r)}, A^{(r)}): clone/split changes family membership only;
U(f), P_f, B(W), q^{(r)}, A^{(r)} unchanged ⇒ Φ unchanged. Across
refresh boundaries invariance is NOT claimed; the preregistered
clone-invariance audit measures the drift. ∎
PROP 3 (merge accounting, repaired). By the canonical ownership map,
J(u) are pairwise disjoint before and after cluster re-formation
(components of a partition-refinement union); each report enters
exactly one stream. Merged-cluster parameters follow the deterministic
aggregation rules of §2. ∎

## 5. Structural ops (transition deltas made exact)

- FISSION(f,k,gap): one episode → two (ΔK = +1): Δstate = +κ +
  Δψ_dur + ψ_gap(new gap); admissible iff K_f < 4, floors fit in
  Ω_free, gap ⊆ plateau with margins.
- TRUNCATE-shorten (ΔK = 0: edge moved): Δstate = Δψ only.
  TRUNCATE-delete (terminal episode removed, ΔK = −1): Δstate = −κ +
  Δψ (κ auto-refunded: state term).
- REACTIVATE(f, e): pre = NO return family at the site (deterministic
  predicate below) AND K_f < 4 AND floors fit. Post: +κ + Δψ; pose
  init = spline-bridge mean at episode center; fresh coefficients and
  origin; fresh moments.
- BIRTH(f′): K=1 with (ℓ_pre, ℓ_post) = (0, 1) — terminal latch:
  d_1 = T + w_m EXACT, represented by the latch bit, NOT by a zero
  slack coordinate (supersedes "slack_post = 0 at init"). State
  a ∈ R^2 = (slack_pre, len_1) via the §1 inverse map with targets
  slack_pre = t_birth + w_m, len_1 = T + w_m − t_birth (admissible iff
  len_1 > floor_len strictly AND t_birth > −w_m, i.e. slack_pre > 0;
  §1 admissibility governs). Δ = +κ + ψ + LEDGER χ if at a return
  site. Admissible under caps; cap-saturated ⇒ INADMISSIBLE this
  round, re-eligible next (logged).
- MERGE(f_old, f_new): pre: return-family predicate holds; episode
  unions disjoint with floors; K_old + K_new ≤ 4; else INADMISSIBLE.
  Post: episodes concatenated (κ·K unchanged in total); radiance from
  the OLDER family (disclosed convention + symmetric ablation);
  pose/motion retained per episode; merged family re-anchored to the
  older family's first episode (exact reparameterization; moments
  reset; unit-tested); clusters re-formed per §2; windows re-derived
  next round; LEDGER +μ.
- PRUNE (episode: len at floor AND micro-render confirms; family:
  episodeless or lifetime-unsupported): Δstate per deletions; ledger
  untouched.
ALL OPS (rev 4): every op writes its child interval state through the
§1 inverse map (child dimension 2K′+1−n_lat′ per the child latch
pattern); latch inheritance, exact-boundary handling, target
admissibility, and moment resets are governed by §1 alone — no op
table entry may encode a boundary as a zero coordinate.
RETURN-FAMILY PREDICATE (deterministic): a family born within radius
r_site of the site with birth time inside W; ties → earliest birth,
then lowest family ID. REACTIVATE/MERGE mutual exclusivity follows.
Path-dependence disclosed; re-tested each round; no undo.

## 6. ε-bound (corrected, two-sided)

Assumptions: §2 floors/caps; r ≤ 1; 0 ≤ ε < 1. Pointwise, for q̃ ≤ ε:
L1 ≤ L0 + ε·p_cap (p_cap = sup p_vis, finite) and
L1 ≥ (1−ε)·L0 (since L1 ≥ r(1−q̃)p_cens + (1−r)p_out ≥ (1−q̃)L0).
Hence |log L1 − log L0| ≤ M(ε) := max{ log(1 + ε·p_cap/p_floor),
−log(1−ε) }, with p_floor = inf L0 > 0. Aggregation: for each bridge b,
|ℓ_b − ℓ_b^cens| restricted to reports with q̃_b ≤ ε is at most
Σ_u α_u · n_{u,b,ε} · M(ε), where n_{u,b,ε} = the number of capped
reports of cluster u in the window with q̃_b ≤ ε (finite: ≤ C_cap ×
|J(u)| × |W|). Since F is 1-Lipschitz in the ℓ∞ norm (∇F = −softmax,
‖∇F‖₁ = 1; τ_B > 0), |Φ shift| ≤ max_b Σ_u α_u·n_{u,b,ε}·M(ε). ∎
Reported with the empirical power curve vs oracle opportunity.

## 7. Acceptance (heuristic; SNIS estimator — consistent, finite-sample biased)

CANONICAL ESTIMATOR (self-normalized importance sampling; supersedes
every "ordinary/unnormalized", "unbiased", or "estimator exact"
description of the sampled render estimate in earlier revisions):
R̂ = Σ_i a_i·ℓ(x_i) / Σ_i a_i with samples x_i drawn from the mixture
m = λ_u·ν + (1−λ_u)·π_D (0 < λ_u ≤ 1), and
a_i = min{ w_max, ν(x_i)/m(x_i) } with w_max := 1/λ_u — the true
weight is ≤ 1/λ_u ALWAYS, so clipping is PROVABLY INACTIVE (retained
as a formal guard). GUARANTEES (all that is claimed): R̂ is a ratio
estimator — strongly consistent for the ν-mean E_ν[ℓ] as n → ∞ and
bounded-weight stable, but NOT unbiased in general at finite sample
size (ratio-estimator bias O(1/n); it vanishes only in degenerate
cases, e.g. λ_u = 1 or constant ℓ); no unbiasedness or exactness of
the sampled estimate is
claimed or used anywhere — acceptance is a preregistered heuristic
(§9), and the paired CRN design cancels the shared normalization
noise between candidate and incumbent without eliminating the bias.
(The unnormalized alternative (1/n)·Σ_i a_i·ℓ(x_i) would be unbiased
for the normalized ν but is NOT the adopted estimator; SNIS is
adopted for weight-noise cancellation under CRN.) Exact (non-sampled)
tracker and prior deltas — computed in closed form, not estimated —
are added outside the sampled render estimate. COMMON RANDOM NUMBERS:
identical {x_i} for incumbent and candidate; ΔÊ is the paired SNIS
difference plus the exact deltas. SE: cluster bootstrap over
(camera, frame) units, B=200 replicates, weights renormalized within
each replicate (the renormalization is exactly the self-normalized
form applied per replicate); SE = sd of paired replicate differences;
SE undefined ⇒ reject. Accept iff ΔÊ + k·SE < 0 (ΔÊ includes the
transaction increment). Rejected candidates: all refit state DISCARDED (incumbent
snapshot restored). Sample partitioning: NO hashing — the reserved
pool is pre-partitioned at iteration 0 into an indexed grid of slots
(round, pass, rank) with rank = the deterministic ordering of conflict
components within the pass; assignment is injective by construction;
unused slots discarded; refits never see confirmation samples.

## 8. Decision classification (flags corrected)

Per committed decision on its confirmation samples at post-refit
parameters (fixed path, disclosed): (a) prior+ledger removed;
render-only flag = passes with tracker AND prior removed; tracker-only
flag = passes with render AND prior removed. DATA-SUPPORTED (an
OPERATIONAL label, not statistical support) = passes (a) AND ≥1
single-term flag; PRIOR-PIVOTAL = fails (a); INTERACTION-SUPPORTED =
passes (a), no single-term flag; EQUIVALENCE-CLASS = all data terms
below preregistered floors. ITT logs every screened candidate;
risk-coverage over the full event inventory.

## 9. Non-claims (extended rev 4)

No statistical validity of acceptance; no unbiasedness or exactness
of the sampled acceptance estimator (SNIS: consistent, finite-sample
biased, §7); no calibrated posteriors; no
physical absence; no identifiability; no global optimality; "tempered
bridge aggregation" ≠ marginalization (stated). The ten viability
conditions of [[operations/elgs-novelty-record]] bind implementation.
