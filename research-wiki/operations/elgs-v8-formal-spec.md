# EL-GS v8.3 — Formal Specification (gate document, revision 3)

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
over the remaining budget (Ω_free computed excluding latched spans).
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
(6) Acceptance estimator: ORDINARY (unnormalized) importance sampling
for the ν-mean (ν is normalized ⇒ unbiased), weights exact and
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

## 1. State space

Families f ∈ F with presence program P_f of K_f ≤ 4 episodes. INTERVAL
PARAMETERIZATION (simplex form; replaces the chain): let Ω = T + 2w_m
and Ω_free = Ω − K_f·floor_len − (K_f−1)·floor_gap (an op is admissible
only if Ω_free > 0). Raw vector a ∈ R^{2K_f+1} → σ = softmax(a) →
slack_pre = Ω_free·σ_0; len_k = floor_len + Ω_free·σ_{2k−1};
gap_k = floor_gap + Ω_free·σ_{2k}; b_1 = −w_m + slack_pre; endpoints
follow by summation, and d_{K_f} ≤ T + w_m holds IDENTICALLY under any
optimizer step. Floors: floor_len = 2w + δ_len, floor_gap = 2w + δ_gap.
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
- BIRTH(f′): K=1 latched-open: episode = (t_birth, T + w_m] in simplex
  form (slack_post = 0 at init). Δ = +κ + ψ + LEDGER χ if at a return
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

## 7. Acceptance (heuristic; estimator exact)

Estimator: R̂ = Σ_i a_i·ℓ(x_i) / Σ_i a_i with samples x_i from the
mixture λ_u·ν + (1−λ_u)·π_D (0 < λ_u ≤ 1), and
a_i = min{ w_max, ν(x_i)/(λ_u·ν(x_i) + (1−λ_u)·π_D(x_i)) } with
w_max := 1/λ_u — the true weight is ≤ 1/λ_u ALWAYS, so clipping is
PROVABLY INACTIVE (retained as a formal guard); the estimator is
self-normalized importance sampling for the declared ν-target. Exact
(non-sampled) tracker and prior deltas are added outside the sampled
render estimate. COMMON RANDOM NUMBERS: identical {x_i} for incumbent
and candidate. SE: cluster bootstrap over (camera, frame) units, B=200
replicates, weights renormalized within each replicate; SE undefined ⇒
reject. Accept iff ΔÊ + k·SE < 0 (ΔÊ includes the transaction
increment). Rejected candidates: all refit state DISCARDED (incumbent
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

## 9. Non-claims (unchanged)

No statistical validity of acceptance; no calibrated posteriors; no
physical absence; no identifiability; no global optimality; "tempered
bridge aggregation" ≠ marginalization (stated). The ten viability
conditions of [[operations/elgs-novelty-record]] bind implementation.
