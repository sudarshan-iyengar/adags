# EL-GS M0/M1 Implementation Plan (APPROVED)

Status: **APPROVED by user 2026-08-11** (plan-mode review; two directed
revision rounds + one focused fresh-context review, all findings resolved).
Authority: implementation semantics per commit `c21de8b`
([[operations/elgs-v8-formal-spec]] rev 4 +
[[operations/elgs-implementation-readiness-errata]] +
[[operations/lgs-method]] substrate via the spec's delegation chain).
Scope: M0 (faithful implementation + verification) and M1 (DiVa-360
evidence/activation census on Apollo/Determined); M2-M5 dependency outline
only. This page is the verbatim preserved plan; section numbers (S1-S17,
S16b/S16c) are internal to it.

---

## Context

EL-GS (Evidence-Lineage Gaussian Splatting) passed its research and mathematical gate at commit
`519626d` (v8.3 formal spec, SURVIVES-WITH-RISKS, hostile novelty 8/10) and its two remaining
implementation-affecting spec inconsistencies were resolved in commit `c21de8b` (spec revision 4:
canonical latch/simplex interval state; canonical SNIS acceptance estimator). `c21de8b` is the
canonical implementation authority. Nothing has been implemented, trained, or submitted.

This plan covers, in evidence-backed detail: (M0) faithful EL-GS implementation into ADAGS with
specification-derived CPU tests, integration tests, checkpoint behavior, and a minimal Apollo GPU
smoke; and (M1) the DiVa-360 evidence-and-activation census gate executed through Determined on
Apollo with full provenance. M2–M5 are outlined at dependency level only.

Execution environment: Apollo cluster, Determined AI scheduler, pools DGX/V100 and Hopper/H100.
The working checkout is `D:\adags` on branch `apollo/csvl-vpl-v2-exploratory`.

## 1. Verified starting state (evidence: git commands run 2026-08-11)

- Repository: `D:\adags`, remote `origin = https://github.com/sudarshan-iyengar/adags.git`.
- Branch: `apollo/csvl-vpl-v2-exploratory`, upstream `origin/apollo/csvl-vpl-v2-exploratory`, **ahead 5** (unpushed; never force-push. Non-force pushes are authorized by plan acceptance per §16b, which supersedes the earlier pushing-is-a-user-decision caution — the first push publishes these 5 wiki commits).
- HEAD = `c21de8b` "wiki: EL-GS implementation-readiness errata (spec rev 4)".
- Ancestry verified: `519626d` ∈ HEAD; `c21de8b` ∈ HEAD; `519626d` is ancestor of `c21de8b`.
- Working tree: clean except two **user-owned untracked files** which must never be modified/staged/committed/deleted:
  `research-wiki/deep-dive-prompt.txt`, `research-wiki/run-deep-dive.ps1`.
- `c21de8b` touched only research-wiki files (log, elgs-experiment-plan, elgs-implementation-readiness-errata [new], elgs-method, elgs-review-history, elgs-v8-formal-spec). No code exists for EL-GS yet [to be re-verified by precursor-code sweep].
- Top-level repo surface relevant to this plan: `main.py`, `train.sh`, `runit.sh`, `arguments/`, `configs/`, `scene/`, `gaussian_renderer/`, `utils/`, `tests/`, `infra/`, `scripts/`, `Dockerfile.apollo-h100`, `Dockerfile.apollo-v100`, `det_cfg_apollo_dgx.yaml`, `det_cfg_apollo_hopper.yaml`, vendored CUDA extensions `diff-gaussian-rasterization/`, `simple-knn/`, `pointops2/`.

## 2. Canonical-semantics implementation summary

Authority chain (all at `c21de8b`): spec rev 4 (`operations/elgs-v8-formal-spec.md`) + errata page, AND — via spec §1's explicit delegation ("Rendering, pose/motion, gauge, pruning, caps: as [[operations/elgs-method]]") plus the method page's "substrate = Loop-1 LGS, family form" — the LGS substrate semantics in `operations/lgs-method.md`. These are settled. The implementation propagates them; it does not re-decide them.

### 2.1 Interval/latch state (spec §1 — single source of truth)
- Canonical family interval state: `(K, latch_pre, latch_post, a)`; latch bits only on outer endpoints b_1/d_K; all four patterns admissible; interior latches inadmissible; latches toggled only by structural ops; no optimizer moments on latches.
- `dim(a) = 2K + 1 − n_lat`, `n_lat = ℓ_pre + ℓ_post`. Canonical serialized coordinate order: `[slack_pre iff ℓ_pre=0], len_1, gap_1, …, gap_{K−1}, len_K, [slack_post iff ℓ_post=0]`.
- Forward map: `Ω = T + 2w_m`; `Ω_free = Ω − K·floor_len − (K−1)·floor_gap` (never depends on latches); σ = softmax(a); unlatched slack = Ω_free·σ; latched slack ≡ 0 (no coordinate); len/gap = floor + Ω_free·σ. Ω-sum identity holds for every pattern ⇒ d_K ≤ T+w_m identically, equality iff ℓ_post=1.
- Exact-boundary rule: softmax coords strictly positive ⇒ exact contact exists ONLY as a latch bit; zero/epsilon coordinates are unrepresentable and forbidden as numeric substitutes; code must branch on latch bits.
- Inverse map (used by EVERY K-changing op): `σ_i = (value_i − floor_i)/Ω_free`; `a_i = log σ_i − max_j log σ_j` (deterministic gauge, max a_i = 0). Targets at exact floor: inadmissible. Exactly-zero outer-slack target ⇒ set latch instead.
- `K=0`: defined empty program — no latch bits, no vector, renders nothing, prune-pending.
- BIRTH: `(ℓ_pre, ℓ_post) = (0,1)` — terminal latch bit, NOT `slack_post = 0`; `a ∈ R²` via inverse map with targets slack_pre = t_birth + w_m, len_1 = T + w_m − t_birth; admissible iff len_1 > floor_len strictly AND t_birth > −w_m.
- Latch inheritance (rev-4 rules): preserved with preserved outer endpoint; discarded when the outermost episode on that side is deleted; cleared by TRUNCATE-shorten of a latched endpoint; REACTIVATE outside a latched endpoint inadmissible; MERGE takes ℓ_pre from parent owning earliest b_1 and ℓ_post from parent owning latest d_K; interior ops never touch latches.
- Optimizer moments: every structural op rewrites `a` via the inverse map ⇒ simplex-logit moment state RESET + logged. Same for re-anchoring reparameterizations (gauge transport: moments of re-expressed params reset, logged).
- Serialization: persist `(K, ℓ_pre, ℓ_post, a)` in canonical order; loader MUST validate `dim(a) = 2K+1−n_lat`; K=0 persists as empty program with no latch bits. No migration framework (no pre-rev-4 checkpoints exist).

### 2.2 Acceptance estimator (spec §7)
- Canonical: SELF-NORMALIZED importance sampling `R̂ = Σ a_i·ℓ(x_i)/Σ a_i`, samples from mixture `m = λ_u·ν + (1−λ_u)·π_D`, weights `a_i = min{1/λ_u, ν(x_i)/m(x_i)}` — true weight ≤ 1/λ_u always ⇒ clipping provably inactive (retained as formal guard, and asserted in tests).
- Properties to propagate into code/comments/logs/docs: strongly consistent; ratio-estimator bias O(1/n), NOT unbiased in general; bounded-weight. NO code, comment, log string, or doc may describe the sampled estimate as "unbiased" or "exact".
- "Exact" survives only for: closed-form tracker/prior deltas (computed outside the sampled estimate) and the weights themselves.
- Paired common random numbers: identical {x_i} for incumbent and candidate; ΔÊ = paired SNIS difference + exact deltas (includes transaction-ledger increment).
- SE: cluster bootstrap over (camera, frame) units, B=200, SAME resample indices for candidate and incumbent, weights renormalized per replicate (SNIS per replicate); SE = sd of paired replicate differences; SE undefined ⇒ reject.
- Degeneracy: ≤5 clusters ⇒ reject. Accept iff `ΔÊ + k·SE < 0`.
- Sample partitioning: NO hashing — reserved pool pre-partitioned at iteration 0 into indexed grid of slots (round, pass, rank), rank = deterministic conflict-component ordering (min lineage ID); injective by construction; unused slots discarded; refits never see confirmation samples. π_D and λ_u frozen before any confirmation draw.
- Logging: record n, ESS, paired ΔÊ per decision (errata implementation consequence 3).
- Acceptance remains a disclosed preregistered heuristic (spec §9 non-claims).

### 2.3 Other settled semantics to propagate (spec §§2–6, 8)
- Evidence restriction: `ℓ_b(P_f)` sums ONLY over clusters `u ∈ U(f) = bind⁻¹(f)`; bind(u) single-valued via canonical cluster point x_u (lowest-ID seed of connected component; recomputed at MERGE by same rule); unassigned ⇒ cluster inactive.
- Report ownership: track j ← exactly one seed s(j) ← exactly one cluster (connected components of seed-overlap graph at binding) ⇒ streams J(u) pairwise disjoint by construction. Merged clusters: r_u = min, d_u = min, α_u recomputed from merged n_cam + preregistered correlation model.
- Cap operator: per bridge, A_{b,j,t} = C_cap cameras of highest q̃, ties by ascending camera ID (bridge-independent tie-break); fewer than C_cap eligible ⇒ take all (both streams).
- Tempered bridge aggregation: Φ = −τ_B·log[Σ_b e^{ℓ_b/τ_B} / Σ_b e^{ℓ_b^cens/τ_B}]; ONE bridge latent per decision; disclosed engineered energy, NOT marginal likelihood.
- PROP 1 (censored zero): q̃=0 across all bridges on a segment ⇒ segment contributes exactly zero to Φ under any structure — must hold exactly in code (unit-tested).
- PROP 2 (clone/split invariance): AT FIXED ROUND SNAPSHOT only; refresh-boundary drift NOT claimed — audited, not asserted.
- PROP 3 (merge accounting): disjoint streams before/after re-formation.
- Observability: q = clip(Σ ω·1_frustum·T_{−(f;j)}·κ_res, 0, 1); deterministic nonnegative sigma-point weights summing to one (no negative-weight UT); q̃ = q·d_u ∈ [0,1]; q is a ROUND SNAPSHOT (E^{(r)} written against q^{(r)}, A^{(r)}); refresh only at round boundaries.
- Frozen-functional snapshot: at round boundary, environment (θ^{(r)}, other families' committed programs) frozen; candidate hypotheses (incl. new births/fissions/merges) evaluated with candidate family counterfactually inserted; MERGE candidates use union anchor set immediately for windows (post-commit re-derivation is bookkeeping only).
- Windows: maximal spans between consecutive anchor intervals; <2 anchors ⇒ NO evidence windows (photometric-only, reported in coverage).
- Energy: E^{(r)} = L_render + β·ΣΦ + κ·K_f + ψ_dur + ψ_gap + C(H); C(H) = χ·N_returnbirth + μ·N_merge is a transaction LEDGER over event history (never refunded); κ per episode is a state term (auto-refunded on deletion). Acceptance compares E^{(r)} including the candidate's transaction increment.
- Structural ops closed set: FISSION, TRUNCATE-shorten/-delete, REACTIVATE, BIRTH, MERGE, PRUNE — each with the rev-4 transition deltas, admissibility preconditions, latch inheritance, and inverse-map child-state writes (spec §5 table). Return-family predicate deterministic (radius r_site, birth time in W; ties → earliest birth then lowest family ID); REACTIVATE/MERGE mutually exclusive.
- MERGE survivor identity: merged family retains OLDER parent's family ID, birth time, birth site, lineage key; younger ID retired, never reused; cluster bindings redirected to surviving ID; all predicates operate on surviving IDs. Radiance from older family (disclosed convention + symmetric ablation).
- ε-bound: two-sided M(ε) with density floors p_floor/p_cap and ESS/α_u dependence (spec §6); reported with empirical power curve — derived constants, not tunables.
- Decision classification (spec §8): fixed-path decomposition labels DATA-SUPPORTED / PRIOR-PIVOTAL / INTERACTION-SUPPORTED / EQUIVALENCE-CLASS with the precedence exactly as specified; every printed label carries the qualifier "(fixed-path decision decomposition, not statistical support)"; ITT logs every screened candidate.
- Rollback: rejected candidates ⇒ ALL refit state discarded, incumbent snapshot restored (bitwise — unit-tested).
- Schedule (from method page): warm-up→2.5k; seeding 2.5k; binding audit 2.8k (see freeze semantics in §2.4a below); structural rounds {3k, 4.5k, 6k} (round 3 truncation/fission only); refit→10k; post-refit classification pass at 10k (spec §8: labels computed on each decision's confirmation samples AT POST-REFIT PARAMETERS — per-decision confirmation-sample references retained in state for this pass).
- Held-out cameras NEVER enter tracking or observability.
- Differentiable-term boundary (binding; prevents renderer self-exoneration by construction): the gradient-based refit optimizes `L_render` + the ψ_dur/ψ_gap barriers (which DO reach the interval logits `a`) w.r.t. θ and a. `Φ`, `q`, `q̃`, cap sets, and bridges are round-snapshot constants — NO gradient flows from them into θ or `a` (bridges are stop-gradient evidence-independent constructors; q is a frozen functional). κ/χ/μ enter only discrete acceptance comparisons. Enforced by a gradient-isolation test (§8.3).
- Binding-freeze semantics (§2.4a): the 2.8k audit freezes the AUDITED binding set. Permitted post-audit mutations are exactly: MERGE redirection of both parents' bindings to the surviving ID with cluster re-formation (spec §2), and late-birth lineage-local seeding under the same delay/audit protocol (method page). Anything else raises. Both permitted paths are logged and unit-tested.

### 2.4 Substrate semantics (LGS family form — binding via the authority chain; `operations/lgs-method.md`)
- Presence function: π_j(t) = S((t−b_j)/w)·S((d_j−t)/w), S = clamped cubic smoothstep (0 below 0, 3u²−2u³ on [0,1], 1 above 1) — exact zero in gaps, exact plateau inside, latched (no mid-episode dip expressible). This is the only differentiable path from L_render to interval endpoints.
- Rendered opacity: σ(o_i)·π_winner(t)·routing; winner by interval lookup, UNIQUE (at most one active episode per family at any t); one row per lineage per timestamp.
- Routing pinning: families with K>1 or any gap are pinned dynamic (frozen route logit, logged); static conversion only for K=1 near-full-span families. Without this, the free `_route_logit` sigmoid (gaussian_model.py:548, multiplied into opacity at renderer :210) is a second unconstrained presence gate that could substitute for episode structure — pinning is enforced in code + config test.
- Motion origins τ: fixed at episode creation, never changed. FISSION children inherit the parent's τ (exact coefficient copy); REACTIVATE episodes get fresh τ; clones copy per episode.
- Pose gauge: first episode is reference; on its removal, exact render-preserving re-anchoring with full gauge transport; transformed moments RESET (logged); render-invariance unit-tested.
- Clone/split of multi-episode families: volume-preserving opacity split uniformly across ALL episodes; per-episode mean perturbation in each episode's own pose frame; content child fresh-init with ZERO moments; pose/motion copied WITH moments; episode-local clones prohibited; atomic transactions. NOTE: `cat_tensors_to_optimizer` (gaussian_model.py:1577) zero-pads ALL new-row moments — the copy-WITH-moments rule for pose/motion requires an explicit post-append moment write for those groups (a identified change to representation-critical code, owned per §13).
- Dual caps + accounting: peak rendered rows ≤ 600k AND total stored trainable scalars ≤ baseline budget; full ledger of episode metadata/accumulators/moments/staging overheads; micro-render/search-cost accounting (cumulative candidate renders, accepted/tried, rasterizer + topology-management time, peak memory, end-to-end GPU-h). K-overflow = reject + log (reported representational-capacity failure metric).
- w FIXED (2 frame intervals) for the whole run; disclosed event resolution: minimum episode and gap span 4 frame intervals.

## 3. Blocker register and underdetermined-decision register

**Formal blockers: NONE.** The rev-4 spec §1/§5/§7 statements are mutually consistent on inspection (BIRTH's (0,1) latch pattern matches the §1 inheritance rule; inverse-map admissibility matches the BIRTH iff; the SNIS description matches the §7 formula; PROP 1–3 and the ε-bound do not depend on either repaired definition, independently confirmed by the errata's fresh review). No contradiction unresolved by `c21de8b` was found. No new mathematical blocker is registered.

**Underdetermined implementation decisions (spec is silent, not contradictory; each must be resolved, disclosed, and preregistered before its consuming module merges — these are design obligations, not spec defects):**
| Decision | Candidate resolutions | Owner | Preregistered in | Oracle |
|---|---|---|---|---|
| Point-truncated transmittance for q (rasterizer exposes only full-ray 1−T; no point-resolved T utility exists in repo) | (a) torch front-set compositing per sigma point (spatially pruned Gaussian subsets, depth-ordered; exact w.r.t. the compositing model); (b) per-(camera,frame) front-subset alpha renders with depth thresholding (approximate, quantization disclosed); (c) small CUDA addition to the vendored rasterizer returning depth-truncated T (exact, higher effort) | **Fable owns the choice** (no separate research/proof stage); selected before module 6 (observability) merges, on exactly: the closed-form small-Gaussian fixture, accuracy, query-source exclusion, gradient isolation, runtime and memory cost. Default: strongest verified non-CUDA option; the CUDA option only if the non-CUDA options fail the criteria — and then as a narrow reviewed change with reference-parity tests against the torch implementation | `prereg_observability_v1.json` + the wiki M0 record (chosen implementation, selection evidence, rejected alternatives, any approximation limitation — recorded durably; compute estimate entered into M0 ceiling) | closed-form T at a point for a 2–3-Gaussian fixture (§8.3) |
| Acceptance sample unit + DSSIM handling (per-pixel ℓ undefined for windowed DSSIM) | tile-cropped (camera, frame) units (LGS precedent: ≤16 tile-cropped pairs) with ν-density correction for tile weighting | Fable | `prereg_acceptance_v1.json` | estimator ν-mean converges to full-render loss on fixture (§8.4) |
| Slot-grid sizing/exhaustion (rank count unknown at iteration 0) | capacity = preregistered candidate-cap × passes × rounds upper bound; exhaustion ⇒ candidate rejected + logged (deterministic) | Fable | `prereg_acceptance_v1.json` | slot audit test (§8.4) |
| Differentiable-term boundary | RESOLVED in §2.3 (refit = L_render + ψ barriers; Φ/q snapshot-frozen) | — | `prereg_structural_v1.json` | gradient-isolation test (§8.3) |
| Initial latch pattern for spanning init families | RESOLVED: (1,1), dim(a) = 2·1+1−2 = 1 (len_1 only) | — | `prereg_structural_v1.json` | init test (§8.5) |

**Infrastructure gaps (actionable inside M0, not blockers):** listed in §10.1 — no Determined submission path, no commit-isolated code materialization, no immutable config record on the training path, no Determined-aware ledger, no DiVa-360 support, mutable image tags, Apollo migration unrecorded in wiki.

**Items requiring user input or environment access at execution time (labeled, not planning blockers):**
- Determined master address + CLI login are untracked (operator environment). Preflight `det whoami` will surface this at M0 smoke time.
- UNVERIFIED assumption: containers on Apollo automount `/apollo` project storage (implied by historical `work_dir` in the det configs; never recorded). Verified by one cheap `det cmd run ls` probe before any GPU smoke.
- DiVa-360 access path (official download; MIT license per loop2 sweep) — inventory on Apollo first; acquisition only if absent; if the official source requires credentials/registration, stop and report to user rather than working around.

## 4. Repository and execution map (evidence: codebase mapper; spot-verified)

### 4.1 Current execution path (file:symbol)
- Entry: `main.py` (~2068 lines); `training()` at main.py:1090–1663; `validation()` (`--val`) at main.py:1031; parser+YAML merge at main.py:1981–2027 (OmegaConf load, recursive setattr; **unknown YAML key ⇒ AssertionError** — every new EL-GS key must be added to `arguments/__init__.py`); iteration guard `DEFAULT_MAX_TRAIN_ITERATIONS = 6000` (main.py:64, `enforce_train_iteration_guard` main.py:1019; override via `ADAGS_MAX_ITERATIONS`/`ADAGS_ALLOW_LONG_RUNS`) — **EL-GS 10k schedule needs this override recorded in config/env**.
- Config surface: `arguments/__init__.py` — `ModelParams`(:47) / `PipelineParams`(:72) / `OptimizationParams`(:84–222, all defaults incl. `motion_lora_rank=8`, lifecycle block :187–221).
- State owner: `scene/gaussian_model.py::GaussianModel` (:78, 2474 lines) — owns all `nn.Parameter`s and its own Adam (`training_setup` :1349–1440, param groups per tensor, `torch.optim.Adam(l, lr=0.0, eps=1e-15)` :1436). Per-primitive tensors incl. `_xyz, _features_dc/rest, _opacity, _scaling, _rotation, _t, _scaling_t, _route_logit, _motion_lora_coeff (N,rank), _motion_lora_basis (rank,anchors,3, global)`, scaffold tensors, densify accumulators, and the **capacity slot-identity ledger** `_capacity_stable_ids/_generation/_last_reassigned` (:196–200, `_capture/_restore_capacity_state` :1215/:1226, schema `phase9-capacity-state-v1`).
- Motion is Python-side pre-rasterization: `render()` (`gaussian_renderer/__init__.py:158`) calls `pc.get_dynamic_xyz(t)` (:200) → `get_lora_motion_offset` (gaussian_model.py:636, centered basis interpolation `_sample_lora_basis` :612) + scaffold offset (:683). Opacity path: `opacity = get_opacity × dynamic_probability` (:210) × `get_marginal_t(t)` (:228/239) × hide-reveal/visibility-event gates (:241–254) → rasterizer. **This is the EL-GS presence insertion point; no CUDA changes needed for presence/motion rendering** (matches LGS substrate note).
- Renderer outputs: `render, viewspace_points, visibility_filter, radii, depth, alpha(=1−T), flow, …` — transmittance available as FULL-RAY alpha only; raw T discarded; no point-resolved transmittance utility exists anywhere in the repo (`gaussian_renderer/diff_gaussian_rasterization.py:165–174`; grep verified). ⇒ q's point-truncated T is an underdetermined-decision-register item (§3) and MAY require torch compositing or a small vendored-rasterizer addition — the "reuse unchanged" classification below applies to the training render path, not to the q probe.
- Optimizer mutators (the transaction toolkit): `replace_tensor_to_optimizer` (:1461, zeroes moments), `_prune_optimizer` (:1477, skips global groups), `prune_points` (:1514), `cat_tensors_to_optimizer` (:1577, zero-padded moments), `densification_postfix` (:1653); **`depth_visibility/capacity.py`**: `CapacityBank` (:15), `_reset_optimizer_rows` (:62, in-place per-row moment reset preserving Parameter identity), `apply_point_neutral_transaction` (:77, asserts identity+budget invariance), donor selection (:197).
- Densify/prune: `densify_and_prune` (:2170–2334), clone (:2018), split (:1864), selection cap `_limit_densify_selection` (:1635) → `densify_until_num_points`; prune criteria :2310–2314; `reset_opacity` (:1451; never fires in 6k lanes since `opacity_reset_interval:30000`).
- Checkpoints: `capture()` (:217–307) → tuple (30 elements for dim-4; element 30 = `routing_motion_params` dict carrying motion/scaffold/route + `capacity_state` + `lifecycle_state`); `restore()` (:309–472) version-sniffs len==30; writes via `Scene.save` (`scene/__init__.py:103`, `chkpnt{iter}.pth`) + best-ckpt (main.py:1527); `save_ply` is a lossy viewer export. **EL-GS state extends this nested-dict convention with its own schema id.**
- Eval: in-training `training_report` (main.py:1681–1969; PSNR/SSIM + dynamic-mask metrics via `MotionPriorCache.get_dynamic_mask`, no LPIPS); `--val` path → `utils/mesh_utils.py::GaussianExtractor` (PSNR/SSIM/LPIPS-alex; writes `stats/validation.json`). Per-run artifacts: `summary.json` (`adags-run-summary-v1`), `capacity-ledger.json`, `lifecycle-ledger.jsonl`, tensorboard events.
- Lifecycle precedent (CSVL-VPL v2, to remain intact as a lane family): `scene/lifecycle.py::LifecycleManager` (protection/exposure/birth via point-neutral transactions; `state_dict`/`load_state_dict` :976/:996), wired at main.py:898/1442/1604; evidence via `depth_visibility/evidence_runtime.py` (frozen consensus dir + `time_shift` control — the pattern for EL-GS frozen tracks).
- Tests: `tests/` = 21 unittest files, ~5.1k lines, CPU-only by convention; CUDA-import avoidance via `importlib` file-path loading (`tests/test_lifecycle.py:20–30`) and `ast` parsing of `main.py` (`tests/test_lane_configs.py:11–17`). No pytest config. Invocation `python -m unittest discover tests`.
- Vendored CUDA extensions (no submodules): `diff-gaussian-rasterization` (JIT-fallback at import), `simple-knn`, `pointops2` — all baked into Apollo images.
- Known broken/latent paths to avoid (not in scope to fix): `scene/__init__.py:90/:96` call nonexistent `create_from_3dgs`/`load_ply`; debug-branch NameError in `diff_gaussian_rasterization.py:159–170`; `environment.yml` stale (py3.7/torch1.12 vs images' torch2.0/cu118).

### 4.2 Reuse classification for EL-GS
- Reuse unchanged: rasterizer + autograd binding; SH/appearance tensors; clone/split/prune core; optimizer-mutator toolkit; `CapacityBank`/`apply_point_neutral_transaction` (+ its per-row moment reset); slot-identity ledger pattern; checkpoint tuple + nested state-dict convention; `MotionPriorCache` (for masks in diagnostics); tensorboard/summary.json writers; test conventions; `validate_apollo_runtime.py`; Dockerfiles/images.
- Extend: `render()` opacity path (add EL-GS winner-lookup presence multiplier, replacing `get_marginal_t` when EL-GS active); `GaussianModel` (family-ID column + EL-GS param groups); `capture()/restore()` (add `elgs_state`); `arguments/__init__.py` (elgs_* config block); `training()` (round-boundary hook: seeding→binding audit→structural rounds→refit); `Scene`/dataset path (DiVa-360 via converter); det task configs (into experiment configs).
- Disable/bypass when EL-GS active (config-gated, never deleted): temporal-Gaussian marginal `get_marginal_t` presence semantics; hide-reveal/visibility-event gates; lifecycle manager (evidence-off substrate uses neither); blur curriculum (decide in dev config, disclosed).
- Genuinely new: `elgs/` package (see §6); tracker preprocessing pipeline; DiVa-360 converter; Apollo submission wrapper + preflight config; census analysis tooling.
- Execution-critical but external/untracked: N3V data on Apollo (`/apollo/users/sri/proj_adags/data/n3v` per det cfg env), Apollo runs/logs roots, Docker Hub image tags, Determined master address (untracked), DiVa-360 (absent everywhere).

## 5. Requirement → module → test traceability matrix

Format: canonical requirement → module/interface → state owner → test oracle (§8 ref) → integration check → logged evidence.

| Requirement (spec §) | Module | State owner | Test (§8) | Integration check | Logged evidence |
|---|---|---|---|---|---|
| Interval/latch state, forward/inverse, K=0 (§1) | `elgs/intervals.py` | `ElgsFamilyBank` | 8.1 | substrate lane renders with plateaus/gaps | per-family (K, latches, a) in checkpoint + episode census |
| Latch inheritance + op preconditions + transition deltas (§1,§5) | `elgs/ops.py` | `ElgsFamilyBank` + transaction ctx | 8.2 | ops fire in integration loop w/o invariant violation | transition records in elgs ledger (op, pre/post state, Δstate, Δledger) |
| Optimizer-moment resets on reparam (§1) | `elgs/ops.py` → `capacity._reset_optimizer_rows` pattern | GaussianModel.optimizer | 8.2 | moments zeroed rows verified post-op in smoke | `moment_reset` events in ledger |
| Family/episode ↔ row binding; clone/split inheritance | `elgs/families.py` (extends capacity stable-ID pattern) | GaussianModel `_elgs_family_ids` | 8.2 | densify/prune keeps family map consistent | family census per round |
| Rendering integration: winner-lookup presence π, routing pinning (§2.4) | `gaussian_renderer/__init__.py` hook + `elgs/presence.py` | render-time only | 8.1 (π analytic + winner uniqueness) + 8.2 (routing pin) + 8.6 | exact-zero absence verified in rendered alpha on synthetic scene | render gate stats + routing-pin log |
| Substrate motion: τ origins, inheritance, gauge re-anchoring, clone/split moment rule (§2.4) | `elgs/families.py` + gaussian_model owner edits | ElgsFamilyBank + optimizer | 8.2 (τ, clone/split, re-anchor) | I2 | moment-reset + τ records |
| Dual caps + search-cost accounting (§2.4) | `elgs/transaction_ledger.py` | ledger | 8.3 (cap accounting) | budget assertions in I3 | caps + cost ledger artifact |
| Evidence reports y, heads L1/L0, ownership U(f) (§2) | `elgs/evidence.py` | frozen tracks artifact (read-only) | 8.3 | dry-run on fixture tracks | evidence coverage report |
| Cluster binding, canonical point, merge aggregation (§2, rev-3 A2) | `elgs/clusters.py` | binding table (frozen at 2.8k) | 8.2/8.3 | binding audit log at 2.8k then immutable | binding manifest (hashed) |
| Observability q, sigma points, snapshot (§3) | `elgs/observability.py` (uses rasterizer probes) | round snapshot cache | 8.4 (determinism) + GPU smoke | q∈[0,1] assertions; snapshot hash stable within round | q distributions per round |
| Windows/anchors/bridges (§3) | `elgs/bridges.py` | round snapshot | 8.3 | <2 anchors ⇒ photometric-only path exercised | window/bridge census |
| Energy: Φ tempered aggregation, PROP 1–3 (§4) | `elgs/energy.py` | pure functions | 8.3 | β=0 arm identical-search check wired | per-decision Φ, ℓ_b vectors |
| Transaction ledger C(H) (§4) | `elgs/transaction_ledger.py` | append-only H | 8.2/8.4 | ledger persisted+restored across checkpoint | ledger JSONL |
| Candidate generation/search (§5 + method) | `elgs/search.py` (semi-Markov engines, conflict graphs, Gauss-Seidel, priority queue) | search state (per round) | 8.2 (preconditions) + ITT log test | round produces ≤cap candidates deterministically under seed | ITT candidate log |
| SNIS acceptance + CRN + bootstrap + degeneracy (§7) | `elgs/acceptance.py` | slot grid (iter-0 partition) | 8.4 | paired eval on GPU smoke; slot audit | n, ESS, ΔÊ, SE, verdict per decision |
| Rollback (§7) | `elgs/transactions.py` (snapshot/restore ctx) | affected rows+moments+RNG | 8.4 (bitwise) | reject path exercised in integration test | rollback events |
| Decision classification (§8) | `elgs/classification.py` | decision records | 8.5 | labels printed with mandatory qualifier | classification table artifact |
| Serialization/checkpoint (§1) | `elgs/state_io.py` + `capture()/restore()` extension | GaussianModel | 8.5 | mid-run resume equivalence (integration) | schema id in checkpoint |
| Metrics/experiment logging (plan B1/B4) | `elgs/reporting.py` + existing summary.json | run dir | 8.5 (schema) | artifacts present after smoke | census/ITT/risk-coverage JSON |

## 6. Module and interface design

New package `elgs/` — pure-Python/Torch, importable WITHOUT CUDA extensions (hard rule so the CPU test suite runs on any node; follows `tests/` convention). GPU-only pieces (rasterizer probes for q; micro-render/confirmation evaluation) live behind a narrow `RenderProbe` interface injected at runtime, so every mathematical module is CPU-testable with a fake probe.

Modules (responsibilities / inputs-outputs / mutation boundary / invariants / failure behavior):
1. `elgs/intervals.py` — canonical interval state `(K, ℓ_pre, ℓ_post, a)`; forward/inverse maps; Ω-sum identity; admissibility predicates. Pure functions + a small `IntervalState` value type. Storage: padded `(F, 9)` logit tensor + per-family latch bits and K, with STRICT active-coordinate masking (softmax over exactly the 2K+1−n_lat active coords; padded coords never read); serialization emits canonical variable-length vectors only. Mutates nothing outside its returned values. Fails loudly on any inadmissible target (raise, never clamp).
2. `elgs/families.py` — `ElgsFamilyBank`: family registry (IDs, birth time/site, lineage keys, clone-descendant row sets), family→row mapping as a per-row `_elgs_family_ids` int64 column on `GaussianModel` (same pattern as `_capacity_stable_ids`; survives clone/split/prune via the same postfix/prune hooks); episode-local pose/motion parameters at immutable per-episode origins (reuses the existing rank-8 LoRA basis machinery: per-episode origin τ_j replaces per-Gaussian `t` in `_sample_lora_basis`-style interpolation). Owns family invariants (≤4 episodes; disjoint; ID retirement never reused).
3. `elgs/presence.py` — winner lookup + plateau/edge-band computation `z_f(t)`, X_f; π multiplier for render; exact zero in gaps.
4. `elgs/clusters.py` — seeds, seed-overlap connected components, canonical cluster point x_u (lowest-ID rule), bind(u), U(f), merge re-formation + (r,d,α) aggregation. Reference `scipy`-free union-find; binding immutable after the 2.8k audit (enforced by a frozen flag; violation raises).
5. `elgs/evidence.py` — report parsing (miss ⊔ [v]×D_img), heads p_vis/p_cens/p_out with floors/caps, L1/L0, censoring equality; per-(u,j,c,t) log-likelihood evaluation. Pure; fitted head params arrive as frozen config.
6. `elgs/observability.py` — q per (bridge, track, camera, frame): sigma points (deterministic nonneg weights), in-frustum, family-present query-source-excluded transmittance via `RenderProbe`, κ_res clipped area ratio, d_u, q̃ = q·d; round-snapshot cache keyed (round, family/candidate hash); clip to [0,1].
7. `elgs/bridges.py` — anchor intervals (capped visible-report floor), windows W(f), bridge family constructors (evidence-independent, stop-gradient), per-decision bridge latent.
8. `elgs/energy.py` — ℓ_b, ℓ_b^cens with cap operator A_{b,j,t} (C_cap, camera-ID ties), α_u, ESS tempering; Φ tempered aggregation; ψ_dur/ψ_gap/κ state terms; E^{(r)} assembly incl. ledger increment. Pure functions over snapshot inputs.
9. `elgs/transaction_ledger.py` (named to avoid collision with `depth_visibility/ledger.py`, which is provenance checking) — event history H; χ/μ transaction charges; append-only JSONL + in-memory state; persisted in checkpoint. Also owns the search-cost/micro-render accounting ledger (§2.4: candidate renders, accepted/tried, rasterizer + topology time, peak memory, GPU-h) and dual-cap accounting.
10. `elgs/ops.py` — the closed op set (FISSION/TRUNCATE×2/REACTIVATE/BIRTH/MERGE/PRUNE): admissibility, child-state via §1 inverse map, latch inheritance, transition deltas, return-family predicate, REACTIVATE/MERGE exclusivity, MERGE survivor rules, gauge re-anchoring equations. Emits a TransactionPlan (pure); never mutates directly.
11. `elgs/transactions.py` — applies/rolls back TransactionPlans against GaussianModel+optimizer: row add/remove via `densification_postfix`/`prune_points` patterns, in-place writes via `apply_point_neutral_transaction`-style guarded ops, per-row + per-family moment resets, bitwise snapshot/restore (params, moments, ledger, RNG, family registry) for rejected candidates. Asserts identity+budget invariants (reuses `CapacityBank` assertions).
12. `elgs/search.py` — screening accumulators, per-lineage constrained semi-Markov interval engines, conflict graphs on current∪bridge footprints, Gauss-Seidel passes, priority queue, per-component confirmation ordering (min lineage ID), candidate caps. Deterministic under seed.
13. `elgs/acceptance.py` — SNIS estimator (weights, provably-inactive clip guard asserted), CRN sample management, slot grid (iteration-0 pre-partition, (round,pass,rank) injective), paired cluster bootstrap (B=200, per-replicate renormalization), degeneracy rejection, accept rule with ledger increment. No "unbiased"/"exact" language anywhere (grep-enforced by a test).
14. `elgs/classification.py` — post-commit fixed-path decision decomposition, flags, precedence, mandatory qualifier string, ITT records, risk-coverage assembly.
15. `elgs/state_io.py` — `elgs_state` schema (versioned), save/load with dimension+pattern validation, explicit incompatibility rejection; integrates into `capture()`/`restore()` element-30 dict alongside `capacity_state`/`lifecycle_state`.
16. `elgs/reporting.py` — census, coverage, ITT, per-round diagnostics into run dir (JSON, schema-tagged). MUST write through the existing `depth_visibility/artifacts.py` (`atomic_write_json`, `atomic_write_json_immutable`, `build_inventory`, `write_terminal_last`) and `canonical.py` (`canonical_json_bytes`, `sha256_file`) — no new artifact-writing machinery.
16b. `elgs/probe.py` — the `RenderProbe` interface + its GPU implementation for q's point-truncated query-source-excluded transmittance, per the §3 register decision (torch front-set compositing or vendored-rasterizer addition — resolved and preregistered before module 6 merges); fake CPU probe for tests lives here too.
17. Trainer integration (in `main.py`): `setup_elgs(...)` mirroring `setup_lifecycle`; round-boundary hook (snapshot refresh → search → confirmation → commit/rollback) at iterations {3k,4.5k,6k}; seeding at 2.5k; binding audit at 2.8k; refit→10k (needs `ADAGS_MAX_ITERATIONS` override, recorded in config); training sampler EXCLUDES the reserved confirmation (camera, frame) units during refit (spec §7 "refits never see confirmation samples" — made real in the sampler, not just in slot indices); post-refit classification pass at 10k.
18. Renderer integration (in `gaussian_renderer/__init__.py`): EL-GS presence multiplier replacing `get_marginal_t` when `elgs_enable`; routing pinning enforcement per §2.4.
19. Config (in `arguments/__init__.py`): `elgs_*` block; static admission test in `tests/` mirroring `test_lane_configs.py`.
20. Tracker preprocessing (offline, new `scripts/build_elgs_tracks.py` + `elgs/tracks_schema.py`): CoTracker3-class per-camera queries from 3D surface seeds (≥2-camera visibility), robust consensus triangulation, identity by common seed, frozen artifact dir + manifest (sha256; the `evidence_runtime` frozen-dir pattern), shift/shuffle control generation. GPU via Determined; ledgered separately per plan.
21. DiVa-360 support (new `scripts/diva360_to_blender.py` converter → existing `transforms_train.json` Blender-branch loader; no `scene/` reader changes unless converter proves impossible — decision recorded after inventory).

## 7. Calibration and preregistration

### 7.1 Constant classification
Category 1 — structural semantics (fixed by spec before implementation, not tunable):
- K_max = 4; w (edge half-width, 2 frame intervals per LGS substrate); w_m margin; floor_len = 2w + δ_len; floor_gap = 2w + δ_gap; softmax gauge (max a_i = 0); latch-inheritance table; canonical coordinate order; C(H) ledger form; SNIS estimator form; B = 200 bootstrap replicates; degeneracy floor (≤5 clusters ⇒ reject); one-bridge-latent-per-decision; camera-ID tie-break; lowest-ID canonical cluster point; older-parent MERGE survivor rule; schedule anchor iterations {2.5k seeding, 2.8k audit, 3k/4.5k/6k rounds, 10k refit}; no-hash slot grid.

Category 2 — evidence parameters (calibrated on declared calibration data, frozen before any evidence-bearing evaluation):
- Evidence heads g_v, g_pos, h_c, π_m^v, π_m^c, p_out parameters (fitted only on calibration scenes, frozen — spec §2); r_u reliability diagnostics mapping; d_u detectability model; α_u correlated-camera model; bridge-family constructor parameters; anchor-interval report-count floor; tracker (CoTracker3-class) inference settings.

Category 3 — development hyperparameters (tuned only on declared dev scenes, frozen before held-out evaluation):
- β, κ, χ, μ, ψ_dur/ψ_gap barrier constants, τ_B, τ (ESS tempering α_ess), k (acceptance SE multiplier), λ_u, C_cap, ε for the reported bound, r_site, screening/candidate-count caps, confirmation sample counts n, learning rates for a-logits.
- (δ_len, δ_gap, δ_tol are Category 1 ONLY — they define op admissibility and the PRUNE at-floor predicate; values fixed once before implementation in `prereg_structural_v1.json`, never tuned on data.)

### 7.2 Category-B preregistration artifacts (committed with the implementation, before the first evidence-bearing job)
Format: JSON under `configs/elgs/` (matches the frozen-JSON convention of `configs/depth_visibility/`), schema-tagged, hash-recorded in run manifests, validated by a static test (mirroring `tests/test_lane_configs.py`). Files:
1. `configs/elgs/prereg_structural_v1.json` — category-1 constants (K_max, w, w_m, δ_len/δ_gap/δ_tol, floors formulas, schedule anchors, B=200, degeneracy floor, C_cap tie rule, survivor conventions, canonical coordinate order id, slot-grid geometry).
2. `configs/elgs/prereg_latch_transition_table_v1.json` — the full op × latch-pattern transition table (pre/post state, parameter init rule, evidence ownership, prior delta, latch inheritance). AUTHORSHIP RULE (breaks self-reference): authored by a FRESH-CONTEXT worker given ONLY spec §1/§5 (no plan, no code); independently reviewed against the spec by a second fresh-context reviewer; reviewer sign-off is a HARD PREDECESSOR of the ops module (§13 row 4) merging; the table's sha256 is asserted inside the §8.2 tests.
3. `configs/elgs/prereg_acceptance_v1.json` — SNIS mechanics (λ_u range + freeze rule with a freeze-before-first-confirmation-draw assertion, π_D construction, k, n per decision, SAMPLE UNIT = tile-cropped (camera, frame) units with DSSIM window handling and ν-density correction (§3 register), bootstrap unit definition, CRN seed derivation, slot-grid indexing incl. sizing/exhaustion rule, one-confirmation-per-component-per-pass).
4. `configs/elgs/prereg_evidence_heads_v1.json` — head families, parameter ranges, calibration-scene list (authorized data), estimation criterion, freeze point, prohibited data (dev/locked/held-out scenes, any post-seeding model state), derived p_floor/p_cap formulas.
5. `configs/elgs/prereg_observability_v1.json` — sigma-point grid scheme (deterministic nonneg weights), κ_res definition, d_u model, α_u correlated-camera model, q-source/update matrix arm definitions.
6. `configs/elgs/prereg_m1_census_v1.json` — M1 census cells, floors, scene/split lists, held-out camera exclusions, gate rule, failure/retry policy (§11.2). FLOOR DERIVATION RULE: floors are DERIVED from the M0 power analysis (which events at which counts give the B3 matrix adequate power), derivation recorded inside the file; primary gated statistics are MODEL-FREE (computed from masks + calibration + tracks, not from any trained model — see §11.2); a fresh-context reviewer signs the floors BEFORE any DiVa-360 statistic is computed.
   AUDITED TRUE ABSENCE — DECIDED NOW, before any DiVa-360 statistic is inspected: **DIAGNOSTIC-ONLY for M1**. It cannot affect the M1 gate or its PASS/FAIL decision under any circumstance. Stated claim limitation (recorded in the M1 result page): M1 therefore CANNOT support a claim-grade estimate of true-event absence or false-positive prevalence — that evidence class is deferred to later blocks. The diagnostic protocol (frozen here for interpretability, still blinded): model-free candidate events from frozen tracks + fg/bg masks under a frozen eligibility predicate; capped stratified sample (≤60, sequence × duration tercile, seeded); two independent fresh-context audits per candidate from the 3 maximal-angular-separation training cameras at a fixed frame stride, blinded to census outputs and each other; confirmed only on agreement; audit records (frames shown, verdicts, agreement rate) preserved as artifacts.
7. `configs/elgs/prereg_metrics_v1.json` — metric spec freeze incl. object-to-lineage mapping protocol, the PERFORMED power analysis (an M0 exit requirement, not a placeholder — it feeds the floor derivation above), classification data-term floors, dose-matched shift/shuffle protocol (for later blocks; frozen now to avoid drift).
Each calibratable entry carries: allowed range, authorized data, estimation/selection criterion, sensitivity-analysis plan, freeze point, prohibited data, and which gate its violation invalidates. First guesses are NOT frozen: category-2 values are placeholders marked `"status":"unfrozen"` until the calibration procedure (declared in the same file) runs on calibration data; the M1 gate refuses to run evidence-bearing cells while any required entry is unfrozen (checked by the submission wrapper).
Validation: the seven prereg schemas are registered in the EXISTING `depth_visibility/schema.py` machinery (`SCHEMA_RULES`/`validate_payload`/`load_json_object`, which already rejects NaN/Infinity constants) rather than a new validator; a thin static test invokes it (plus the lane-config-style admission for YAML configs).

## 8. M0 test plan (CPU-exact unless stated; oracles independent of implementation)

Derived from spec §§1–8 + errata "Required M0 (B0) tests" + experiment-plan B0 + substrate (§2.4). Each test lists its oracle type.
ORACLE-INDEPENDENCE RULE (applies throughout): every "separate reference implementation" oracle is authored by a fresh-context worker from the governing spec section ALONE (no access to `elgs/` code or the implementer's rationale), frozen as hash-recorded test data before the module under test merges. Truth-table and fixture rows quote the governing spec sentence inline so reviewers can check row-by-row.

### 8.1 Interval/latch state and presence
| Test | Oracle |
|---|---|
| Forward map on LITERAL hand-computed fixtures: fixed (K, pattern, a) → exact endpoint values, one per latch pattern (catches shared misreadings a round-trip cannot) | analytic hand-computed values (in-test literals) |
| Inverse-map gauge: `max(a) == 0` exactly; serialized coordinate order matches the canonical order LITERALLY (element-by-element on a fixture) | analytic |
| Forward∘inverse round-trip, all 4 latch patterns × K∈{1..4}, randomized admissible states (seeded sweep) | metamorphic identity (‖x−x′‖ < float tol) |
| Presence function π_j(t): smoothstep values at hand-picked t (edge, band interior, plateau, gap); exact 0 in gaps; exact plateau value; latched shape (no mid-episode dip) | analytic (closed-form smoothstep) |
| Winner lookup: uniqueness (≤1 active episode per family at any t); boundary behavior; K=0 renders nothing | enumerated fixtures |
| dim(a) = 2K+1−n_lat enforced for all patterns | enumerated expected dims (analytic) |
| Ω-sum identity per pattern (slack+len+gap sums to Ω exactly) | analytic expected value |
| d_K ≤ T+w_m identically under random optimizer steps; equality iff ℓ_post=1 | invariance relation under perturbation |
| Strict positivity of unlatched slacks after arbitrary gradient steps | invariance relation |
| K=0 empty program: renders nothing, no latch bits, no vector | enumerated state |
| Rejection: invalid dims, interior latches, exact-floor targets, zero-slack numeric substitutes, epsilon-thresholded latches | deliberately invalid cases |
| Serialization round-trip + loader dimension validation (accept valid, reject corrupted dims/patterns) | deterministic fixture + invalid cases |

### 8.2 Structural operations
| Test | Oracle |
|---|---|
| Latch inheritance across EVERY op (BIRTH, FISSION, TRUNCATE-shorten, TRUNCATE-delete terminal/interior, REACTIVATE, MERGE, PRUNE) incl. discard/clear cases | enumerated transition table (hand-derived from spec §1/§5, written as data before code) |
| BIRTH terminal-latch encoding: (0,1), a∈R², admissibility iff (spec §5) | analytic + invalid cases |
| Operation preconditions (K<4, floors fit Ω_free, return-family predicate, REACTIVATE/MERGE exclusivity, cap-saturated ⇒ inadmissible+logged) | enumerated + invalid cases |
| Transition deltas Δstate/Δledger per op vs hand-computed values on fixtures | analytic expected values |
| MERGE survivor identity (older ID, birth data retained; younger retired, never reused; bindings redirected) | enumerated state transition |
| Ownership redirection + cluster re-formation at MERGE (x_u lowest-ID rule; r/d/α aggregation) | enumerated + separate reference implementation of connected components |
| Optimizer-moment reset on every reparameterizing op + reset logged | deterministic fixture (inspect moment tensors + log record) |
| Gauge re-anchoring: render-preserving reparameterization (exact equations), moments reset | metamorphic (render output identical pre/post re-anchor at fixed params) |
| τ inheritance: FISSION children inherit parent τ + exact coefficient copy; REACTIVATE gets fresh τ; clones copy per episode | enumerated fixtures (§2.4) |
| Clone/split of multi-episode families: volume-preserving opacity split across ALL episodes; content child ZERO moments; pose/motion copied WITH moments; episode-local clones rejected | deterministic fixture inspecting moment tensors (exercises the post-append moment write over `cat_tensors_to_optimizer` zero-padding) |
| PRUNE-episode predicate: len ≤ floor + δ_tol AND micro-render confirmation required; PRUNE-family: episodeless or lifetime-unsupported; dormancy alone never prunes | enumerated + deliberately-ineligible fixtures |
| Routing pinning: K>1 or any-gap family ⇒ route logit frozen + logged; K=1 near-full-span exempt | deterministic fixture |
| Post-audit binding mutations: MERGE redirection and late-birth seeding permitted + logged; any other post-2.8k binding write raises | enumerated + invalid cases |

### 8.3 Evidence and energy
| Test | Oracle |
|---|---|
| Censored-segment exact-zero (PROP 1): q̃=0 segments contribute exactly 0 to Φ across ALL admissible transitions/re-partitionings | analytic (constant-shift cancellation) on synthetic fixtures; exact equality, not tolerance |
| Censoring equality L1(q̃=0) ≡ L0 | analytic |
| Likelihood algebra L1/L0 incl. outlier mass, miss tokens, truncated-Gaussian g_pos normalization over D_img | separate reference implementation (numpy, straight-line transcription of spec §2) |
| Cap operator: C_cap selection, ascending-camera-ID ties, <C_cap ⇒ all, bridge-independent eligibility | enumerated fixtures |
| Tempered aggregation Φ; one bridge latent per decision; F(x+c·1)=F(x)−c property | analytic property + reference impl |
| Clone/split invariance of Φ at fixed snapshot (PROP 2) | metamorphic (Φ identical pre/post clone) |
| Refresh-boundary non-invariance AUDIT path (drift measured and logged, not asserted zero) | deterministic fixture exercising the audit code |
| ε-bound: |Φ shift| ≤ bound on synthetic sweeps with known q̃ ≤ ε | analytic inequality check |
| U(f) restriction: reports of unbound clusters never enter any ℓ_b | deliberately mis-bound fixture (must contribute 0) |
| Window derivation: <2 anchors ⇒ no windows; merged families union-anchor windows at candidate time | enumerated fixtures |
| Segment construction S(P_f,W): maximal runs of {t ∈ W : t ∉ X_f}, constant z per run; edge band X_f STRICT inequality at boundaries; re-partitioning under a transition | enumerated fixtures (spec §4 + §1:194–196) |
| q analytic oracle: point-truncated transmittance on a 2–3-Gaussian closed-form fixture matches the chosen probe method within its declared error model; q ∈ [0,1]; query-source exclusion verified (excluding the source changes T as computed by hand) | analytic closed-form |
| Gradient isolation: backward through E^{(r)} refit loss reaches θ and `a` ONLY via L_render+ψ; zero grad into θ/`a` from Φ, q̃, cap sets, bridges | autograd graph assertion on fixture |
| Dual-cap accounting: peak rendered rows and stored-scalar budget computed correctly on fixtures; K-overflow reject+log | analytic counts |

### 8.4 Snapshot, acceptance, rollback
| Test | Oracle |
|---|---|
| Frozen-functional snapshot: candidate-inserted q deterministic; identical across repeated evaluation; environment not mutated | determinism + state-hash comparison |
| Frozen candidate evaluation: candidate scoring never mutates incumbent state | state-hash comparison (bitwise) |
| Rejected-candidate rollback restores incumbent bitwise (params, optimizer moments, ledger, RNG streams) | state-hash comparison |
| SNIS weight bound: a_i ≤ 1/λ_u always; clipping provably inactive (assert clip never fires on valid inputs) | analytic bound |
| Empirical SNIS bias → 0 with n against closed-form ν-mean on synthetic ℓ | analytic expected value (closed-form integral) |
| CRN determinism: identical {x_i} incumbent/candidate; ΔÊ reproducible bit-exact under fixed seed | determinism check |
| Bootstrap: per-replicate weight renormalization; same resample indices both arms; SE = sd of paired diffs; B=200 | separate reference implementation |
| Degeneracy: ≤5 clusters ⇒ reject; SE undefined ⇒ reject | deliberately degenerate fixtures |
| Slot grid: pre-partitioned at iter 0; (round, pass, rank) injective; exhaustion ⇒ reject+log; refits never see confirmation samples (index disjointness) | enumerated + disjointness assertion |
| Reserved-pool training exclusion: training sampler never draws reserved (camera, frame) units during refit (the §7 guarantee made real, not just index-disjoint) | integration assertion (I3/I4) on sampler draw log |
| λ_u/π_D freeze: any confirmation draw before freeze raises; one confirmation per component per pass | deliberately out-of-order fixture |
| Sample-unit correctness: estimator ν-mean converges to the full-render loss on a small real-render fixture under the preregistered tile unit + DSSIM handling | analytic (full render computed directly) |
| Acceptance rule ΔÊ + k·SE < 0 incl. transaction increment | analytic fixtures |

### 8.5 Decision classification, config, checkpoint
| Test | Oracle |
|---|---|
| Classification precedence (DATA-SUPPORTED / PRIOR-PIVOTAL / INTERACTION-SUPPORTED / EQUIVALENCE-CLASS) over all flag combinations; evaluated at POST-REFIT parameters on the decision's own confirmation samples | enumerated truth table from spec §8 (spec sentences quoted per row) |
| Mandatory qualifier string on every printed label | fixture + string assertion |
| Spanning-init families: latch pattern (1,1), dim(a)=1 | analytic |
| Checkpoint save/load: full EL-GS state round-trip (incl. per-decision confirmation-sample references, slot grid, ledger); incompatible schema rejected explicitly | deterministic fixture + invalid cases |
| Config validation: unknown keys, missing preregistration entries, out-of-range values rejected | deliberately invalid configs |
| Launcher validation: submission refused on execution-relevant dirty content [depends on infra design] | deliberately dirty fixture |

### 8.6 Test conventions, integration tests, and GPU smoke
Conventions (from repo evidence): `unittest`, CPU-only, no CUDA imports — `elgs/` is importable without extensions by design, so tests import it directly (no importlib workaround needed). "Property tests" = seeded randomized sweeps inside unittest (hypothesis is NOT in the repo or the pinned image; no new dependency). Every tolerance explicit per test; exact-equality assertions (PROP 1, rollback, serialization) use `==`/`torch.equal`, never `allclose`. New test files: `tests/test_elgs_intervals.py`, `_ops.py`, `_evidence_energy.py`, `_acceptance.py`, `_transactions.py`, `_classification.py`, `_state_io.py`, `_configs.py` (static admission incl. forbidden-word grep: "unbiased"/"exact estimator" must not appear in `elgs/`), `_integration.py`.

Integration tests (CPU, synthetic micro-scene of a few dozen Gaussians, fake RenderProbe):
- I1: substrate loop — EL-GS families active in a miniature `training()`-shaped loop; winner-lookup presence multiplies opacity; exact-zero absence in gaps (rendered contribution exactly 0 for out-of-episode timestamps).
- I2: densify/prune interop — clone/split/prune preserve family-ID mapping and interval state (family census identical modulo row counts).
- I3: round execution — one full structural round on fixtures: snapshot → candidates → frozen evaluation → accept one/reject one → commit + rollback verified; ledger and ITT records written.
- I4: checkpoint mid-run — save at iteration k, restore, continue: bit-identical family state, ledger, slot-grid bookkeeping; incompatible schema rejected.
- I5: config/launcher — EL-GS dev config admitted by static test; submission wrapper dry-run produces a complete manifest and refuses a dirty execution-relevant tree.

Minimal Apollo GPU smoke (M0, within the ~5 GPU-h B0 ceiling; N3V `cut_roasted_beef`, data already on Apollo — DiVa-360 NOT required for M0):
- S0 (per pool, ~min): `det cmd run` → `validate_apollo_runtime.py --require-gpu --expected-capability {7.0|9.0} --repo . --scene <n3v scene>` + `/apollo` mount probe. Runs on BOTH pools once (validates both images; ~no GPU-h).
- S1 (one pool, Hopper primary): `smoke_apollo.yaml`-derived `configs/elgs/smoke_elgs.yaml` (200–600 iters, tiny caps, schedule compressed like `lane_smoke`) with elgs_enable — asserts: EL-GS code path executed (log markers), one structural round ran, checkpoint written + reloadable, `summary.json` + elgs ledger + census artifacts present, provenance manifest complete, run resumable.
- S2 (same pool): S1 config restored-from-checkpoint continuation — resume equivalence on GPU.
Completion rule: scheduler-terminal + logs inspected + intended commit/config/image/hardware verified from the manifest + artifacts readable. Submission is not completion.

## 9. Checkpoint strategy

- EL-GS state serialized per family: `(K, ℓ_pre, ℓ_post, a)` canonical order + episode-local pose/motion params + tied content + family/cluster identity keys + binding map + transaction ledger H + snapshot bookkeeping + RNG/slot-grid state needed for resume.
- Schema version identifier: single integer/string constant (starts at rev-4 semantics); loader validates schema id AND per-family `dim(a) = 2K+1−n_lat`; any mismatch ⇒ explicit rejection with message (no silent coercion, no migration framework — none needed, no pre-rev-4 checkpoints exist).
- Optimizer state persisted; moment-reset events logged and the log persisted.
- Mechanism: extend `GaussianModel.capture()`/`restore()` (scene/gaussian_model.py:217/:309) — add `elgs_state` dict into the element-30 `routing_motion_params` container, sibling to `capacity_state` (`phase9-capacity-state-v1`) and `lifecycle_state`; schema id `elgs-state-v1`. `restore()` treats missing `elgs_state` as "baseline checkpoint" (see boundary below); present-but-invalid ⇒ explicit rejection.
- `elgs_state` contents: schema id; family registry (IDs incl. retired-ID watermark, birth time/site, lineage keys); per-family `(K, ℓ_pre, ℓ_post, a)` canonical variable-length vectors; per-row family-ID column; episode pose/motion origin (τ) bookkeeping; cluster binding table + audited-freeze flag; transaction ledger H; slot-grid state (partition + consumed slots); per-decision confirmation-sample references (required by the post-refit classification pass, spec §8); round/snapshot bookkeeping; moment-reset log; routing-pin log; RNG stream states needed for CRN reproducibility.
- Baseline-substrate boundary (verified against `restore()` behavior): a baseline ADAGS 30-tuple checkpoint CAN initialize the non-EL-GS substrate — reusable: `_xyz, _features_*, _opacity, _scaling, _rotation, _route_logit, motion (LoRA) tensors, capacity state, optimizer state for those groups`. Newly initialized on EL-GS activation: family registry (initial-cloud rows grouped into K=1 spanning families per the LGS "spanning-then-carve" route, latch pattern (1,1), dim(a)=1), interval states, `_elgs_family_ids` column, EL-GS param groups (a-logits, episode-local params) with fresh moments, empty ledger, unfrozen bindings. NOT reusable across the boundary: `_t`/`_scaling_t` temporal-Gaussian marginal as presence (bypassed under EL-GS; tensors retained untouched for reversibility). This boundary is exercised by test 8.5 (load baseline ckpt → activate EL-GS → validate).

## 10. Apollo execution and provenance

### 10.1 Existing-infrastructure assessment (evidence: infra mapper, verified against files)

What exists (reusable as historical evidence / patterns):
- Runtime images: `Dockerfile.apollo-v100` / `Dockerfile.apollo-h100` — base `determinedai/environments:cuda-11.8-pytorch-2.0-gpu-0.31.1`, `determined==0.38.0` client installed, all python deps pinned, 3 CUDA extensions (`simple-knn`, `pointops2`, `diff-gaussian-rasterization`) baked and build-checked via `scripts/validate_apollo_runtime.py --build-check`. Repo code NOT in image (`.dockerignore` excludes `**`); `WORKDIR /apollo/users/sri/proj_adags/repo/adags` exists in image as empty dir. Images pushed as mutable tags `sudarshaniyengar/adags:apollo-{v100,h100}-v1` (no digest pin; `force_pull_image: true`).
- Task configs: `det_cfg_apollo_dgx.yaml` (pool `dgx`) / `det_cfg_apollo_hopper.yaml` (pool `hopper`), slots 1, env vars `ADAGS_PROJECT_ROOT/REPO_ROOT/DATA_ROOT(=…/data/n3v)/RUNS_ROOT/LOGS_ROOT`, `WANDB_MODE=disabled`. NO `entrypoint`, NO `bind_mounts`, NO `workspace/project` ⇒ valid only for `det cmd run --config-file`; not experiment configs. The four `ADAGS_*_ROOT` data/run vars have zero consumers in code.
- Preflight: `scripts/validate_apollo_runtime.py` (imports, extensions, commands, `--require-gpu` capability check {7.0, 9.0}, `--repo`, `--scene`) — runtime modes never invoked by any tracked artifact. Smoke config `configs/n3v/smoke_apollo.yaml` (200 it / 50k pts / res 4) exists, referenced only by the validator.
- Provenance patterns (Slurm-era, reusable as PATTERNS): Tier-2 exact-worktree binding (`scripts/prepare_csvl_vpl_stage1*.py`: O_CREAT|O_EXCL immutable JSON, per-file sha256, `git merge-base --is-ancestor` freeze gate); Tier-3 preregistered run matrix with config/launcher/source sha256s (`scripts/build_phase9_run_matrix.py`); append-only `JOB_LEDGER.jsonl` with duplicate-submission blocking (`scripts/submit_phase9_depth_visibility.sh`); `depth_visibility/canonical.py` (`sha256_file`, `canonical_json_bytes`), `depth_visibility/ledger.py` (`recursive_provenance_check`).
- Trainer-side git stamping: `main.py` `get_git_commit/get_git_branch/get_git_dirty/get_job_metadata` — currently surfaced only via W&B (disabled on Apollo).

Concrete gaps proven by inspection (this justifies a new thin submission wrapper — no existing launcher can be extended, because no Determined launcher exists):
1. No `det` invocation, master address, workspace/project binding, or entrypoint recorded anywhere.
2. No recorded mechanism for code to reach the container (no bind_mounts; code not in image); "which commit ran" is currently undefined on Apollo.
3. No isolated commit materialization anywhere (`git archive`/`git worktree`: zero hits); Slurm jobs ran from the shared mutable worktree with a self-declared `worktree_clean_claim: false`.
4. No immutable config record on the generic path (`cfg_args` is a `str(Namespace)` blob).
5. No Determined-side experiment ledger, no Determined task/trial ID capture (`main.py` job-ID scan knows only OAR/SLURM/PBS).
6. Metrics/metadata discarded on Apollo (`WANDB_MODE=disabled`; W&B is the only consumer of git metadata).
7. No DiVa-360 loader/config/path anywhere (N3V "Blender"-branch loader via `transforms_train.json` in `scene/dataset_readers.py`; dispatch `scene/__init__.py:52-58`).
8. Mutable image tags; no digest pinning.
9. Apollo migration undocumented in `research-wiki/operations/` (institutional-memory gap to close as part of M0 wiki work).

### 10.2 Execution design

**Submission wrapper** — new `scripts/submit_apollo.py` (justified: no Determined launcher exists to extend, §10.1; reuses `depth_visibility/canonical.py::sha256_file` + `artifacts.py::atomic_write_json_immutable` O_EXCL patterns and the JOB_LEDGER append-only shape). Responsibilities:
1. Execution closure check: resolve the declared execution-relevant set (elgs/, scene/, gaussian_renderer/, utils/, arguments/, main.py, configs used, the wrapper itself, det config) against `git status`; any dirty/untracked file in that set ⇒ REFUSE (unrelated dirt, e.g. the two user-owned wiki files, is allowed and listed in the manifest as excluded). `--dirty-smoke` overrides but stamps `evidence_bearing: false` into the manifest and run dir name.
2. Isolated commit materialization: `git archive <commit>` → temp context dir (+ generated run manifest as the only non-repo file, content-hashed) → submitted as the Determined experiment context (`det e create` includes the context; Determined snapshots it). Entrypoint runs from the uploaded context, `PYTHONPATH` pinned to it; the experiment template OMITS `work_dir` (or sets it to the context dir) and does NOT export a worktree `ADAGS_REPO_ROOT` — and the entrypoint ASSERTS `elgs.__file__` (and `main.__file__`) resolve under the context dir, aborting otherwise (requirement 3; the historical det cfgs point `work_dir` at the mutable shared worktree and must not be inherited). CUDA extensions come from the image (already baked). Evidence-bearing commits must be pushed to origin (or a `git bundle` with recorded sha stored beside the run dir) BEFORE submission, so the exact snapshot exists off the workstation.
3. Config identity: canonical-JSON hash of the fully-merged run config recorded pre-submission; entrypoint re-hashes at runtime and aborts on mismatch.
4. Manifest (O_EXCL via `depth_visibility/artifacts.py::atomic_write_json_immutable`, in run dir + ledger): commit SHA, config path+hash, prereg file hashes, image DIGEST (see 6), pool, slots, seed, dataset manifest hash, output run dir, wrapper argv+hash, submitter host, UTC stamp; post-submit: Determined experiment ID + task/trial IDs; post-run: terminal state, restart count, artifact inventory+hashes (sealed last — `terminal.json` pattern).
5. Ledger + duplicate prevention (race-safe, not check-then-act): before `det e create`, the wrapper CLAIMS the cell name by O_EXCL-creating `claims/<cell_name>.json` on shared Apollo storage — EEXIST ⇒ blocked (inspect the claim + `det experiment list` to resolve); the experiment ID is written into the claim after submission. Ledger: append-only `experiment-ledger.jsonl` (schema `elgs-apollo-ledger-v1`); the APOLLO copy is authoritative (single ≤4 KiB O_APPEND write per record); the local copy is a read-only pull.
6. Experiment config: ONE template `det_exp_apollo.yaml` with pool + image injected by the wrapper; for every evidence-bearing run the image field is a DIGEST reference (`sudarshaniyengar/adags@sha256:…`, tag kept as label only) — digest resolved once per image per session (`docker manifest inspect` or registry API from the submitter; recorded in the wiki authority page), because mutable tags + `force_pull_image: true` cannot pin bits. `max_restarts: 0`; if Determined nevertheless restarts/re-runs a task, the entrypoint detects an existing manifest/checkpoint in the run dir and either resumes from checkpoint (explicit resume mode) or ABORTS — never a silent second run under the same ID; any restart is a distinct ledger event. Originals `det_cfg_apollo_*.yaml` stay untouched for `det cmd run` preflights.
7. Cancellation + monitoring procedures (verified once in M0): submission (`det e create`), status (`det experiment describe/list`), logs (`det experiment logs` / `det task logs`), CANCEL (`det experiment kill` + ledger `cancelled` event + claim-file annotation), artifact inspection (run-dir listing + manifest check), resumption (explicit resume mode above). Each procedure exercised during S0/S1 and recorded in the wiki authority page.
Not planned: any Determined searcher/HP tuning, W&B re-enablement (metrics travel via artifacts; decision recorded), interactive `det shell` as an execution path (debug only, never evidence-bearing).

**Run/output layout** (requirement 5–7): raw data read-only under `/apollo/users/sri/proj_adags/data/` (dataset manifest with per-file sha256 for the subset actually read); runs under `/apollo/users/sri/proj_adags/runs/elgs/<run_id>/` with `run_id = <UTCstamp>_<cell>_<seed>_<commit7>`; Determined logs additionally captured via `det experiment logs` into the run dir at completion-audit time.

**Preflight (smallest sufficient, at submission time; no broad requalification):** `det` CLI reachable + version; workspace/project exist; target pool has capacity (informational); image digest resolvable; ledger writable; dataset manifest validates; output root writable + quota headroom (df probe); one-time-per-session container mount probe (`det cmd run ls <roots>`). In-container gate for the first task of a session: `validate_apollo_runtime.py --require-gpu --expected-capability <cap> --repo . --scene <path>` (existing tool, unmodified).

**Hardware policy:** M0 S0 preflight on both pools (validates both images, negligible cost); M0 S1/S2 smoke + all M1 census cells on Hopper/H100 only. DGX/V100 reserved as fallback; any pool switch is a new ledger entry, never silent; every manifest records pool + GPU name from `nvidia-smi`/torch. Comparisons never mix pools. Hardware failures (CUDA/OOM/node) classified `infra_failure` in the ledger, distinct from `scientific_failure`; infra failures are retryable (max 2, same cell name + retry counter) — these are instances of the verified-defect invalidation rule (§11.2), and each retry records the defect. Scientific failures are results, never retried.

## 11. M1 — DiVa-360 data and census

### 11.1 Data (facts from `operations/loop2-sweep-2026-08.md` §F: arXiv 2307.16897, CVPR24 Highlight; 53-camera true 360° surround; 25 hand-object + 21 object-centric + 8 long-duration sequences; 120 fps; fg/bg masks; MIT license; no published dynamic-GS baselines)
1. Apollo inventory FIRST (assume nothing): `det cmd run` listing of `/apollo/users/sri/proj_adags/data/` + local checkout `data/` check. Only if absent → acquisition.
2. Acquisition procedure: official DiVa-360 release channel (project page/repo per arXiv 2307.16897); verify license text at source (recorded in wiki page). Minimum dev subset: 2–3 hand-object interaction sequences (dev census scenes; exact list fixed in `prereg_m1_census_v1.json` BEFORE download completes, chosen from the paper's sequence table, not from data inspection — selection-bias guard), all 53 cameras' calibration + the frames needed for the census window, fg/bg masks. Expected size estimated from the official release listing before download; storage+quota preflight against the estimate; download onto persistent Apollo storage `data/diva360/` (outside Git), via a logged Determined CPU task or login host [whichever the cluster permits — recorded]; if credentials/registration are required: STOP, report to user (per instruction).
3. Integrity: per-file sha256 manifest written at acquisition (`data/diva360/MANIFEST.sha256.json`, O_EXCL); raw tree set read-only (chmod -w); all preprocessing writes to separate derived dirs with their own manifests + provenance (source manifest hash + code commit).
4. Splits: per-sequence held-out camera list fixed in `prereg_m1_census_v1.json`; held-out cameras excluded from tracker inputs, seeds, observability, and any calibration — enforced structurally (preprocessing emits only training-camera track/seed artifacts; census code never receives held-out camera IDs).
5. Preprocessing: `scripts/diva360_to_blender.py` converter → existing Blender-branch layout (`transforms_train/test.json`, `points3d.ply`) so `scene/` is untouched; camera-convention correctness validated by reprojection checks against provided calibration (reusing `depth_visibility/camera.py` validators) on the Apollo host AND a load smoke inside the actual Determined runtime (both recorded).
6. Tracker validation: `scripts/build_elgs_tracks.py` (CoTracker3-class, per spec: per-camera queries from 3D surface seeds visible ≥2 training cameras, robust consensus triangulation, identity by common seed) — the pipeline CODE dry-runs on synthetic fixtures in M0 (B0 requirement); first real-data run on one short window in M1; output = frozen tracks artifact + manifest; shift/shuffle control variants generated at the same time from the same frozen artifact. Tracker WEIGHTS get the same provenance treatment as the dataset: official source + license recorded in the wiki, sha256 manifest, read-only storage on Apollo, stop-and-report if access requires credentials. GPU tracker runs via Determined; ledgered separately from the 25 GPU-h census ceiling (per experiment plan: "frozen-tracker preprocessing ledgered separately"). Accounting is for REPRODUCIBILITY ONLY — no hard ceiling and no compute-based approval pause applies to preprocessing: the first short-window real-data run is timed; its per-(camera×frame) GPU cost is extrapolated to the full dev subset and the projection recorded BEFORE the full tracker submission; every tracker job carries `category: "preprocessing"` in the experiment ledger with hardware provenance and processed scope (sequences, cameras, frame ranges); the M1 result page reports projected vs actual preprocessing GPU-h alongside (never inside) the census ceiling.
7. M1 image revision (required; the pinned images lack any tracker stack — Dockerfile pip list verified): one budgeted image update adding the tracker dependency set, built via the existing Dockerfiles + `validate_apollo_runtime.py --build-check`, pushed as a NEW tag and pinned by digest, build record in the wiki authority page. §15's "Dockerfiles never touched" is amended accordingly (this is the one planned exception, done once at M1 start).

### 11.2 Census execution and gate (experiment-plan B1; ceiling ~25 GPU-h)
- Cells (frozen in `prereg_m1_census_v1.json` before any submission; ordered so the cheapest kill runs first and the GATE never depends on a trained model):
  M1-A0 MODEL-FREE opportunity screen (CPU, ~0 GPU-h, FIRST): upper bounds on occlusion / true-absence / same-object-return opportunity computed ONLY from shipped masks + 53-camera calibration + frozen tracks — no trained model anywhere in the statistic. **The gate floors apply exactly to these model-free statistics** (a substrate that reconstructs occluders poorly must not be able to fail — or pass — the gate for the data). Below floors ⇒ M1 FAILS here for ≈0 GPU-h.
  M1-A0b audited true-absence (CPU, DIAGNOSTIC-ONLY): the frozen §7.2-item-6 audit protocol executed on M1-A0's candidate set; produces the diagnostic audited-confirmed statistic + audit records; never enters the gate or PASS/FAIL.
  M1-A track-coverage census (CPU/1-GPU short): fraction of dynamic content bound to clusters; coverage by camera-count strata.
  M1-B substrate trainability smoke: LGS-substrate (evidence-off) short runs on the dev sequences (conditional on M1-A0 pass; produces model snapshots + sanity panels).
  M1-C model-conditioned opportunity diagnostic: q̃ distributions under M1-B snapshots — SECONDARY diagnostic only, reported next to M1-A0, never gated.
  M1-D shift/shuffle sanity on census statistics (controls behave as expected — cheap validity probe, not causal claim).
  (Baseline 4DGS/STG port smoke MOVED to M2's budget — the gate does not consume it; recorded as a deviation from the B1 prose with this justification.)
- Naming: `m1_<cell>_<seq>_<seed>`; config freeze: all census configs + prereg hashes recorded in each manifest; cell definitions frozen before any DiVa-360 statistic is computed (including CPU cells); submission sequence: M1-A0 → M1-A0b → M1-A → M1-B → M1-C/M1-D; duplicate prevention per §10.2 (retry claim key `<cell_name>__r<n>` so O_EXCL claims don't self-block); monitoring via `det experiment list/logs` polls recorded in the ledger.
- Failure classification: infra vs scientific per §10.2; a scientific failure of a census cell is a RESULT (preserved), not a retry candidate.
- Expected artifacts per cell: census JSON (schema-tagged), diagnostics panels (PNG), summary.json, manifest, terminal.json; for M1-A0b additionally the audit records (frames shown, verdicts, agreement rate). Interpretability requirement: every census statistic traceable to (sequence, camera set, frame window) in the artifact.
- Independent verification (independence defined): a fresh-context worker recomputes the gated statistics from PRIMARY inputs — the frozen tracks artifact, masks, calibration, and saved probe outputs (the M1-A0b audit records serve the same role for the diagnostic recomputation) — writing its OWN reduction from the definitions in `prereg_m1_census_v1.json`, never reading the census code's summaries or reusing its reduction functions. Integrity audit checks manifest completeness, commit/config/image-digest/hardware match, and that the intended code path ran (log markers + config echo).
- Gate: preregistered floors (derived from the M0 power analysis, reviewer-signed BEFORE any DiVa-360 statistic was computed — §7.2 item 6) applied EXACTLY to the model-free statistics (M1-A0) ONLY; M1-A0b is diagnostic and never gate-bearing; PASS ⇒ M1 complete, M2 unblocked scientifically — starting M2 still requires user approval (§16b).
- Failure and retry policy (preregistration integrity — binding):
  - A scientifically VALID failure against the original M1 gate is a FAILURE, permanently preserved (wiki result page + artifacts + ledger). Floors, eligibility definitions, audit requirements, gate cells, AND this failure/retry policy itself are FROZEN at M0 before any DiVa-360 data is inspected, and immutable thereafter — no post-hoc change may convert a failure into a pass.
  - A RETRY (same cell, same gate) may replace an original run ONLY when a verified implementation, instrumentation, corrupted-data, infrastructure/hardware (`infra_failure` per §10.2), or protocol-execution defect made the original result unusable; the defect, its verification, and the invalidated run are recorded in the ledger (`invalidated_by_defect` event referencing the replacement) AND on the M1 result page alongside the replacement. A defect discovered AFTER the cell's gate outcome is known additionally requires fresh-context reviewer confirmation — recorded before the replacement is submitted — that the defect (a) was identified independently of the gate outcome and (b) is demonstrably result-affecting; at most two such post-outcome invalidations across all of M1.
  - A valid frozen M1 failure is the FINAL M1 result. There is no revision, redesign, or exploratory-relabel mechanism inside this plan; a run may be replaced only under the verified-defect rule above. Whatever follows a final M1 failure is outside this plan's scope and is a user decision.

## 12. M2–M5 dependency outline (high level only)

- M2 (B2, ~80 GPU-h): ported external baselines (documented, compute-matched; absorbs the baseline-port smoke moved out of M1) + LGS substrate evidence-off lanes + oracle-structure probes (claim C1). Depends on: M0 code, M1 census pass, frozen category-2/3 constants for substrate.
- M3 (B3, ~120 GPU-h): decisive q-source/update matrix (5 arms × dev scenes × 3 seeds), β=0 identical-search, naive-loss, dose-matched shifts/shuffles, oracle bridge/linkage. Depends on: M2 substrate + baselines; preregistered kill rules bind.
- M4 (B4, ~40 GPU-h): decision-classification distributions, risk-coverage, ITT, lineage-removal sensitivity, ε-bound power curve, self-censoring and clone-invariance audits, feasibility ledger. Depends on: M3 artifacts.
- M5 (B5, ~90–120 GPU-h): freeze → held-out DiVa-360 → Ego-Exo4D stress → HOT3D/ADT metric validation → N3V/Technicolor continuity → failure taxonomy. Depends on: M4 complete, constants frozen.

## 13. Worker ownership and integration order

Representation-critical writes are SERIAL (one writer at a time on `elgs/` core, `scene/gaussian_model.py`, `main.py`, renderer). Single-owner-file rule: `scene/gaussian_model.py` has EXACTLY ONE owner (Fable) for all of M0 — rows needing edits there (family column row 2; moment writes row 4; capture/restore row 10; param groups row 11) submit their required edits to that owner as an ordered sub-list with explicit handoffs; the owner lands them serially. Same rule per file for rows 13/14 (one owner per file; Sonnet scaffolding is handed off to Fable sequentially, never concurrent on one file). Sonnet workers only where semantics are already fixed; every Sonnet artifact inspected by Fable before integration. Independent reviewers are fresh-context (no implementation rationale shared). Reference-implementation oracles (§8 rule) and the transition-table prereg (§7.2 item 2) are authored by fresh-context workers from spec sections alone — never by the module implementer — and their sign-off precedes the consuming module's merge.

| # | Module(s) | Owner | Owned files | Spec §§ | Required tests | Depends on | Reviewer | Stop condition |
|---|---|---|---|---|---|---|---|---|
| 1 | intervals | Fable | elgs/intervals.py | §1 | 8.1 | — | Opus fresh-context math reviewer | all 8.1 green + reviewer sign-off |
| 2 | families, presence | Fable | elgs/families.py, presence.py | §1 + method repr | 8.1/8.2 subset | 1 | same | invariants green |
| 3 | renderer hook + probe | Fable | gaussian_renderer/__init__.py (surgical), elgs/probe.py | §3 | I1 + S1 | 2 | Opus (module boundary) | exact-zero absence verified |
| 4 | ops + transactions | Fable (Opus consult on rollback design) | elgs/ops.py, transactions.py | §1, §5 | 8.2, 8.4 rollback | 1,2 | Opus | transition-table tests green vs data oracle |
| 5 | clusters, evidence, bridges, observability | Fable | elgs/clusters.py, evidence.py, bridges.py, observability.py | §2, §3 | 8.3 | 1,2 (probe fake) | fresh-context implementation-semantics reviewer | PROP-1 exactness green |
| 6 | energy + ledger | Fable | elgs/energy.py, ledger.py | §4, §6 | 8.3 | 5 | Opus math reviewer | Φ reference-impl parity |
| 7 | acceptance + slot grid | Fable | elgs/acceptance.py | §7 | 8.4 | 4,6 | Opus math reviewer | SNIS bias-convergence + CRN determinism green |
| 8 | search | Fable | elgs/search.py | §5 + method | 8.2 preconditions, ITT | 4,5,6 | fresh-context reviewer | deterministic-under-seed + caps |
| 9 | classification + reporting | Sonnet (semantics fixed by §8) | elgs/classification.py, reporting.py | §8 | 8.5 | 7 | Fable | truth-table green |
| 10 | state_io + checkpoint | Fable | elgs/state_io.py, scene/gaussian_model.py capture/restore | §1 serialization | 8.5, I4 | 2,4 | Opus (checkpoint design) | round-trip + rejection green |
| 11 | config + trainer wiring | Sonnet (mechanical) then Fable integration | arguments/__init__.py, main.py, configs/elgs/* | schedule | 8.5 config, I5 | 1–10 | Fable | static admission green |
| 12a | transition-table prereg | FRESH-CONTEXT worker (spec §1/§5 only; no code/plan access) | configs/elgs/prereg_latch_transition_table_v1.json | §1, §5 | table-vs-spec review | — | second fresh-context reviewer | sign-off REQUIRED BEFORE row 4 merges |
| 12b | remaining prereg tables | Fable | configs/elgs/prereg_*.json (rest) | G3 cat-B | table-vs-spec review | — (parallel) | independent fresh-context reviewer | reviewer confirms table ≡ spec |
| 12c | reference-impl oracles (§8.3 likelihood, §8.4 bootstrap, §8.2 components) | FRESH-CONTEXT worker per oracle (spec section only) | tests/ref_impls/ (frozen, hash-recorded) | §2, §4, §7 | — | — | Fable (integration only, no edits) | frozen before consuming module merges |
| 13 | submission wrapper + det template | Sonnet draft → Fable owns final file | scripts/submit_apollo.py, det_exp_apollo.yaml | — | I5 + S0 | — (parallel) | Fable | dry-run manifest complete; refusal paths tested |
| 14 | DiVa converter + tracks pipeline | Sonnet scaffolding → sequenced handoff to Fable (one owner per file at any time) | scripts/diva360_to_blender.py, build_elgs_tracks.py, elgs/tracks_schema.py | §2 evidence | fixture tests + M0 dry run | — (parallel; real data M1) | fresh-context reviewer | reprojection validation green |
| 15 | spec-to-code verification | Independent fresh-context reviewer (Opus) | none (read-only) | all | — | 1–11 done | n/a | statement-by-statement map, zero unresolved divergences; PRECEDES GPU smoke S1 |

Integration commits: narrow, one module-group each, in dependency order 1→2→3→4→5→6→7→8→9→10→11 (12/13/14 land in parallel when ready, subject to 12a preceding 4); each commit message names the spec sections implemented; wiki M0 record updated at milestones, not per-commit.

## 14. Goal → evidence → verification ledger; pass/fail; ceilings; preservation

### M0 phases (order: 0 → 1 → 2 → 3 → 4 → 5 → 6; spec-to-code verification PRECEDES GPU smoke)
0. FIRST EXECUTION ACTION — preserve the accepted plan: save it as a tracked page `research-wiki/operations/elgs-m0-m1-implementation-plan.md`; add only minimum discoverability references (a `research-wiki/log.md` entry + a one-line pointer from `operations/elgs-experiment-plan.md`); verify `git status --short` + `git diff --stat`; commit narrowly. After this preservation commit, implementation proceeds with no further planning phase.
1. Goal: Apollo execution decision recorded + submission path exists → Evidence: wiki operations page (Apollo/Determined execution authority), `submit_apollo.py` + det experiment template committed → Verification: S0 preflight passes on both pools; dry-run refusal paths demonstrated; cancel/status/logs/resume procedures exercised.
2. Goal: EL-GS core implemented faithfully → Evidence: modules 1–10 committed with tests → Verification: full CPU suite green (`python -m unittest discover tests`) twice — once with randomized test order, once with a different global seed; every §2/§2.4/§5 requirement row has a passing test or a written justified exception in the M0 wiki record.
3. Goal: ADAGS integration → Evidence: module 11 committed; I1–I5 green → Verification: substrate lane trains on CPU micro-scene; existing 278-test suite still green (no regression).
4. Goal: preregistration authored + B0 analysis work performed → Evidence: 7 prereg files committed (category-2 entries marked unfrozen with declared calibration procedures); the POWER ANALYSIS performed and recorded in `prereg_metrics_v1.json`; tracker-pipeline dry run on synthetic fixtures executed (both are B0 items — kept inside M0, not deferred) → Verification: independent reviewer table-vs-spec sign-off (12a already merged earlier); static admission test green; power-analysis numbers feed the M1 floor derivation.
5. Goal: independent spec-to-code verification → Evidence: reviewer report mapping spec §§1–8 + §2.4 substrate semantics to file:line → Verification: zero unresolved divergences; divergences found are fixed (new commit + retest) before proceeding.
6. Goal: GPU validation → Evidence: S1+S2 manifests, artifacts, ledger entries (S1/S2 stamped `evidence_bearing: false`; N3V smoke-scene subset gets an inventory + sha256 manifest before use) → Verification: completion rule of §8.6 (terminal state + logs + provenance match + artifacts readable + resume equivalence).
M0 PASS ⇔ all six verified. M0 FAIL states are specific: any spec-to-code divergence unresolved, any 8.x test red, or GPU smoke provenance incomplete ⇒ M0 not complete (no partial credit).

### M1 phases
1. Goal: data present + validated → Evidence: inventory record; acquisition manifest; read-only raw tree; converter outputs + reprojection validation on host AND in-container → Verification: manifests hash-verified; load smoke inside Determined runtime green.
2. Goal: frozen tracks → Evidence: tracks artifact + manifest + controls → Verification: dry-run report; seed/identity rules audited by fresh-context reviewer; held-out cameras verified absent from artifact.
3. Goal: census executed → Evidence: M1-A0/A0b/A/B/C/D cell artifacts (incl. audit records per §7.2 item 6) + ledger + terminal manifests → Verification: completion rule per cell; independent recomputation of the gated statistics from primary inputs (§11.2).
4. Goal: gate applied → Evidence: gate report citing prereg floors + census numbers → Verification: integrity audit (provenance + evaluator execution verified) precedes gate; result page committed to wiki EITHER way (pass or preserved negative).
M1 PASS ⇔ floors met under verified provenance. M1 FAIL ⇒ permanently preserved negative and the FINAL M1 result; replacement only under the verified-defect invalidation rule; what follows is a user decision (§11.2).

### Compute ceilings
- M0: ≤5 GPU-h total (S0 ≈ minutes; S1/S2 ≤ ~2 GPU-h; margin for one infra retry).
- M1: ≤25 GPU-h census ceiling. Tracker preprocessing has NO hard ceiling and no compute-based approval pause — projected and actual usage, hardware provenance, and processed scope are recorded for reproducibility only (§11.1 item 6).
- Guard: the wrapper writes projected GPU-h into each manifest; ledger running total checked before each submission; exceeding a ceiling requires a user decision, never silent.

### Durable preservation
- Every M0/M1 conclusion (positive or negative) → `research-wiki/operations/` pages: `apollo-determined-execution-authority` (new, closes the §10.1 institutional-memory gap), `elgs-m0-implementation-record`, `elgs-m1-census-result`; `research-wiki/log.md` entries per campaign with Determined experiment IDs; ledgers and manifests preserved on Apollo storage; nothing depends on chat memory.
- Commits narrow and intentional; `git status --short` + `git diff --stat` + static checks before each; research-wiki pages tracked normally (no `git add -f`); the two user-owned untracked files never touched.

## 15. Expected changed files

M0 (new unless noted): `elgs/` — the §6 module list is canonical (intervals, families, presence, clusters, evidence, observability, bridges, energy, transaction_ledger, ops, transactions, search, acceptance, classification, state_io, reporting, probe, tracks_schema, `__init__.py`); `tests/test_elgs_*.py` (9 files) + `tests/ref_impls/` (frozen fresh-context oracles); MODIFIED: `arguments/__init__.py` (elgs block), `main.py` (setup_elgs + round hook + sampler exclusion + post-refit pass + iteration-guard env note), `gaussian_renderer/__init__.py` (presence hook + routing pinning), `scene/gaussian_model.py` (single-owner: family column, param groups, capture/restore, pose/motion moment-copy write); `configs/elgs/` (smoke_elgs.yaml, dev configs, 7 prereg JSONs); `depth_visibility/schema.py` (register elgs prereg schemas — additive only); `scripts/submit_apollo.py`; `det_exp_apollo.yaml` (new single template; originals untouched); wiki: 3 new operations pages (`elgs-m0-m1-implementation-plan` — the preserved plan, `apollo-determined-execution-authority`, `elgs-m0-implementation-record`) + log entries + a one-line pointer edit to `operations/elgs-experiment-plan.md` carrying the B1-supersession note (§17).
M1 (new): `scripts/diva360_to_blender.py`, `scripts/build_elgs_tracks.py`, census configs under `configs/elgs/`, ONE budgeted image revision (Dockerfile tracker-stack addition, new tag, digest-pinned, build-checked — the single planned Dockerfile exception), wiki result pages; NO changes to `scene/` loaders expected (converter route).
Never touched: `research-wiki/deep-dive-prompt.txt`, `research-wiki/run-deep-dive.ps1`, vendored CUDA extensions (unless the §3 q-probe decision selects the rasterizer-addition path — then a narrow, tested, justified change to `diff-gaussian-rasterization/` owned by Fable), Dockerfiles outside the one M1 revision, `depth_visibility/` (read-only reuse except the additive schema registration), existing lane configs.

## 16. M2–M5 note

See §12. No detail planned beyond dependencies, per instruction.

## 16b. Execution authority granted by plan acceptance

Acceptance of this plan authorizes, without further per-step approval, the ordinary reversible work M0–M1 require:
- creating the implementation branch; normal non-force pushes (this supersedes §1's "pushing is a user decision" note: the first push of the implementation branch necessarily publishes the 5 pre-existing unpushed wiki commits of §1 — plan acceptance covers this);
- adding code, tests, configs, and prereg files per this plan;
- using an appropriate existing Determined workspace/project (created-if-absent, recorded);
- downloading publicly accessible required data (DiVa-360 dev subset, tracker weights) after the provenance and storage checks of §11.1; setting acquired raw data read-only;
- building, PUSHING to the registry, and digest-pinning the one planned runtime-image revision (registry credentials from the operator environment; if unavailable, stop and report);
- submitting, monitoring, cancelling (on defect, or once in M0 to exercise the documented cancellation procedure), and auditing M0–M1 Determined experiments within the §14 ceilings;
- making in-scope implementation fixes; committing durable results and wiki records.

Still excluded regardless: force-push; overwriting or deleting data; modifying unrelated work or the two user-owned files; exceeding M0–M1 scope; proceeding to M2.

### 16c. Execution operating rules
- When enough information exists to act, act. Continue autonomously until an approved stop condition is genuinely reached.
- Follow the existing worker-ownership (§13), delegation, and goal → evidence → verification (§14) rules as written — they are not duplicated or expanded here.
- Use relevant ARIS skills when they materially help with implementation, Determined execution, monitoring, experiment-integrity auditing, or result-to-claim analysis; read and follow the applicable `SKILL.md` instructions first. Never use them to reopen planning or method discovery.
- Make the minimum faithful changes the specification and this plan require.
- Add no speculative flexibility, unrelated features, broad refactors, or adjacent cleanup.
- Every changed line must trace to an approved requirement.
- Preserve all unrelated user changes.
- Update `research-wiki/` throughout execution with durable engineering decisions, provenance, failures, results, and claim limitations.
- Before reporting progress or completion, verify every claim against tool output or an artifact from the current run; state plainly what remains unverified.
- Do not end a turn with a promise, plan, question, or statement of intent while authorized work remains and an approved stop condition has not been reached.
Pause for user input ONLY on: destructive or irreversible actions; material scientific scope change; any change that would alter the approved EL-GS claims; any access requiring credentials, login, or registration not already available in the operator environment (§3 Determined login; §11.1 data and tracker weights; registry push); a failed M1 gate (what follows is a user decision, §11.2); overrun of the M0 or M1-census GPU-h ceilings (§14; tracker preprocessing is explicitly exempt — reproducibility accounting only). Ordinary implementation steps already authorized here are not re-asked.

## 17. Audit labels and review record

- UNVERIFIED (to be verified at stated preflight): /apollo automount inside containers; N3V preprocessed data present on Apollo (det cfg env implies raw root only); DiVa-360 absence on Apollo storage (inventory is the first M1 act); Determined workspace/project names (none recorded — wrapper will create-or-use a named workspace, recorded in the wiki page). (Image-vs-Dockerfile correspondence is NOT claimed verifiable — evidence-bearing runs pin by digest instead, §10.2 item 6.)
- All file:symbol claims in §4/§10 verified by direct read or mapper evidence; Fable spot-checked the load-bearing reuse claims directly (`depth_visibility/capacity.py` transaction + moment reset; `det_cfg_apollo_hopper.yaml` full contents; `gaussian_model.py::capture` 30-tuple + nested state dicts; renderer opacity path :196–254).
- No plan item contradicts `c21de8b`: the §2 semantics summary was transcribed from the rev-4 spec text itself; §2.4 was transcribed from `lgs-method.md` (binding via the spec's §1 delegation); the M0 tests mirror the errata's required B0 list verbatim plus spec/substrate-derived additions.
- Deviations from experiment-plan B1 prose, recorded here and to be recorded in the wiki: (1) baseline 4DGS/STG port smoke moved from B1 to M2's budget (gate does not consume it). (2) "Audited true absence" is DIAGNOSTIC-ONLY for M1 (user decision, fixed before any DiVa-360 statistic is inspected); the stated consequence is that M1 cannot support a claim-grade estimate of true-event absence or false-positive prevalence. (3) B1's "one preregistered revision cycle" is SUPERSEDED by §11.2's failure/retry policy (a valid frozen failure is the final M1 result; replacement only for a verified technical or protocol-execution defect) — the phase-0 pointer edit to `operations/elgs-experiment-plan.md` carries this supersession note so the durable record holds one policy, not two.
- Fresh-context adversarial review of this plan: completed (Opus, hostile). 2 FATAL + 24 MAJOR findings; ALL resolved by revision (substrate authority §2.4; q-probe decision register §3; oracle-independence rule §8; ownership/concurrency §13; provenance items §10.2; census restructure §11.2; classification timing §2.3/§6.17; δ classification §7.1; and the enumerated minors). No finding was left open.
- Focused fresh-context review of the user-directed revision (M1 failure/retry policy; audited-true-absence decision; external-action authority; cross-plan consistency): completed (Opus). Verdict was "not pass" with 7 must-fix + 13 one-clause findings — all 20 applied. Nothing outside the four areas was reopened.
- Final user-directed corrections (this revision, superseding the corresponding earlier record entries): audited true absence → DIAGNOSTIC-ONLY with the stated claim limitation; tracker preprocessing → reproducibility accounting only, no hard ceiling or compute-based pause; M1-R1 exploratory mechanism REMOVED (a valid frozen M1 failure is final; replacement only under the verified-defect rule); execution operating rules added as §16c.
