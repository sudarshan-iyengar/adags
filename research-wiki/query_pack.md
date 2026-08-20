# Query Pack

## 2026-08-14 MEASUREMENT CLOSURE — the absence instrument has a material defect

Read [[operations/elgs-absence-diagnostic-result]] before citing any
DiVa-360 true-absence figure. Verdict **status_2 (material defect),
UNANIMOUS across 144 sensitivity readings**, independently recomputed.

- Corrected tranche-1 pooled totals: **239,545 occlusion / 597
  true-absence / coverage 0.8212** (the durable "~240k / ~700" was the
  defective-era 237,821 / 679).
- **ZERO of 597** true-absence windows are corroborated as genuine
  full-multiview disappearance — pooled and in every sequence. In
  **96.6%** an eligible foreground component sustained
  multi-view-consistent occupancy of the anchor while the tracker's report
  failed to qualify: **87.6%** visibility flag < 0.5, **12.2%**
  never-queried cameras, C2 (true track loss) = **0 everywhere**.
- Coverage and absence share the `v >= 0.5` threshold, so their
  anticorrelation is partly an instrument identity. Occlusion (needs
  association in >= 2 cameras) is barely coupled and its supply claim
  STANDS.
- **Tracks and conversions are REUSABLE**: `v` is stored per report, so any
  census-level instrument correction re-evaluates all 20 tranche-1
  sequences for ~1 CPU-hour and ZERO GPU-hours. Only a query-construction
  change (fixing the 12.2% never-queried limb) needs re-tracking.
- NOT established: physical presence. The frozen M1-A0b audit sample (73
  windows) was emitted but NOT run; no physical-absence claim is permitted
  before it returns. G-OA's valid FAIL is NOT reopened.
- Screening scope and cost: [[operations/elgs-exhaustive-screen-scope]].

Compressed project memory for ideation. Updated 2026-08-20 after the
Loop-3 block produced [[operations/ccr-method-2026-08-20]] (CCR, the
current lead method candidate) and the first POSITIVE episodic-presence
result. The 2026-07-29 section remains the last user-approved direction
record.

## 2026-08-20 LOOP 3 — localized presence WINS on the fixture; CCR frozen; substrate at STG parity

- **Lane B positive (first ever for episodic presence).** The corrected
  LOCALIZED cell — per-primitive oracle membership (~84 rows, 8
  families), ordinary temporal marginal retained for every non-gated
  row, TOTAL opacity gate (static twin included), routing pins off —
  beat the matched temporal control on LRV3 `event_return` by
  **+1.0496 dB** (frozen floor 0.5) at matched capacity (1,126 FEWER
  primitives), `event_episode1` +1.24, `ordinary_all` −0.39, first
  return frame +2.0 dB. **The total gate rendered EXACT absence
  (infinite PSNR, zero error) on 21 of 27 gap frames**; the pooled
  ghost deficit is entirely the two designed ramp frames. The
  2026-08-19 −5.23 dB negative is now FULLY attributed to wiring
  (global swap + static-twin leak + voxel oracle + pinned routing):
  same event, same budget, localized wiring is **+6.15 dB** over the
  old A1. [[operations/lrv3-local-presence-corrected-cell-2026-08-20]].
  Small-mistiming control ran as experiment 191 (see the page for the
  outcome). Ordinary-region −0.39 dB cost real, unattributed.
- **Two instrument repairs that touch every historical number.**
  (1) `route_logit_init` in YAML NEVER controlled a fresh run —
  create_from_pcd materialized route logits from the constructor
  default 4.0 before training_setup read the YAML; every historical
  cell trained from p_dyn ≈ 0.982. Repaired; corrected cells declare
  4.0 explicitly. (2) Both eval call sites channel-split PSNR
  (+0.268 dB bias); now pooled, with tests. The `--val` path clamps,
  the training-time path does not — never mix them in one table
  ([[operations/n3v-baseline-registry-2026-08-20]] for every trap,
  including that the survey CSV's `csvl_vpl_v2_exploratory` 34.48 row
  is the WRONG-TIME CONTROL L5, never a baseline).
- **Substrate position, canonical:** pooled+clamped **33.5050 dB /
  SSIM 0.9593 / LPIPS-alex(norm) 0.0814** on `cut_roasted_beef` frames
  0-49 at 1352x1014, cam00 held out, 6000 iterations — vs STG's
  published 33.52 read at 25,000 iterations. Parity to 0.015 dB at a
  quarter of the schedule ([[operations/stg-n3v-protocol-parity-2026-08-19]]
  Appendix C). The substrate is NOT the blocker.
- **Lane C selected method: CCR** — observation-born packet birth
  (budget-neutral spacetime relocation, packet ids) + a POST-TRAINING
  certified consolidation pass: directional donor appearance reuse
  (DC primary arm) admitted per-edge by paired counterfactual trial
  renders on reserved units, sequential in a prespecified order, one
  all-or-nothing joint veto, B2 byte-identical to B1 outside the
  pointer column. Frozen after 2 generators + cross-model triage +
  16-query kill-search + 3 full-text threat reads + a 3-round hostile
  fresh-context review ([[operations/ccr-method-2026-08-20]],
  [[operations/ccr-novelty-record-2026-08-20]]). Novelty is NARROWED,
  alive: CubifyGS occupies object-level asset reuse after absence;
  the unoccupied cell is per-primitive, appearance-only,
  trial-render-certified tying with exact restoration. Claims split:
  identity only on the synthetic fixture, utility only on the frozen
  N3V segment (frames 0-49,
  [[operations/ccr-segment-selection-2026-08-20]]). Rejected on the
  record: annealed soft tying (the 5.23 dB pathology class), ballistic
  matching as headline, channel-selective sharing as standalone.
- **Ladder state at block end: see the 2026-08-20 handover** for
  experiment IDs (B0/B1 seeds on the STG-matched protocol with
  reserved parity, preflight, consolidation pass tooling all landed:
  `scene/packet_birth.py`, `scene/appearance_edit.py`,
  `scripts/consolidate_packets.py`, `main.py --val --appearance_edit`).
- **Unchanged:** DiVa claim-grade instrumentation PAUSED (the Lane B
  positive is fixture-level; real-data event supply is exactly as
  unresolved as before). EL-GS structural rounds, evidence heads: off.

## 2026-08-08 (night) EL-GS — Loop-2 lead candidate (gate pending)

- User-relaxed constraints for Loop 2: external priors ALLOWED; any
  public dataset ALLOWED; per-scene optimization FIXED.
- Method: EL-GS = episodic lineage representation (LGS substrate in
  family form) + RENDERER-CONDITIONED CENSORED EVIDENCE for
  data-supported structural selection: frozen multi-view point tracks
  interpreted through likelihood-ratio factors whose informativeness is
  gated by counterfactual observability computed BY the current scene
  model (family-present query-source-excluded transmittance, bridge
  family, censoring equality ⇒ censored segments contribute exactly
  zero); structural ops (fission/truncation/reactivation/birth/merge)
  selected under one energy with disclosed permanence+complexity
  priors; conditional claims only. Full spec + v8 fix set:
  [[operations/elgs-method]]; reviews:
  [[operations/elgs-review-history]] (refine 5.7→8.9; five fresh
  adversarial rounds, hostile novelty 4→6→7→7→8); novelty:
  [[operations/elgs-novelty-record]] (8.0 PROCEED WITH CAUTION,
  conditional; fall-back 6.5-7.0 if q degrades to a confidence
  weight); plan: [[operations/elgs-experiment-plan]] (~360-390 GPU-h,
  gated); sweeps: [[operations/loop2-sweep-2026-08]].
- Unoccupied cells occupied by EL-GS (verified): tracker visibility
  states as representation-level presence/identity evidence;
  measurement-model existence inference in differentiable rendering;
  reactivation with OWN trained content (relocation family is the
  mechanistic opposite).
- Datasets: primary DiVa-360 (53-cam surround, 25 hand-object seqs,
  MIT; GS baselines must be established); Ego-Exo4D cooking stress;
  HOT3D/ADT pose GT for metric validation only; N3V/Technicolor
  continuity.
- GATE (before any implementation): complete the v8 formal write-out
  (tempered-mixture bridge aggregation; ratio factors; one bridge
  latent per decision; ε-bound derivation; full transition table) and
  pass one further fresh-context adversarial round; then user approval.
- Program trajectory on one referee scale: STAR-GS 5.5 → LGS 6.5 →
  EL-GS 8.0 conditional. Target was 8.5+: the referee's stated ceiling
  reasons are inherited statistical machinery, heuristic
  search/acceptance, tracker/bridge dependence, and ported-baseline
  benchmark risk.

## 2026-08-08 (evening) LGS — representation-level candidate (awaiting user decision)

- Method: LGS (Lineage Gaussian Splatting) — the dynamic primitive
  becomes a LINEAGE: tied radiance (world-frame SH + base opacity) +
  ordered disjoint compact-support EPISODES (K≤4; per-episode
  translation/rotation-offset/scale-offset; rank-8 motion coefficients
  at immutable per-episode origins; EXACT-zero absence between episodes;
  latched presence, chain-invariant intervals). Structural search
  (supporting, not claimed): screening accumulators + counterfactual
  micro-render acceptance (ΔL̂ + λ·ΔS < 0, codec-mode-decision
  ancestry); reactivation = deterministic voxel-hash retrieval of
  dormant lineages at predicted pose, re-enabling OWN trained content
  (no merge — predictable-pose returns only, disclosed). Full spec:
  [[operations/lgs-method]]; plan: [[operations/lgs-experiment-plan]]
  (~350-420 GPU-h, activation-census gate kills cheaply); reviews:
  [[operations/lgs-review-history]]; novelty:
  [[operations/lgs-novelty-record]]; boundary:
  [[operations/repr-sweep-2026-08]]; rejected candidates:
  [[operations/rejected-representations-2026-08]].
- Provenance: 5 verified sweeps + 9 deep-dives → 4-round refine loop
  (9.2 READY) → 5 FRESH-context adversarial rounds (4 SINKS with full
  redesigns: statistics-carving v5, ratio-economics v6, soft-assignment
  v7 — each ledgered; v9 SURVIVES-WITH-RISKS, no unrepairable defect)
  → calibrated novelty check **6.5/10 PROCEED WITH CAUTION** (same
  scale as STAR-GS's 5.5; CC1 multi-episode latched presence HIGH; CC5
  conjunction HIGH field-specific; referee: 8+ requires a new inference
  principle/foundational primitive). Run target was 8.5+ → decision
  escalated to user: approve LGS at 6.5, or direct another loop under
  changed constraints.
- Verified-unoccupied kernel (five sweeps + dedicated gap searches):
  multi-interval/reactivating presence per primitive; latched presence;
  per-primitive changepoints; exact compact-support absence;
  counterfactual trial-render acceptance (closest: L2D2-GS 2606.29374 —
  offline policy reward, not live acceptance); dormant reactivation
  with OWN content (nearest family is the mechanistic OPPOSITE:
  3DGS-MCMC code-verified donor-clone overwrite; FreeTimeGS++
  2605.03337 ablates the donor-respawn family).
- Largest risks (binding on any test): mechanism activation rarity on
  N3V (B1 census gate); hard-sharing appearance bias (object-frame +
  soft-share probes); no-merge irreversibility; search-cost ledger.
- Wiki corrections recorded (not silent): 2606.23212 self-names VAD
  (≠ VAD-GS 2510.09364); kang2025 "CEC-4DGS" actual title "Clustered
  Error Correction with Grouped 4DGS" (2511.16112), repo CEM-4DGS.

## 2026-08-08 STAR-GS candidate (preserved, not lead)

- Method: STAR-GS v9 — budget-neutral correction of
  depth-deficient dynamic Gaussian models via occlusion-aware multiview
  residual-space carving (SRC): parent-free time-local births at carved
  deficit sites, funded by audited donor retirement; model-internal only;
  zero new trainable components. Full mechanism:
  [[operations/star-gs-v9-method]]; test plan:
  [[operations/star-gs-v9-experiment-plan]]; review substance:
  [[operations/star-gs-v9-review-history]]; sweep findings:
  [[operations/sota-sweep-2026-08]].
- Provenance: 5-round research-refine + 5 fresh-context adversarial
  rounds (4 SINKS resolved by redesign, final SURVIVES-WITH-RISKS) +
  novelty check 5.5/10 PROCEED WITH CAUTION. Four rejected families
  recorded in [[operations/rejected-approaches-2026-08]] (residual
  routing; support-matched momentum; observation clocking; calibrated-
  consensus claims) — claim inflation (exactness/calibration/priority)
  was fatal every time; conservative proposer + causal validation
  survived.
- NEW closest work (mechanism-verified): [[papers/kang2025_cec_4dgs]]
  (CEC-4DGS, SIGGRAPH Asia 2025) — error-clustered, cross-view-checked,
  TIME-LOCAL 4D birth at single-view rendered depth, unbudgeted; the
  mandatory primary baseline (faithful + budget-matched ports).
  FreeTimeGS (CVPR 2025, N3V 33.19 avg SOTA peer-reviewed) does periodic
  budget-neutral relocation (0.5·∇g+0.5·σ) — "budgeted reallocation is
  occupied" now includes it; SharpTimeGS (arXiv 33.57) leads
  self-reported. TAD-GS + 4D Scaffold-GS (AAAI 2026) both occupy
  presence-weighted densification statistics.
- Round-1 exploratory evidence reading (unchanged, load-bearing):
  presence-weighted densification = the only contract-valid perceptual
  win (dyn-mask +0.72 dB, LPIPS −5.2%); targeted birth > generic churn
  at matched counts (+0.29 dB); generic capacity hurts; external
  evidence alignment refuted (L5 ≥ L3) → mechanisms must be
  model-internal.
- Experiment plan: [[operations/star-gs-v9-experiment-plan]] (~405-460
  GPU-h; Phase-A constructor gate kills cheaply; CEC ports mandatory;
  localization evidence make-or-break; annotations evaluation-only).

## 2026-07-29 CSVL-VPL v2 direction (last approved direction record)

- Stage 1 of CSVL-VPL v1 (temporal surface association over sealed P03) was
  executed 2026-07-26 and returned three no-gos: association could not beat
  camera-swapped flow; flow was non-causal (98.6% selected-edge overlap
  without it); and all 19 scanned windows contain zero front/rear cross-order
  candidates. The P03 multilayer-bin occlusion representation is structurally
  wrong for a frontal rig and is retired.
- The approved v2 method keeps the two-part frame and replaces both halves:
  primitive-centric evidence (E1 external reprojection visibility per
  primitive/camera/frame; E1-int rendered transmittance; E2 model-deficit
  birth targeting) and a from-scratch lifecycle (protection by update
  freezing, occlusion-aware exposure-normalized densification, budget-neutral
  E2 birth, hysteretic retirement) that never manipulates rendered opacity.
- Sequencing changed: Phase 0 evidence-opportunity census with preregistered
  floors gates everything; the oracle-capacity attribution lane (B02) is
  restored and runs before/alongside inferred-evidence lanes; trainer limbs
  that consume no external evidence are no longer blocked on Gate A.
- Scene allocation: dev = cut_roasted_beef + cook_spinach; locked =
  flame_steak + sear_steak; stress = coffee_martini + flame_salmon_1; final
  comparisons all six. Capacity: matched-capacity + generic-extra-capacity +
  shuffled-evidence controls are a hard gate for any visibility-attribution
  claim; capacity deltas allowed for disclosed Pareto-reported results.
- Closest-work ranking was corrected after a full sweep with public-code
  reading: TAD-GS, PersistGS, VAD-GS, RiGS, Mono4DGS-HDR lead; Proxy-GS and
  OccluGaussian demoted. GauSTAR is the cleanest foil (re-create vs
  hide/reveal). Budget-neutral reassignment is occupied (SharpTimeGS,
  3DGS-MCMC) and is a control, never a contribution.

## 2026-07-25 post-B01 direction (superseded 2026-07-29)

- The corrected 256-slot B01 continuation produced only `+0.048315468 dB`
  global PSNR, `+0.011161912 dB` dynamic-mask PSNR, and `+0.055157407 dB`
  static PSNR. It establishes transaction and optimizer-state stability, not a
  visibility-mechanism win.
- The selected direction was CSVL-VPL v1: a calibrated surface visibility
  ledger coupled to a visibility-conditioned primitive lifecycle. Fixed-count
  reassignment remains a matched-count control and reusable transaction
  substrate.
- The sealed P03 artifact was believed to contain useful calibrated multilayer
  opportunity evidence; Stage 1C subsequently showed it contains zero
  cross-order opportunities, so the "temporal surface association" first stage
  recorded here was executed and no-go'd. See the 2026-07-29 section.

## Project direction

ADAGS studies dynamic Gaussian reconstruction on calibrated N3V cooking scenes.
The approved objective has two independent parts:

1. infer foreground/background order and occluded, hidden, and newly revealed
   surface state from calibrated multiview-temporal depth, appearance, camera
   geometry, and correspondence, with uncertainty and abstention; and
2. couple Gate-A-passing evidence to surface-owned primitive birth, promotion,
   protection, and retirement, while retaining matched-count and generic-extra-
   capacity controls so intermittently visible content is learned while visible
   and reconstructed after reveal without static harm.

The deterministic/frozen geometry-first ledger remains the evidence route, but
CSVL-VPL replaces one-shot fixed-budget reassignment as the lead method.
Reassignment remains a control. Local layered surface memory is the explicit
fallback if a single primitive bank cannot preserve hidden and visible states.
A learned visibility field remains deferred until deterministic Gate A passes.

## Approved experimental discipline

- Development: `cut_roasted_beef` only.
- Locked transfer: `flame_steak` and `sear_steak`.
- New human reference: at least 24 event tracks, target 30-36, at least 20%
  independently double annotated.
- R009 is historical continuity only, never an unbiased holdout or tuning set.
- Gate A has separate engineering-admission and claim-grade transfer tiers.
- Gate B practical targets: event PSNR `+0.20 dB` and LPIPS `-5%`; R009 `3/5`
  and oracle-gap recovery are secondary diagnostics.
- Conditional one-scene envelope: capacity-only and oracle-capacity first; only
  after oracle admission, visibility-only, coupled, and shuffled evidence.
  Maximum five lanes, 6000 iterations, 600k points, 15 hours per lane, about 80
  GPU-hours total.
- Cross-dataset evaluation is deferred until N3V Gate B admission.
- No implementation or compute is authorized while state is
  `objective_approved_awaiting_method_refinement`.

## Corrected prior evidence

R031-R033 did **not** test calibrated multiview-temporal depth. They used only
`cam00`, omitted known cameras, ran independent adjacent two-frame DA3 calls,
normalized depth per frame, compared time at the same pixel without warping,
and inferred edges/change rather than surface order. Their overlap auditor then
scored `cam00` support directly in historical `cam10` crop coordinates without
reprojection. The qualitative audit found coherent coarse depth but brittle
R032 contours, blocky R033 tile expansion, and clear cam00/cam10 viewpoint
differences. Treat the negative as specific to that heuristic and evaluator.

R030 rejects a 400-step, unwarped rectangle-weighted clone/split continuation
on the existing bank. R037 used R020 boxes, not DA3, and rejects fixed opacity
attenuation. Neither tested verified visibility plus hidden-surface capacity.

R013/R015 remain image-space oracle upper bounds. R017 opacity gating, R025
candidate refinement, R027 boundary micro-densification, and R030 oracle-crop
micro-densification failed checkpoint-backed event tests. Extra continuation
alone also failed in R029.

## Closest literature and novelty pressure

- [[papers/zhang2026_vad_gs]] is the closest capacity precedent: voxel
  visibility and calibrated cross-frame MVS initialize missing Gaussian
  geometry under urban LiDAR/box/rigidity assumptions.
- [[papers/gao2026_proxy_gs]] uses proxy occlusion depth for culling and
  surface-guided anchor densification. Visibility-guided capacity is not new in
  general.
- [[papers/zhou2026_4c4d]] applies different learned opacity-decay policies to
  view/time-active and inactive 4D Gaussians. Visibility-conditioned dynamic
  optimization is established.
- [[papers/rai2026_packuv]] uses flow-guided keyframes, layered UV Gaussians,
  projected dynamic labels, and static freezing for temporal consistency and
  disocclusion.
- [[papers/liu2025_occlugaussian]] uses static camera co-visibility for scene
  partitioning and region culling.
- Supporting geometry/representation context:
  [[papers/lin2025_depth_anything_3]], [[papers/zhang2020_vis_mvsnet]],
  [[papers/zhang2024_monst3r]], [[papers/li2021_neural_scene_flow_fields]],
  [[papers/liu2021_neuray]], [[papers/lin2021_deep_3d_mask_volume]],
  [[papers/luiten2023_dynamic_3d_gaussians]],
  [[papers/li2023_spacetime_gaussians]], and [[papers/guo2026_usplat4d]].

Novelty is a working hypothesis, not a fact: calibrated uncertainty-bearing
non-rigid surface order/reveal evidence plus budget-neutral preservation and
reassignment may differ from opacity modulation, proxy-guided growth,
keyframing, region partitioning, and VAD-GS new-geometry initialization. A full
mechanism matrix remains required before any novelty claim.

## Top gaps

- G5: capacity allocation must be matched, budgeted, and dynamic-aware.
- G7: event and static diagnostics are required; global PSNR is insufficient.
- G9: uncertainty and occlusion confidence are missing from current priors.
- G13: occlusion/disocclusion require causal visibility state rather than only
  smooth deformation or lifespan effects.
- G14: transient detail needs identity-preserving promotion/demotion.
- G15: prior confidence must be separated from counterfactual usefulness.

## Failed or weak ideas to preserve

- Single-camera normalized depth-edge/change support with component/tile caps.
- Copying 2D crop coordinates across cameras without reprojection.
- Another posthoc ROI, tile-fill, opacity-gate, or clone/split refinement as the
  primary route.
- Hard static/dynamic conversion, Gaussian blur curriculum, unanchored
  scaffold residuals, broad flow supervision without reliability, and early
  part-basis motion without strong initialization/priors.
- Treating a passing Gate A as proof of Gate B, or blaming depth when an
  oracle-capacity lane also fails.

## Implemented surface and fixed baseline

The base branch has reversible routing, LoRA motion, route0, MotionPriorCache,
optional flow/mask supervision, route0/scaffold residual paths, ordinary
clone/split densification, opacity/size pruning, and dynamic/static diagnostics.
It does not have ordered surface visibility, hidden-surface protection, or
visibility-driven budget-neutral reassignment. The fixed comparison is LoRA
route0 at 6000 iterations and a 600k point ceiling.

## Active chains

- R031-R033 camera-confounded edge support -> calibrated multiview-temporal
  surface ledger -> Gate A.
- R030/R037 existing-bank intervention failures -> oracle-capacity admission ->
  preservation plus budget-neutral reassignment -> Gate B.
- VAD-GS/Proxy-GS/4C4D/PackUV-GS novelty pressure -> mechanism-specific
  ablations, conservative claims, and one dominant contribution.

## Open unknowns

- Which independent depth alignment and correspondence stack is reliable on
  non-rigid hands, food, utensils, flame, and specular surfaces?
- How should event tracks encode occluder, hidden surface, ordering pairs,
  boundaries, and uncertainty without leaking locked transfer evidence?
- Can a correctly reprojected annotated oracle actually make the selected
  budget-neutral capacity operator improve `cut_roasted_beef`?
- Which slots can be retired safely, and how should reassignment be initialized
  without adding a second contribution?
- Does Route 1 remain novel after implementation-level comparison with VAD-GS,
  Proxy-GS, 4C4D, PackUV-GS, OccluGaussian, and newer 2026 work?
