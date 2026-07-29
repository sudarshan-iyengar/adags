# Phase 9 CSVL-VPL v2 Direction

Date: 2026-07-29
Status: approved by the user as the project direction (this session's written
proposal and its Section 8 actions 1-3). Phase 0 execution authorized; Phase 1+
requires a PHASE0_GO decision and renewed user approval.
Supersedes: [[operations/phase9-post-b01-csvl-vpl-direction]] as the lead
method scope. Preserves its controls, accounting, and data discipline.
Preserves: [[operations/phase9-csvl-vpl-stage1-result]],
[[operations/phase9-csvl-vpl-stage1b-result]],
[[operations/phase9-csvl-vpl-stage1c-result]] as binding negative evidence.
Parent objective: [[objectives/depth-visibility-capacity-v1]] (Section 19
addendum records this refinement).

## Why the v1 scope is retired

Stage 1 of CSVL-VPL v1 was implemented and executed on 2026-07-26 and returned
three consecutive no-gos (jobs 50246056, 50250624, 50321533):

1. `STAGE1_NO_GO`: the association could not distinguish valid flow from
   camera-swapped flow (which scored better: confidence 0.922 vs 0.889); 46
   reappearances, zero reveals.
2. `STAGE1B_CONTROL_OR_BINDING_DEFECT`: flow is non-causal (no-flow replay
   keeps 98.565% of selected edges); P03 uncertainty and depth/order carry
   zero score weight — the association is arithmetically a geometry-proximity
   matcher.
3. `STAGE1C_NO_INFORMATIVE_INTERVAL`: all 19 pre-specified windows contain
   zero front/rear cross-order candidates. The sealed evidence layer contains
   no occlusion-order transitions on the only scene it covers.

Root cause of the empty evidence layer (X03 waterfall,
[[operations/phase9-overnight-report]]): P03 represents occlusion as
multi-camera-confirmed multilayer bin occupancy; 93.4% of 3,067,491 projected
bins were rejected for insufficient camera co-support. On a frontal rig, an
occluded surface is by construction visible in other cameras, so two-layer
multi-camera co-support in one bin is structurally rare. Occlusion here is a
per-(surface, camera, time) reprojection relation, not a per-bin stack.

## v2 method scope

The two-part scientific frame is retained: calibrated visibility evidence
coupled to a visibility-conditioned primitive lifecycle, with Gate A / Gate B
separation, causal controls, resource accounting, and static no-harm.

### Evidence redesign (CSVL-R): primitive-centric, no external track graph

The primitives themselves are the persistent identities (stable slot IDs from
the B01 transaction substrate). Three evidence channels:

- **E1 (external per-primitive visibility)**: project each primitive into each
  training camera at each frame; compare its depth against DA3-derived
  first-surface depth with sigma-aware margins (reuse
  `depth_visibility/surfaces.py` `dense_depth_order` semantics); emit
  visible / occluded / uncertain with order-risk and abstention. Occlusion
  signal under this representation is dense, not 158-bins-sparse.
- **E1-int (internal rendered visibility)**: per-primitive per-view
  transmittance/contribution from the rasterizer. Exact and model-consistent;
  drives protection and exposure normalization; requires no Gate A because it
  never claims to discover missing content.
- **E2 (model-deficit detection)**: observed-but-unmodeled regions from
  render-vs-evidence depth and accumulated-opacity disagreement (VAD-GS
  trigger family, adapted, without patch-match MVS); multiview seed
  back-projection for birth targeting.

### Lifecycle redesign (VPL-R): from scratch, never touching rendered opacity

Runs 0-6000 through the densification window. Deterministic policies only;
zero new trainable components. Limbs: (1) protection = masked update freezing
plus split/donor vetoes while rendered visibility is ~zero (never opacity
clamping — R017/R037 twice-refuted); (2) occlusion-aware exposure-normalized
densification (TAD-GS-adapted, transmittance-aware); (3) E2-triggered
multiview surfel birth, budget-neutral via the B01 transaction; (4) hysteretic
retirement with protection veto; (5) reveal-time unfreeze.

An oracle-evidence capacity lane (the B02 obligation, restored) runs in
parallel: if oracle evidence plus the lifecycle cannot beat route0, the
mechanism family is retired regardless of Gate A.

### Binding design constraints from the failure record

C-1 no rendered-opacity manipulation from external evidence; C-2 separate
occluder from hidden surface; C-3 supply revealed content, not only remove
occluders; C-4 no posthoc/late-checkpoint interventions for mechanism claims;
C-5 oracle-capacity attribution before evidence investment; C-6 opportunity
census before evidence machinery; C-7 matched real-data paired controls with
separation margins as Go criteria; C-8 pre-registered effect-size floors;
C-9 learned LPIPS, >=2 seeds for claim-grade, >5 events.

## Scene allocation (decided)

- Development: `cut_roasted_beef` + `cook_spinach` (second scene requires DA3
  sidecars and masks; preregistered cost).
- Locked claim-grade transfer: `flame_steak` + `sear_steak` (unchanged; single
  post-freeze evaluation; annotation stratification preserved).
- Stress tier: `coffee_martini` + `flame_salmon_1` — evaluated only at
  preregistered checkpoints, every reveal logged, never tuned on.
- Final admitted comparisons: all six scenes (contract Section 11); six-scene
  route0 600k/9000 baselines completed 2026-07-22.

## Capacity policy (decided)

Post-B01 softened form with teeth: the causal claim ("visibility-conditioning
improves allocation") requires beating (i) the capacity-matched generic
control, (ii) generic-extra-capacity at equal resources, and (iii) surviving
shuffled/misaligned evidence. Headline reconstruction numbers may use capacity
deltas only with the full resource ledger and Pareto reporting. The +/-2%
tolerance applies to designated matched lanes. Budget-neutral reassignment is
a control/substrate, never a claimed contribution (SharpTimeGS fixed-count
stage-2 and 3DGS-MCMC relocation occupy it).

## Phase plan

- **Phase 0 (authorized now)**: wiki repair; preregistered primitive-centric
  evidence-opportunity census on `cut_roasted_beef` with frozen numerical
  floors committed before execution; annotation contract draft. Ends at a
  formal PHASE0_GO / PHASE0_NO_GO decision.
- Phase 1: E1/E2 production + validation (held-out-camera consistency, matched
  paired controls with margins); DA3 sidecars for cook_spinach; annotation
  started. Requires PHASE0_GO and user approval.
- Phase 2: mechanism falsification — from-scratch 6000-iter lanes on cut
  (route0 rerun, protection-only, exposure-only, presence-VAD control,
  oracle-evidence lifecycle). Kill: oracle lane fails -> retire the family.
- Phase 3: coupled causal admission vs the ten-control matrix, 2 dev scenes,
  2 seeds. Gate B targets unchanged (+0.20 dB event PSNR, -5% real LPIPS,
  static no-harm).
- Phase 4: freeze, annotate, locked-pair single evaluation, six-scene sweep,
  stress-tier reveal.

Estimated 400-600 GPU-hours total plus annotation; every phase preregistered;
no phase starts before its predecessor's floor is met.

## Novelty positioning (summary)

Closest works, re-ranked after full sweep + public-code reading:
[[papers/sandu2026_temporally_aware_densification]] (top threat; must be a
reimplemented baseline), [[papers/ramlal2026_persistgs]],
[[papers/zhang2026_vad_gs]], [[papers/wu2026_rigs]],
[[papers/liu2025_mono4dgs_hdr]], [[papers/zhou2026_4c4d]],
[[papers/zheng2025_gaustar]] (cleanest foil: re-create vs hide/reveal),
[[papers/liao2026_sharptimegs]] (occupies fixed-count densification).
[[papers/gao2026_proxy_gs]] and [[papers/liu2025_occlugaussian]] are demoted
to tier 3 (static-scene culling). The defensible claim is the conjunction:
calibrated + abstaining + non-rigid surface-order evidence + persistent
identity across the hidden interval + preservation-not-recreation +
budget-accounted lifecycle + annotated disocclusion-event evaluation. Every
individual limb is published; the borrowed-mechanism attribution ledger in the
session proposal is mandatory in any paper draft. The annotated
disocclusion-event benchmark + causal-control matrix is itself a defensible
contribution and insurance against a modest effect size.

## Honest odds (recorded at approval time)

E1 abundance on N3V ~75%; protection+exposure beats route0 events at matched
budget ~40-50%; full coupled Gate B pass ~30-35%; publishable if Gate B passes
~65%. If the Phase 2 oracle lane fails, the recorded intent is to publish the
negative plus the benchmark rather than iterate a twelfth variant.
