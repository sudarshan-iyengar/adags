# Query Pack

Compressed project memory for ideation. Updated 2026-07-29 after the Stage
1/1B/1C no-gos and the approved [[operations/phase9-csvl-vpl-v2-direction]].

## 2026-07-29 CSVL-VPL v2 direction (current)

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
