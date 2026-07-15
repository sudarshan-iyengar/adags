# Query Pack

Compressed project memory for ideation. Updated 2026-07-15 after canonicalizing
[[objectives/depth-visibility-capacity-v1]].

## Project direction

ADAGS studies dynamic Gaussian reconstruction on calibrated N3V cooking scenes.
The approved objective has two independent parts:

1. infer foreground/background order and occluded, hidden, and newly revealed
   surface state from calibrated multiview-temporal depth, appearance, camera
   geometry, and correspondence, with uncertainty and abstention; and
2. couple Gate-A-passing evidence to budget-neutral preservation plus
   reassignment/reinitialization of Gaussian capacity so intermittently visible
   content is learned while visible and reconstructed after reveal without
   static harm.

Route 1, a deterministic/frozen geometry-first visibility ledger plus one
capacity component, is approved as lead. Route 3, explicit layered surface
memory, is fallback. Route 2, a learned visibility field, is permitted only if
deterministic Gate A passes but remains incomplete.

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
