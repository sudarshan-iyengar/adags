# Literature and Code Sweep — 2026-08-08 (durable findings)

Five parallel verified sweeps (N3V leaderboard/protocol; occlusion/
visibility mechanisms; motion representation; densification/capacity;
adjacent fields) plus nine paper deep-dives executed during the STAR-GS
discovery run. Every item below was verified against arXiv abstract/HTML
pages or code repositories during the run — none is search-snippet-only.
Companion pages: [[operations/star-gs-v9-review-history]],
[[operations/rejected-approaches-2026-08]]. Extends (does not replace)
[[operations/sota-code-audit-2026-07-25]].

## N3V leaderboard and protocol (verified numbers, self-reported)

- **FreeTimeGS** (CVPR 2025, 2506.05348): 33.19 dB avg — strongest
  peer-reviewed. Per-scene: cut_roasted_beef 34.52, flame_steak 34.98,
  sear_steak 34.06. Mechanism verified from full text: 4D Gaussians with
  linear velocity + temporal Gaussian opacity; ROMA-match triangulation
  init; **periodic budget-neutral relocation every 100 iterations of
  low-opacity primitives to regions scoring high on 0.5*grad+0.5*opacity**
  (i.e., budgeted reallocation in dynamic GS is OCCUPIED — this corrected
  a wrong "genuine gap" claim from one sweep agent); 4D opacity
  regularization (stop-gradient form, weight 1e-2).
- **SharpTimeGS** (arXiv 2602.02989): 33.57 avg self-reported, arXiv-only
  as of 2026-08-08 (wiki note's CVPR tag unconfirmed by the sweep). No
  code.
- TAD-GS (arXiv 2606.23212): 32.42 global / 24.68 M-PSNR; per-scene cut
  33.99, flame_steak 34.04, sear_steak 33.89; does NOT compare against
  FreeTimeGS/SharpTimeGS (reviewer-visible gap). Multi4D (ECCV 2026):
  32.30. Swift4D (ICLR 2025): 32.23. Ex4DGS (NeurIPS 2024): 32.11. STG
  (CVPR 2024): 32.05 (cut 33.20 per TAD-GS's table). 4DGS-Yang
  (2310.10642): 32.01 (cut 33.85). ADAGS route0 baseline at 6000 it/600k:
  34.37 on cut — inside the competitive band for that scene.
- Protocol consensus: 1352x1014 (half res), cam00 held out, first 300
  frames, LPIPS-Alex where stated (TAD-GS, FreeTimeGS explicit; several
  papers unstated — cross-paper LPIPS not comparable); SSIM vs DSSIM
  conventions mixed; **coffee_martini cam13 has a known temporal desync**
  (dropped from train+test by some papers, undisclosed by others);
  4DGS-Wu used 200 random frames rather than first-300 — a real
  inconsistency. MonoDyGauBench (TMLR 2025) is the protocol critique but
  monocular-scoped; 2605.12437 (CVPR 2026 WS) critiques multiview
  protocol variance via a synthetic benchmark.
- Temporal-stability metrics: no field standard; tOF and tPSNR are
  borrowed ad hoc (MoRel, Detail-Enhanced GS); adopt rather than invent.
  ViDAR ([[papers/nazarczuk2025_vidar]]) quantifies co-visibility-mask
  static bias (mean 26% dynamic pixels; 4% worst-case) and computes
  -D metrics by substituting a dynamic mask into the DyCheck masked-metric
  code — the citable precedent for dynamic-region evaluation.

## Occupied mechanism axes (verified; capsule per axis)

- Presence/duration-weighted densification statistics: TAD-GS (VAD/TAT/
  TOW; drop-in gains on STG/Ex4DGS/Swift4D) and
  [[papers/cho2026_4d_scaffold_gs]] (AAAI 2026; anchor growing weighted
  by temporal opacity with inverse-duration term; own stated open
  failure: 1-2-frame content).
- Error-driven, time-local 4D birth: [[papers/kang2025_cec_4dgs]]
  (SIGGRAPH Asia 2025; single-view rendered-depth backprojection;
  unbudgeted) — closest work to STAR-GS.
- Budgeted relocation: FreeTimeGS (above), 3DGS-MCMC, SharpTimeGS
  stage-2 (fixed-count error+opacity+motion score).
- Parent-free error birth (static or rigid): STG guided sampling (self
  depth), ConeGS (2511.06810, iNGP-proxy depth cones), AdpSplit
  (2605.06876), VAD-GS (voxel z-buffer + patch-match MVS; rigid urban).
- Densification theory (static scenes): SteepGS
  ([[papers/wang2025_steepgs]], CVPR 2025 — saddle-point escape; loses
  0.3 dB on Mip-NeRF360 for compactness), GDAGS
  ([[papers/zhou2026_gdags]], ICLR 2026 — gradient-coherence ratio),
  Structure-Aware Densification
  ([[papers/lyu2026_structure_aware_densification]], SIGGRAPH 2026 —
  aliasing-vs-misplacement). Cite, don't re-derive.
- Visibility-gated gradient/token masking: WildRayZer
  ([[papers/chen2026_wildrayzer]], CVPR 2026 Highlight — analysis-by-
  synthesis pseudo-masks, token dropping; transients EXCLUDED not
  reconstructed); PackUV/MAPo/Ex4DGS external-mask freezing; U-4DGS
  (2602.06343, learned pixel uncertainty, monocular humans).
- Object permanence through occlusion: [[papers/mazur2026_4dpm]]
  (CVPR 2026 Oral — SE(3) primitive pose graph, parent-child motion
  extrapolation, monocular piecewise-rigid, NOT Gaussian-splatting);
  PersistGS (CVPRW only — differentiable physics, rigid);
  [[papers/zheng2025_gaustar]] (CVPR 2025, code — mesh-bound tracking,
  unbind/re-create).
- Sparse/selective optimizer semantics: gsplat sparse_adam,
  Taming-3DGS selective Adam (throughput, static) — a control family,
  closed as a contribution axis by
  [[operations/rejected-approaches-2026-08]].
- Temporal anti-aliasing: Alias-free 4DGS (2511.18367). Lifespan shaping:
  SharpTimeGS flat-top. Time warps: TAD-GS TOW. Motion bases/splines:
  SE(3) B-spline GS (CVPR 2026), TRiGS, WebSpline, shared trajectory
  bases (2508.07182); LoRA-of-motion possibly occupied by 4DSurf IMT
  (2603.28064, unconfirmed detail). Hand-object fast motion:
  Interaction-Aware 4DGS (2511.14540). Dynamic-GS overfitting diagnosis:
  "Incoherent Deformation, Not Capacity" (2604.16747).
- Feedforward/prior stacks (context, not method components): VGGT-Omega,
  PAGE-4D (ICLR 2026, code), StreamVGGT, NoPo4D, TrackerSplat (SIGGRAPH
  Asia 2025 — track-triangulated INITIALIZATION), GSFixer (ICML 2026),
  VidSplat (SIGGRAPH 2026), SpatialTrackerV2, TAPIP3D, ViGeo, StableDPT,
  Amodal SAM.

## Verified dead ends (searches that found no occupant)

Layered/occlusion-ordered dynamic GS representation; disocclusion-event-
specific quality benchmark; occlusion-protection lifecycle for per-scene
multiview optimization; residual/error-volume carving for birth (beyond
CEC-4DGS's depth-based variant and classical silhouette carving — the
slice claimed by [[operations/star-gs-v9-method]]).

## Deep-dived paper pages added/updated this run

[[papers/kang2025_cec_4dgs]] (mechanism-verified),
[[papers/cho2026_4d_scaffold_gs]], [[papers/wang2025_steepgs]],
[[papers/zhou2026_gdags]], [[papers/lyu2026_structure_aware_densification]],
[[papers/mazur2026_4dpm]], [[papers/chen2026_wildrayzer]],
[[papers/nazarczuk2025_vidar]], [[papers/zheng2025_gaustar]] (upgraded to
deep-dived, code inspected), [[papers/zoomers2026_nvgs]] (upgraded to
deep-dived).
