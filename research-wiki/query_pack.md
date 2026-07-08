# Query Pack

Compressed project memory for ideation. Updated 2026-06-30 after latest-literature mapping and high-rigor idea tournament.

## Project Direction

ADAGS studies dynamic Gaussian Splatting for N3V cooking scenes. Current direction: keep reversible LoRA route0 as the stable base, then target fast-motion blur and static/dynamic ghosting using dynamic masks, reliable flow/track priors, scaffold or part/expert residual motion, and motion-aware densification. The 2026 literature narrows the claim: ADAGS needs to justify its route0/residual split against rigid-aware methods, its sharpness story against frequency-aware/Gabor representations, and its speed story against feedforward 4D reconstruction.

## Selected Direction After Tournament

Primary direction: [[ideas/self-calibrated-prior-reliability-field]].

Paper package:
- [[ideas/boundary-aware-static-anchor-negative-space]] is the deterministic baseline and static-leakage safeguard.
- [[ideas/rendered-flow-gated-supervision]] is the first rendered/track-flow instantiation.
- [[ideas/adags-failure-atlas-mechanism-screen]] is the diagnostic protocol and backup direction.

Hard gate: reliability must beat LoRA route0 on dynamic-region quality while preserving static quality and matched realized budget. Kill the direction if reliability selects mostly easy pixels, suppresses hard boundaries, degrades static background, or cannot improve dynamic metrics over route0/render-gate.

## Problem-First Redo Addendum

Generated 2026-06-30 after the user explicitly rejected over-constraining ideation around ADAGS. The redo treats ADAGS as an experimental sandbox rather than the final method boundary.

Updated stance:

- High-upside lead remains [[ideas/event-causal-visibility-gaussians]], but current concrete forms R017, R025, and R027 are negative.
- Other SOTA-track seeds: [[ideas/identity-conserving-detail-carriers]], [[ideas/frequency-adaptive-temporal-support]], [[ideas/counterfactual-prior-usefulness-routing]].
- Safe fallback remains [[ideas/self-calibrated-prior-reliability-field]] plus [[ideas/adags-failure-atlas-mechanism-screen]] and [[ideas/boundary-aware-static-anchor-negative-space]].
- Do not pitch reliability-gated flow as the main intellectual bet unless the high-upside representation ideas fail feasibility or novelty checks.

New field-level gaps: G13 visibility events, G14 identity-conserving detail promotion, G15 counterfactual prior usefulness.

Event-crop memory: R013/R015 remain oracle upper-bound evidence only. R017 actual opacity gating, R025 non-oracle candidate refinement, and R027 non-oracle boundary-gated micro-densification all failed the frozen R009 real-window gate. R027 was closest directionally but still only had 2/5 strict all-baseline PSNR+L1 wins and less than 1% oracle recovery. R028 posthoc support audit found R026 boundary support had essentially zero frozen-crop coverage, so R027 mainly falsifies that support+training recipe; R025 is stronger evidence against current posthoc local refinement because R020 candidate support overlapped 4/5 windows.

Important new literature memory: [[papers/zhao2026_ground4d]] adds geometry-consistency pressure and supports the idea that photometric dynamic GS needs stronger geometry/prior routing.

## Implemented Surface

- Reversible soft routing, LoRA motion, older hard/static gate paths, W&B logging, Slurm launchers, and fixed-budget W&B analysis.
- Current branch adds `MotionPriorCache`, dynamic ROI loss, static exclusion loss, optional rendered-flow/track-flow supervision, scaffold residual motion, motion-aware densification, and dynamic/static diagnostics.
- Remote Codex branches add LoRA flow/mask variants: phase-2 LoRA, file-mask residual flow, core-mask ramp, render-gated flow, soft-border, boundary-ring, and static-anchor lanes.

## Literature Map

- Baselines: 4D-GS, Deformable 3DGS, SWinGS, Compact Dynamic 3DGS, Fully Explicit DGS, Grid4D.
- Priors/tracks/scaffolds: MoSca, Shape of Motion, Prior-Enhanced GS.
- Flow/motion guidance: MotionGS, SplatFlow, PaMoSplat.
- Specialized motion/capacity: HiCoM, MAPo, SharpTimeGS, SpeeDe3DGS, SE(3) B-spline continuous motion, MoE-GS.
- Latest omitted 4D cluster now ingested: RiGS, AdaGaR, D4RT, MotionScale, Hybrid 3D-4DGS, Disentangled4DGS, Mono4DGS-HDR, 4C4D, Flow4DGS-SLAM, and Multi4D.
- Problem-first redo added Ground4D as geometry-consistency pressure.
- Implication of latest cluster: "add scaffold/flow/capacity" is not enough; the defensible axis is reliable specialization with diagnostics: rigid/coherent base vs local residual, dynamic-frequency/detail preservation, fixed-budget allocation, and static leakage control.
- Evaluation and reliability: MonoDyGauBench, USplat4D.

## Top Gaps

- G1: dynamic-region sharpness needs direct objectives and metrics.
- G3: long-range tracks/depth priors are now table stakes; ADAGS hook is inactive.
- G4: scaffold residual motion is crowded by MoSca and Prior-Enhanced GS.
- G6: single global motion models are a known weakness; specialize without losing route0 stability.
- G7: claim needs dynamic diagnostics and qualitative panels, not only PSNR.
- G8: flow supervision needs reliability gating, not broad application.
- G9: uncertainty/occlusion confidence is missing from ADAGS masks and flow losses.
- G11: representation frequency/detail is now a core sharpness axis.
- G12: feedforward 4D reconstruction raises the baseline for speed/generalization claims.
- G13: occlusion/disocclusion need visibility-event treatment rather than smooth deformation everywhere.
- G14: high-frequency transient detail needs identity-conserving promotion/demotion rules.
- G15: prior confidence must be separated from counterfactual prior usefulness.

## W&B Evidence

Project: `models-ku-leuven/adags`.

- 126 total runs; 126 finished; no active/crashed/failed/killed runs found.
- Fixed-budget cohort: 36 finished runs. Mean PSNR: `lora_route0` 33.9558/34.4265/34.2963; `scaffold_lora_route0_noreg` 33.9639/33.8198/34.5127; `scaffold_lora_route0_dyn` 32.9705/34.0401/34.0693.
- Mechanism screen: 15 finished 600k runs. Mean PSNR: `lora_route0` 34.2596, `lora_route0_dyn` 33.7631, `scaffold_lora_route0_noreg` 33.8535, `scaffold_lora_route0_reg` 33.9445, `scaffold_lora_route0_dyn` 33.1592.
- Eval-6000 flow/phase-2 cohort: render-gated flow is stronger than naive flow lanes but needs controlled dynamic diagnostics.
- No synced runs found with tag `diagnostics:dynamic_static`.

## Failed Or Weak Ideas To Preserve

- Gaussian blur curriculum alone.
- Hard static/dynamic conversion as main method.
- Early part-basis motion without route0 initialization or strong priors.
- Scaffold without active correspondence.
- Naive broad flow supervision without reliability gating.
- Unanchored scaffold residual motion as a paper lead.

## Active Idea Seeds

- [[ideas/self-calibrated-prior-reliability-field]]
- [[ideas/boundary-aware-static-anchor-negative-space]]
- [[ideas/adags-failure-atlas-mechanism-screen]]
- [[ideas/dynamic-mask-static-exclusion]]
- [[ideas/track-prior-scaffold-motion]]
- [[ideas/rendered-flow-gated-supervision]]
- [[ideas/motion-aware-densification-budget]]
- [[ideas/part-aware-reversible-routing]]
- [[ideas/dynamic-region-diagnostic-benchmark]]
- [[ideas/route0-residual-specialist-motion]]
- [[ideas/confidence-aware-prior-gating]]
- [[ideas/dynamic-frequency-detail-budget]]
- [[ideas/competitive-route0-residual-budget-allocator]]
- [[ideas/reliable-temporal-detail-transport]]
- [[ideas/adags-paper-direction-discovery-20260630]]
- [[ideas/event-causal-visibility-gaussians]]
- [[ideas/identity-conserving-detail-carriers]]
- [[ideas/frequency-adaptive-temporal-support]]
- [[ideas/counterfactual-prior-usefulness-routing]]

## Latest Idea-Creator Ranking

Generated 2026-06-30. Top recommendation: reliability-gated rendered flow plus dynamic-region diagnostics. Higher-novelty backup: route0-coherent plus residual-specialist motion, only if diagnostics show localized route0 failures. Conditional supports: confidence-aware prior gating and dynamic frequency/detail under fixed realized budgets. Do not pitch unanchored scaffold, naive broad flow, hard static conversion, blur curriculum, or early part-basis as lead methods.

## Latest Tournament Ranking

High-rigor tournament completed 2026-06-30 with three independent reviewer roles and rebuttal rulings. Result: C11/self-calibrated prior reliability field wins conditionally; C15/failure atlas is backup. I1/render-gated flow and C14/boundary-static-anchor are components, not standalone leads. I3/route0 residual specialization and C13/competitive budget allocation remain backup seeds only. I5/detail metrics, I2/diagnostics, and fixed-budget audits are required evidence axes, not standalone paper leads.

## Open Unknowns

- Are dynamic masks good enough, or are residual masks leaking too much background?
- Does track-flow loss activate and produce sensible rendered flow when `enable_rendered_flow` is true?
- Does scaffold coefficient/basis norm grow, or does the residual path stay unused?
- Can ADAGS beat LoRA route0 on dynamic-region metrics while matching realized point budget?
- Is the defensible novelty: training-only priors, flow reliability gating, reversible route0-specialized residuals, or dynamic-region diagnostics?
- Can route0 be interpreted as coherent rigid-ish motion while residual/scaffold paths handle local non-rigid failures, or is that story unsupported by current metrics?
- Do dynamic edge/detail metrics capture the high-frequency failures emphasized by AdaGaR and Multi4D?
- What is the right baseline class for the final claim: per-scene monocular GS, sparse multi-view 4DGS, or feedforward 4D reconstruction?

## Literature Watchlist

Unresolved candidate names from the June 2026 search pass: SPIN-4DGS, FAGS/Frequency-Aware Dynamic Gaussian Splatting, and GP-4DGS. Do not ingest these until title, authors, venue, and a stable paper/project URL are verified.
