# SOTA-Code Audit Before Gaussian Overhaul

Date: 2026-07-25
Status: complete decision memo
Decision: targeted mechanism

## Intent

This memo implements the "SOTA-Code Synthesis Before Gaussian Overhaul" step. The goal is not to run new experiments. The goal is to decide whether current ADAGS evidence justifies a broad Gaussian representation overhaul, one constrained mechanism, or no new mechanism.

The existing route0 LoRA failure modes remain accepted baseline evidence: route0 LoRA is the comparator line, not a space for more blind sweeps. Human annotations and evaluator labels remain evaluation/reference only and must not enter training.

## Current ADAGS Surface

The current checkout already has the relevant local implementation hooks:

- `main.py`: Slice-B capacity modes and transaction/ledger entry points (`normalized_slice_b_capacity_mode`, `load_slice_b_capacity_sidecar`, `maybe_apply_slice_b_capacity_transaction`, `write_slice_b_capacity_ledger`).
- `main.py`: eval-time diagnostics for `test/dynamic_mask_psnr` and `test/static_region_psnr`, with `MotionPriorCache`.
- `scene/gaussian_model.py`: LoRA motion tensors, optimizer-state handling, densification copies, and stable ID/capacity-related state.

That means the audit did not justify a fresh trainer rewrite before the Phase9 visibility/capacity path. The safe implementation action is to keep the existing Slice A -> Slice B funnel and add stronger prior-art controls in the memo/table.

## Audited Code Sources

Machine-readable audit tables:

- `research-wiki/operations/sota-code-audit-2026-07-25-candidates.csv`
- `research-wiki/operations/sota-code-audit-2026-07-25-mechanism-scores.csv`

Local convenience copies:

- `analysis/sota_code_audit_20260725/candidates.csv`
- `analysis/sota_code_audit_20260725/mechanism_scores.csv`

Representative code audits covered:

- 3D-4DGS: static 3D plus dynamic 4D bank, temporal-scale dynamic-to-static conversion, separate static densification.
- 4C4D: visibility/time-active opacity decay and depth/flow renderer outputs.
- USplat4D: uncertainty graph, key/non-key interpolation, MoSca-based dynamic Gaussian scaffold.
- VAD-GS: voxel visibility, vacancy detection, depth propagation, and actor Gaussian growth.
- Proxy-GS: proxy/depth visible filtering and high-loss anchor growth.
- MoSca, Shape of Motion, RiGS: scaffold or motion-basis static/dynamic/transient bank families.
- AdaGaR, PaMoSplat, MoE-GS: high-frequency, part-aware, or expert temporal capacity.
- DepthRegularizedGS and DepthSplat: depth-supervised or depth-to-Gaussian geometry priors.
- Ground4D and code-limited repos: adjacent context or availability limitations.

License and reuse risk is uneven. Several repos are permissive MIT/Apache, but Proxy-GS, DepthRegularizedGS, and MoE-GS contain Gaussian-Splatting noncommercial-license material. AdaGaR, PaMoSplat, RiGS, V4D, MEGA, OccluGaussian, PackUV, and Prior-Enhanced GS did not provide a directly reusable license/code surface in the audited clone/page state. Treat those as design evidence only unless rechecked.

## Decision

Selected decision: `targeted mechanism`.

Do not launch a major Gaussian representation overhaul now. Do not run more route0 LoRA sweeps. Continue with the current Phase9 visibility-capacity path:

1. Complete deterministic/eval-only CSVL Slice A evidence.
2. Only if Gate A passes, run Slice B point-neutral surface ownership/reassignment.
3. Keep the matched point budget unless the claim explicitly changes to quality-capacity tradeoff.
4. Keep cam00 RGB, human labels, and evaluator masks out of training.

## Why Not `overhaul`

The audited SOTA code supports almost every broad idea in the handoff, but mostly in forms that would weaken ADAGS novelty if copied directly:

- Static/dynamic banks and layered/transient representations already appear in 3D-4DGS, Shape of Motion, and RiGS.
- Visibility-aware updates already appear in 4C4D, VAD-GS, and Proxy-GS.
- Depth/proxy geometry growth appears in VAD-GS, Proxy-GS, DepthRegularizedGS, and DepthSplat.
- Motion scaffolds, uncertainty graphs, and motion bases appear in MoSca, USplat4D, Shape of Motion, and RiGS.
- Local temporal/high-frequency capacity appears in AdaGaR, PaMoSplat, and MoE-GS.

What did not appear in the audited code is the exact ADAGS claim shape: a target-independent, non-human-label training path that builds calibrated surface visibility/ownership evidence and then performs a point-neutral reassignment/preservation transaction under a matched Gaussian budget. That gap is narrower than a full overhaul and more defensible than another capacity/loss sweep.

## Ranked Mechanisms

1. CSVL-ISR point-neutral surface ownership and reassignment.
   - Decision: selected targeted mechanism.
   - Claim shape: visibility-calibrated dynamic surface ownership reduces motion blur and static leakage under matched Gaussian budget.
   - Reason: closest SOTA methods either grow geometry or apply visibility/uncertainty without ADAGS' fixed-budget event-ownership transaction.

2. Layered/transient surface memory with explicit front/rear ownership.
   - Decision: fallback overhaul only.
   - Reason: RiGS and 3D-4DGS make layered/static/dynamic banks crowded prior art. This is only worth doing if Gate A passes and single-bank Slice B fails.

3. Uncertainty-gated dynamic motion anchors.
   - Decision: diagnostic/refinement.
   - Reason: USplat4D supports uncertainty calibration, but making it the first implementation would turn the method into another learned dynamic field before deterministic evidence is proven.

4. High-frequency local temporal capacity.
   - Decision: fallback mechanism.
   - Reason: AdaGaR, PaMoSplat, MoE-GS, and Shape of Motion support this family, but it risks repeating the failed "more capacity" story unless visibility diagnostics show blur remains after ownership correction.

5. Proxy/depth-guided net growth.
   - Decision: not primary.
   - Reason: this is mature and plausible, but changes the claim to a quality-capacity tradeoff.

6. Global depth/flow regularization.
   - Decision: no viable novelty alone.
   - Reason: useful as selective evidence/control only; too much prior art and too close to "added more losses."

## Implementation Implications

The next implementation work should stay within the existing Phase9 contract:

- Preserve route0 LoRA 600k as the baseline/comparator.
- Use existing Slice-B capacity hooks rather than replacing the Gaussian model wholesale.
- Add SOTA-derived controls to any experiment memo: visibility-only, capacity-only, shuffled/generic sidecar, and full CSVL-ISR.
- Treat VAD-GS/Proxy-GS style birth/growth as a separate quality-capacity experiment, not a matched-budget claim.
- Treat uncertainty and depth/flow as calibration/evidence fields, not global training supervision.
- If a major layered/transient overhaul becomes necessary, require a new method memo and approval because it changes the representation claim and optimizer-state risk.

## Verification Boundary

This memo did not submit SLURM jobs, run training, or alter trainer/model code. It is a code/paper synthesis deliverable. Reconstruction validation remains gated as:

- cut-scene 6000 run before all-scene expansion;
- six N3V scenes at 6000 before larger claims;
- frozen annotation/evaluator alignment after labels are complete;
- small external sanity check before top-tier positioning.
