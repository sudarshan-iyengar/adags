# Phase 9 Post-B01 Direction: CSVL-VPL

Date: 2026-07-25
Status: SUPERSEDED 2026-07-29 by [[operations/phase9-csvl-vpl-v2-direction]].
The Stage 1 approved below was implemented and executed on 2026-07-26 and
returned three consecutive no-gos: [[operations/phase9-csvl-vpl-stage1-result]]
(`STAGE1_NO_GO`), [[operations/phase9-csvl-vpl-stage1b-result]]
(`STAGE1B_CONTROL_OR_BINDING_DEFECT`), and
[[operations/phase9-csvl-vpl-stage1c-result]]
(`STAGE1C_NO_INFORMATIVE_INTERVAL`). The P03 multilayer-bin evidence route and
the external temporal-association-first sequencing are retired; the controls,
accounting, and data discipline below remain reusable. Known defect of this
page: its Later Causal Requirements list omits the oracle-capacity attribution
lane required by objective Section 8 and the Phase 8A review; v2 restores it.
Supersedes as lead mechanism: CSVL-ISR one-shot point-neutral reassignment
Preserves: [[operations/phase9-slice-b-v13-b01-decision]] as operator evidence
Parent objective: [[objectives/depth-visibility-capacity-v1]]

## Decision

Continue with a calibrated visibility ledger, but change its representation
role. The selected direction is **CSVL-VPL: Calibrated Surface Visibility
Ledger with a Visibility-Conditioned Primitive Lifecycle**.

The method hypothesis is that a persistent surface identity, foreground/rear
order, and uncertainty-bearing visibility state should control one coherent
primitive lifecycle:

- initialize or promote surface-owned primitives from calibrated multiview
  observations while the surface is first-visible;
- normalize learning by usable visible exposure;
- protect owned primitives through evidence-supported occlusion;
- retire or release them only after hysteretic low-utility evidence.

Discovery may use additional capacity. It must report point, parameter, memory,
compute, and runtime changes and must compare against both matched-capacity and
generic-extra-capacity controls. A matched-budget claim is made only when final
and integrated resource budgets are actually matched.

## Why The Lead Changed

The corrected B01 continuation executed one event-blind `K=256` in-place
transaction at 562,147 points. It gained `+0.048315 dB` global PSNR and
`+0.011162 dB` dynamic-mask PSNR at iteration 5250, with slightly worse static
ghost score. This establishes that the transaction is executable and that
optimizer-state surgery plus stable slot accounting can be reused. It does not
establish visibility benefit, missing-surface initialization, or persistent
surface identity.

The B01 operator is therefore retained as a matched-count control and
implementation substrate, not the lead method.

## Current Ledger Boundary

The sealed P03 `cut_roasted_beef` artifact contains 158 ordered multilayer bins
over 123/300 frames and keeps cam00 RGB unopened. It explicitly records
`temporal_identity_status=not_propagated_in_p03_v7`. Its temporal changes are
target-bin occupancy proxies.

P03 proves nonzero calibrated ordered-depth opportunity. It does not yet
provide persistent surface identity, visibility transitions, hidden-surface
ownership, or a trainable sidecar.

## External Mechanism Reconciliation

- TAD-GS establishes temporal-presence-normalized densification and lifespan
  allocation. Adapt exposure normalization; use temporal-presence densification
  as a control.
- VAD-GS and Proxy-GS establish visibility/depth-guided missing-geometry growth.
  Adapt multiview surface initialization, not their LiDAR/box/proxy/anchor
  pipelines.
- RiGS and Hybrid 3D-4DGS establish lifecycle and bank conversion. Adapt
  explicit state transitions and accounting, not a wholesale multi-bank port.
- 4C4D establishes visibility-conditioned opacity optimization. Use opacity-only
  as a control.
- USplat4D establishes uncertainty-selected propagation. Retain deterministic
  risk and abstention; do not revive the scaffold route.
- PackUV establishes layering/disocclusion precedent. Keep local layered memory
  as the representation fallback, not the first implementation.

Generic visibility densification, opacity modulation, proxy-guided growth,
multi-bank promotion, and layering are crowded. The remaining hypothesis must
be tested as a conjunction: calibrated nonrigid foreground/rear identity and
abstention drive a controlled surface-owned lifecycle that improves difficult
reconstruction beyond generic capacity.

## Route Disposition

1. **Selected:** CSVL-VPL surface-owned lifecycle.
2. **Control:** one-shot point-neutral CSVL-ISR/v13 reassignment.
3. **Control:** temporal-presence visibility-aware densification.
4. **Fallback:** local explicit front/rear surface memory if a Gate-A-passing
   ledger plus a strong-reference lifecycle test shows a single bank cannot
   preserve supported rear content.

Do not implement a learned visibility field, global depth/flow loss, opacity
gate, full scaffold/uncertainty graph, or broad representation rewrite now.

## First Approved Stage

Implement only the label-free surface-identity ledger:

- consume the existing sealed P01 DA3, P02 flow, and P03 ordered-depth
  artifacts;
- wire deterministic temporal surface association, split/merge abstention,
  bounded dormancy/reappearance, per-camera order/state/risk, and complete
  ancestry into a new immutable ledger version;
- add controlled hide/reveal, corrupted-flow, camera-swap, z-sign, temporal
  offset, deterministic replay, and provenance fixtures;
- produce one bounded `cut_roasted_beef` engineering diagnostic.

This stage does not modify `main.py`, `scene/gaussian_model.py`, the rasterizer,
optimizer, densification, pruning, routing, or training configs.

One CPU Slurm diagnostic of at most two hours is approved after static tests. It
must reuse sealed sidecars and may not rerun DA3, train, render evaluation
images, open cam00 RGB, consume annotations/evaluator masks, submit multiple
lanes, or write W&B.

## Stage 1 Go/No-Go

Go when controlled identity/order/state fixtures pass, scientific payload
hashes repeat exactly, every accepted record has complete target-free ancestry,
split/merge never propagates identity, and the real diagnostic emits nonzero
multi-frame rear-surface tracks with explicit uncertainty.

No-go when the output remains an occupancy proxy, controlled hide/reveal
identity fails, real rear support has zero multi-frame yield, or correctness
requires target RGB, labels, or outcome-tuned thresholds. Redirect the evidence
or association representation before touching the trainer.

Stage 1 remains engineering evidence. Genuine independent human evaluation is
still required for Gate A, and Gate A engineering admission remains required
before representation training.

## Later Causal Requirements

Before attributing reconstruction gains to visibility, compare:

- matched route0;
- null optimizer reset;
- v13 generic point-neutral reassignment;
- generic extra capacity with identical K/schedule/initializer;
- generic multiview surface initialization without identity;
- visibility-only weighting;
- temporal-presence densification;
- preserve-only and birth-only lifecycle ablations;
- shuffled/misaligned ledger;
- full CSVL-VPL.

Record final/peak/integrated points, trainable parameters, optimizer bytes,
allocated/reserved VRAM, host memory, wall time, GPU-hours, throughput, active
splats, render FPS, and checkpoint/storage cost. Report "better reconstruction,"
"better allocation at the same budget," and "quality-capacity tradeoff" as
separate conclusions.

## Detailed Artifacts

The full verification, inventory, external evidence, route comparison,
claim-to-experiment matrix, controls/accounting plan, and bounded Stage 1 prompt
are under:

`refine-logs/phase9_csvl_vpl_direction_20260725/`
