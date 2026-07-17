# Phase 9 overnight report: depth–visibility–capacity

Status: Slice A implementation and A02/A03 execution path independently validated; checkpoint pending; no jobs submitted
Started: 2026-07-15
Branch: `codex/hpc-orchestrator-bootstrap`
Phase 8B checkpoint: `94cd67df53cfc487989c71dc16a60fe853f53550`
Objective: [[objectives/depth-visibility-capacity-v1]]
Method candidate: [[operations/phase9-csvl-isr-v1-method]]
Experiment plan: [[operations/phase9-csvl-isr-v1-experiment-plan]]

## Objective and method version

Phase 9 continues the two-gate objective: first establish meaningful calibrated
surface visibility/order/event/uncertainty evidence; only then ask whether it
causally improves representation capacity. The selected v1 is CSVL-ISR:
Calibrated Surface Visibility Ledger with Intermittent-Surface Reassignment.

The admitted first implementation scope is Slice A. It reconstructs local
micro-surfaces from training-camera DA3 optical-z predictions, calibrated
reprojection, appearance, and flow; emits target-free cam00 and transitively
leave-one-camera-out auxiliary visible/occluded/uncertain states plus bounded
hide/reveal/reappearance; and freezes annotation, baseline, metric, risk,
identity, uncertainty, and provenance contracts. It explicitly makes no claim
about deforming geometry while a surface is absent from every camera.

The later Slice B candidate is one fixed, post-densification, in-place,
point-count-neutral reassignment transaction. Its executable contract and
181-entry causal DAG are independently admitted, but it is not implemented and
no representation run is admitted before A00/A01, code review, I01, and Gate A
prerequisites. Route 3 remains fallback. Route 2 remains disallowed unless a
reliable but incomplete deterministic Gate A first exists.

## Phase 8B checkpoint

- Saved Phase 8B diff and regenerated worktree diff both hashed to
  `650f9792bb0f1c8cfc3b33e6790e71e6835c03c54c70606b4f45aaa2f0f089bf`.
- The intended path set matched exactly; eleven extra EOF blank lines were the
  only validation correction.
- Final staged patch hash:
  `8a44ebd9e4208c930ab27e7b6c8e83d1dacf5484a86507427b574bbca46fbb96`.
- Commit `94cd67d` was pushed to `origin/codex/hpc-orchestrator-bootstrap`
  without rewriting history. The repository was clean and synchronized.

## Method decisions and rejected alternatives

| Decision | Evidence/rationale | Rejected alternative |
| --- | --- | --- |
| cam00 is the scored target | Live transforms metadata shows cam00 in test and absent from train for all six scenes; cam10 is training. | Inactive Technicolor/COLMAP cam10 split. |
| Strict target-free prediction graph | Cam00 supplies calibration/time only until sealed prediction; image/labels are evaluation-only. | Target RGB/depth self-consistency presented as held-out evidence. |
| Optical-axis z in supplied scene gauge | Pinned DA3 geometry unprojects `K^-1[u,v,1] depth`. | Metric claim, ray distance, per-frame normalization, or fitted scale/shift. |
| Local micro-surfaces and observed-track states | Non-rigid all-camera gaps are not geometrically observable. | A fabricated persistent hidden xyz/order state. |
| Deterministic/frozen Gate A teacher | Clean causal separation and one dominant contribution. | Learned visibility field before deterministic evidence is admitted. |
| Evaluation state apertures and union spatial metrics | Occluded support comparator and unmatched spatial FP require explicit test-only denominators. | Hidden polygons, matched-track-only precision, or method-defined denominators. |
| Slice A before Slice B | Independent Gate A failure must be distinguishable from capacity/optimization failure. | End-to-end coupled implementation before evidence validation. |
| One in-place reassignment transaction for first Slice B | Smallest existing dynamic-bank seam and exact budget matching. | New bank, CUDA change, continuous allocator, birth/death, or layering as v1. |

## Independent methodology review trail

Fresh isolated reviews progressively rejected under-specified versions rather
than lowering the gate:

| Round | Score | Verdict | Blocking focus and disposition |
| --- | ---: | --- | --- |
| 0 | 4.5 | NOT READY | Observability, state hierarchy, depth gauge/type, labels/splits, risk, controls, budget/transactions. Reframed to bounded observed-track states and two slices. |
| 1 | 6.2 | NOT READY | Exact q/foreground, surface/track, uncertainty, annotation, association, and transaction operators. Added executable Slice A candidate. |
| 2 | 4.0 | NOT READY | Target-transitive provenance, patch/raster, temporal identity, partitions, matching, baseline pins. Added rounds 3-4. |
| 3 | 6.8 | NOT READY | Metric math, fusion/exclusivity, raster, gap ID, annotation, fraction matching, DA3 pin. Closed in round 4. |
| 4 | 6.0 | NOT READY | Risk domain, missing inconsistency comparator, FP aggregation, annotation denominators, grouping/conformance. Closed in round 5. |
| 5 | 6.5 | NOT READY | Occluded comparator rows, unfair temporal collapse, unmatched spatial FP. Closed in round 6 and cohesive method. |
| 6 | 8.4 | NOT READY | Event-track qualification, unknown-frame masking, and spatial reduction. Closed in round 7. |
| 7 | 7.6 | NOT READY | Median/hash/zero-cost semantics, risk estimators, double annotation, and R031-MT threshold. Closed in round 8. |
| 8 | 7.8 | NOT READY | Transitive target ancestry, track-level double annotation, DA3 hash/tolerance policy, stable identities. Closed in round 9. |
| 9 | 7.9 | NOT READY | Role-keyed discovery binding, exact assignment ties, aggregate node ID, PCA eigenspace/sign. Closed in round 10. |
| 10 | 9.1 | READY | No blockers. Slice A admitted; human labels, DA3 authority, and runtime conformance remain execution gates. |

Raw proposals, closure documents, reviewer provenance, and operational decisions
are preserved under
`$WORK/proj_adags/agent-control/phase9-depth-visibility-capacity/`.

The independent execution-contract sequence then scored plan/Slice B as
7.6/7.0 (v3), 8.8/8.2 (v4), 9.0/9.3 (v5; Slice B ready but plan blocked on
explicit DA3 call arguments), and finally 9.4/9.3 `READY` (v6). V6 also scored
the DA3 pin delta 10/10. The final review SHA-256 is
`2e60fcfc65b6a88f9e12396e1edbc790b3ce597878a85221ad1c5ebb5f8f673d`;
the 181-entry generator check and all nine source bindings pass.

## Claim map and run matrix

| ID | Purpose | Scene/data | State | Slurm/W&B | Expected artifacts |
| --- | --- | --- | --- | --- | --- |
| A00/A01 | Static and controlled correctness admission | analytic/synthetic | implementation validated; registered executions pending | login only | tracked tests and exact fixture reports |
| I00/I01 | Independent code review and pushed implementation freeze | tracked code/config/env | review PASS; commit/push and I01 pending | no Slurm until resolved | review decision, command/config/code authority |
| A02/A03 | DA3 weight authority and camera/depth conformance | read-only model + cut frame 0 | implementation validated; not submitted | CPU/GPU Slurm | authority and conformance manifests |
| A04 | Target-free tiny real diagnostic | cut frames 125-127 | registered, not submitted | GPU Slurm <=2 h | sealed sidecars/controls/report |
| A05 | Empty blinded packet and genuine label freezes | 54 fixed windows | manifest frozen; labels unavailable | CPU Slurm + external humans | packet, two-stage labels/adjudication |
| P01/P02/P03 | Full calibrated depth, flow semantics, and CSVL ledgers | all six scenes | registered, not submitted | 6 GPU + 12 CPU Slurm producers | exact array/ledger/provenance inventories |
| P04/P05 | Training sidecars, evaluators, cut oracle | all six / annotated scenes | registered, not submitted | CPU Slurm | leakage/read-set/evaluator freezes |
| A06 | Target-consuming baselines and genuine Gate A | cut then locked flame/sear | not evaluable: no labels | 3 GPU baseline + CPU score jobs | explicit cut/transfer decisions |
| B00-B02 | Operator, matched pilots, oracle, sole repair | cut | contract admitted; implementation gated | static + short Slurm | checkpointed feasibility/attribution |
| B03 | Seven-lane causal comparison including null-reset | cut | gated | 7 matched GPU lanes | explicit Gate B decision/checksum |
| B04 | Frozen all-six seed 0 then conditional seeds 1/2 | all N3V | gated | fully enumerated | reuse proofs, checkpoints, metrics, decisions |

No job IDs, W&B run IDs, checkpoints, or generated Phase 9 sidecars exist yet.
JOB_LEDGER.jsonl is empty. This is intentional: method and plan reviews precede
substantial computation.

## Gate A criteria

| Criterion | Engineering | Claim grade | Current state |
| --- | ---: | ---: | --- |
| Controlled camera/z/order/cycle fixtures | all pass | all pass | static code PASS; registered A01 execution pending |
| Ordering accuracy / AUROC / coverage | .70 / .75 / .60 | .75 / .80 / .70 per transfer scene | not evaluable |
| Event F1 / recall | .45 / .60 | .60 / .70 | not evaluable |
| Boundary F1 and region IoU over baseline | +.05 each | +.10 each | not evaluable |
| Cross-view and temporal error reduction | 15% each | 25% each | not evaluable |
| Ordering/transition ECE | <=.15 | <=.10 | not evaluable |
| Track/scene coverage | every event; recall on >=80% tracks | no family completely missed | not evaluable |

Missing human annotations leaves label-dependent items `not_evaluable`; no
threshold is changed. A04 will report only valid target-exclusion, numerical,
repeatability, projection, cycle, compactness, and support diagnostics.

## Gate B criteria

| Criterion | Requirement | Current state |
| --- | --- | --- |
| Checkpoint-backed event PSNR / LPIPS | practical +0.20 dB / -5% | not run |
| Events improved on both | majority | not run |
| Static no-harm | all-300-frame PSNR >=-0.05 dB; perceptual/reconstruction-L1 <=+2% per scene | not run |
| Flicker / reveal ghost | mean regression <=5%; no scene failure | not run |
| Point budget | final/integrated within +/-2%; <=600k | not run |
| Causal controls | full > visibility-only and capacity-only; shuffle no gain | not run |
| Six-scene/seed consistency | frozen all-six first, matched seeds next | not run |

R009 continuity and oracle-gap recovery remain secondary. A successful Gate A
cannot pass Gate B, and an operator/strong-reference failure prevents blind
coupled sweeps.

## Quantitative and qualitative evidence

No new quantitative or qualitative method outcome has been observed. The only
new evidence is forensic/static:

- active camera metadata corrects cam10 to cam00 as held-out target;
- DA3 depth semantics are optical z in the provided gauge;
- active loader already carries K and calibrated transforms;
- existing renderer depth is accumulated alpha-weighted z, so it is not used as
  a first-surface label;
- current Gaussian total-budget accounting must include dynamic plus hard-static
  rows before Slice B;
- in-place dynamic-bank row reassignment is the narrowest capacity seam, but all
  optimizer auxiliaries and restart transactions must be enumerated first.

These findings do not establish that CSVL works.

## Slice A implementation checkpoint evidence

The implementation package now has deterministic adapters, schemas, fixtures,
artifact authorities, target-exclusion ledgers, metrics, execution manifests,
Slurm wrappers, and fail-closed unimplemented actions. The immediate A02/A03
path received an independent narrow code review.

The reviewer first confirmed the four requested binding fixes and found one
additional blocker: pinned DA3 returns aligned extrinsics as `N x 3 x 4`, while
the adapter required `N x 4 x 4`. The adapter now copies pinned output into
fresh homogeneous `N x 4 x 4` w2c matrices and validates the last row. The
reviewer's post-fix verdict is `PASS`; reviewer
`/root/a02_a03_final_review`, artifact SHA-256
`ac39ecd18652862ee8a747436301ffc33a14599322588b10545e34fe25be4c5c`.

Validation after the final correction:

- full Phase 9 unittest discovery: 71/71 PASS twice;
- focused camera and execution tests: 16/16 PASS;
- Python compilation: PASS;
- both Slurm shell syntax checks: PASS;
- 181-entry matrix generator check: PASS;
- `git diff --check`: PASS.

This is engineering evidence only. No real depth, visibility, representation, or
Gate A/B outcome has been observed.

## Compute used

- GPU-hours: 0.
- Slurm jobs: 0.
- W&B writes/runs: 0.
- Registered reservation maximum: 416 unconditional plus 424 conditional
  GPU-hours (840 total), not consumed and not a spending target.
- Login-node work: lightweight repository/metadata inspection, static document
  generation/validation, checksums, Git, Slurm capability probes, and isolated
  reviews only.

## Methodological changes during the cycle

1. Replaced unobservable all-view hidden geometry with explicit unobserved state.
2. Corrected the active held-out camera to cam00.
3. Removed depth scale/shift fitting and per-frame normalization.
4. Made cam00 prediction fully target-free.
5. Defined point-to-patch-to-track-to-raster identity and uncertainty.
6. Added fail-closed DA3 pin/hash/conformance.
7. Added blinded annotation apertures without inventing hidden shapes.
8. Made spatial metrics semantic unions so unmatched predictions are penalized.
9. Registered a fair metric-specific R031-MT comparator.
10. Kept all trainer/capacity mutations outside Slice A.
11. Made leave-one-camera-out exclusion transitive through DA3 groups, fusion,
    temporal edges, and first-visible witnesses.
12. Separated numeric DA3 conformance from immutable production-sidecar hashes.
13. Froze two independent discovery passes, a union roster, explicit
    `not_found`, two roster responses, and blind adjudication for all 54 windows.
14. Froze canonical group/node/fused/patch/track identities, assignment ties,
    PCA eigenspace/sign rules, and model-space color initialization.
15. Replaced a 128-entry training list with a 181-entry executable producer and
    decision DAG, including six-scene preprocessing and exact checkpoint reuse.
16. Made exact-zero event PSNR finite and single-valued at the frozen MSE floor.
17. Replaced implementation-dependent shuffling with domain-separated SHA-256
    cyclic permutations for targets and complete confidence maps.
18. Explicitly froze all DA3 inference-mode arguments that previously relied on
    pinned API defaults.

## Unresolved scientific questions

- Will pinned DA3 be repeatable and sufficiently cross-view coherent on N3V?
- Will calibrated local surfaces survive hands, utensils, flame, specular food,
  and topology change without excessive abstention?
- Can genuine annotators identify enough non-ambiguous state apertures/tracks?
- Does a strong training-only reference sidecar make point-neutral reassignment
  useful, or is the representation/operator itself inadequate?
- If the operator works, does inferred visibility add causal benefit beyond a
  rate-matched trigger?

## Exact next action

Inspect and stage only the intended Phase 9 implementation/config/test/wiki
paths, run the cached-diff checks, create and push one narrow implementation
checkpoint, then execute registered local A00/A01 and I01. Only after their
terminal manifests and hashes pass should A02 be submitted and monitored. A03
runs only after successful A02 and must be analyzed before any A04, production
inference, training, Gate B, or broad matrix execution. Genuine human fields
remain empty unless external reviewers supply them.
