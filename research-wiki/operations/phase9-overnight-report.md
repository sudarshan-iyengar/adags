# Phase 9 overnight report: depth–visibility–capacity

Status: Slice A checkpoint pushed; A00 engineering retry fix validated; no jobs submitted
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

## Local authority execution

`P9-A00-STATIC-S20260715` attempt 1 failed before running any test because the
launcher passed `top_level_dir` to unittest discovery while `tests/` is
intentionally not a Python package. The failed terminal is preserved in
operational state with SHA-256
`578d028fd7b5bd3d5b338adae7ea422df3eb54050af13d1b412a692277819e3a`.
This is an engineering launcher failure, not scientific evidence. Removing that
single discovery argument makes the exact registered discovery form pass 71/71
twice; Python compilation and diff checks also pass. One retry is admitted
after the narrow fix is committed and pushed.

## Exact next action

Commit and push the one-line A00 discovery correction with this failure record,
then retry A00 once. If it passes, run A01 and I01 and verify all terminal and
artifact hashes. Only then dry-run, submit, and monitor A02. A03 runs only after
successful A02 and must be analyzed before any A04, production inference,
training, Gate B, or broad matrix execution. Genuine human fields remain empty
unless external reviewers supply them.

## 2026-07-17 execution update: A02, A03, and fail-fast X01

This section supersedes the earlier zero-job/zero-outcome statements above.

### Branch and checkpoints

- Branch: `codex/hpc-orchestrator-bootstrap`.
- Phase 9 scaffold commits: `719d451f` and `879bb850`.
- Fail-fast pilot commit: `de768ad2b057c1451324e8cc6c828f3b50c93adf`,
  `Add fail-fast visibility geometry pilot`.
- The branch was clean, pushed, and equal to upstream before the cycle-v2
  implementation freeze.
- Cycle-v2 freeze bound 51 tracked sources and 2 exact commands. Terminal
  SHA-256: `be79f2d43943ff417f69891208014ad387a3b27dcb0e56a34662810a1ea0e026`.

### Completed run matrix

| Run | Job | State | Elapsed / GPU-hours | Decisive result |
| --- | ---: | --- | --- | --- |
| A00 static | login | completed after one preserved launcher fix | 0 GPU | 71/71 then 76/76 after X01 code |
| A01 synthetic | login | completed | 0 GPU | front visible and rear occluded |
| A02 DA3 authority | 49621046 | COMPLETED 0:0 | 00:00:59 / 0 GPU | weight SHA `8ebe871a...` sealed |
| A03 frame-0 conformance | 49621286 | FAILED 1:0 | 00:03:07 / 0.05194 | same-group depth missed registered 1e-5 allclose |
| X01 frame-125 fail-fast | 49624624 | COMPLETED 0:0 | 00:03:20 / 0.05556 | raw two-group geometry blocked by 8.30% shared-anchor max relative MAD |

Total measured A100 consumption through X01 is approximately 0.1075 GPU-hours.
There were no W&B writes or training runs.

### X01 repeatability result

X01 preserves the A03 failure and measures its magnitude rather than changing
the threshold. For two repeated frame-125 inferences of the same six-view
group:

- 99.999825% of depth values satisfy the registered 1e-5 tolerance;
- median absolute delta is 0;
- 90th percentile is (4.768\times10^{-7});
- 99.9th percentile is (9.537\times10^{-7});
- maximum absolute delta is 0.0328665;
- maximum symmetric relative delta is 0.00269887;
- maximum duplicate relative MAD is 0.00134944;
- confidence arrays are exactly identical;
- processed-K corner error is (1.330\times10^{-5}) pixels.

Thus the strict allclose failure is caused by an extremely sparse depth
outlier. It remains a valid negative conformance result, but broad same-group
nondeterminism is not the material problem.

### X01 cross-group result

The same physical `cam01` anchor was then inferred in the second six-view
group with five different companion cameras. Its maximum cross-group relative
MAD was 0.0829967, exceeding the frozen 0.05 duplicate limit. Both groups'
processed intrinsics remained correct to (1.330\times10^{-5}) pixels.

The registered fail-fast rule therefore rejected raw group fusion before any
cam00-projected geometry was evaluated:

- geometry admitted: no;
- frame ledgers produced: 0;
- cam00 RGB opened: no;
- label-dependent Gate A: not evaluable;
- Gate B: not run.

Terminal SHA-256:
`5b718ab9d36735fb49762a20cd8ebb35bb242e5d5eafd30b4f447e832d9c78a5`.
Report SHA-256:
`7e611f78b1fbc9fce51c740a6e7f1baf1d6a890990a731ff7a8873a5588112b1`.

### Gate status after X01

| Gate item | Status | Evidence boundary |
| --- | --- | --- |
| Camera/K and target exclusion | pass | calibrated inputs; K error (1.33\times10^{-5}) px; cam00 RGB unopened |
| Strict same-group 1e-5 repetition | fail | A03 and X01; threshold unchanged |
| Same-group fusion-scale repeatability | pass diagnostically | max relative MAD 0.00135 < 0.05 |
| Raw cross-group common gauge | fail | shared-anchor max relative MAD 0.082997 > 0.05 |
| Cross-view visibility/support | not evaluated | correctly blocked before fusion |
| Human Gate A | not evaluable | no fabricated labels |
| Gate B representation benefit | not run | no visibility ledger admitted |

### Failure attribution and next experiment

The current failure is a support-discovery input/gauge problem, not evidence
against depth and not yet a representation-capacity or optimization failure.
`align_to_input_ext_scale=True` does not by itself make independently grouped
DA3 depths directly fuseable under the global maximum rule.

The next registered diagnostic should remain small: measure the full shared
anchor cross-group error distribution, fraction within the existing 0.05
limit, spatial concentration, and deterministic calibration/held-out gauge
alignment. Prefer explicit uncertainty and abstention on inconsistent
train-view anchor pixels. Only use a gauge correction if it improves held-out
anchor pixels without cam00 or frozen event-crop tuning. If reliable coverage
remains adequate, run one-frame geometry with rejected pixels excluded;
otherwise change grouping or depth representation rather than relaxing the
threshold.

## 2026-07-20 execution update: X02 reconciliation and compute-forward v4

This section supersedes the X01 next-experiment recommendation above.

### X02 terminal result

- Run: `P9-V3-X02-CUT-ANCHOR-ABSTENTION-S20260717`.
- Job: `49625472`, `COMPLETED/0:0`, elapsed `00:03:24`, one A100, 0.0567 measured GPU-hours.
- Terminal SHA-256: `d99b188ca130b06a14a2541db271fdc3645f1768a046c56fc6cc6a83a731424b`.
- Report SHA-256: `604e3ebae40845d162ac7e790281273082cca3487b29993ef1d6e76976c11323`.
- Total measured Phase 9 A100 consumption through X02: approximately 0.1642 GPU-hours.

X02 reused the frozen 0.05 per-pixel duplicate relative half-difference rule: inconsistent shared-anchor pixels abstain; retained pixels use the confidence-weighted aggregate. It did not apply a gauge correction, did not change thresholds, and did not open cam00 RGB.

### X02 scientific result

| Item | Result |
| --- | --- |
| Coordinate handling | admitted for frames 125-127 |
| Anchor retained fractions | 0.999858, 0.999858, 0.999612 |
| Supported target bins | 689, 692, 674 |
| Ordered multilayer bins | 0, 0, 0 |
| Label-dependent Gate A | `not_evaluable` |
| Gate B | not run |

Interpretation: per-pixel abstention reduced the X01 group-gauge blocker enough to execute calibrated target-projected geometry, but the ordered-layer opportunity chain has zero yield on the toy window. This is not evidence that depth or visibility is useless, and it is not an authorization for capacity training. It is a specific zero-yield failure of the current ordered-layer predicate/operator chain on the three-frame diagnostic.

### Decision

Stop tiny diagnostics. The next admitted evidence run is one production-facing full `cut_roasted_beef` opportunity-mining pass that records candidate-count waterfalls and ordered-layer rejection reasons from supported target bins to accepted multilayer hypotheses. If full-cut opportunity mining still yields zero ordered layers across real opportunities, pivot to the registered Route 3 layered surface-memory path rather than relaxing thresholds or adding another mask/tile diagnostic.

Annotation should start in parallel from the frozen raw-RGB 54-window packet with two independent annotators. Until genuine labels exist, Gate A and event-region Gate B remain `not_evaluable`.

Slice B engineering may proceed only as B00/B01 scaffolding and matched feasibility. B03/B04 remain blocked until genuine A06 engineering admission and B02 oracle-capacity admission.

### v4 implementation checkpoint

The v4 source update adds:

- ordered-layer opportunity diagnostics to `evaluate_frame_geometry`; every frame report now includes a target-bin candidate waterfall, raw/accepted layer histograms, depth-span distribution, and rejection counts;
- `cut-opportunity-mining`, a Slurm action that consumes exact A02 and X02 terminals, runs the X02 abstention rule across all `cut_roasted_beef` frames, and writes `phase9-cut-opportunity-mining-v1`;
- `depth_visibility.capacity`, a CPU Slice B/B00 transaction fixture for point-neutral in-place reassignment, null-reset moment surgery, hard-static budget accounting, and event-blind donor selection;
- `operator-static`, a runner action that seals those B00 static tests;
- `configs/depth_visibility/phase9_cut_opportunity_cycle_v4.json`, a four-entry cycle covering v4 I01, A05 packet build, B00 static operator test, and the X03 full-cut opportunity-mining Slurm job.

Validation before any v4 submission:

- focused fast-pilot plus capacity tests: 10/10 PASS;
- full Phase 9 unit discovery: 84/84 PASS;
- Python compilation: PASS;
- Slurm shell syntax checks: PASS;
- cycle-v4 matrix sanity check: PASS (4 runs);
- `git diff --check`: PASS.

### v4 publication and X03 submission

After the local v4 checkpoint above, commit `00602a5731f0295bcd2471608bcba9b4a134f1df` was pushed to `origin/codex/hpc-orchestrator-bootstrap` and branch state was clean/synchronized. The v4 implementation freeze completed with terminal SHA-256 `9abb66d5c9e592fc3c166f3ae39b04655b6cb720bd4588e060f5f4465c494bab`; implementation freeze SHA-256 `f12e750030e910061378459dfe8a71fef5b2ba94508cc9f6310517b69f3fec63`; command registry SHA-256 `9d0a4766f829ce2457e255779e8b2585a4e93a5b1bb3dd9e961be8805777ab06`.

The raw-RGB annotation packet producer completed with terminal SHA-256 `343f60f1c98aa400bba68da2464482cfb164a188649766cedb99307730fa0189`; packet SHA-256 `98e42669149b90116291569e4f33077050058b56fd83291a00953370a2a3ce6b`; R009 separation proof SHA-256 `6f9d52c2114c7438d5dc6a34e7cd6d852079eb5921ec5d1fcbdf01f4ac45a469`. The B00 operator-static producer completed with terminal SHA-256 `b0e9285cf26cecbe848f2a036ab88621678b54abcb70c3cb78501037479fe6c7` and 5/5 CPU Slice B fixture tests passing.

`P9-V4-X03-CUT-OPPORTUNITY-MINING-S20260720` was submitted through Slurm as job `49909443` at `2026-07-20T18:28:03Z`; initial `squeue` state was `PENDING`. This submission is still only the production-facing opportunity-mining unblocker. It admits no Route 3 pivot, threshold change, B03/B04 training, or claim update until the terminal report is hashed and interpreted.

### X03 full-cut opportunity result

`P9-V4-X03-CUT-OPPORTUNITY-MINING-S20260720` completed as Slurm job `49909443` with `COMPLETED/0:0` in `00:11:12` (0.1867 A100 GPU-hours). Terminal SHA-256: `3ed30bcfbdc5a2a843d72261a51e5900ecdb7c118fcd202a8703b011a7eb7ce1`. Report SHA-256: `78be452f3a8cb19d7fe91a51fab7954701af0fba0f6844673bf9d8dcd4dd14c8`.

The full-cut opportunity miner executed geometry for all 300 `cut_roasted_beef` frames. All 300 frames had supported target bins. The aggregate candidate waterfall was:

| Stage | Count |
| --- | ---: |
| projected target bins | 3,067,491 |
| bins with minimum camera support | 201,030 |
| bins with at least two raw depth clusters | 27,740 |
| bins with at least two minimum-camera-supported layers | 158 |
| ordered multilayer bins | 158 |

The 158 accepted ordered multilayer bins appeared across 123 frames; the maximum per frame was 3. Rejection counts were 2,866,461 bins with insufficient bin cameras, 173,290 single-depth-cluster bins, and 27,582 depth-cluster bins with insufficient camera support.

Interpretation: the X02 zero-layer result was a local three-frame yield issue, not a full-cut zero-yield failure. This does not pass Gate A, because human labels are still absent, but it removes the zero-opportunity stop condition. The next admissible work is full CSVL ledger/evaluation production plus the human annotation workflow. Route 3 should not be triggered on zero ordered-layer yield from the current evidence.
