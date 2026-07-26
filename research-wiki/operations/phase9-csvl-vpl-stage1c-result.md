# Phase 9 CSVL-VPL Stage 1C Result

Date: 2026-07-26
Branch: `codex/hpc-orchestrator-bootstrap`
Stage-1B freeze commit: `d68b25db613ae245bdd83a7b9bfcfe9f6ff608cb`
Stage-1C C0 implementation/authority commit:
`0dd0aa64fa261c7ece2c41907ecc02f0d6538ff2`
Run: `P9-VPL-S1C-C0-CUT-S20260726`
Slurm job: `50321533`
Outcome: `STAGE1C_NO_INFORMATIVE_INTERVAL`

## Decision

Gate C0 admits no interval in the existing sealed `cut_roasted_beef` evidence.
All 19 pre-specified windows contain zero front/rear cross-order candidates.
This remains true in windows with substantial P03 observations, ambiguity,
multi-frame gaps, valid-mask coverage, motion, directional diversity, and
complete candidate provenance. The missing quantity is specifically cross-order
candidate evidence, not general occupancy or candidate density.

The ordered-gate rule therefore stops Stage 1C before Gate C1. No support-level
transport, backward-cycle score, control replay, association, tracking, reveal,
or reappearance computation was implemented or run. This is a data-admission
result for the existing sealed artifacts, not evidence that support transport or
temporal surface association is scientifically impossible.

This outcome grants no Gate-A, trainer, Gaussian, primitive-lifecycle,
reconstruction, or physical-surface-identity authority.

## Stage-1B preservation

The complete Stage-1B implementation and durable negative result were committed
first at `d68b25db613ae245bdd83a7b9bfcfe9f6ff608cb`. The result records primary
outcome `STAGE1B_CONTROL_OR_BINDING_DEFECT`, secondary flow-insensitivity, and
the zero-cross-order limitation.

The matched-temporal descriptive correction separates 1,911 immutable
chain-level selections from 5,335 constituent step offsets. Read-only replay
preserved candidate-row hash
`1b246511559beeb43558a4890ac9dc2fd251ab950a35fbbfd1f698329488d870`,
full-track hash
`5973436bee97107f66456f8b84155ab81ebbc6331b6109891b7ff91b067dee17`,
and Stage-1B canonical hash
`a82cedda1f20a17e1e9eb42c94a5158fef397fd2141185e503a40a7a8dc1fa77`.
No Stage-1B output was replaced and no second Stage-1B job was submitted.

## Frozen Gate-C0 rule

The exact pre-observation authority is
`research-wiki/operations/phase9-csvl-vpl-stage1c-authority.md`. Config
`configs/depth_visibility/csvl_vpl_stage1c_c0_v1.json` has SHA-256
`6b77b9ec224d2661e1d147405f1ad3c09e6b8e1b286e4515146c3f2b115f39d8`.

The scanner uses the exact sealed Stage-1 P03 geometry-prefiltered candidate
universe with at least two common permitted cameras and sealed P02 valid-mask
support, before association score or admission. It enumerates closed 30-frame
windows at stride 15 plus one deterministic end-aligned tail window. Both edge
endpoints must be inside.

A window requires:

- at least one front/rear cross-order candidate;
- at least one source with multiple plausible candidates;
- complete P03 endpoint, camera/time, calibration, and P02-record provenance.

At most one development interval is selected. No secondary interval is allowed.
Admissible windows would be ranked lexicographically by cross-order count,
cross-order gap count, ambiguity, later-gap opportunity, camera disagreement,
directional diversity, candidate count, and start frame. No score, admission,
confidence, annotation, evaluator, or reconstruction outcome enters selection.

## Window results

| Frames | P03 observations | Candidates | Multi-candidate sources | Later-gap opportunities | Cross-order | Decision |
|---|---:|---:|---:|---:|---:|---|
| 0-29 | 80 | 171 | 43 | 28 | 0 | reject |
| 15-44 | 72 | 235 | 56 | 17 | 0 | reject |
| 30-59 | 38 | 88 | 23 | 4 | 0 | reject |
| 45-74 | 18 | 4 | 0 | 0 | 0 | reject |
| 60-89 | 10 | 2 | 0 | 0 | 0 | reject |
| 75-104 | 14 | 7 | 2 | 1 | 0 | reject |
| 90-119 | 28 | 43 | 11 | 1 | 0 | reject |
| 105-134 | 26 | 38 | 9 | 0 | 0 | reject |
| 120-149 | 36 | 78 | 22 | 8 | 0 | reject |
| 135-164 | 52 | 118 | 32 | 8 | 0 | reject |
| 150-179 | 26 | 26 | 8 | 2 | 0 | reject |
| 165-194 | 16 | 20 | 6 | 2 | 0 | reject |
| 180-209 | 18 | 44 | 12 | 4 | 0 | reject |
| 195-224 | 24 | 18 | 6 | 2 | 0 | reject |
| 210-239 | 42 | 48 | 14 | 4 | 0 | reject |
| 225-254 | 48 | 117 | 26 | 7 | 0 | reject |
| 240-269 | 36 | 79 | 20 | 3 | 0 | reject |
| 255-284 | 14 | 28 | 8 | 2 | 0 | reject |
| 270-299 | 2 | 0 | 0 | 0 | 0 | reject |

There are 316 unique observations and 666 unique candidate edges globally.
The overlapping windows are not summed to infer unique counts. Because every
candidate gap is at most five frames and adjacent 30-frame windows overlap by
15 frames, every candidate edge is covered by at least one window.

The densest rejected window, frames 15-44, contains 72 observations over 25
observed frames, evenly split between 36 front and 36 rear observations. It has
235 candidates from 62 sources, 56 ambiguous sources, 1-6 candidates per source
(median 4), and temporal gap 1-5 (median 3). Candidate provenance completeness
is 100%; camera support is 2-3 cameras (median 3).

In that window, P02 manifest validity coverage has mean `0.98976` and median
`0.99522`. Flow-chain magnitude has median `0.34261 px`, 90th percentile
`11.22829 px`, and maximum `25.22722 px`; all eight direction octants occur and
circular directional diversity is `0.49005`. Mean camera direction disagreement
is `0.06940` and mean camera magnitude standard deviation is `0.58457 px`.
Projected P03 quantization support can overlap for a mean fraction `0.77021` of
per-candidate camera rows. Despite these nontrivial label-free quantities, its
cross-order count is exactly zero.

Across windows, maxima are 80 observations, 235 candidates, 56 ambiguous
sources, 28 later-gap opportunities, directional diversity `0.86720`, and mean
camera-direction disagreement `0.21979`. No window is reveal-admissible.

The empty tail window reports provenance completeness as zero because the
descriptive fraction uses zero for an empty candidate set. This is not a missing
binding for a real candidate and does not affect rejection: it independently
fails both cross-order and ambiguity requirements.

## Gate C1 and requested association diagnostics

Gate C1 is not applicable because Gate C0 rejected every window. Accordingly:

- support-transport, zero-flow, valid-flow, and cycle formulas were not frozen;
- true backward, zero, sign-negated, camera-swap, temporal-offset,
  direction-rotated, and corrupted controls were not run;
- candidate paired discrimination and ranking were not computed;
- geometry-only, zero-flow support, forward-flow support, cycle, camera-only,
  full, and no-flow ablations were not run;
- selected edges, tracks, rear/cross-order tracks, reveal/reappearance,
  confidence, risk, and abstention results are not applicable.

Reporting Stage-1B track or reappearance counts here would violate the ordered
gate because those occupancy-dominated associations are exactly the evidence
Stage 1C was asked not to redesign against.

## Inputs, outputs, and hashes

Direct scientific inputs:

- Stage-1 ledger: SHA-256
  `3dc1cff29ee34ca6141c0ee40171605ad8fbe9f2ad3c31d1c71bbbc26b0651b8`,
  canonical
  `f0d1b8678574b8e1c52216a75fc5cf04ebaa7635fd68c1d79330a96e573dbdc2`;
- Stage-1B audit: SHA-256
  `d3aabcf8754088f4c7e765ecf36c7080598011e175ee686df3099d62002a6468`,
  canonical
  `a82cedda1f20a17e1e9eb42c94a5158fef397fd2141185e503a40a7a8dc1fa77`.

Transitive bindings include P02 manifest
`67c4ac7ed23b143f1410397f508a781bdd138e79688b62613508b09964e46f93`,
P03 ledger
`50d3d2abc1e6c0ce9dd9cd29bbe610e8cdb69fd164e22cba8fdc8d50b0baf35c`,
train/test calibrations
`5f45aa1b9aadbcb6d623ae6376a97d9f3c8188a86e934f406da450d7083bb679`
and
`1900f8b23ff91cde46a1a65f467b51e11ae877c7015f22bb3249303eceb7ab89`,
and all 1,635 consumed P02 record bindings with canonical binding hash
`3bb0707b40ede2b2ea9068a9b930babfce7edcf6ae2132377eb8b12d90c4c1bc`.
P03 retains the sealed P01 ancestry.

Output root:
`$WORK/proj_adags/runs/phase9-depth-visibility-capacity/stage1c-association-redesign/`.

- frozen run matrix SHA-256
  `3b0c7d7cce7532bc92e780f4cf716fc81186599d7c4647dc83d00bfdc2576c30`;
- exact-worktree authority SHA-256
  `e789f57cac149b9fd49cece6c2c536e38c8d6c2589a5331994ca4f5f183b05bb`;
- command registry SHA-256
  `c8170cc84901171b25fd24e1572c2f36392922611cf8a6e287c0c1c3a8f40ab6`;
- resolved execution SHA-256
  `a8142ad12f41e2d0df8e81c79b12d70ae84372b9991d31bb7a887c6f27b2730c`;
- interval selection SHA-256
  `94c343a7dff800e50df57fa8b3ac3d48e911ac3f01926e6c05e9e80426d279bd`;
- diagnostics SHA-256
  `57206fe0a8e299f07a9811a6493ef7b794b688f33d8afd041243d9e63e44b1c6`;
- terminal SHA-256
  `08c7d6df77286abd8a7e025f2a73fbbadfbdb5802cd60e20e76b61dbf1cf8d2d`;
- canonical Stage-1C C0 scientific-content SHA-256
  `02d8611bbdad1dfff74e165c16b126854bc5992bb92e3e5a67f9df9ae827518d`.

Independent recomputation matches the canonical hash. In-job repeat with
alternate excluded timestamp, Slurm ID, and output root also matches exactly.
All 40 exact-worktree source hashes still match after runtime.

## Tests, read boundary, and geometry proof

The focused Stage-1B/1C suite passes 13 tests. The complete
`tests/test_depth_visibility*.py` suite passes 124 tests with zero failures or
errors. Tests cover frozen-rule mutation, positive cross-order admission,
zero-cross-order rejection, deterministic tail windows, no emitted geometry,
candidate-universe mismatch, prohibited reads, and exact canonical repeats.

An independent read-only audit is `PASS`. It independently reproduced 316
observations, 666 candidate edges, all 19 windows, the zero cross-order count,
and canonical hash
`02d8611bbdad1dfff74e165c16b126854bc5992bb92e3e5a67f9df9ae827518d`.
It also mutated every score/admission-named candidate field and reproduced the
identical selection object. Audit files:

- `audit/EXPERIMENT_AUDIT.md`, SHA-256
  `c7d66bc75783b0c684dded8d1f2937a08f199182fec20afaffea823fc2366ad7`;
- `audit/EXPERIMENT_AUDIT.json`, SHA-256
  `ed5a4d54e69ac4aea606e907178b73f5a02fa2ded2060b4a38ce13e30ec999b6`.

The application opened only the immutable Stage-1 ledger and Stage-1B audit as
scientific data. It read no RGB, cam00 image, annotation, evaluator mask,
reconstruction output, checkpoint, model, W&B data, or label. Static inspection
shows the scanner never accesses Stage-1B `control_scores` or admission flags.
The output contains no `world_xyz`, optical depth, hidden geometry, surface
ownership, or temporal identity field.

## Runtime and CPU-only proof

Job `50321533` completed `0:0` in 59 seconds on `boost_usr_prod`. Resources were
one node, one task, 8 CPUs, 64 GiB RAM, 2 hours, account `euhpc_d21_034`, QoS
`boost_qos_lprod`, and exactly one A100 reservation. Batch peak RSS was 183,000
KiB.

CUDA devices were hidden before application Python. The application imported
no Torch, loaded no model, moved no tensor to CUDA, made zero CUDA API calls,
launched zero kernels, mapped no CUDA runtime/driver library, appeared in no
NVIDIA compute-process row, and allocated zero GPU memory.

Logs:

- `logs/P9-VPL-S1C-C0-CUT-S20260726_50321533.out`, SHA-256
  `04482062134c68ffbe42f871733c0a07e35de01bd4fee25cbc515d0de5241860`;
- `logs/P9-VPL-S1C-C0-CUT-S20260726_50321533.err`, SHA-256
  `1c4c78b28684d24a641c560210f8ed2e1f06d178cb6d6a2195b7177ef62519da`.

One job was submitted. Both `squeue` and `sacct` were checked before submission;
no prior job existed. No retry, resubmission, second interval, or second lane was
used.

## Scope deviations and next action

There was no scientific scope deviation. Gate C1 was intentionally not begun.
The empty-tail provenance fraction disclosed above cannot change the
zero-cross-order result or admit an interval. The independent audit records one
additional non-material caveat: the sealed Stage-1B carrier physically contains
score/admission fields, so JSON parsing deserializes them even though selection
code never accesses or emits them. Mutation invariance demonstrates that they
do not influence the result. This is not a byte-level non-deserialization claim.

The smallest justified next action is a separately approved, label-free
scene/interval acquisition or sealing procedure that first demonstrates actual
P03 cross-order opportunities with permitted-camera and P02 support. It must be
specified before inspecting annotations or association outcomes. Only after
such evidence is independently admitted should a support-transport association
redesign be proposed. Existing occupancy-only evidence is insufficient.

Independent Gate-A admission remains required. This result does not authorize
trainer or primitive-lifecycle integration.
