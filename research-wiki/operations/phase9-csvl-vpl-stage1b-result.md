# Phase 9 CSVL-VPL Stage 1B Result

Date: 2026-07-26
Branch: `codex/hpc-orchestrator-bootstrap`
Stage-1 commit: `70a9a678df290a2ae9510f313fdb704cae2632f4`
Run: `P9-VPL-S1B-D01-CUT-S20260726`
Slurm job: `50250624`
Outcome: `STAGE1B_CONTROL_OR_BINDING_DEFECT`

## Decision

Stage 1B demonstrates a concrete Stage-1 control-definition defect: the control
reported as reversed flow is not a sealed backward `t+1 -> t` field. It negates
each sampled sealed forward vector and follows that negated trajectory. Stage 1B
repairs the scientific disclosure, keeps the original association problem fixed,
and adds magnitude/mask-matched direction, camera, and temporal controls before
the single authorized diagnostic.

The corrected diagnostic does not rescue the association evidence. Removing
flow changes only 3 of 206 selected edges and retains 98.565% selected-edge
overlap. Valid flow ranks the lowest-P03-displacement proxy first on 56.02% of
multi-candidate sources, below camera swap (68.90%), reversed/sign-negated flow
(58.28%), and temporal offset (59.04%). Valid flow increases endpoint error
relative to zero flow for 78.98% of the frozen candidates. These are strong
secondary indications that flow is not causal for the current output, but the
required selected outcome is the control-defect outcome because the original
reversed control is not semantically what its label claimed.

A second descriptive instrumentation defect was found after runtime: the
matched-temporal offset distribution combines one offset per camera chain with
one offset per P02 step. This changes only that reported distribution; it runs
after scoring and cannot affect sampled chains, paired scores, admissions,
selected edges, tracks, or ablations. The immutable raw rows give the correct
one-value-per-chain distribution: 1,911 chains, counts `-2:276`, `-1:647`,
`+1:684`, `+2:304`, mean `+0.04867`, median `+1`. The single-job authority was
already exhausted, so no output was overwritten and no second job was submitted.

The corrected code also reports the constituent P02-step distribution separately:
5,335 steps, mean `+0.05455`, median `+1`. Read-only replay of the summary function
left the canonical candidate-row hash
`1b246511559beeb43558a4890ac9dc2fd251ab950a35fbbfd1f698329488d870`,
full-current-score track hash
`5973436bee97107f66456f8b84155ab81ebbc6331b6109891b7ff91b067dee17`,
and the Stage-1B canonical scientific hash unchanged.

This outcome does not pass Gate A. It grants no trainer, primitive-lifecycle,
reconstruction, or physical-surface-identity authority.

## Scope and changed files

Stage 1B adds a forensic replay module, immutable audit/diagnostic schemas, a
strict config, one-run matrix, exact-worktree authority preparer, CPU-isolated
runner action, and focused tests. It reuses the Stage-1 observation/candidate
universe and association state machine exactly; no association threshold or
lifecycle rule is redesigned.

Changed Stage-1B paths:

- `depth_visibility/association_audit.py`;
- `depth_visibility/schema.py`;
- `depth_visibility/surface_tracks.py` (read-only bound-step audit access only);
- `scripts/run_phase9_depth_visibility.py`;
- `scripts/run_phase9_depth_visibility_job.sh`;
- `scripts/prepare_csvl_vpl_stage1b.py`;
- `configs/depth_visibility/csvl_vpl_stage1b_v1.json`;
- `configs/depth_visibility/phase9_csvl_vpl_stage1b_v1.json`;
- `tests/test_depth_visibility_stage1b.py`;
- this result page.

Unrelated pre-existing research-wiki edits and the prior experiment-audit trace
were preserved and excluded from the Stage-1 commit and Stage-1B source binding.
The narrow Stage-1B freeze commit is identified by the subsequent Stage-1C
authority and a post-freeze pointer on this page. No push was made.

## Exact inputs and authority

The exact-worktree authority binds 37 source files while honestly making no
clean-worktree claim. Its ordinary SHA-256 is
`cb4ba3d5750325b46e7f72b8e6292071478d3654e85bc1f2d7a0d8d77cb2cd52`.
The command registry SHA-256 is
`976d63648fb7c3e8b678064feebad9c8e4867ac5f04e85247c50c3e5497347c4`.
The resolved-execution SHA-256 is
`3f20ab6dc97b6af3e0320b2d2cb11f67897b00c55c2ecf9077cbfb8f375051dd`.

Top-level consumed bindings:

- Stage-1 terminal:
  `8db32fa81aeb541cb53aeae74a7abdfc2849f00be8a163a51eee31f00b8767a3`;
- Stage-1 ledger:
  `3dc1cff29ee34ca6141c0ee40171605ad8fbe9f2ad3c31d1c71bbbc26b0651b8`,
  canonical content
  `f0d1b8678574b8e1c52216a75fc5cf04ebaa7635fd68c1d79330a96e573dbdc2`;
- Stage-1 diagnostics:
  `80154edd5e8eb1ca0c5b2831e63e5b4f11919e55630175d6acc9a41bc733767c`;
- P03 ledger:
  `50d3d2abc1e6c0ce9dd9cd29bbe610e8cdb69fd164e22cba8fdc8d50b0baf35c`;
- P03 terminal:
  `dd45de6b3e0cbe36eec4a88aa1f7537a8761042b43e5154508fc044754612c67`;
- P02 manifest:
  `67c4ac7ed23b143f1410397f508a781bdd138e79688b62613508b09964e46f93`;
- P02 terminal:
  `97c79317d455e31aed5b1a0291ba2fa9afd9282a5fff08a8c8c09eca20f0c32b`;
- train calibration:
  `5f45aa1b9aadbcb6d623ae6376a97d9f3c8188a86e934f406da450d7083bb679`;
- test calibration:
  `1900f8b23ff91cde46a1a65f467b51e11ae877c7015f22bb3249303eceb7ab89`;
- Stage-1 config:
  `1629c46f60b38d6df412d31e1d44f722730d8efa406e87580bb837509cdab942`;
- Stage-1B config:
  `02c0ba14e7898757860e6a402fda1a187c1d7fedabb755ebd4ddd8efd67a7272`.

The audit binds all 1,635 consumed P02 NPZ records by exact path, schema,
ordinary file hash, flow-content hash, validity-mask hash, generator revision,
camera, source frame, and target frame. Stage-1/P03 bindings preserve the
transitive P01 ancestry.

## Controls and verified bindings

All sampled arrays are native `1352 x 1014`, pixel units at source resolution,
integer pixel centers, bilinear sampling, no resize/scale transform, and sealed
forward `t -> t+1` records. Every step is filtered by the P02 forward-backward
validity mask and binds the mask and array hashes.

- valid: original camera and logical time, unchanged forward vectors;
- reversed/sign-negated: original forward records and time, exact vector
  negation at all 5,224 sampled steps; this is not an independent backward flow;
- camera swap: next lexicographic P02 camera, with original P03 endpoints,
  original pixels, original calibration, time relation, candidates, and
  thresholds fixed; all 5,205 sampled steps use a different camera;
- temporal offset: original camera and candidate problem, with each sampled
  P02 relation shifted exactly `+1` frame at all 5,256 steps;
- corrupted: original camera/time, transform `(dx,dy) -> (dy+16,dx-16)` at all
  5,092 sampled steps;
- direction matched: rotate the valid accumulated displacement by 90 degrees,
  preserving valid magnitude, valid-mask ancestry, time, and candidate support;
- camera matched: select among the two nearest-baseline alternative cameras by
  relative magnitude plus mask-coverage error, with deterministic ties;
- temporal matched: select among offsets `-2,-1,+1,+2` by relative magnitude
  plus mask-coverage error, with deterministic ties.

All 1,913 camera evidence rows retain original source/destination calibration
IDs, source/destination times, and P03 source/destination observation IDs.
Projection calibration is unchanged across every control. No camera, time,
calibration, or P03 metadata was unintentionally swapped with the tested flow.

## EPE and current score

For each original camera, the reported endpoint error is

`||(project(source_xyz,camera_t) + supplied_flow_chain) - project(destination_xyz,camera_t+gap)||_2`.

It is measured in native pixels, requires every sealed mask sample to be valid,
and is aggregated by median across common cameras. Every control uses the same
fixed P03-derived destination projection and original calibration, so a control
is not evaluated against itself. This is still an internal P03/calibration
consistency proxy, not physical correspondence ground truth.

Normalized EPE divides by `2 px + projected source half-bin radius + projected
destination half-bin radius`. The tolerance has mean 14.553 px and median
14.086 px. Current candidate cost is
`0.8*min(2,normalized_EPE) + 0.2*min(2,geometry_cost)` and admits at cost `<=1`.
Candidates are already P03-geometry-prefiltered at
`0.05*R_scene*frame_gap`. Current risk is the maximum of clipped normalized EPE
and missing-camera fraction. P03 uncertainty and depth/order have zero score
weight; forward-backward consistency is used only as a binary P02 mask.

Across all 666 candidates, median geometry cost is 0.1295, median valid
normalized EPE is 0.0980, and median valid EPE is 1.4097 px. Median zero-flow
reprojection residual is only 0.6779 px. Valid flow direction has median cosine
0.1607 against the required P03-projected displacement, explaining why valid
flow frequently worsens the proxy EPE. Camera-swapped chains also have smaller
motion (mean 1.647 px; matched camera 1.223 px) than valid (2.430 px), which is
favored by the near-static P03 candidate endpoints.

## Replay ablations

| Frozen candidate score | Accepted | Selected | Changed selected vs full | Tracks | Multi-frame | Rear multi-frame | Abstained | Reappeared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry/P03 only | 666 | 188 | 24 | 107 | 47 | 24 | 21 | 40 |
| flow only | 594 | 209 | 3 | 107 | 52 | 27 | 0 | 47 |
| camera/reprojection only | 604 | 209 | 3 | 107 | 52 | 27 | 0 | 47 |
| geometry + camera, no flow | 604 | 209 | 3 | 107 | 52 | 27 | 0 | 47 |
| geometry + flow magnitude, no camera-specific consistency | 606 | 204 | 4 | 108 | 51 | 27 | 4 | 47 |
| full current score | 601 | 206 | 0 | 108 | 51 | 27 | 2 | 46 |

The no-flow selected-edge overlap with full is 0.98565. Removing flow therefore
does not materially alter the selected associations.

## Paired discrimination and matching

Paired control-minus-valid candidate cost results:

| Control | Pairs | Mean delta | Valid lower | Control lower | Admission changes |
|---|---:|---:|---:|---:|---:|
| reversed/sign-negated | 653 | +0.00614 | 52.53% | 45.33% | 4 |
| camera swap | 657 | -0.03935 | 52.51% | 45.36% | 5 |
| temporal +1 | 663 | +0.00018 | 45.85% | 51.43% | 5 |
| corrupted | 654 | +1.26285 | 97.25% | 0.00% | 594 |
| 90-degree direction matched | 666 | +0.00901 | 51.50% | 45.80% | 3 |
| nearest-camera matched | 666 | -0.04551 | 42.79% | 54.80% | 5 |
| temporal matched | 666 | +0.00584 | 52.10% | 45.05% | 3 |

Direction matching exactly preserves magnitude/mask/time. Temporal matching is
close in magnitude and mask coverage. Camera matching remains only partial:
matched camera mean magnitude is 1.223 px versus valid 2.430 px, although mask
means are 0.9843 versus 0.9880. No maximum match-error criterion was imposed,
so the camera-matched comparison is not by itself a decisive specificity test.
No association threshold was tuned against these outcomes.

## Tracks, states, and interval

The full replay has 316 observations, 108 tracks, 51 multi-frame tracks, 27
multi-frame rear tracks, 206 propagated observations, 2 split abstentions, and
99.367% association coverage. Track duration has mean 3.611 frames, median 1,
and maximum 22. States are 156 visible, 158 occluded, 616 dormant, and 2
abstained. There are 46 reappearance events and zero reveals. Mean confidence
is 0.88891 (median 0.92888); mean risk is 0.11109 (median 0.07112).

The P03 evidence covers 123 frames from 0 through 281, with 159 intervening
unobserved frames. There are 166 source observations with multiple candidates,
up to 6 candidates per source. Motion is mixed: valid chain magnitude median is
0.492 px, 90th percentile 7.360 px, maximum 25.227 px. However, there are zero
front/rear cross-order candidates and zero admitted cross-order candidates.
P03 exposes no temporal surface identity. Thus the absence of a reveal reflects
the selected P03 interval/candidate evidence, not a tested failure or success of
the reveal state-machine transition. Reappearance records describe bounded
gaps in observations; they do not establish rear-to-front reveal identity.

## Determinism, outputs, runtime, and independent audit

Output root:
`$WORK/proj_adags/runs/phase9-depth-visibility-capacity/stage1b-association-audit/`.

- audit `preprocess/cut_roasted_beef/association-audit-v1.json`:
  SHA-256 `d3aabcf8754088f4c7e765ecf36c7080598011e175ee686df3099d62002a6468`;
- diagnostics `preprocess/cut_roasted_beef/diagnostics-v1.json`:
  SHA-256 `f409e57b71473fbfcfa4ad1a79da75fae925599a00fab0607a2daab9a35211e2`;
- terminal `executions/P9-VPL-S1B-D01-CUT-S20260726/terminal.json`:
  SHA-256 `0543587003eb0196b3ea2af6b287dfb64295cb0276a8817b037d48c919dcf8c7`;
- canonical scientific-content SHA-256:
  `a82cedda1f20a17e1e9eb42c94a5158fef397fd2141185e503a40a7a8dc1fa77`.

The in-job hash repeats exactly with alternate excluded runtime metadata, and
an independent post-run recalculation reproduces it. All 37 authority source
hashes matched after runtime. The full depth-visibility suite passes: 116 tests,
0 failures.

Job `50250624` completed `0:0` in 8:19 on `boost_usr_prod`. Resources were 1
node, 1 task, 8 CPUs, 64 GiB RAM, 2 hours, account `euhpc_d21_034`, QoS
`boost_qos_lprod`, and exactly one A100 reservation. Batch peak RSS was
20,044,584 KiB. CUDA devices were hidden before Python; Torch/model were not
loaded; CUDA API calls, tensor moves, kernel launches, mapped CUDA libraries,
application GPU-process rows, and application GPU memory were all zero.

Logs:

- `logs/P9-VPL-S1B-D01-CUT-S20260726_50250624.out`, SHA-256
  `c9e3959c630bc2c9e4ed5a203d40a76b36cb9d0f884256a33b9cf3276a3d1012`;
- `logs/P9-VPL-S1B-D01-CUT-S20260726_50250624.err`, SHA-256
  `1c4c78b28684d24a641c560210f8ed2e1f06d178cb6d6a2195b7177ef62519da`.

Independent audit is `WARN`, with no leakage/authenticity failure and negative
association-discrimination evidence:

- `audit/EXPERIMENT_AUDIT.md`, SHA-256
  `0ba607af7b467fe959c0919aa4c74b6323bc595e82e4939dafa098d5e41cc96f`;
- `audit/EXPERIMENT_AUDIT.json`, SHA-256
  `9433e031e34e5bbe35bbabc7003e7d8cf7592371f424c5e076df084ab729aa3e`.

## Read boundary, deviations, and next action

No cam00 RGB, train RGB, annotations, evaluation masks, model weights, new DA3,
new flow, reconstruction output, W&B, trainer, Gaussian lifecycle, rendering,
training, or evaluator threshold was used. One job was submitted; no retry,
or second lane occurred. The required Stage-1 commit is recorded above; no
Stage-1B push was made. The GPU was reserved only for scheduling. The runtime
authority continues to bind the exact pre-correction source bytes.

The only scope-relevant deviations are the disclosed control/instrumentation
defects: sign-negated forward flow had previously been reported as reversed
flow, and the matched temporal-offset descriptive aggregate double-counts chain
and step levels. Neither justifies a positive association claim.

The smallest scientifically justified next action is to freeze this negative
forensic record and stop integration. Any continuation requires a separately
approved proposal that (1) defines a semantically valid backward/direction
control or explicitly retires that label, (2) fixes the versioned offset
aggregate, (3) preselects an interval/scene using label-free evidence criteria
that actually contains cross-order candidate opportunities, and (4) redesigns
association before evaluation because the current no-flow replay is nearly
identical. Independent Gate-A review remains mandatory. No thresholds are
frozen from this audit.
