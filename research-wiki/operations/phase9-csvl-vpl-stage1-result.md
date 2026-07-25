# Phase 9 CSVL-VPL Stage 1 Result

Date: 2026-07-26  
Branch: `codex/hpc-orchestrator-bootstrap`  
Run: `P9-VPL-S1-D01-CUT-S20260726`  
Slurm job: `50246056`  
Outcome: `STAGE1_NO_GO`

## Decision

The Stage-1 implementation, controlled fixtures, provenance, immutability,
determinism, and CPU-only execution pass. The real `cut_roasted_beef` diagnostic
does not provide adequate camera-/flow-specific association evidence for positive
Gate-A admission.

The result is a scientific no-go for the current association design, not an
ordinary implementation failure. The valid run is not materially stronger than
the mismatched controls: camera-swapped flow improves mean confidence and error,
while reversed and temporally offset flow are nearly indistinguishable from
valid flow. Sensitivity only to the deliberately extreme corrupted-flow control
is insufficient to rule out a P03 geometry/occupancy proxy.

Do not integrate this ledger into training, primitive lifecycle, reassignment,
or reconstruction. Independent Gate-A admission remains required for any future
reconsideration.

## Scope and implementation

Stage 1 adds immutable schemas
`phase9-csvl-vpl-stage1-ledger-v1` and
`phase9-csvl-vpl-stage1-diagnostics-v1`, strict P02 validation, calibration-only
scene indexing, a focused surface-hypothesis association module, a Stage-1
runner action, a one-run Slurm matrix, exact-worktree authority, and controlled
tests.

`track_id` means only a deterministic algorithmic association hypothesis. It is
not evidence of proven physical surface identity.

Observed records expose source observation, camera/time ancestry, P01/P02/P03
bindings, target-camera front/rear order, visibility state, confidence, risk,
uncertainty, and abstention. Dormant records are bounded to five frames, retain
identity descriptors only, set current xyz and optical depth to null, and set
depth order to `unknown_not_observed`. Reveal and reappearance are transition
events; ambiguity has no track ID.

## Sealed inputs

Top-level ordinary SHA-256 bindings:

- P01 arrays:
  `fc5a5ae31bfdf674ca5d6e5027c869feadd829ef2147ae3b4931a5b631e0bb42`
- P01 manifest:
  `c14ea2d4432470c24f7bad043f3a5e4308a8cedca4fe4c60843cfd7f5069f22b`
- P01 terminal:
  `b68b25b574a239dbe7136b0ca3b940628ea9cffbf5a56e7868500940c1e89cc0`
- P02 manifest:
  `67c4ac7ed23b143f1410397f508a781bdd138e79688b62613508b09964e46f93`
- P02 terminal:
  `97c79317d455e31aed5b1a0291ba2fa9afd9282a5fff08a8c8c09eca20f0c32b`
- P03 ledger:
  `50d3d2abc1e6c0ce9dd9cd29bbe610e8cdb69fd164e22cba8fdc8d50b0baf35c`
- P03 terminal:
  `dd45de6b3e0cbe36eec4a88aa1f7537a8761042b43e5154508fc044754612c67`
- train calibration:
  `5f45aa1b9aadbcb6d623ae6376a97d9f3c8188a86e934f406da450d7083bb679`
- test calibration:
  `1900f8b23ff91cde46a1a65f467b51e11ae877c7015f22bb3249303eceb7ab89`
- Stage-1 config:
  `1629c46f60b38d6df412d31e1d44f722730d8efa406e87580bb837509cdab942`

The ledger additionally binds all 1,107 consumed P02 NPZ records by exact path,
schema, producer revision, ordinary file SHA-256, flow-content SHA-256, and
validity-mask SHA-256.

## Output and determinism

Output root:
`$WORK/proj_adags/runs/phase9-depth-visibility-capacity/stage1-v1/`

- ledger:
  `preprocess/cut_roasted_beef/surface-hypothesis-ledger-v1.json`
  - ordinary SHA-256:
    `3dc1cff29ee34ca6141c0ee40171605ad8fbe9f2ad3c31d1c71bbbc26b0651b8`
  - canonical scientific-content SHA-256:
    `f0d1b8678574b8e1c52216a75fc5cf04ebaa7635fd68c1d79330a96e573dbdc2`
- diagnostics:
  `preprocess/cut_roasted_beef/diagnostics-v1.json`
  - ordinary SHA-256:
    `80154edd5e8eb1ca0c5b2831e63e5b4f11919e55630175d6acc9a41bc733767c`
- terminal:
  `executions/P9-VPL-S1-D01-CUT-S20260726/terminal.json`
  - ordinary SHA-256:
    `8db32fa81aeb541cb53aeae74a7abdfc2849f00be8a163a51eee31f00b8767a3`
- resolved execution:
  ordinary SHA-256
  `d0da50dea1ec584975b4dbfbd3316211cb6d0c98904ea5eb4855d005a3dd5bb8`
- exact-worktree authority:
  ordinary SHA-256
  `cf45d8ce30597d40d53794eff0b78e151f0aaee18207cd389639a79ab05b7fbf`

The canonical payload excludes only `timestamp_utc`, `slurm_job_id`, and
`absolute_output_root`. The in-job exact replay and an independent post-run
recalculation both reproduce
`f0d1b8678574b8e1c52216a75fc5cf04ebaa7635fd68c1d79330a96e573dbdc2`.
All 31 exact-worktree source hashes still match the authority artifact.

## Controlled verification

The focused depth-visibility suite passes exactly: 111 tests, 0 failures.
It covers stable two-layer hide/reveal, bounded dormancy/reappearance, split
abstention, merge abstention, no fabricated hidden geometry/order, complete
permitted-camera ancestry, prohibited reads, z/depth inversion, deterministic
hash replay, fail-closed missing/mismatched/corrupt/reversed flow, and the
corrupted/reversed/camera-swap/temporal-offset directionality fixtures.

## Real diagnostic

Job `50246056` completed with exit `0:0` in 4 minutes 58 seconds on
`boost_usr_prod`, node `lrdn0085`. Requested resources were 1 node, 1 task,
8 CPU cores, 64 GiB memory, 2 hours, account `euhpc_d21_034`, QoS
`boost_qos_lprod`, and exactly one GPU reservation.

The application remained CPU-only:

- `CUDA_VISIBLE_DEVICES=""`;
- `NVIDIA_VISIBLE_DEVICES=none`;
- application device `cpu`;
- Torch not imported;
- no model loaded;
- zero tensor-to-CUDA moves;
- zero CUDA API calls or kernel launches;
- no CUDA runtime/driver library mapped;
- application PID absent from NVIDIA compute-process rows;
- zero meaningful application GPU memory.

The batch step used 3 minutes 5 seconds total CPU and 13,396,288 KiB peak RSS.

Valid-run distributions:

- 316 observations, 314 associated, 2 split-ambiguity abstentions;
- 108 tracks, 51 multi-frame tracks, 27 multi-frame rear tracks;
- rear tracks: 16 with at least 3 observations, 10 with at least 5, and 3
  with at least 10; maximum 15 observed rear frames across an 18-frame interval;
- 156 front and 158 rear observed-order records;
- 156 visible, 158 occluded, 616 dormant, 2 abstained records;
- 46 reappearance events across 28 tracks; no real front/rear reveal transition;
- mean confidence 0.88891, mean association risk 0.11109;
- median confidence 0.92888, median risk 0.07112;
- mean reprojection endpoint error 1.41836 pixels;
- mean normalized temporal-consistency error 0.10028;
- association coverage 99.367%, propagation coverage 65.190%, abstention 0.633%;
- maximum frame concentration 6 observations; top-frame fraction 1.91%;
- 158 unique P03 regions; top-region fraction 0.637%;
- provenance complete;
- 616 dormant records, maximum dormancy age 5, and zero fabricated dormant
  xyz/depth/order records;
- 158 rear records and zero missing permitted-camera ancestry records.

Control comparison:

| Input | Multi-frame | Rear multi-frame | Mean confidence | Mean risk | Mean EPE px |
|---|---:|---:|---:|---:|---:|
| valid | 51 | 27 | 0.88891 | 0.11109 | 1.41836 |
| corrupted | 0 | 0 | n/a | n/a | n/a |
| reversed | 52 | 27 | 0.88833 | 0.11167 | 1.42879 |
| camera swap | 52 | 27 | 0.92200 | 0.07800 | 1.01023 |
| temporal offset | 51 | 27 | 0.88580 | 0.11420 | 1.43804 |

The rear yield is nontrivial and not concentrated in one track or region, but
the mismatched-control behavior prevents scientific admission.

## Read boundary and deviations

No cam00 RGB, train RGB, annotations, evaluation masks, R009, model weights,
reconstruction outputs, W&B, DA3 regeneration, flow regeneration, render,
training, trainer, rasterizer, optimizer, densification, pruning, routing, or
primitive lifecycle action was used or modified.

One GPU was reserved only as the approved scheduling choice; it was not used by
the application. One job was submitted. No resubmission or additional lane was
launched. No commit or push was made.

There were no deviations from the approved scientific scope. The diagnostic
did reveal a scientific limitation not visible in the controlled fixtures:
insufficient specificity to camera/direction/time mismatch.

## Evidence required for future independent Gate-A admission

This run cannot be admitted positively. A future, separately authorized
candidate would need a pre-specified association design, frozen before
evaluation, that:

1. retains the current controlled fixture, immutability, provenance,
   uncertainty, abstention, and no-hidden-geometry guarantees;
2. preserves nontrivial multi-frame rear yield across multiple frames and
   regions;
3. makes valid camera-correct forward-time evidence materially stronger than
   camera-swap, reversed-flow, temporal-offset, and corrupted controls in
   association survival and calibrated risk/error distributions;
4. demonstrates that identity is not assigned mainly by P03 geometry proximity
   or occupancy;
5. receives independent Gate-A review before any trainer or primitive-lifecycle
   integration.

No thresholds are frozen from this outcome.

