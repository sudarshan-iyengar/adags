# Phase 9 CSVL-VPL Stage 1C Gate-C0 Authority

Date: 2026-07-26
Branch: `codex/hpc-orchestrator-bootstrap`
Stage-1B freeze commit: `d68b25db613ae245bdd83a7b9bfcfe9f6ff608cb`
Run: `P9-VPL-S1C-C0-CUT-S20260726`
Status: frozen before observation

## Scope and ordering

This authority admits only CSVL-VPL Stage 1C Gate C0: a deterministic,
label-free scan of existing sealed Stage-1 observations and their pre-score
P03/P02-supported candidate universe. It does not implement, execute, or admit
the Gate-C1 association redesign. Gate C1 may begin only if the immutable C0
artifact selects an admissible interval.

If no frozen window contains a front/rear cross-order candidate opportunity,
the task stops with `STAGE1C_NO_INFORMATIVE_INTERVAL`. It must not continue on
an occupancy-only interval.

No trainer, Gaussian, primitive lifecycle, capacity, reconstruction, learned
association, new DA3, new optical flow, annotation, evaluator, or W&B work is
authorized.

## Stage-1B freeze

The complete Stage-1B implementation and durable negative result are frozen by
commit `d68b25db613ae245bdd83a7b9bfcfe9f6ff608cb`. The durable result records:

- primary outcome `STAGE1B_CONTROL_OR_BINDING_DEFECT`;
- secondary finding that removing flow changes only 3 of 206 selected edges;
- secondary finding that the evaluated evidence has zero cross-order
  candidates and therefore cannot test reveal.

The post-runtime matched-temporal description was corrected without replacing
any immutable output. The chain-level distribution has 1,911 values with counts
`-2:276`, `-1:647`, `+1:684`, `+2:304`; the constituent step-level distribution
has 5,335 values. Read-only replay preserves candidate-row hash
`1b246511559beeb43558a4890ac9dc2fd251ab950a35fbbfd1f698329488d870`,
full-track hash
`5973436bee97107f66456f8b84155ab81ebbc6331b6109891b7ff91b067dee17`,
and canonical Stage-1B hash
`a82cedda1f20a17e1e9eb42c94a5158fef397fd2141185e503a40a7a8dc1fa77`.

## Frozen label-free selection rule

The exact config is
`configs/depth_visibility/csvl_vpl_stage1c_c0_v1.json`, SHA-256
`6b77b9ec224d2661e1d147405f1ad3c09e6b8e1b286e4515146c3f2b115f39d8`.
It is part of the exact-worktree runtime binding.

The scan:

1. uses the sealed Stage-1 observation set and exact Stage-1 candidate edges;
2. uses candidates only after the P03 world-displacement prefilter, common
   permitted-camera count of at least two, and sealed P02 validity-mask support;
3. does not read candidate score, admission, confidence, track, evaluator, or
   reconstruction success when selecting an interval;
4. enumerates closed 30-frame windows with stride 15 and one end-aligned tail
   window when the regular stride omits it;
5. assigns a candidate to a window only when both endpoints are inside;
6. admits a window only when it has at least one front/rear cross-order
   candidate, at least one source with multiple plausible candidates, and
   complete P03 endpoint, camera, calibration, flow-record, and time provenance;
7. selects at most one development interval and no secondary interval.

Admissible windows are ordered lexicographically, without fitted weights, by:

1. cross-order candidate count, descending;
2. cross-order candidates with temporal gap greater than one, descending;
3. multi-candidate source count, descending;
4. later compatible edges without an earlier compatible destination from the
   same source, descending;
5. camera-direction disagreement, descending;
6. flow directional diversity, descending;
7. total candidate count, descending;
8. start frame, ascending.

The rule freezes no claim-grade association threshold. Its only numeric choices
are window enumeration and the already sealed Stage-1 candidate support bounds.

## Reported quantities and boundaries

Every window reports:

- P03 observation and observed-frame counts;
- source and candidate counts, multiple-candidate sources, and candidates per
  source;
- front/rear observation counts and all cross-order transition counts;
- later compatible gap opportunities and temporal-gap distribution;
- permitted-camera support;
- P02 manifest validity coverage and sampled boundary support;
- flow magnitude, occupied directions, and circular directional diversity;
- per-candidate camera magnitude and direction disagreement;
- projected P03 quantization-support overlap possibility;
- complete endpoint/camera provenance fraction;
- every failed admission reason.

The gap metric is a candidate opportunity, not proof of physical disappearance.
The projected-overlap metric asks whether quantization disks may overlap; it is
not observed support-mask IoU. Candidate IDs remain geometry hypotheses and do
not establish temporal or physical identity.

## Immutable inputs

- Stage-1 terminal SHA-256
  `8db32fa81aeb541cb53aeae74a7abdfc2849f00be8a163a51eee31f00b8767a3`;
- Stage-1 ledger SHA-256
  `3dc1cff29ee34ca6141c0ee40171605ad8fbe9f2ad3c31d1c71bbbc26b0651b8`;
- Stage-1B terminal SHA-256
  `0543587003eb0196b3ea2af6b287dfb64295cb0276a8817b037d48c919dcf8c7`;
- Stage-1B audit SHA-256
  `d3aabcf8754088f4c7e765ecf36c7080598011e175ee686df3099d62002a6468`.

Stage-1 and Stage-1B preserve exact transitive P01/P02/P03, calibration,
schema, version, path, and cryptographic bindings. Gate C0 opens only the two
derived immutable scientific files and verifies those transitive bindings.

## Read boundary

Before selection is frozen and executed, no annotations, evaluator masks,
cam00 RGB, train RGB, reconstruction outputs, checkpoints, labels, or W&B data
may be read. This authority does not authorize a post-selection human audit;
none is necessary if no interval is admitted.

No hidden geometry, xyz, ownership, depth, or temporal identity is generated.
Window outputs contain counts, distributions, exact observation/candidate IDs
through the bound inputs, and provenance summaries only.

## Runtime authority

Exactly one Slurm job may be submitted after static tests and exact-worktree
preparation. Before any attempted submission or resubmission, both `squeue` and
`sacct` must be checked. No second job is authorized.

Resources are frozen as one node, one task, 8 CPUs, 64 GiB RAM, 2 hours,
partition `boost_usr_prod`, account `euhpc_d21_034`, QoS
`boost_qos_lprod`, and exactly one A100 reservation solely for scheduling.
The wrapper hides CUDA before Python. The application must import no Torch or
model, map no CUDA libraries, launch no CUDA kernels, appear in no NVIDIA
compute-process row, and allocate zero application GPU memory.

Outputs are under
`$WORK/proj_adags/runs/phase9-depth-visibility-capacity/stage1c-association-redesign/`.
Logs are `logs/P9-VPL-S1C-C0-CUT-S20260726_%j.{out,err}`.

The immutable artifact schema is
`phase9-csvl-vpl-stage1c-interval-selection-v1`. Canonical scientific content
excludes only timestamp, Slurm job ID, and absolute output root. Repeated
construction with alternate excluded metadata must reproduce exactly.

## Decision boundary

`STAGE1C_NO_INFORMATIVE_INTERVAL` is mandatory if the admissible-window count
is zero. If C0 admits a window, this job reports only that admission; Gate-C1
implementation and any additional runtime require new, explicit authority.
Neither branch authorizes trainer or primitive-lifecycle integration.
