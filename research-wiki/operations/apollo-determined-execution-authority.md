# Apollo/Determined Execution Authority

Status: ACTIVE operational authority for EL-GS M0/M1 execution
(created 2026-08-11 with the M0 implementation; closes the previously
unrecorded Apollo migration — commits `4935ec3`, `f2ec5f8`, `03ba4c8`,
`e055248` of 2026-08-07/08 built runtime images and smoke task configs
but recorded no wiki decision and no submission path). Full design
rationale: [[operations/elgs-m0-m1-implementation-plan]] §10; this
page is the operational summary implementers and auditors read.

## Cluster facts (historical evidence + this migration)

- Scheduler: Determined AI (`determined==0.38.0` client; base images
  `determinedai/environments:cuda-11.8-pytorch-2.0-gpu-0.31.1`).
- Pools: `dgx` (V100) and `hopper` (H100). Hardware policy: M0 S0
  preflight on both pools; M0 smoke and all M1 census cells on
  `hopper` only; any pool switch is a new ledger entry, never silent.
- Images: `sudarshaniyengar/adags:apollo-{v100,h100}-v1` (Docker Hub).
  EVIDENCE-BEARING runs reference images by DIGEST
  (`sudarshaniyengar/adags@sha256:...`) — mutable tags never pin bits.
  The one planned M1 image revision was executed 2026-08-11:
  `apollo-h100-v2` = the v1 Dockerfile + commit-pinned CoTracker3
  (`facebookresearch/co-tracker@82e02e80`) + imageio 2.34.2 /
  imageio-ffmpeg 0.5.1; built locally with the in-Dockerfile
  `validate_apollo_runtime.py --build-check` gate; pushed; MANIFEST
  DIGEST for all evidence-bearing M1 references:
  `sha256:a2877f26cb8528454fe45e701ce638a6042dd68155fb5359cb7edc608a4a7816`.
  Tracker weights are NOT baked — separately manifested artifact.
- Storage roots (inside containers): project
  `/apollo/users/sri/proj_adags`, raw data `data/` (read-only),
  runs `runs/elgs/<run_id>/`, logs `logs/`. VERIFIED 2026-08-11 by
  the one-time S0 mount probes (`det cmd run ls`, hopper task
  `c25ac11f`, dgx task `0c098195`): /apollo automounts on BOTH pools,
  both images pull and run, and `data/n3v` holds all six scenes.
  Master `determined.intern.denayer.be` (0.38.0), CLI 0.38.1, user
  `sri`; pools live at probe time (hopper 3 slots, dgx 8).
- Code reaches the container ONLY as the uploaded `det e create`
  context produced by `git archive <commit>` — never from the shared
  worktree `/apollo/users/sri/proj_adags/repo/adags` (the historical
  `work_dir`), which the in-container `runtime_assertions()` refuses.
- Workspace/project: `adags` / `elgs` (owner decision 2026-08-11;
  created-if-absent by preflight).

## The submission path

`scripts/submit_apollo.py` (subcommands: `preflight`, `submit`,
`status`, `logs`, `cancel`, `audit`) + the single experiment template
`det_exp_apollo.yaml` (placeholders NAME/POOL/IMAGE_REF/
ENTRYPOINT_ARGS/RUN_DIR; every deviation from the historical
`det_cfg_apollo_*.yaml` documented in its header, schema-verified
against Determined 0.38). The historical cfg files stay untouched for
`det cmd run` probes.

Submit flow (evidence-bearing): execution-closure check (dirty or
untracked content inside the execution-relevant set ⇒ REFUSE;
unrelated dirt listed as excluded) → O_EXCL cell claim
(`claims/<cell>__r<n>.json` on shared storage; EEXIST = duplicate
blocked) → `git archive` context materialization + canonical config
hash → O_EXCL run manifest (commit, config+prereg hashes, image
digest, pool, seed, dataset manifest hash, run dir, argv, projected
GPU-h, `evidence_bearing`) written into the context → `det e create`
→ experiment ID into the claim sidecar + one O_APPEND ledger line
(schema `elgs-apollo-ledger-v1`; the APOLLO copy is authoritative).
`--dirty-smoke` overrides closure but stamps
`evidence_bearing: false`. `max_restarts: 0`; a Determined-initiated
re-run is detected by the entrypoint (existing manifest/checkpoint ⇒
resume explicitly or abort) and ledgered as a distinct event.
Cancellation: `det experiment kill` + `cancelled` ledger event +
claim annotation. Completion is audited (`audit` subcommand:
terminal state + artifact inventory + immutable `terminal.json`) —
submission is never completion.

Evidence-bearing commits must exist off-workstation before
submission (push to origin, or a recorded `git bundle` beside the
run dir).

## Preflight (smallest sufficient; no broad requalification)

Per session: `det --version` reachability; workspace/project
create-or-verify; one `det cmd run ls` mount probe of the project
roots (closes the UNVERIFIED automount assumption); image digest
resolution. Per submission: template render check, closure, claim,
dataset-manifest presence, claims/ledger writability. In-container
gate of a session's first task:
`validate_apollo_runtime.py --require-gpu --expected-capability
{7.0|9.0} --repo . --scene <path>`.

## Failure classification and retries

`infra_failure` (CUDA/OOM/node/scheduler) vs `scientific_failure` in
the ledger. Infra failures: retryable max 2 via claim key
`<cell>__r<n>` under the verified-defect invalidation rule of
[[operations/elgs-m0-m1-implementation-plan]] §11.2. Scientific
failures are results, never retried. W&B stays disabled on Apollo
(historical `WANDB_MODE=disabled` retained): metrics travel as run
artifacts; git/config provenance travels in the manifest, replacing
the W&B-only metadata path the trainer had.
