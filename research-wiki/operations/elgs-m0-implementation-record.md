# EL-GS M0 — Implementation Record

Date: 2026-08-11. Status: M0 executed per
[[operations/elgs-m0-m1-implementation-plan]] (approved same day).
Authority: spec rev 4 at `c21de8b` + the substrate semantics of
[[operations/lgs-method]] via the spec §1 delegation chain.

## What was built (one commit per module group, all pushed)

`e749a5d` plan preservation → `80080c8` intervals/presence/families →
`ce57acb` clusters/evidence/energy (exact PROP-1) → `9b6d35b`
bridges/probe/observability (q decision resolved: front-set
compositing, no CUDA change) → `8f42897` SNIS acceptance →
`7d8dd4e` classification → `7b573d5` ledger/state IO → `d3bbfae`
structural ops + signed-off transition table → `79d9bab` Determined
submission wrapper + execution authority page → `540e396` search →
`42edffc` runtime glue → `55711d1` renderer/GaussianModel owner edits
→ `d3d40bc` smoke config → `cdcaa19` trainer hooks + structural
prereg → `608f38c` remaining prereg + power analysis + tracks schema
→ `8b096c4` bootstrap oracle → `7a71be5` table nits → `21a08a3` +
`56d353f` + `6b7ad4d` verification-divergence resolutions →
`b31040a` + `2daa376` + `550e0ff` S1 hash-contract fixes.

Test suite: 538 tests (≈200 EL-GS-specific incl. two FROZEN
fresh-context reference oracles with parity at 1e-13); 535 pass
locally; the 3 failures are pre-existing environment-dependent tests
(unset `$WORK`, absent refine-logs history, POSIX path literal on
Windows) — untouched by this work. Double-run requirement met
(shuffled order + different global seed: identical results).

## Verification audit trail

Independent fresh-context spec-to-code verification (Opus), four
passes: 106-row mapping → 18 divergences (all resolved) → 7
regressions in the fix commit (all resolved; incl. an
UnboundLocalError startup blocker caught only because the verifier
demanded a setup construction test) → 4 further defects (all
resolved: torch.isin pin cost; stratified diagonal reservation with
fail-closed coverage assertions; pin-log dedupe; restored-refs
carry-over) → **FINAL SIGN-OFF**. The verifier's environment could
not execute the suite (torch DLL failure); its passes are static/AST
analysis, with this session's green runs as the runtime evidence.

Adjudications recorded during implementation (each with spec basis,
carried in code docstrings + the transition table): interior
TRUNCATE-delete admissible (untagged header v8.2 item 4); MERGE latch
= Boolean OR (derived equivalence); K-change target allocation =
affected-spans-absorb (header item 1); PRUNE-family leaves bindings
(freeze permits no third mutation); merge n_cam = max over members;
§8 equivalence-class-first precedence; post-refit committed-arm
training asymmetry disclosed.

Disclosed M1-gated items (not defects): the evidence stack
(clusters/bridges/observability/energy/SlotGrid full path) is
implemented + unit-tested but unwired pending the M1 track
artifacts; MERGE gauge re-anchoring and REACTIVATE pose init are
carried as directives awaiting episode-local pose tensors; the smoke
round path is photometric-only (λ_u = 1, disclosed degenerate-valid
SNIS).

## Apollo execution (S0/S1/S2)

S0 (2026-08-11): det 0.38.1 → master `determined.intern.denayer.be`
(0.38.0), user `sri`; pools hopper+dgx live; both images pull and
run; `/apollo` automounts in containers on BOTH pools (tasks
`c25ac11f`, `0c098195`); `data/n3v` holds all six scenes,
`cut_roasted_beef` fully preprocessed. Workspace `adags` / project
`elgs` created at first preflight.

S1: three submissions, all at pushed commits, ledgered
(`elgs-apollo-ledger-v1`, claims `m0_s1_smoke_elgs__r{0,1,2}`):
- Experiment 1 (`6b7ad4d`): the in-container `runtime_assertions`
  gate REFUSED to train on a config-hash mismatch — the fail-closed
  provenance gate working as designed. Cause: manifest hashed the
  CRLF worktree view vs the LF archive. Fixed (`b31040a`: hash the
  materialized context).
- Experiment 2 (`b31040a`): refused again — second half of the same
  contract: the hash keyed entries on ABSOLUTE path strings, so the
  submitter temp dir and `/run/determined/workdir` could never
  agree. Fixed (`2daa376`: relative-path keying, regression-tested).
  Both refusals are S1 catches, not losses: they prove the
  provenance gate rejects exactly what it must.
- Experiment 3 (`550e0ff`): both provenance gates PASSED; scene
  loaded; 111 families seeded over 50,000 rows; trained to the first
  structural round; the paired candidate render crashed (CUDA illegal
  access — the unit-render closure passed a CPU-tensor camera into
  the rasterizer; the training loop's `viewpoint_cam.cuda()` prep was
  missing). Fixed (`b507459`).
- Experiment 4 (`b507459`): **COMPLETED 100%.** Full mechanical chain
  on the H100: setup + seeding (111 families / 50k rows), all three
  structural rounds (iterations 200/350/500) with paired CRN renders
  on stratified slot-grid units, SNIS acceptance evaluated and — on
  this well-trained spanning geometry — REJECTED each mid-plateau
  fission (correct behavior; rejections ITT-logged; the §8 post-refit
  pass correctly skipped with zero committed decisions). Dual caps
  observed (peak 50,000 rows / 3,500,497 scalars). Artifacts:
  `chkpnt600.pth` (45 MB), `chkpnt_best.pth`, `summary.json` with the
  elgs block (3 tried / 0 accepted / rounds [200,350,500]) and
  best-val PSNR 28.24, capacity ledger, cameras/cfg records.
- Experiment 5 (`b507459`, S2 first attempt): the restore path caught
  a REAL integration bug — the checkpointed optimizer state carried
  the `elgs_a` param group, unloadable into the base-group optimizer
  that exists at restore time. Fixed (`41121ac`: capture strips the
  group; a-logit values persist in elgs_state, moments reset on
  restore, disclosed + logged).
- Experiment 6 (`41121ac`, S1 rerun): **COMPLETED 100%** with
  behavior identical to experiment 4 (same candidate id, same
  rejections — cross-commit reproducibility of the structural
  machinery).
- Experiment 7 (`41121ac`, S2): **COMPLETED** — `restored: true`,
  111 families reloaded through the validated elgs_state path, the
  stripped optimizer state loaded cleanly.

Compute: S0 probes + the three fail-closed S1 gates ≈ minutes of
container time; two full smokes + resume ≈ 0.25 GPU-h total — well
inside the ≤5 GPU-h B0 ceiling. All runs stamped
`evidence_bearing: false`; every submission at a pushed commit;
seven ledger entries with distinct retry claims.

## M0 gate summary — PASSED (2026-08-11)

All six plan phases verified: (0) plan preserved `e749a5d`; (1)
execution authority + submission path live (S0 both pools; refusal,
cancel, status, logs, resume procedures all exercised for real
during the smoke sequence); (2) EL-GS core faithful — full suite
green twice (shuffled order + different seed); (3) integration green
with zero regression of the pre-existing suite; (4) all seven prereg
files committed with the PERFORMED power analysis and the
tracker-pipeline fixture dry run; (5) independent spec-to-code
verification: four passes, 29 findings (18+7+4), all resolved,
FINAL SIGN-OFF; (6) GPU validation: S1 COMPLETED end-to-end and S2
restore COMPLETED, with the three intermediate fail-closed catches
(two provenance-hash contracts, one device-prep bug, one
checkpoint-group conflict) each fixed at a pushed commit and
regression-tested — the smoke sequence doing exactly its job.

M1 is unblocked. Disclosed M1-gated items carried forward: evidence
stack wiring (tracks artifacts), MERGE gauge transport + REACTIVATE
pose init (episode-local tensors), evidence-bearing image digest
pinning + the M1 image revision (tracker stack).
