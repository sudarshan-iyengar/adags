---
title: "Stage-2 resume packet (2026-09-15) — decision 5 = option (iv), realdata_gate to D:, GO; what is done, what the blocked session left, exact steps"
date: 2026-09-15
evidence_bearing: false
---

# Stage-2 resume packet (2026-09-15)

The 2026-09-14/15 session prepared wave-2 stage 2 through spec §13.21
(commit `65b4a075a9f4d6b97771e2a76ba94c36731029da`, pushed; Leonardo
fast-forwarded) and received the user's decisions: **option (iv)** for
G-wrongmem-L, **move `runs/realdata_gate` to D:**, **GO**. It could not
execute them: the desktop app's permission classifier began refusing
every shell action (including remote reads) after the archive job text
containing a deletion was proposed, and states it keeps refusing for
the rest of that session. Nothing below has been run unless marked DONE.
Resume in a fresh session (or the same one switched out of auto mode).

## 0. State on the record (all DONE)

* §13.21 appended; four decisions applied; reducer guards + own-gap
  validation; launcher `ADAGS_SAVE_ITERATIONS`; 154 reducer tests pass
  locally and on Leonardo; hashes match the §13.21 table.
* L draws (decision 1, `draws_prefix<S>_final`, job 57752673), G-est
  programs (jobs 57753670–57753690), round-4 recount (57753922/925/927),
  plan of 84 cells (`stage2/stage2_plan.json` at the amendment commit,
  `stage2_inputs.sha256` `030a2c21…`), stage-2 scripts uploaded to
  `$W/agent-control/realdata/absfix2/stage2/` (cell template, warden,
  cross-check, submitter, montage, collector, STG chain, collect).
* Wave-1 GMIS/GONES precondition re-extraction: job 57761259 FAILED
  (needed `meta/train.log`), resubmitted as **57761925** with the
  metadata links. Partial result read at 00:16 CEST (job still RUNNING,
  ~10 min per cell): GMIS s0–s3 and GONES s0–s1 extracted at frames
  244 / 291 with rc 0, all rows in the box (7,002–7,326), zero-presence
  frames inside their own gaps (232… / 288…); GONES s2–s3 pending. Read
  the rest in `logs/absfix2_precond_wave1_shams_57761925.out` and hash
  the eight `wave1_cells/*/precondition.json` into the freeze record.
* Composites to `D:\adags-archive\leonardo\runs\realdata\absfix2\`:
  7 of 8 dirs copied and VERIFIED against
  `stage2/composite_manifest_*.sha256` (manifests also at
  `D:\adags-archive\leonardo\manifests\`); the copy of all 8 dirs
  COMPLETED at 23:48 (flame 4.4 GB, sear 4.3 GB on D:), but
  `sear_steak/build_id114_60_89_v2` is NOT yet verified (run
  `sha256sum -c` against its manifest first). Nothing removed from
  Leonardo yet.
* Charge census DONE and promoted to
  [[charge-absence-census-2026-09-15]] (negative at scene level on 6 of
  8 scenes; commit that page).

## 1. Files prepared but NOT applied (scratchpad of the blocked session)

`C:\Users\sucar\AppData\Local\Temp\claude\D--adags\38206e4c-8f32-4aec-8d7e-07414396e9e0\scratchpad\stage2\`:

* `edit_round4.py` — adds `local_floor_contribution_matched_draw`,
  `--l_mode local_floor --l_min_rows 1100 --l_cap 1.0` to
  `scripts/draw_membership_shams.py` (run from the repo root with the
  repo's python; asserts every anchor).
* `edit_round4_tests.py` — appends seven `local_floor` tests to
  `tests/test_draw_membership_shams.py`.
* `section_13_22_draft.md` — the §13.22 text with FILL_ placeholders.
* `archive_realdata_gate.sbatch` — manifest → rsync to scratch_large →
  verify → remove original → symlink (the step that triggered the block;
  run it deliberately, outside auto mode).
* `draws_final.sbatch` (adapt: `--l_mode local_floor --l_min_rows 1100
  --l_cap 1.0`, output `draws_prefix<S>_iv/`, job name `absfix2_draws_iv`).

If the scratchpad is gone, the rule is fully specified in
`section_13_22_draft.md`'s text above and in §13.21 option (iv).

## 2. Exact steps (in order; `$W=/leonardo_work/EUHPC_D36_068/sri/proj_adags`, `S2=$W/agent-control/realdata/absfix2/stage2`)

1. Locally: `python edit_round4.py`, `python edit_round4_tests.py`,
   `python -m pytest tests/test_draw_membership_shams.py -q` (expect all
   pass; the test arithmetic: 40 truth rows × 1.0, 60 local × 0.7, cap
   40 → 57 rows; floor 36 at row 52). Commit `scripts/
   draw_membership_shams.py` + the test file; push; `git pull --ff-only`
   on Leonardo.
2. Leonardo: redraw L on all 12 prefixes with the tracked CLI
   (`--l_mode local_floor --l_min_rows 1100 --l_cap 1.0 --contribution_key
   w_total --gap 60 89`, inputs exactly as `draws_final.sbatch`) into NEW
   `draws_prefix<S>_iv/`; ledger line; assert A/B row identity 24/24;
   record draw_n, positive/zero rows, mass ratio, sha256 per prefix.
3. Recount round 5 (`stage1/recount.sbatch` with
   `L_DRAWS_SUFFIX=_iv`), after `cp -n` of the round-4 reports to
   `box_recount_round4_L_final/`; L in-box must be ≥ 100 on 12/12
   (acceptance), else refuse that prefix and report.
4. Read job 57761925; if COMPLETED, the eight `wave1_cells/*/
   precondition.json` exist (frames 244/291).
5. `submit_stage2.py`: change `program_path` for `GWRONGMEM_L` to
   `draws_prefix%d_iv/program_gwrongmem_l.json`; `--plan`; check the 84
   planned cells and 4 missing (sear GESTMEM) are unchanged except the
   L hashes.
6. Write §13.22 from the draft with every FILL_ filled; JSON:
   `arm_roles.GWRONGMEM_L`, `FREEZE_LIST_STATUS.programs_wrongmem_L`,
   `pending_user_decisions: []`, `stage2_go`, `STAGE_2.frozen_commit`;
   regenerate `stage2_freeze.json` (the script text is in the ledger's
   2026-09-14 lines and in the blocked session's `leo_step18.sh`) and
   copy it to `research-wiki/assets/absfix-stage2-freeze.json`; commit;
   push; ff Leonardo (this commit is `FROZEN_COMMIT` for every cell).
7. `cindata`; verify `sear_steak/build_id114_60_89_v2` on D: against its
   manifest; then remove the eight composite dirs on Leonardo (ledger
   line with D: path + manifest hashes). Submit
   `archive_realdata_gate.sbatch` (ledger line); after it completes, scp
   the scratch copy to `D:\adags-archive\leonardo\runs\realdata_gate\`
   and verify against `stage2/archive_manifest_realdata_gate.sha256`.
8. `bash $S2/stg_chain.sh --go` (63 jobs with dependencies; ledger lines
   written by the script).
9. `python $S2/submit_stage2.py --go --frozen_commit <commit>` (84 cells;
   per-cell ledger lines, `submit/<tag>.sh`, `jids/<tag>.txt`).
10. `nohup bash $S2/stage2_warden.sh > $S2/warden.nohup 2>&1 &` on the
    login node; check `$S2/status.txt` every cycle; the certificate
    lasts 12 h (jobs run on regardless; only new ssh fails).
11. When all cells are COMPLETED: `python $S2/stage2_collect.py` (writes
    `manifest_stage2.json`; point the calibration GMIS/GONES entries at
    `wave1_cells/<tag>/`), run the reducer and hash its output BEFORE
    viewing montages (`stage2_montage.py` per scene/prefix), then the
    verdict page from the reducer output and primary inputs, every job
    id in the ledger; STG profiles from `runs/stg/<scene>_v3_full_seed<S>/
    f_box_profile.json` reported descriptively.

## 3. Rules that bit this session (carry)

* A shell action containing a deletion of a run directory can trip the
  desktop permission classifier for the whole session; run archive
  steps deliberately in default mode, and never let a job script delete
  before its copy verifies.
* Python edits via shell heredocs failed twice on quoting; write patch
  scripts to files and run them.
* The stage-1 recount used frame 245 for GMIS; the reducer's fixture and
  §13.21 freeze 244 (floor of the midpoint); `submit_stage2.py` and the
  wave-1 re-extraction use 244.

## 4. Executed (2026-09-15, 00:00–01:40 CEST)

Steps 1–6 ran as written (spec §13.22 is the record: draw job 57765868, recount round 5 jobs 57766941/57766942/57766945, plan regenerated, wave-1 sham preconditions job 57761925 COMPLETED 8/8, composites 8/8 verified on D:, archive-copy job 57767135 copy-only). The deletion steps of 7 were NOT run: the removal commands are proposed to the user separately and wait for approval. Steps 8–10 follow the freeze-record commit F (ledger `FROZEN_COMMIT F =`).
