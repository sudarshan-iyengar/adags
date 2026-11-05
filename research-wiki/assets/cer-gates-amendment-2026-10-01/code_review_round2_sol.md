## Verdict

MATERIAL DEFECT

Reviewed on my own judgment; no ARIS skill was referenced.

## Confirmed defects

- **NOT FIXED — replacements can still fabricate replication.** [cer_reduce.py:93](/D:/adags-cer/agent/cer_reduce.py:93) checks distinct **labels**; [line 114](/D:/adags-cer/agent/cer_reduce.py:114) checks only `method` and `scene`. Neither checks `run_checks.run`. In memory, I supplied one scored run under three distinct replacement labels: the reducer accepted it and reported **18 observations**. The replacement input also remains a bare label mapping, with no infrastructure-failure classification. **Fix:** validate distinct underlying run identities and require the replacement ledger’s failure classification.

- **NEW DEFECT — inherited environment can disable the new commit guard.** Both submissions use `--export=ALL,...,EXPECT_COMMIT=$EXPECT`: [Ex4DGS:44](/D:/adags-cer/agent/submit_cer_ex4dgs.sh:44), [Multi4D:53](/D:/adags-cer/agent/submit_cer_multi4d.sh:53). Slurm documents that inherited variables “take precedence over any explicitly given environment variables.” An exported empty `EXPECT_COMMIT` therefore overrides the intended pin, and the launchers’ `if [ -n "${EXPECT_COMMIT:-}" ]` skips verification entirely: [Ex4DGS:37](/D:/adags-cer/agent/train_ex4dgs_cer.sbatch:37), [Multi4D:42](/D:/adags-cer/agent/train_multi4d_cer.sbatch:42). **Fix:** bind the intended pin in the submission environment or use an explicit export list; require it for confirmatory jobs. [Slurm documentation](https://slurm.schedmd.com/sbatch.html).

- **NOT FIXED — final-render evidence is still trusted rather than reconciled.** [ars_score.py:147](/D:/adags-cer/agent/ars_score.py:147) correctly rejects invalid discrepancies and wrong iterations, but never recomputes the discrepancy or checks the two recorded means. Executing the actual validation block accepted `mean_psnr_float=30`, `native_mean_psnr_debug_metrics=10`, `abs_diff_vs_native=0`; it also accepted both means as NaN with discrepancy zero. The debug-metrics read extracts only `iteration` at [line 130](/D:/adags-cer/agent/ars_score.py:130). **Fix:** require finite means, read the final native PSNR, and recompute and compare the discrepancy.

## Suspected issues

- **Runtime code stability remains procedural.** The hand-off says “Never pull while a job runs” at [line 103](/D:/adags/research-wiki/operations/cer-handoff-2026-10-01.md:103), but the launchers enforce no lock or immutable snapshot after their initial check. A snapshot or enforced exclusion covering training and rendering would settle this.

- **Analysis decisions remain pending.** The specs explicitly say “not frozen until accepted” for [§13.4](/D:/adags/research-wiki/operations/cer-ex4dgs-spec-2026-09-30.md:513) and require principal confirmation for [§14.2](/D:/adags/research-wiki/operations/cer-multi4d-spec-2026-09-30.md:304). A dated acceptance covering the statistical choices and benchmark selection would settle this.

## Unverified claims

- Census PASS is recorded in the plan log, but the referenced census JSON is unavailable locally.
- `ars_sub.sh` is unavailable; its duplicate-name refusal and concurrent-submission behavior remain unverified.
- Historical cluster tests, deployment, dry runs and submission state lack reachable execution evidence. I reproduced all **12 tests and five mutation detections in memory**; native execution remains blocked by SciPy’s temporary-file requirement.

## Missing

- Regression tests for duplicate underlying run identities, inherited empty/wrong pins, and inconsistent final-render records.
- The round-1 complete-input preflight remains unresolved: [submit_cer_multi4d.sh:38](/D:/adags-cer/agent/submit_cer_multi4d.sh:38) checks selected files and an identity directory’s existence, not complete readable image, mask and window-package inventories.