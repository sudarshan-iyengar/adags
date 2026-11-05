# Re-review (round 2): fixes for your code review of the CER scorer, reducer and submission scripts

You reviewed these files in round 1; your review is `code_review_sol.md` in the folder this prompt was saved in, and is
reproduced at the end of this prompt. Your working directory is the same (`agent/` holds the current files). Check ONLY
whether each round-1 finding is now fixed correctly, and whether the fixes introduced a new defect. Do not re-review
unchanged code.

Responses to round 1:
1. CER failures: `cer_reduce.py` takes `--failures` (JSON {CER label: {job, reason}}; numeric job id and a reason
   required; labels must be planned CER runs). A declared failure is not loaded; its scene is design-without-power with
   the record in `rule4_preconditions[scene].cer_runs_failed`; values use the runs that exist. Any other missing score
   still refuses.
2. Replacements: unknown replacement keys refuse; duplicate labels within any role refuse; every loaded score's
   `run_checks.method` and `run_checks.scene` must equal the manifest's (the scorer writes both).
3. Smoke overrides: `submit_cer_multi4d.sh` refuses if EXTRA or RENDER_AT is set and passes `EXTRA=,RENDER_AT=`
   explicitly in `--export`; the scorer already refuses `extra != ""`.
4. Multi4D entry gates: plan §4b gate 1 (Ex4DGS proof first) was DROPPED by the principal on 2026-09-30 (plan log:
   "CER is built on both Ex4DGS and Multi4D, in parallel. This changes §4b ... Gate 1 no longer applies"). Gate 2
   (census) passed on 2026-09-30; the script now requires `checks/wave1/m4d_census_summary.json`
   `gate_pass_all_layouts == true`. FROZEN must still name the plan-log entry.
5. Mutable worktree: both launchers now refuse, before creating anything, unless the worktree HEAD equals
   EXPECT_COMMIT (passed by both submission scripts) with porcelain 0 when the job starts (diffs `fix_launch_*.diff`).
   Snapshots were not adopted: the worktrees are only changed by a manual bundle pull, which the hand-off forbids while
   a job runs from them, and the check now closes the window between submission and start.
6. Tests: 12 tests; rule 3 PASS asserted; whole-frame-only and ordinary-only harm; Multi4D FAIL and
   design-without-power; benchmark-proof boundary (6 with each event type, 5, 6 missing an event type); failures,
   duplicate replacement, unknown replacement key, scene mismatch, replacement read. Five mutations (your rule-3 flip,
   no method/scene check, no duplicate check, no failure-record validation, benchmark threshold 5) are each caught.
7. Final-render record: `ars_score.py` now requires a numeric `abs_diff_vs_native` in [0, 1e-3], `n_frames == 300`
   and `iteration == 20000` (both Multi4D methods).
8. Ordinary share: `None` when `memory_mass_gap != 0`; excluded from the per-scene summaries.
Suspected/missing: the zero-share wording is in the reducer output (`note`); the "came from the pinned gates()" claim is
removed from the reducer and the spec. The statistical reading (unadjusted return bound as frozen; arm + scene blocking
in rule 3; Multi4D run 1 as the benchmark as frozen) is left to the principal and listed for decision. Duplicate
submission: `ars_sub.sh` refuses a job name that already has a `jids/NAME.jid` (written at submission), and both
preflights refuse existing run directories or job records; submissions are sequential from one shell.

Return: per finding FIXED / NOT FIXED / NEW DEFECT with file:line, then a verdict.
## Verdict

MATERIAL DEFECT

Reviewed on my own judgment; no ARIS skill was referenced. Scores: reducer **5/10**, tests **4/10**, scorer **6/10**, preflights **3/10**.

## Confirmed defects

- **Major — CER failures cannot receive the promised verdict.** Amendment §13.5 requires a CER assertion failure to make its scene DESIGN_WITHOUT_POWER. [cer_reduce.py:93](/D:/adags-cer/agent/cer_reduce.py:93) instead collects every absent score and raises `"missing … planned score(s)"` at lines 98–99. There is no failure-record input. **Fix:** accept explicit, evidenced CER-failure records and propagate their scenes through rules 1–3; continue refusing unexplained missing scores.

- **Major — replacements can fabricate replication.** [cer_reduce.py:78](/D:/adags-cer/agent/cer_reduce.py:78) substitutes labels without checking uniqueness or score identity. I mapped replicates 2 and 3 to replicate 1: the reducer accepted three identical labels and still reported **18 observations**, although only 16 distinct input files were used. It also never validates score method/scene against the manifest. **Fix:** require unique run identities, validate method and scene, reject unknown replacement keys, and require the replacement ledger’s failure classification.

- **Major — confirmatory submissions inherit smoke overrides.** [submit_cer_multi4d.sh:50](/D:/adags-cer/agent/submit_cer_multi4d.sh:50) uses `--export=ALL`; [train_multi4d_cer.sbatch:29](/D:/adags-cer/agent/train_multi4d_cer.sbatch:29) reads inherited `RENDER_AT` and `EXTRA`, then applies them at lines 78–86. Neither is checked by the submission preflight. Consequently, an exported smoke `EXTRA` can submit nine shortened confirmatory jobs. [Slurm documents that ALL propagates the caller’s environment](https://slurm.schedmd.com/sbatch.html). **Fix:** reject these overrides and clear them before submission; assert the confirmatory settings again inside the launcher.

- **Major — Multi4D’s entry gates are unenforced.** The supplied §4b requires passing Ex4DGS causal proof and the persistent-bank census threshold. [submit_cer_multi4d.sh:24](/D:/adags-cer/agent/submit_cer_multi4d.sh:24) checks only that `FROZEN` is nonempty; the remaining preflight contains no proof or census check. **Fix:** require and record evidence for both entry gates, together with the accepted design record.

- **Major — submission-time pins do not bind the eventual training.** Both launchers execute from mutable worktrees. [train_ex4dgs_cer.sbatch:49](/D:/adags-cer/agent/train_ex4dgs_cer.sbatch:49) and [train_multi4d_cer.sbatch:53](/D:/adags-cer/agent/train_multi4d_cer.sbatch:53) merely *print* the branch, commit and cleanliness before training. A checkout change while jobs queue therefore permits training the wrong code; later scoring refusal only detects the wasted run. **Fix:** use immutable snapshots and verify expected commit/config/input hashes immediately before execution.

- **Major — the tests miss a broken verdict path.** [test_cer_reduce.py:95](/D:/adags-cer/agent/test_cer_reduce.py:95) checks rule 3’s `"status" == "OK"`, not its PASS verdict; the Multi4D test similarly checks only scene count at line 212. In an in-memory execution, changing every rule-3 PASS to FAIL left **all ten tests passing**. **Fix:** assert rule-3 PASS explicitly and cover ORD-only harm, Multi4D FAIL/DESIGN_WITHOUT_POWER, and benchmark-proof boundary cases.

- **Major — final-render validation accepts non-finite evidence.** [ars_score.py:146](/D:/adags-cer/agent/ars_score.py:146) rejects `None` or `abs_diff_vs_native > 1e-3`. **NaN passes that comparison**, as do negative discrepancies. The recorded iteration is copied but never required to equal 20,000. **Fix:** require finite, valid numeric fields and iteration 20,000; recompute the discrepancy against the final debug-metrics record.

- **Minor — a failed memory gate produces a mislabeled bank share.** [cer_reduce.py:129](/D:/adags-cer/agent/cer_reduce.py:129) always reports `1 − residual − transient` as `"ordinary_persistent_share"`. When `memory_mass_gap != 0`, that remainder includes memory rows, contrary to §14.3’s stated condition. **Fix:** report it as unavailable or explicitly as the persistent remainder on those runs.

## Suspected issues

- **The statistical reading requires an explicit principal decision.** [cer_reduce.py:234](/D:/adags-cer/agent/cer_reduce.py:234) uses Holm rejection for the gap, but the return clause uses only the unadjusted bound. With three gap p-values of 0.00001 and three return p-values of 0.03, return Holm-adjusted p-values are 0.09, yet clause (b) can pass. This follows the literal bound clause but does not establish Holm-significant return non-inferiority. Settle the intended reading, the conventional pooled-variance standard error, rule-3 scene blocking, and the **Multi4D run-1 benchmark choice** at [cer_reduce.py:266](/D:/adags-cer/agent/cer_reduce.py:266) before results. The implementation’s Holm arithmetic matches the [reference implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html).

- **Duplicate submission protection may be incomplete.** Both launchers separate the existence check from `mkdir -p`; neither reserves a run atomically. Whether duplicate jobs can actually be submitted depends on the unavailable `ars_sub.sh`. Inspect that helper and test concurrent submission.

## Unverified claims

- Cluster installation, smoke/negative checks, historical mutation results, dry runs, “nothing submitted,” timings, census values, reproduction bands and identity-agreement results: their logs/artifacts are absent.
- Native execution of the ten-test suite was blocked by SciPy’s import-time temporary-file requirement in this read-only environment. The in-memory adaptation passed; that does not verify cluster execution.
- `ars_sub.sh`, environment scripts, `m4d_final_render.py`, evaluator and run-report implementation are unavailable, preventing an end-to-end orchestration review. Multi4D’s ECCV 2026 identification is supported by its [author project page](https://batfacewayne.github.io/Multi4D.io/).

## Missing

- Input preflight validates selected files/directories, not complete readable image, identity-mask and window-package inventories.
- The required Ex4DGS zero-share ambiguity wording is absent from reducer output.
- Cut the claim that matching the superseded `pass` formula proves a file **“came from the pinned gates()”**. It proves internal consistency; the supplied doctored-run check itself demonstrates the distinction.