# Review request: CODE review round 2 (focused) and the revised readings, "CER on Multi4D, early gate with a gated transient copy path"

Same reviewer. Your round-5 review (inlined at the end) gave CODE READY WITH FIXES and PREMISE CHECK QUALIFIED. Check:
1. Fix 1: D:\adags-cer\agent\train_multi4d_cer_early_tr.sbatch now records exit_code after the tree check (end of file); F9 in
   D:\adags-cer\multi4d-early-tr\test_cer_early_tr.py (commit c087731) requires trees_changed 0 and equal before/after states;
   commit 96f6fe0 changes F9's §14.2 clause to residual mass outside the gap exactly 0 only (the residual share and the stale
   `pass` field are descriptive per spec §14.2/14.6, D:\adags\research-wiki\operations\cer-multi4d-spec-2026-09-30.md). The F9
   test now passes on the integration run (job 60750535). The test diffs are inlined below.
2. The revised readings (draft plan section 14 and draft plan-log entry, inlined): do they now state only what the evidence
   supports (your findings 2-5)? Anything still overstated or wrong?
3. Any remaining CODE defect (5afa119 is unchanged)?
Output: CODE verdict (READY / READY WITH FIXES / MATERIAL DEFECT), READINGS verdict (SOUND / QUALIFIED / UNSOUND), numbered
findings with evidence and fix. Terse.

=============================== TEST DIFF 5afa119..96f6fe0 (F9 parts) ===============================
diff --git a/test_cer_early_tr.py b/test_cer_early_tr.py
index 1ef74e3..73c9790 100644
--- a/test_cer_early_tr.py
+++ b/test_cer_early_tr.py
@@ -1752,7 +1752,13 @@ def f9(a, R):
     tr_lab = state.get("tr_labels")
     # the clause that needs a labelled donor: vacuous (not passed) if none was ever eligible
     eligible = lifts.get("labelled_eligible", 0)
+    # the launcher records the worktree before as commit, branch and porcelain count, and the other trees as one state line
+    before = {"substrate_state": "%s %s %s" % (prov.get("substrate_commit"), prov.get("substrate_branch"), prov.get("substrate_porcelain")),
+              "early_tree_state": prov.get("early_tree_state"), "cer_tree_state": prov.get("cer_tree_state"),
+              "pinned_tree_state": prov.get("pinned_tree_state")}
     c = {"launcher exit code 0 (no sentinel fired)": prov.get("exit_code") == "0",
+         "launcher: trees_changed 0 and every tree's state after equals its state before": (
+             prov.get("trees_changed") == "0" and all(v and v == prov.get(k + "_after") for k, v in before.items())),
          "schedule order coarse <= merging == begin < phase3 < densify_until < iterations": coarse <= merge == b < p3 < du < it,
          "cfg_args written by train.py holds cer_transient_gate=1": bool(m_flag) and m_flag.group(1) == "1",
          "cer_begin.json: early mode, begin_iter = the command-line value, cer_transient_gate 1": (
@@ -1777,8 +1783,8 @@ def f9(a, R):
          "14.2 (transient rows counted) memory share after the return >= 0.5": gates["memory_share_return"] >= 0.5,
          "14.2 (transient rows counted) memory mass in the gap exactly 0": gates["memory_mass_gap"] == 0.0,
          "the label-1 transient share of the gap region exactly 0": gates.get("memory_transient_share_gap_region") == 0.0,
-         "14.2 residual mass outside the gap exactly 0, residual share in the gap >= 0.2, gates pass": (
-             gates["residual_mass_outside_gap"] == 0.0 and gates["residual_share_gap"] >= 0.2 and gates["pass"] is True),
+         # spec 14.2: the residual share and the stale `pass` field (the superseded 0.20 rule) are descriptive, never read
+         "14.2 residual mass outside the gap exactly 0": gates["residual_mass_outside_gap"] == 0.0,
          "rows frozen from the freeze to the end": drift["rows"] == gates["rows_at_end"] if "rows" in drift else False,
          "saved label lengths = saved rows = rows at the end, all three banks": (
              state["fg_labels"].shape[0] == rows_ply["fg"] == gates["rows_at_end"]["fg"]
@@ -1798,7 +1804,8 @@ def f9(a, R):
     out["lifts"] = lifts
     out["surviving_label1_transient_rows"] = {"at_freeze": drift["per_bank"]["tr"]["rows_label1"], "at_end": gates["memory_rows"].get("tr")}
     out["scene"] = prov.get("scene")
-    out["scene_is_carton"] = "carton" in prov.get("scene", "").lower()
+    out["scene_is_carton"] = prov.get("scene", "") == "ARS-L2-A0"  # ARS-L2-A0 is the carton same-pose scene
+    out["descriptive"] = {"residual_share_gap": gates["residual_share_gap"], "pass_superseded_rule": gates["pass"]}
     out["mean_psnr"] = rec.get("mean_psnr_float")
     vacuous = [] if eligible > 0 else ["no labelled row was ever eligible: the labelled-copy clause is vacuous, not passed"]
     check("F9 integration", all(c.values()), {"failed": [k for k, v in c.items() if not v], "vacuous": vacuous, **out}, R)

=============================== LAUNCHER (tail) ===============================
rc=$?

# after: the three trees must be exactly as before (read-only); a failing git call or any change fails the job
ST_EARLY_AFTER=$(git_state "$WT") || ST_EARLY_AFTER="git failed"
ST_CER_AFTER=$(git_state "$CERT") || ST_CER_AFTER="git failed"
ST_EARLYT_AFTER=$(git_state "$EARLYT") || ST_EARLYT_AFTER="git failed"
ST_PINNED_AFTER=$(git_state "$PINNED") || ST_PINNED_AFTER="git failed"
TREES_CHANGED=0
[ "$ST_EARLY_AFTER" = "$ST_EARLY" ] && [ "$ST_EARLYT_AFTER" = "$ST_EARLYT" ] && [ "$ST_CER_AFTER" = "$ST_CER" ] && [ "$ST_PINNED_AFTER" = "$ST_PINNED" ] || TREES_CHANGED=1
{
  echo "finished=$(date -Is)"
  echo "substrate_state_after=$ST_EARLY_AFTER"
  echo "cer_tree_state_after=$ST_CER_AFTER"
  echo "early_tree_state_after=$ST_EARLYT_AFTER"
  echo "pinned_tree_state_after=$ST_PINNED_AFTER"
  echo "trees_changed=$TREES_CHANGED"
  # the begin iteration as Multi4D itself resolved it (train.py writes cfg_args from the parsed arguments)
  echo "cfg_args_cer_begin_iter=$(grep -o 'cer_begin_iter=[-0-9]*' "$RUN_DIR/model/cfg_args" 2>/dev/null | head -n 1 | cut -d= -f2)"
  echo "cfg_args_cer_transient_gate=$(grep -o 'cer_transient_gate=[-0-9]*' "$RUN_DIR/model/cfg_args" 2>/dev/null | head -n 1 | cut -d= -f2)"
} >> "$RUN_DIR/provenance.txt"
cat "$RUN_DIR/model/cer_gates.json" 2>/dev/null
if [ "$TREES_CHANGED" != 0 ]; then echo "FAIL: a tree changed during the job"; [ "$rc" = 0 ] && rc=1; fi
# the exit code is recorded only after every check of the job, so a recorded 0 means the job itself exits 0
echo "exit_code=$rc" >> "$RUN_DIR/provenance.txt"
exit $rc

=============================== DRAFT PLAN SECTION 14 ===============================

## 14. Implementation, tests, and the premise check (2026-10-16 evening; appended)

**Review state, stated first.** Codex reviewed the plan in four rounds (MATERIAL DEFECT each: 6, 3, 3 and 2 confirmed
findings; round 4's were all in the task-B probe; every finding was fixed, sections 10–13). Round 5 could not run at once:
Codex hit its usage limit at 17:00, resetting at 21:28. Round 4 raised nothing against the task-A design, so the
implementation was started during the wait and the GPU tests below ran before the first code review. **Round 5 (21:31, one
combined review): PLAN READY (consensus); CODE READY WITH FIXES; PREMISE CHECK QUALIFIED** (record
`review_r5_plan_r1_code_premise_codex.md`; dispositions at the end of this section).

**Code** (branch `cer-early-tr` from `cer-early@7b30ef8`; local worktree `D:/adags-cer/multi4d-early-tr`; Leonardo worktree
`repo/multi4d-cer-early-tr`, made by bundle fetch into `repo/multi4d` and `git worktree add`; extensions
`build/torch_extensions_multi4d_cer_early_tr`). `repo/multi4d`, `repo/multi4d-cer` and `repo/multi4d-cer-early` were never
checked out or edited; their HEAD and porcelain were recorded before and after the deployment and by every job.

| commit | content (Sonnet 5.5 unless marked) |
|---|---|
| `5edd293` | tests F0–F9, written before the code |
| `5afa119` | the code: D1–D9 (8 files, no CUDA file) |
| `e857935` (orchestrator) | tests: loaded transient weight buffers sized in the F0/F5 subprocesses; F0 split into parts for the 30-min debug QOS |
| `36024cf` | tests: F5 attributes positive-control anomalies to the escalated defects |
| `65e0eb9` (orchestrator) | tests: an "unwritten" static entry counts as unattributed |
| `c7866b6` | tests: F5c, the controls with the static set out of view |
| `b5254ee` (orchestrator) | tests: F5c requires zero witness contribution in the gated gap view |
| `c087731` (orchestrator) | tests: F9 requires unchanged trees from the launcher record (Codex code round 1) |
| `96f6fe0` (orchestrator) | tests: F9 reads spec §14.2 (the residual share and the stale `pass` field are descriptive) |

The code has not changed since `5afa119`. Decisions the implementer took that the plan does not state (for the code review):
`begin` sets `cer_gate.TRANSIENT = bool(flag)`; with flag 0 the armed lift payload carries an all-zero `cer_labels` that the
unlabelled transient bank accepts (no numerical effect, one extra device sync); the new JSON fields appear only with flag 1
except `cer_transient_gate`; E1a of the early suite fails on this tree by construction (it forbids a labelled transient class
and an edited renderer) and F0 tolerates exactly those two entries, checked separately by F0's static part.

**Launchers** (local masters in `D:/adags-cer/agent/`, installed in `agent-control/ars/sbatch/`):
`train_multi4d_cer_early_tr.sbatch` (requires `CER_TRANSIENT_GATE`, pins all four trees) and `cer_tests_m4d_early_tr.sbatch`.

**GPU tests (debug QOS unless marked; stand-in model: bottle same-pose base run 1; not evidence).**

| test | result | job | what it showed |
|---|---|---|---|
| F0 static | PASS | 60733454 | every new path is flag-guarded; renderer gains one line; gate only gains; no CUDA change |
| F0 dynamic | PASS | 60733454 | flag 0 equals `cer-early@7b30ef8`: begin labels, donors and `cer_begin.json` counts, lifecycle and lift sequences hash-identical (6,059 lifted rows, the `randperm` exercised) |
| F0 suites | PASS | 60733456, 60734698, 60734699 | T0–T6 and E1–E8 on the new tree with flag 0 (E1a tolerated by construction, as above) |
| F1 | PASS | 60730613 | transient propagation exact through lift, clone, split, prune, densify-and-prune, weight-prune |
| F2 | PASS | 60730613 | labels neutral to the model: 50 tensors bit-identical, RNG equal |
| F3 | PASS | 60730613 | with flag 1 the lift selection equals the unarmed one (7,096 eligible, 3,571 labelled, subsample case exercised) |
| F4 | PASS | 60730613 | the transient gate's render identity at gap and non-gap frames (200 labelled rows) |
| F5 | gate clauses PASS; positive controls FAIL | 60733454, 60736116 | see below |
| F5c | gate clauses PASS; positive controls FAIL | 60743389 | see below |
| F6 | PASS | 60730613 | memory mass in the gap exactly 0 in all three banks (transient memory gap mass at q = 0: 69,535); gates and drift correct |
| F7 | PASS | 60730613 | flag 1: transient labels all 0 at begin, the masses reported and recomputed |
| F8 | PASS | 60730613 | `tr_labels` saved, reload refused, every transient lifecycle method raises after the freeze |
| F9 | PASS | run 60736117 (lprod), test 60750535 (60750297 failed on the test's superseded 0.20 residual rule, fixed in `96f6fe0`) | carton, shortened schedule; the three refusals and the plumbing pass: exit 0, trees unchanged, flag resolved 1; memory mass in the gap 0.0, residual outside 0.0, label-1 transient share of the gap region 0.0, frozen; **8 labelled rows eligible over 3 lift selections, 2 labelled copies of 6,000** |

**F5 and F5c.** In every run (two fresh processes each, four jobs) every witness (200 label-1 transient rows that paint the
gap view when the gate is open) has **all nine parameter groups exactly 0 with the gate shut**, under both losses, and the
deterministic fixture row has all nine nonzero with the gate open and all nine exactly 0 with it shut (F5c, and F5 in job
60733454). The **positive controls** fail:

- **Colour channels:** with the static set in view, 31–36 witnesses have a zero colour gradient in an unclamped channel; all
  36 are explained by the static row at the same index writing its clamp flag into the transient bank's buffer (the escalated
  `clamped` defect, `rasterizer_impl.cu:160`). With the static set moved out of view (F5c) these anomalies vanish (594 of
  597 unclamped channels positive).
- **Whole-row zeros:** 10 witnesses (11 at the gap view), the same rows in both processes, have exactly zero rendering
  gradient in every group on the training build although they contribute (contribution at least 11.5). This persists with
  the static set out of view and is not resolved on the build that writes `max_contrib`. Cause not established.
- **Run-to-run:** transient probe gradients differ between fresh processes (up to 319 absolute at the gap view), also with
  the static set out of view.
- **Reading (corrected after Codex round 5):** the exact zeros with the gate shut, and the forward skip of a zero opacity
  (`F:651-654`), support the gate's exactness. The failed positive controls limit the row-by-row gradient evidence: a
  clamp flag overwritten from true to false could also *enable* a suppressed derivative, so "the defect can only add zeros"
  (rev of this draft) is withdrawn. The failures are not isolated to the substrate by a paired gate-off control; Codex
  found no label-path suppression at `q = 0` by reading the code (factor 1; donor labels on the final mask; split order
  matches). **Escalated to the principal** as a third substrate observation beside `clamped` and `max_contrib`: some
  transient rows get no training-build gradient although they render, and transient gradients differ between fresh
  processes (a CUDA initialisation check, e.g. compute-sanitizer initcheck, would settle uninitialised reads).

**The premise check (descriptive; this changes the recommendation).**

1. **Low measured exposure of memory rows to the lift on the early trajectory.** The counter
   `labelled_candidates_excluded` sums, over all 139 lift selections of each confirmatory early run, the labelled rows above
   the threshold at each selection, before the subsample (`cer_drift.json`): bottle 0 / 0 / 0, carton 45 / 12 / 16, mug
   1 / 0 / 6 candidate occurrences. Each selection draws 2,000 of about 15,000 eligible rows; there were 64,000–85,000
   transient rows at the freeze. F9 agrees (8 eligible, 2 labelled copies). **Qualification (Codex round 5):** this is
   exposure on the early trajectory, not a bound on the fix's effect; a restored pool also changes the `randperm` draw and
   so later unlabelled selections and training; realised copies, descendants and their contribution are not counted.
2. **The transient rows that paint the object interior in base runs are mostly the gap occluder's tails.** Job 60742905
   (`agent-control/ars/cer/return_diag/transient_time/transient_time_interior.json`), cam00, interior mass at frames 40 and 57:
   the share of transient interior mass from rows centred in time inside the gap is 0.83–1.00 on the mug, 0.52–0.99 on the
   bottle at frame 57, and 0.10–1.00 on the carton (runs 1 and 3 partly centred before the gap); 0.43–1.00 of it comes from
   rows with any contribution (mass > 0) in the gap region at frame 74. In the early arm (run 1 per scene) the transient share
   of interior mass is 0.000–0.001. Limits: two frames, cam00, composited contribution mass (not the bank-alone alpha of
   the 2026-10-16 diagnostic), one early run per scene, no re-check of the diagnostic's renders against the saved renders.
3. **Reading (qualified after Codex round 5).** The data are consistent with base Multi4D's transient "detail" on the object
   being mostly the time tails of rows centred in the gap, and with the lift exclusion having had little to exclude. They do
   not establish that these rows are an occluder, that the detail causes the interior PSNR difference, or that the fix could
   not recover detail by another route (a contribution or colour ablation would be needed). The 2026-10-16 entry's "most
   likely mechanism" is no longer the most likely one on this evidence (append-only: the entry stands as written).

**Recommendation (for the principal).** Defer the section 8 comparison (≈ 26 GPU-h): the measured exposure of memory rows to
the lift is low, so the fix is unlikely to change much, although this is a prioritisation, not a proof. The code stays on
its branch, tested as above; no scorer method was written.

**Codex round 5 dispositions.**

| # | finding | disposition |
|---|---|---|
| 1 | the new launcher records the exit code before its tree check; F9 does not check the trees | accepted: exit code recorded last (`train_multi4d_cer_early_tr.sbatch`); F9 requires `trees_changed 0` and equal before/after states (`c087731`). The same order exists in the pre-existing early launcher (not edited) |
| 2 | "the defect can only add zeros" is false | accepted: withdrawn (F5 reading above) |
| 3 | candidate counts do not bound the fix's effect | accepted: "low measured exposure", qualified |
| 4 | temporal overlap does not establish occluder tails or causation | accepted: "consistent with"; the causal claim needs an ablation |
| 5 | the task-B blanket conclusion contradicts the foreground result | accepted: the plan-log reading is rewritten (foreground memory rows starved; mixed-batch loss unmeasured) |
| suspected | the F5 failures not isolated to the substrate | stated as open; a paired gate-off control or initcheck would settle it |
| missing | the time diagnostic lacks a saved-render fidelity check; its limits | the limits are stated; the fidelity check is not run |


=============================== DRAFT PLAN-LOG ENTRY ===============================

### 2026-10-16 evening: the two sources of the early arm's object deficit, tested (exploratory; no training of an arm)

Brief from the principal (2026-10-16): (A) plan, review, implement and test a fix for the missing transient detail (memory rows
lift donors again, their transient copies gated as memory); prepare, not submit, a comparison. (B) Measure whether memory rows
are densified less in the early arm before any densification change. The record is
[cer-multi4d-early-tr-plan-2026-10-16](cer-multi4d-early-tr-plan-2026-10-16.md) (plan rev 5, sections 1–13; implementation,
tests and premise check, section 14). Reviews: `assets/cer-early-tr-2026-10-16/`. Every job id below is in the ledger.

**Task A, the fix (transient copies of memory rows gated as memory).**

- **Plan:** five revisions; Codex rounds 1–4 each MATERIAL DEFECT (6, 3, 3, 2 findings), all fixed; round 5 (21:31) PLAN READY.
  The implementation went ahead during the Codex usage-limit wait (17:00–21:28), after round 4 had found nothing in task A.
  Code review round 1 (same call): READY WITH FIXES (the new launcher's exit-code order, F9's tree check; both fixed).
- **Code:** branch `cer-early-tr`, code at `5afa119` (tests to `b5254ee`), flag `cer_transient_gate` (0 = the early arm, shown
  bit-identical by F0). GPU tests: F0–F4, F6–F8 PASS; F9 integration PASS; F5/F5c gate clauses PASS (exact zeros for every
  witness in all nine parameter groups) with positive-control failures: one part explained by the escalated `clamped` buffer
  (they vanish with the static set out of view), and 10 rows with entirely zero training-build gradient, cause not
  established and not isolated from the code by a paired control.
- **The premise is weakened (Codex: QUALIFIED).** On the early trajectory the lift exclusion met few labelled candidates: 0–45
  candidate occurrences per run over 139 selections (bottle 0/0/0, carton 45/12/16, mug 1/0/6). This is low exposure, not a
  bound on the fix's effect (a restored pool also changes the random draw and later training). In base runs the transient rows
  that paint the object interior before the gap are mostly centred in time inside the gap (mug 0.83–1.00, bottle 0.52–0.99 at
  frame 57, carton 0.10–1.00 of their mass), and 0.43–1.00 of that mass is from rows with some contribution in the gap region:
  consistent with time tails of gap rows, not shown to be an occluder or to cause the PSNR difference. The 2026-10-16 entry's
  "most likely mechanism" is no longer the most likely one.
- **Recommendation:** defer the comparison (≈ 26 GPU-h), as a prioritisation. Nothing was submitted for training.

**Task B, densification of memory rows (bottle same-pose, one run each, exploratory diagnostics).**

- **Probe:** `agent-control/ars/densify_probe.py` (runpy around `cer-early@7b30ef8` `train.py`), its test (17 checks PASS, jobs
  60728104/60728741/60728898/60729394), reader `densify_read.py`. Runs: early arm 60729395 (2:49), shadow base 60729396 (2:32;
  CER off, the memory rule applied at fine 3,000 without arming the gate). Output
  `agent-control/ars/cer/densify_probe/densify_read_bottle_r1.json`.
- **Background memory rows (3,349 of the early arm's 3,388 memory rows at 3,000), steps 3,100–15,900, early / shadow:** mean average gradient 1.011×
  (median 1.026×); clone selections 5 / 5; split 0 / 0; rows at 16,000 2,652 / 2,667 (0.994×); visibility count per window
  99.6 / 99.6. Increments that carry no gradient for a gated row (every view of the batch that saw the row was a gap view) are
  1.8% of the count, not ≈ 10%; deleting them raises the average by 1.9% and changes no selection.
- **Other background rows:** gradient 1.02×; clones 213 / 208; splits 668 / 850 (0.79×).
- **Foreground memory rows:** 39 (early) / 93 (shadow) at 3,000; in the early arm their average gradient is 0 and all are
  gone by 16,000; the shadow base keeps 18 after 54 splits. A small cohort (39 of 3,388 memory rows in the early arm).
- **Reading (corrected after Codex round 5):** background memory rows are densified about as often in the early arm as in the
  shadow base (similar gradient, clone 5/5, split 0/0, rows 0.99×). The foreground memory rows are starved in the early arm:
  mean average gradient 1.1e-7 against 1.4e-4, splits 0 against 54, none left at 16,000 (18 in the shadow base). The deletion
  sensitivity removes only all-zero increments; the gradient a gated row misses in mixed batches (one gap view, one non-gap
  view; gradients summed before the norm, `train.py:855`) is not measured. So the diluted-statistics candidate is small for
  background memory rows; the foreground starvation and the mixed-batch loss are open, and the image impact of either is not
  measured. No densification change is proposed yet. Unverified: one run per arm, one scene; the two runs already differ at
  fine 3,000 by GPU nondeterminism.

**Cost.** Task B 5.9 GPU-h (probe tests 0.5, runs 5.4). Task A tests 3.3 GPU-h (F9 1.1 on lprod). Premise check 0.05 GPU-h.

**For the principal.** (1) Whether to drop the fix (recommended) or still run the comparison. (2) The new substrate finding:
some transient rows get no training-build gradient although they render, and transient gradients differ between fresh
processes. (3) Where the object deficit comes from remains open: the transient part is consistent with gap rows' time tails; for the
persistent part (bottle −0.28, carton −0.16, mug +0.20 dB) the candidates left are the starved foreground memory rows and
the gradient missed in mixed batches.

=============================== YOUR ROUND-5 REVIEW ===============================
## Verdict

MATERIAL DEFECT

PLAN: READY  
CODE: READY WITH FIXES  
PREMISE CHECK: QUALIFIED

Independent judgment; no ARIS skill used. Opened [cer.py:18](/D:/adags-cer/multi4d-early-tr/scene/cer.py:18): `import torch`. The round-4 probe fixes are present, including rejection of changed worktree states in an in-memory check.

## Confirmed defects

1. **Moderate — code: F9 can accept a failed training job.** The launcher records `exit_code=$rc` before changing `rc` when trees changed ([launcher:176](/D:/adags-cer/agent/train_multi4d_cer_early_tr.sbatch:176), line 187). F9 checks that recorded code but never checks `trees_changed` or the before/after states ([test:1755](/D:/adags-cer/multi4d-early-tr/test_cer_early_tr.py:1755)). **Fix:** finalize checks before recording the exit status; require unchanged tree states in F9.

2. **Moderate — code interpretation: “They can only add zeros” is false.** The shared clamp flags multiply colour derivatives by either zero or one ([backward.cu:160](/D:/adags-cer/multi4d-early-tr/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu:160)). Overwriting a true clamp flag with false can enable an otherwise suppressed derivative; the test explicitly recognizes that anomaly ([test:1002](/D:/adags-cer/multi4d-early-tr/test_cer_early_tr.py:1002)). **Fix:** remove this argument. Forward exclusion has direct support from the alpha skip; failed positive controls limit the row-by-row gradient evidence.

3. **High — premise: candidate counts do not bound the intervention to “a few dozen rows.”** The counter correctly sums **labelled candidate occurrences above threshold**, before subsampling ([early cer.py:197](/D:/adags-cer/multi4d-early/scene/cer.py:197), line 209). It does not count unique donors, realized copies or descendants. Restoring candidates also changes the pool supplied to `randperm`, potentially changing unlabelled selections and subsequent training ([gaussian_model.py:430](/D:/adags-cer/multi4d-early-tr/scene/gaussian_model.py:430)). **Fix:** describe low measured exposure on the early trajectory, not a counterfactual effect bound. Track realized copies, descendants and contribution mass if that bound matters.

4. **High — premise: temporal overlap does not establish “occluder’s tails,” causation or impossibility of recovery.** “Also paints the gap” means merely `gap_mass > 0` ([diagnostic:73](/D:/adags-cer/agent/transient_time_interior.py:73)); the reported percentage weights those rows’ **pre-gap** mass ([line 84](/D:/adags-cer/agent/transient_time_interior.py:84)). It measures neither meaningful gap opacity nor occluder identity. These are composited contribution masses, whereas the earlier alpha diagnostic used the bank alone ([cer.py:93](/D:/adags-cer/multi4d-early-tr/scene/cer.py:93)). **Fix:** call this evidence consistent with temporal tails. A contribution/colour ablation is needed for the causal claim. Deferring 26 GPU-h is defensible prioritization; “the premise is false” and “cannot restore detail without the occluder” are not established.

5. **High — task B: the blanket conclusion contradicts the foreground result and excludes an unmeasured mechanism.** The supplied reader output gives foreground memory split selections **0 versus 54**, with all early foreground memory rows gone. Its early mean gradient is **1.077×10⁻⁷**, not exactly zero. Background totals alone cannot establish the blanket claim, and cohort size does not bound image impact. Moreover, the deletion sensitivity removes fully zero increments ([probe:103](/D:/adags-cer/agent/densify_probe.py:103)); it does not recover missing gradient contributions in mixed batches, whose gradients are summed before taking the norm ([train.py:855](/D:/adags-cer/multi4d-early-tr/train.py:855)). **Fix:** report similar background selections in this pair, suppressed foreground densification, and unresolved effects of fewer gradient views.

## Suspected issues

- **The remaining failures have not been isolated to the substrate.** I found no direct label-path suppression at `q=0`: the factor is one, donor labels use the final geometry mask, and split inheritance has matching repeat order. Settle attribution with paired gate/bypass backward checks and initialization diagnostics; NVIDIA documents [Initcheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) for uninitialized device-memory reads.

## Unverified claims

- GPU pass results, ten whole-row failures, historical PSNRs, per-run exclusion totals and costs were not independently reproduced. I did not reach the Leonardo checkpoints, raw logs and provenance; local Python has no PyTorch.

## Missing

- The time diagnostic lacks its own saved-render fidelity check. Importing the loader does not execute the bank tool’s comparison against saved renders ([banks:145](/D:/adags-cer/agent/cer_m4d_banks.py:145)).
- Explicit limits on the premise reading: two frames, cam00, and only one early replicate per scene. The time mapping, erosion convention and frame-59 gap mask agree with the inspected dataset and diagnostic code.