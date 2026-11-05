## Verdict

MATERIAL DEFECT

Opened: both early sbatch files in full and their legacy counterparts; the requested diff; prior review requests/reviews; early-gate plan and CER spec; relevant portions of `train.py`, both tests, scene/model/argument files, gate, parameter merge, hybrid CUDA sources, and scorer files. Checked API semantics against the [Bash manual](https://www.gnu.org/s/bash/manual/bash.html) and [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.isfinite.html).

## Confirmed defects

1. **Major — E8 drops failed positive controls from subsequent checks.** The [plan:270](/D:/adags/research-wiki/operations/cer-multi4d-early-gate-plan-2026-10-06.md:270) defines witnesses by positive instrument contribution at q=0. Instead, [test_cer_early.py:395](/D:/adags-cer/multi4d-early/test_cer_early.py:395) rebuilds `w_gap` from `PASSED["gap_q0/" + n]`, retaining only rows whose opacity/DC gradients already passed. [Line 977](/D:/adags-cer/multi4d-early/test_cer_early.py:977) accepts `2 * w_gap >= w_ident`.

   Consequently, half the identified witnesses can lack the required gradients and disappear from q=1 zero checks. An extracted-predicate reproduction with **2 identified, 1 passing, 1 retained per bank passed all 18 flags**. Retained rows have valid controls; coverage of the original witness set remains incomplete.

   **Fix:** preserve the instrument-defined witness set, require each row’s opacity/DC positive control, and check q=1 zeros on those same rows.

## Suspected issues

- **Unfrozen short runs may save a different model from the one rendered/reported.** [train.py:896](/D:/adags-cer/multi4d-early/train.py:896) saves before maintenance at 961–979; evaluation and gates follow. At F1’s final iteration 3500, densification remains eligible. Settle with saved-versus-final row/label comparisons on an unfrozen short run. This does not invalidate F1’s stated reading from `cer_begin.json`.

**Escalated, excluded from this verdict:** inherited CUDA `clamped` sizing and drift-pass/`max_contrib` interaction.

## Unverified claims

- Reported GPU PASS results, memory counts `[0, 2525]`, six donors, lifecycle hashes, and remote before/after tree pins: no corresponding primary results were found in the inspected locations.
- Numerical correctness on the two new commits: local Python has no PyTorch; only static and isolated predicate/shell probes ran.
- Live cluster wrapper/environment behavior was unavailable for inspection.

## Missing

- GPU validation of the reviewed commit, especially E2/E3/E5–E9 and the revised E4.
- Runtime coverage of `frozen == false`: [E9:1092](/D:/adags-cer/multi4d-early/test_cer_early.py:1092) explicitly checks acceptance “by the range rule alone, no training run.”
- The promised early-arm scorer implementation. The inspected [scorer:74](/D:/adags-cer/agent/ars_score.py:74) has no `multi4d_cer_early` method; enforcement of `frozen` and the full schedule remains pending.