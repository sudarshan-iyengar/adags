# Fable 5.1 review of the CER Ex4DGS diff (f37694c + cd5ef1e vs 7fac649), 2026-09-30

Read-only subagent review; condensed transcription of its report.

| # | dimension | score | reason (short) |
|---|---|---|---|
| 1 | correctness | 9 | every spec §1 property implemented; all post-30k row-count paths and all render callers traced, no bypass or desynchronisation; `getattr(args, "cer_scene", "")` in the loader is necessary (render.py's sentinel parser); deductions: cam00 exclusion and `time_pad_type == 0` unasserted; TypeError-shaped failure if `begin` never runs |
| 2 | shapes and dtypes | 9 | every tensor walked; only edge unguarded is N_d = 0, impossible on these scenes |
| 3 | minimality | 9 | nothing beyond the spec; small excess (`.clone()`, `zeros_like(...).repeat`, unused return) |
| 4 | surgical scope | 9 | every upstream line traces to the spec; one style nit (four-target assignment with a trailing comment) |
| 5 | tests | 8 | gaps: T3 gradient check could pass with `.grad is None`; FORECAST fails instead of reporting; ply refusal accepts any AssertionError; exact-zero check lacks a memory-mass witness; resume refusal and per-bank assertion untested |
| 6 | launchers | 7 | `ITER` decoupled from the config's iterations (M1); provenance captured only before training; rendered iteration not recorded |

**M1 (major).** With `ITER` ≠ the trained final iteration, `render.py` loads another checkpoint; at 30,000 it is the
pre-`begin` 40-tuple, so an **ungated** base evaluation lands in the CER model directory's metrics JSON. Fix: derive
`ITER` from the config, record it; optional loader guard asserting CER state after `restore` when `cfg_args` names a
CER scene.

**Minor.** m1 assert `densify_until_iter < iterations`; m2 assert no cam00 in `train_views`; m3 assert
`time_pad_type == 0` and use `dataset.duration`; m4 T3 vacuity; m5 FORECAST should report; m6 re-record provenance at
render time; m7 check the refusal message; m8 require memory mass before the gap in the exact-zero check; m9 resume
refusal and per-bank assertion untested (acceptable, say so). Nits: style of the attribute line, `.clone()`,
`zeros((k, 3))`, unused return; the test launcher's `RUN` applies to every scene; no refusal of an existing run
directory; a `cer_scene` key inside a config JSON would override the CLI; T7's 17,200 iterations never reach the
error-based prune.

**Confirmed correct.** Residual rows sit between the banks (stable-sort argument holds, T2 tests it);
`contributor++` precedes the skip and `last_contributor` is set only after a contribution; `begin` runs after the
30,000 step and NaN prune, outside `no_grad`; `mass` uses `autograd.grad` on the colour leaf only with the config's
near/far; newborn protection from `prune_invisible` and `prune_small`; identical keep masks in `_prune_optimizer` and the
label masks; `restore` sets CER state before `training_setup`; capture is a 40-tuple at 30,000 and a 41-tuple at 40,000;
`render.py` has no second render path.

**Verdict: the substrate diff is ready for the principal's review; fix M1 before the first submission.**

All findings were applied in commits 73e9b65 (tests) and f71c1b0 (code), and in the launcher and scorer.
