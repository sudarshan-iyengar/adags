## Verdict

MATERIAL DEFECT

## Confirmed defects

- **High — invalid confirmatory numbers can produce PASS/PASS.** [cer_reduce.py:234](/D:/adags-cer/agent/cer_reduce.py:234) accepts `isinstance(v, (int, float))`, which includes booleans ([Python documentation](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool)). Reproduced strict-JSON PASS/PASS with every CER P1 value `True` and each layout’s base P1 values `[False, False, True]`. [Gate validation:246](/D:/adags-cer/agent/cer_reduce.py:246) also accepted boolean masses and `memory_share_return=2.0`, producing PASS/PASS. **Fix:** reject booleans in numeric fields and enforce field-specific bounds. These holes predate this patch.

- **High — descriptive DWP fields still prevent publication.** [Line 456](/D:/adags-cer/agent/cer_reduce.py:456) computes an unguarded descriptive subset contrast. With L3 failing a precondition and every PSNR `0.0`, it raises `"zero residual variance"` instead of writing `DESIGN_WITHOUT_POWER`. Separately, [line 461](/D:/adags-cer/agent/cer_reduce.py:461) computes failed-scene diagnostics that the verdict never consults: L3 CER P1 values `[1e308]*3` overflow `layout_gap_diff.L3`, and the new [blanket guard:561](/D:/adags-cer/agent/cer_reduce.py:561) refuses. **Fix:** report unavailable descriptive contrasts; validate only evidence used by verdicts and sanitize excluded-scene diagnostics.

- **High — Ex4DGS sensitivity can still block the official verdict.** The recursive call at [line 552](/D:/adags-cer/agent/cer_reduce.py:552) can raise before reaching `_sensitivity`’s return. Reproduced with all PSNRs zero except replacement-base scores near 30: the official calculation yields finite FAIL/FAIL, but original-base sensitivity raises `"zero residual variance"`, withholding output. **Fix:** isolate sensitivity failures and report that analysis unavailable. This path is pre-existing.

- **Medium — the new Welch regression test does not exercise its claimed failure.** [test_cer_reduce.py:490](/D:/adags-cer/agent/test_cer_reduce.py:490) uses `30.0 + (x - 30.0) * 1e-160`; every modified value rounded to exactly `30.0`. The test produced no sanitized paths and also [accepts refusal:495](/D:/adags-cer/agent/test_cer_reduce.py:495). **Fix:** multiply values directly by `1e-100`, which reproduced non-finite Welch intervals, and require successful serialization plus replacement paths.

## Suspected issues

None retained.

## Unverified claims

Claude’s native “all 15 tests pass locally” run has no supplied log. My read-only, in-memory harness passed 15/15 using Student-t adapters; native execution was blocked by SciPy’s temporary-file requirement.

## Missing

Regression coverage for invalid confirmatory numbers, DWP diagnostic failures, and unavailable original-base sensitivity.