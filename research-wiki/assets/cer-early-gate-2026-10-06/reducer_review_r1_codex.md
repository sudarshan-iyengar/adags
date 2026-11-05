## Verdict

MATERIAL DEFECT

## Confirmed defects

- **High — a valid failure declaration aborts reduction.** With all nine early runs declared CER failures, the descriptive comparison still calls `contrast(...)` at [cer_reduce.py:393](D:/adags-cer/agent/cer_reduce.py:393). Reproduced: `Refuse: design matrix is rank-deficient`. The [failure policy:552](D:/adags/research-wiki/operations/cer-ex4dgs-spec-2026-09-30.md:552) requires the affected scene to be “DESIGN_WITHOUT_POWER.” **Fix:** report an unavailable descriptive contrast when either arm is absent; preserve confirmatory verdicts and guard the print at line 464.

- **Medium — undefined drift becomes measured zero.** The producer explicitly returns `None` when target mass is zero: `share = lambda x: x / denom if denom > 0 else None` at [cer.py:310](D:/adags-cer/multi4d-early/scene/cer.py:310). The reducer filters out those values and sums the remainder at [cer_reduce.py:197](D:/adags-cer/agent/cer_reduce.py:197). Reproduced: both banks’ `null` shares become `0`; missing bank fields also become zero. **Fix:** preserve undefined shares as `None` and validate both banks before aggregation.

## Suspected issues

- **Descriptive inputs can block confirmation.** [cer_reduce.py:106](D:/adags-cer/agent/cer_reduce.py:106) adds the 16k arm to the mandatory manifest; line 142 refuses missing scores. Reproduced: removing one 16k score prevents any result. The [frozen entry:1732](D:/adags/research-wiki/operations/ars-fast-path-plan-2026-09-23.md:1732) defines confirmation against nine base runs and labels the 16k comparison descriptive. It does not explicitly resolve whether descriptive inputs must be mandatory. Clarify that requirement; if confirmation must remain available independently, separate their loading.

## Unverified claims

- Live queue status, “nothing has been scored,” and the exact historical mutation outcomes lack primary evidence in the referenced local artifacts.
- The ordinary test command could not run unchanged because SciPy’s import attempted a blocked temporary-file write. All 14 tests passed using an in-memory filesystem and a shim for an unused sampling import.

## Missing

- Tests for all early runs declared failed, undefined drift, missing drift fields, and nonfinite drift.
- Strict JSON validation. [test_cer_reduce.py:61](D:/adags-cer/agent/test_cer_reduce.py:61) round-trips using Python defaults. A reproduced drift `NaN` survives that check; those defaults permit nonstandard JSON numbers. **Fix:** validate finite shares and serialize with `allow_nan=False`. [Python documentation](https://docs.python.org/3.12/library/json.html#json.dump)