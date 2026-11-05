## Verdict

MATERIAL DEFECT

Code opened; clean HEAD is `7b30ef8`. [scene/cer.py:225](D:/adags-cer/multi4d-early/scene/cer.py:225): `if cer_gate.ARMED and hasattr(fg, "cer_labels"):`

Code references below use the plan’s prefixes.

## Confirmed defects

1. **Major — the transient gate lacks the existing NaN safeguard.** `C:134–137` explicitly documents “A NaN opacity times a 0 factor renders at alpha 0.99”, but checks only foreground/background. D6 adds transient multiplication without extending this check; `F:651–654` therefore does not guarantee exclusion for NaN opacity. **Fix:** conditionally validate transient opacity under flag 1 at begin/freeze/gates, and test refusal on a NaN transient memory row.

2. **Major — F4’s non-gap reference is incorrect.** Outside the gap, memory factor is 1 (`G:30–33`); its proposed `−inf` reference removes memory rows. Correct code can fail this test. Existing [T3:257–265](D:/adags-cer/multi4d-early/test_cer.py:257) selects the disabled cohort separately for each frame. **Fix:** use `−inf` only for transient rows whose factor is zero; outside the gap compare against unchanged opacity.

3. **Major — F5 requires positive gradients that visibility does not establish.** E8 already handles clamped colour channels and loss cancellation (`test_cer_early.py:380–399`). It also restricts pre-gap controls to witnesses visible there (`:474–480`); transient temporal culling makes this restriction necessary (`F:281–285`). F5 drops those qualifications and additionally demands nonzero `xyz` gradients. **Fix:** retain E8’s controls and pre-visible subset; use a controlled, nondegenerate fixture for each additional gradient group.

4. **Major — Task B overstates its counterfactual.** `denom` counts OR-visible **batches**, while the numerator uses the norm of summed view gradients (`GM:114–117`; [T:590](D:/adags-cer/multi4d-early/train.py:590), `:593`, `:855–860`). Subtracting all-gap increments is valid for deleting those recorded zero increments. It leaves mixed batches where a row was visible only in the gap view. **Fix:** call this an *all-gap-batch deletion sensitivity*, or track row-level zero increments more completely. Specify zero-denominator handling. It cannot reconstruct training without gating.

5. **Minor — F2 conflicts with D3’s refusal rule.** F1 inserts a dictionary containing nonzero labels. Repeating that invocation on F2’s unlabelled control must trigger D3’s proposed assertion. Also, ignoring metadata can pass a neutrality test, contradicting “F1–F9 fail on the unmodified tree.” **Fix:** strip labels from the control payload while preserving model inputs; classify F2 as a regression test.

6. **Minor — Task B gives the wrong foreground observation window.** Foreground density control stops before phase 3 (`T:46`, `:948`): its last scheduled densification is 9,900, versus background’s 15,900. **Fix:** report bank-specific windows and actual callback counts.

## Suspected issues

1. **D5 can be an additional effective intervention.** It gates a pre-existing seed independently of restoring future lift copies, potentially changing the residual-donor pass (`C:253–257`). One row does not bound its contribution. **Settle:** report its label and ungated all-pixel gap mass. If nonzero, use a seed-only control or describe Task A as a bundled exploratory intervention rather than isolated lift attribution.

2. **F9’s donor guarantee is unsupported.** Carton memory-row counts do not establish eligible labelled donors: selection also requires `dynamic_score > threshold` (`GM:422–427`). **Settle:** record qualifying labelled donors at the smoke run’s lift calls.

## Unverified claims

- **“Both runs carry the same perturbation” is unestablished.** Early’s second mass pass is gated (`C:253–255`); shadow stays ungated. Equal pass counts cannot establish equal effects from the acknowledged uninitialised buffer. Moreover, the available [densify_probe.py:214](D:/adags-cer/agent/densify_probe.py:214) performs only one shadow mass pass. Reconcile that implementation with the plan.
- Historical numbers match the research record (`ars-fast-path-plan-2026-09-23.md:1858–1946`), but local primary diagnostic outputs are absent. Previous durations support approximately 2.6 hours for the **early arm**, not measured runtime for the proposed arm or probe. The 26 GPU-hour total remains an estimate. GPU tests and Leonardo-only wrapper internals were not verified.

## Missing

- Explicit assignment/restoration of transient `cer_clock` and `cer_q`, both required by `G:30–33`; D5/D8 explicitly mention only labels. The bank-alone loop must also reset `TRANSIENT` between heterogeneous runs (`cer_m4d_banks.py:119–127`).
- Task-B-specific tests for the phase-2 standalone background rasterizer, pure/mixed batches, counter reset/prune/append alignment, and corrected denominator zero. E8 exercises the hybrid path (`test_cer_early.py:304`).
- Separate cumulative lift births from surviving descendants. Clone/split/prune alter live counts, and D5 introduces labelled ancestry unrelated to lift copies; cumulative copy counters alone are not full lineage tracking.