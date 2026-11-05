## Verdict

SOUND WITH QUALIFICATIONS

CODE: READY WITH FIXES. READINGS: QUALIFIED.

## Confirmed defects

1. **F9 does not enforce the required carton scene.** It assigns `out["scene_is_carton"]` at [test_cer_early_tr.py:1807](/D:/adags-cer/multi4d-early-tr/test_cer_early_tr.py:1807), but determines success using only `all(c.values())` at line 1811. A different scene can pass. **Fix:** put the scene requirement into `c`.

2. **The revised premise heading still asserts occluder identity.** Section 14 says “mostly the gap occluder’s tails,” then explicitly says the evidence does not establish an occluder. The diagnostic measures temporal centres and merely `paints_gap = g_in[2] > 0` ([transient_time_interior.py:73](/D:/adags-cer/agent/transient_time_interior.py:73)). **Fix:** replace the categorical heading with the measured temporal-overlap description. “The previous mechanism is unestablished” is also better supported than declaring it no longer the most likely mechanism.

3. **The foreground exact-zero statement remains wrong.** The plan-log bullet says “their average gradient is 0”; the reachable `scratchpad/densify_read_bottle_r1.json:235` reports `"grad_mean_seen": 1.0772971137384957e-07`. The later corrected paragraph does not repair the contradictory bullet. **Fix:** use approximately `1.1e-7` consistently.

## Suspected issues

- **Attribution of the unexplained gradient failures remains unresolved.** Calling them a “substrate finding” still outruns the acknowledged absence of a paired bypass control. Initcheck detects uninitialized accesses; it does not by itself establish which defect causes the observed gradient failures. NVIDIA also recommends memcheck first. **Settle:** paired gate/bypass backward checks, sanitizer diagnostics, then demonstrate that correcting the identified fault removes the failures. [NVIDIA documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#initcheck-tool)

## Unverified claims

- GPU pass results, especially job **60750535**, were not independently reproduced: local Python has no PyTorch, and I did not reach the remote raw results/provenance.
- Exclusion totals, detailed F5/F5c counts, historical PSNRs, costs and deployment history remain unverified. Copied diagnostic outputs support checking arithmetic, not authenticating their execution.

## Missing

- **Unequal foreground observation windows:** the reader output reports only **2** early windows with memory rows versus **69** shadow windows (`densify_read_bottle_r1.json:234,275`). The reader averages only windows where the cohort exists ([densify_read.py:38](/D:/adags-cer/agent/densify_read.py:38)). State this beside the gradient comparison; report disappearance timing and matched-window results before interpreting starvation causally.