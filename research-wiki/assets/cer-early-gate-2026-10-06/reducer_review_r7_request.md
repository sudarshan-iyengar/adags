# Code review request: the CER reducer's early-gate extension, round 7 (focused)

Your round 6 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r6_codex.md) found: booleans and
out-of-range confirmatory values accepted; design-without-power descriptive subsets and failed-scene diagnostics able to
block output; the Ex4DGS sensitivity re-run able to block the official verdict; a Welch test that did not exercise its case.
Changes in D:\adags-cer\agent\cer_reduce.py (diff cer_reduce_m4dearly.diff):
- every confirmatory PSNR goes through _psnr_value (numeric, not bool, float-convertible, finite, 0..1000 dB) -> Refuse;
- gate fields: bool rejected; *_share* fields in [0, 1]; other numeric fields (masses) >= 0; `pass` must be a bool -> Refuse;
- rule 1: a contrast for an OK event type is the verdict's evidence (failure refuses); for a design-without-power event type
  the subset contrast is descriptive (failure -> {"unavailable": ...}); same for rule 3 under design-without-power;
- the Ex4DGS original-base sensitivity re-run is wrapped; a Refuse there -> {"unavailable": ...};
- end of reduce(): finiteness is REQUIRED only of verdict-bearing parts (each OK rule-1 event type; rule 3 when OK);
  design-without-power parts and all other descriptive output are sanitized (non-finite -> None, path listed).
Tests (diff test_cer_reduce_m4dearly.diff): t_invalid_confirmatory_values_refuse (bool PSNR, negative PSNR, share 2.0,
bool mass, negative mass, non-bool pass, Ex4DGS share 1.5), t_dwp_descriptives_and_sensitivity_never_block (zero-variance
DWP subset written as unavailable; the same zero variance in a verdict-bearing contrast refuses; Ex4DGS sensitivity
failure reported, not blocking), Welch test now scales PSNRs by 1e-100 and requires listed replacement paths. A share of
1e308 in a gate file is now refused as out of range. 17 tests pass locally.
A compatibility re-run of the new reducer on the real 2026-10-02 Ex4DGS and Multi4D scores is being run on the cluster.
Check the changes; whether a legitimate result could now be refused (e.g. could a real Ex4DGS residual_share or any real
gate field legitimately lie outside these bounds - check D:\adags-cer\ex4dgs and D:\adags-cer\multi4d-early scene\cer.py
gates() for how each field is computed); anything still open. Verdict line READY / READY WITH FIXES / MATERIAL DEFECT;
findings with severity, file:line, fix. Terse.
