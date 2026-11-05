# Code review request: the CER reducer's early-gate extension, round 6 (focused)

Your round 5 (D:\adags\research-wiki\assets\cer-early-gate-2026-10-06\reducer_review_r5_codex.md) found absurd finite
inputs making pre-existing descriptive aggregates non-finite (a 1e308 transient share's scene mean; Welch 0/0 from tiny
variances), which strict JSON then refused. Instead of patching each aggregate, D:\adags-cer\agent\cer_reduce.py now ends
reduce() (verdict of record only; the ex4dgs sensitivity sub-call returns before it) with one structural rule:
- any non-finite value inside rule1_causal or rule3_do_no_harm REFUSES (a verdict must rest on finite numbers);
- every other non-finite float anywhere in the output (descriptive) is replaced by None and its JSON path listed in
  nonfinite_descriptive_replaced, so a descriptive value can never stop the verdicts being written.
Also: drift shares must be finite numbers in [0, 1] (bool rejected; 0.0 accepted).
New test t_nonfinite_descriptive_never_blocks (1e308 transient share on multi4d and multi4d_early; PSNRs scaled by 1e-160;
drift 0.0 and a boolean). All 15 tests pass locally. Diffs: cer_reduce_m4dearly.diff, test_cer_reduce_m4dearly.diff.
Check: is there still ANY input path by which a descriptive value prevents the verdicts being written, or by which a
verdict could be written from a non-finite or invalid number? Is the rule1/rule3 refusal correct (could it refuse a
legitimate result)? ex4dgs and multi4d unchanged apart from strict JSON and this end-of-reduce rule?
Verdict line READY / READY WITH FIXES / MATERIAL DEFECT; findings with severity, file:line, fix. Terse.
