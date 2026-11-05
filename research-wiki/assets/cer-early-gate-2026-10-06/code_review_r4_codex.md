## Verdict

SOUND WITH QUALIFICATIONS

## Confirmed defects

None found in `96e92ee`. The round-3 defect is fixed: [line 389](/D:/adags-cer/multi4d-early/test_cer_early.py:389) defines witnesses by `(tot_gap[i] >= FLOOR) & (m.cer_labels == 1)`, independently of gradients. Both gap controls receive that mask; [line 983](/D:/adags-cer/multi4d-early/test_cer_early.py:983) requires `n_failing == 0` and every witness’s positive control. Extracted-predicate checks reject the former half-passing case.

The [process comparison](/D:/adags-cer/multi4d-early/test_cer_early.py:1018) retains index equality, finite-gradient checks, numerical agreement, and identical zero/nonzero patterns. Its finiteness check matches the [PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html).

## Suspected issues

- **Unfrozen save/render consistency remains unresolved.** [train.py:896](/D:/adags-cer/multi4d-early/train.py:896) calls `scene.save` before maintenance and evaluation. F1’s restricted `cer_begin.json` reading avoids this concern for its forecast; it does not establish saved-versus-rendered agreement. Settle with final saved/reloaded row, label, and render comparisons.

## Unverified claims

- **GPU validation “runs now”:** no job ID or primary result accompanies the claim. Local PyTorch is absent; only static and extracted-predicate checks ran here.
- **F1 provides runtime coverage of `frozen == false`:** no completed F1 artifacts were available. [E9:1103](/D:/adags-cer/multi4d-early/test_cer_early.py:1103) explicitly says “range rule alone, no training run.” Require completed F1 provenance and `cer_gates.json` showing `frozen: false`.

## Missing

- **Early scorer implementation and review before confirmatory submission.** [ars_score.py:74](/D:/adags-cer/agent/ars_score.py:74) still excludes `multi4d_cer_early`. Submission-time deferral is documented, but the promised frozen/mode/begin/full-schedule enforcement remains pending.

