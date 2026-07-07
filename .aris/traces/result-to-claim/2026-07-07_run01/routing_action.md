# Routing Action

Verdict: `claim_supported: no`, confidence `high`.

Action taken:

- Recorded R025 as a failed M1 method attempt in `refine-logs/EVENT_CROP_METHOD_TRACKER.md`.
- Recorded R018-R025 in `refine-logs/EXPERIMENT_TRACKER.md`.
- Wrote the decision memo `refine-logs/hide_reveal_poc/r025_event_candidate_refine_decision_memo.md`.
- Preserved the negative result in `research-wiki/experiments/r025-event-candidate-refine-real-window-check.md`, `research-wiki/event-crop-fix.md`, `research-wiki/graph/edges.jsonl`, `research-wiki/log.md`, and `findings.md`.
- Next method work should not tune M1 on the frozen windows; it should switch mechanism or decompose candidate support versus refinement damage.
