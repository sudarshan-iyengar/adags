# Synthetic Hide/Reveal PoC Report

## Heldout Metrics

- `n`: 120
- `true_events`: 72
- `reveal_events`: 72
- `normal_controls`: 48
- `candidate_recall`: 1
- `candidate_false_positive_rate`: 0
- `candidate_score_auc`: 1
- `margin_auc`: 1
- `accepted_precision`: 1
- `accepted_recall`: 1
- `false_event_rate_normal`: 0
- `mean_delta_true`: -0.566367
- `mean_delta_normal`: 0.393649
- `identity_reconnection_accuracy`: 1
- `matched_lifespan_accept_recall`: 1
- `matched_lifespan_identity_reconnection_accuracy`: 0
- `no_identity_accept_recall`: 1
- `no_identity_identity_reconnection_accuracy`: 0
- `unnormalized_accept_recall`: 1
- `unnormalized_false_event_rate`: 0
- `no_hysteresis_false_event_rate`: 0

## Stop / Go

- `pass_candidate_recall`: True
- `pass_margin_separation`: True
- `pass_matched_lifespan_gate`: True
- `pass_no_identity_deletion`: True
- `proceed_to_real_windows`: True

## Notes

- Synthetic labels carry the identity claim; real windows should be sanity checks only.
- Matched lifespan can accept the same patch event, but gets no identity-reconnection credit because it has no hidden-identity reveal matching.
- The no-identity deletion can accept patch events, but should not pass the identity reconnection gate.
