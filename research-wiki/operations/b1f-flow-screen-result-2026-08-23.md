# B1-F flow screen — machinery health CONFIRMED; metrics PENDING
# (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Design, arms and gates frozen
before any output in [[b1f-flow-birth-prior-spec-2026-08-23]]; launch
rule and preflight in [[b1f-preflight-result-2026-08-23]]. This page is
opened while the screen is still running so that the machinery-health
finding — which is already terminal for three arms — is on the record
independently of the metrics.

## 1. Cells

| arm | seed | exp | val exp | training state |
|---|---:|---:|---:|---|
| plain B1 (fresh on-pool comparator) | 0 | 237 | 251 | COMPLETED |
| plain B1 | 1 | 238 | 252 | COMPLETED |
| **B1-F** correct flow | 0 | 239 | 253 | COMPLETED |
| **B1-F** correct flow | 1 | 240 | — | running |
| **B1-X** camera-swapped wrong flow | 0 | 241 | — | running |
| **B1-X** camera-swapped wrong flow | 1 | 242 | — | running |

All commit `789595a`, pool `hopper`, admitted H100 image
`sha256:0d577168…`, 6,000 iterations, frames 0-49, cam00 sealed,
`route_logit_init: 4.0`, `elgs_reserved_parity: true`, 600k cap.

**Seeds are genuinely different in this family** — `--seed` is passed
through to the trainer as `--extra-arg=--seed`, verified in the rendered
entrypoint (`main.py … --seed 0 --test_iterations 6000`). That is a
departure from the historical ladder, where the wrapper's `--seed` never
reached `main.py` ([[seed-threading-defect-2026-08-23]]), and it is why
this screen trains its OWN plain-B1 comparators rather than reusing
experiments 197/200.

## 2. GATE CONDITION 6 (machinery health) — PASSES

The frozen gate requires flow to have actually applied, with no
missing-flow fallback, no invalid-unit pathology, and
`flow_sites_valid > 0` at every event. Read from `packet_state.pt`,
experiment 239, all seven birth events:

| iter | sites | valid | %valid | mask | bounds | non-finite | outlier | mean speed | coeff norm | **realized ratio** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 2174 | 2103 | 96.7 | 14 | 0 | 0 | 57 | 0.02051 | 0.0755 | **1.0000** |
| 1500 | 2540 | 2466 | 97.1 | 42 | 0 | 0 | 32 | 0.02108 | 0.0662 | **1.0000** |
| 2000 | 2873 | 2777 | 96.7 | 54 | 0 | 0 | 42 | 0.01335 | 0.0322 | **1.0000** |
| 2500 | 2998 | 2930 | 97.7 | 24 | 0 | 0 | 44 | 0.00969 | 0.0194 | **1.0000** |
| 3000 | 2998 | 2877 | 96.0 | 71 | 0 | 0 | 50 | 0.01672 | 0.0366 | **1.0000** |
| 3500 | 2997 | 2892 | 96.5 | 49 | 0 | 0 | 56 | 0.01408 | 0.0231 | **1.0000** |
| 4000 | 2998 | 2828 | 94.3 | 28 | 0 | 0 | 142 | 0.02669 | 0.0366 | **1.0000** |

* **94.3-97.7% of birth sites received valid flow at every event** —
  far above the 50% floor, and never zero.
* **`flow_realized_ratio_mean` is exactly 1.0000 at every event**: the
  LoRA probe basis reproduces the requested displacement in full, so the
  cell is measuring flow rather than basis rank-deficiency. That was the
  specific failure mode Appendix B's funnel fields were added to expose.
* **Zero out-of-bounds and zero non-finite failures** across all seven
  events. The only per-row rejections are the validity mask (14-71) and
  the 99th-percentile outlier rule (32-142, i.e. 1.5-4.7%).
* Coefficient norms are **0.019-0.076** — small, as the measured flow
  field predicts (p50 0.06 px, p99 3.1 px), and two orders of magnitude
  below the ~1.6 population median that the withdrawn magnitude guard
  would have clamped against. This confirms on real data what
  [[b1f-preflight-result-2026-08-23]] §3 recorded: **the original guard
  would not have bound here, so the amendment's urgency was overstated
  even though its reasoning was right.**

## 3. Capacity parity and flag-off behaviour — both verified

| arm | rows relocated over 7 events |
|---|---:|
| plain B1 (exp 237) | 19,538 |
| B1-F (exp 239) | 19,578 |

**Within 40 rows**, so the arms are capacity-matched and the only
difference is what the relocated rows' motion coefficients start at.
Recorded ladder B1 relocated 19,555 / 19,568 rows, so the fresh on-pool
comparator **reproduces the recorded operator's behaviour** despite the
pool and seed changes.

Flag-off behaviour verified from the artifact rather than assumed: the
plain-B1 record carries `flow_init: false` with every flow counter at
zero — `flow_sites_total 0`, `flow_mean_speed 0.0`,
`flow_realized_ratio_mean 0.0`. The funnel is written on EVERY record,
including arms where flow is off, which is what makes "a run whose flow
never applied" detectable from the record alone.

## 4. What is NOT yet established

**Everything the screen exists to decide.** No metric has been read. The
decisive comparison is **B1-F vs B1-X**: if correct flow does not beat
camera-swapped flow on the event endpoint, the result is UNATTRIBUTABLE
and the flow birth prior is rejected regardless of how B1-F compares to
plain B1 — a rule that exists because CSVL-VPL Stage 1 measured
camera-swapped flow scoring ABOVE valid flow.

Two pre-registrations bear on how a null must be read, both recorded
before any output: the measured flow field is small, so a null is
**likely and legitimate** ([[b1f-preflight-result-2026-08-23]] §4); and
the 0-49 masks this screen scores on are **not confirmed by ground
truth** and score mostly static pixels, raising the false-negative risk
([[b1f-flow-birth-prior-spec-2026-08-23]] Appendix C).

**Metrics and the frozen gate application are appended to this page when
all six arms and their evaluations are terminal.**
