# RESULT — B1-F preflight: the flow mechanism is healthy; the six-cell
# screen is LAUNCHED (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Launch rule frozen before the
funnel was read in [[b1f-flow-birth-prior-spec-2026-08-23]] Appendix D.
Cell: Determined experiment **234** (`ladder_b1f_preflight` r0, commit
`2ba6a62`, pool `hopper`, admitted H100 image `sha256:0d577168…`,
1,200 iterations), COMPLETED in ≈25 min ≈ **0.42 slot-h**.

## 1. The funnel at the single birth event (iteration 1,000)

Read from `packet_state.pt` in the run dir, schema
`ccr-b1-packet-birth-v1`, config confirming `flow_init: true`,
`flow_source: "correct"`:

| field | value |
|---|---:|
| camera / timestamp | `cam18_0027` / 0.9 |
| donors requested = realized | 2,159 (shortfall 0) |
| packets stamped / rows assigned | 84 / 1,708 |
| **`flow_sites_total`** | **2,159** |
| **`flow_sites_valid`** | **2,129 (98.6%)** |
| `flow_failed_mask` | 8 |
| `flow_failed_bounds` | 0 |
| `flow_failed_nonfinite` | 0 |
| `flow_failed_outlier` (99th-pct rule) | 22 |
| **`flow_realized_ratio_mean`** | **1.0** |
| `flow_coeff_norm_mean` / `max` | 0.0518 / 0.8236 |
| `flow_mean_speed` | 0.01537 |

## 2. The frozen launch rule, applied

* **(a) flow actually applied** — `flow_sites_valid / flow_sites_total` =
  **98.6% ≥ 50%**, and the event is not zero. **PASS.**
* **(b) the basis can represent the velocities** —
  `flow_realized_ratio_mean` = **1.0 ≥ 0.9**; the LoRA probe basis
  reproduces the requested displacement **in full**. **PASS.**
* Budget remaining well above the 12 slot-hour floor. **PASS.**

**The six-cell screen is LAUNCHED**: experiments **237/238** (plain B1,
on-pool comparator), **239/240** (B1-F, correct flow), **241/242**
(B1-X, camera-swapped wrong-flow control), all `hopper`, seeds 0 and 1.

Two further mechanism facts the preflight establishes incidentally: the
run crossed iteration 1,000 **without raising**, which proves the
`main.py` trainer wiring reaches `MotionPriorCache` — an unwired
`flow_init: true` run is designed to fail closed at exactly that point;
and the rendered entrypoint was inspected before submission and contains
no `Program Files` substring.

## 3. An honest correction to my own amendment

[[b1f-flow-birth-prior-spec-2026-08-23]] Appendix B withdrew the
coefficient-norm median clamp on the strength of an implementation-time
measurement: "~328 coefficient norm needed against a population median of
~1.6", i.e. a ~200× shrink that would have made the cell vacuous.

**On real N3V data that measurement does not apply.** The measured
coefficient norm here is **0.0518 mean, 0.8236 max** — far BELOW the
~1.6 population median the amendment was argued against. **The original
guard would very probably not have bound at all on this data.**

The discrepancy is explained, as INFERENCE rather than measurement, by
the displacement scale: the implementation-time figure came from a
synthetic case whose flow mapped to a much larger world displacement,
and `|c| ≈ |displacement| / |B|` with the LoRA basis still near its
`motion_lora_init_scale: 0.01` at iteration 1,000 — which is consistent
with the observed `0.0518 ≈ 5e-4 / 0.01`.

**So the amendment remains correct in PRINCIPLE — a LoRA coefficient is
not a physical quantity and a population-median clamp on it measures the
wrong thing — but its stated URGENCY was overstated, because the
evidence offered for it did not come from this data.** The one
behavioural change that did take effect is the replacement guard: the
99th-percentile input-side outlier rule rejected **22 of 2,159 sites
(1.0%)**, a small and disclosed effect.

Recorded because the failure mode generalizes: a guard justified by a
synthetic measurement should have its scale checked against the real
data before the justification is written down as a number.

## 4. What the preflight does NOT establish

Exactly what Appendix D said in advance: **whether the effect is worth
measuring.** `flow_mean_speed` is 0.01537 in the trainer's velocity
unit, and the independently measured flow field is small (p50 0.06 px,
p99 3.1 px, only 8.6% of pixels moving more than half a pixel), so the
initialization is a small perturbation to rows that then train for
thousands of further Adam steps. Whether it survives is the question the
six cells exist to answer and cannot be inferred from 1,200 iterations.

## 5. Bookkeeping

Claim consumed: `ladder_b1f_preflight` r0 (experiment 234). Point count
grew 366,366 → 446,329 over the run, so densification was active and the
birth event fired against a densifying cloud, as it will in the screen.
