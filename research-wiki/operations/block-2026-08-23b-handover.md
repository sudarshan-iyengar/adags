# HANDOVER — 2026-08-23 block 2

Self-contained. EXPLORATORY throughout, `evidence_bearing: false` on
every cell.

## 1. State

| item | value |
|---|---|
| branch | `apollo/csvl-vpl-v2-exploratory` |
| local == origin | `eb293a27a252a0ad773cd1ea14aa1cc34fce6c4a` |
| divergence | **0 / 0** |
| block start | `81bbac8` |
| protected files | `research-wiki/deep-dive-prompt.txt`, `run-deep-dive.ps1` — untracked and untouched |
| other outstanding | `overnight-handover-23aug.md` (user-owned, not touched by me) |

Commits: `2a06043` closure repair · `1190f58` LRV4 diagnostic +
frozen spec · `b8e41cf` three result pages + query_pack + gap_map ·
`eb293a2` replicate-floor spec.

## 2. RUNNING — pick this up first

Experiments **261 / 262 / 263** (`repl_floor_b1_{a,b,c}`, r0, pool
`hopper`, commit `eb293a2`, image `sha256:0d577168…`), all
`STATE_RUNNING` at handover, ~2.5 slot-h each, ≈7.5 total.

Spec and the **interpretation frozen in advance**:
[[same-code-replicate-floor-spec-2026-08-23]]. Do not move a threshold
after reading them.

```bash
python scripts/det_monitor.py experiment --experiment-id 261
```

When all three are `STATE_COMPLETED`, evaluate each at `chkpnt6000` with
`main.py --val` (pooled+clamped) and score the frozen event union with
`scripts/event_ray_metrics.py` on
`configs/n3v/ladder_event_masks_crb0_49.json`, 8-bit saved-render basis —
the same convention the flow screen used. Then apply the spec's §4 table:
`R` ≥ 0.341 retires the two-seed 50-frame protocol for utility claims;
0.10 ≤ `R` < 0.341 means every future delta must be reported beside `R`;
`R` < 0.10 leaves two-seed power as stated. Append the result to the spec
page. **Experiment 237 is not a fourth replicate — different commit.**

## 3. Results (all terminal, all recorded)

**Membership — priority 1, zero GPU**
([[lrv3-membership-diagnostic-2026-08-23]]). The membership that ACTED
binds on the fresh 50k seeding cloud: **precision 0.0446, recall
0.1786** (336 gated, 15 correct, 84 in-sphere). The trained substrate's
96.9% precision is a different cloud and describes nothing that happened
in training — the T2 page's over-gating attribution was right.
**Recall is hard-capped at 0.1786**: only 15 of 84 in-sphere rows lie in
the two gated cells; the other 69 are in the six the estimator abstained
on. A 64³ occupancy grid improves precision **6×** and moves recall
**not at all**, so every downstream repair lands in the partial-membership
regime measured at 24.67 against 27.14 ungated. **No retraining was run —
no clear pre-output repair exists.** T1's exact boundaries are untouched.

**LRV4 — priority 2, experiments 259/260**
([[lrv4-lo-distribution-result-2026-08-23]]). Reading **(b)**: `P2` = 8
rows against LRV3's 3,925, support widths **27× larger and disjoint**.
277 rows are in-region at the return with median width **25.6 s — the
whole sequence**. A one-frame return produces no localized rows at all;
no floor rescues it. Observation-supply claim still **UNTESTED**, and
LRV4 cannot test it.

**Flow — priority 3, zero GPU** ([[b1f-flow-postmortem-2026-08-23]]).
**EXPLAINED.** Arms differing by two config lines run identical code to
iteration 1000 yet separate by 0.089–0.236 relative RMS training loss in
501–999 — indistinguishable from post-intervention. Densification at 500
amplifies 1400×. Two corrections: the camera swap is only **~84%
decorrelated** (β ≈ 0.16; rejection stands, gate 3's FAIL is not
decisive), and the birth-gating "dynamic mask" is a **top-15% residual
quantile** for every training camera because real masks exist only for
held-out `cam00`.

**Paper path** ([[paper-path-decision-2026-08-23b]]):
**representation-first**; authored membership is a quantified declared
limitation, not a pending repair.

## 4. Cost

| item | slot-h |
|---|---:|
| 259 + 260 (LRV4/LRV3 diagnostics) | ≈0.1 measured |
| 261 + 262 + 263 (replicate floor, running) | ≈7.5 projected |
| everything else | **0** |

## 5. Permitted and forbidden

**Permitted:** membership by 8³ spatial partition is refuted, quantified
at the moment it binds. Non-oracle recall on this fixture is capped at
0.1786. LRV4's null is not a threshold artefact. The flow null is
explained by the measurement channel.

**Forbidden:** that non-oracle membership is impossible in general (only
this partition on this fixture is refuted); that a finer estimator would
work (precision improves, recall cannot); that flow is useless as a prior
(only velocity initialization for relocated packets at this protocol was
tested, and the channel could not resolve it); that observation supply
does or does not drive headroom (untested); that the replicate floor is
known (running).

## 6. Open blockers

* The N3V utility lane is **suspended behind 261–263**. Nothing should be
  scaled until the replicate floor returns.
* A view-starved fixture (fewer cameras, unchanged 3-frame return) is the
  only remaining way to test observation supply. Per
  [[lrv3-fixture-hazards-2026-08-23]] §2 it needs a new named fixture and
  a tool-guard relaxation. **Not proposed, not authorized.**
* `test/track_flow_l1` = 0.06124 identically across all six flow-screen
  models — an inert metric, recorded not chased.
* Per-row survival of flow-initialized velocity needs
  `_motion_lora_coeff` from the 536 MB per-arm checkpoints — present, not
  pulled.
