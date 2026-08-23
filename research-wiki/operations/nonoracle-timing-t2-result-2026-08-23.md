# RESULT — phase T2: gating on INFERRED boundaries is WORSE than not
# gating, and the cause is MEMBERSHIP, not timing (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Phase T1 and its gate:
[[nonoracle-episode-timing-result-2026-08-23]]. Cells: **245**
(`lrv3_a_est_r0`) and **246** (`lrv3_a_est_r1`), commit `bbf1c4f`, dgx,
admitted V100 image, both at `main.py`'s default seed 6666 to stay
seed-matched to the recorded comparators; evaluations **249**/**250**
via `scripts/eval_lrv1_event.py`, the same evaluator that produced the
recorded figures.

## 1. The measurement, all four arms from PRIMARY artifacts

| arm | `event_return` | `event_episode1` | `ordinary_all` | `ghost_gap` |
|---|---:|---:|---:|---:|
| **A0′** — no gate (exp 184) | **27.1432** | 29.6022 | 28.3515 | 29.0248 |
| **A1-LOCAL** — oracle gate (exp 185) | **28.1928** | 30.8449 | 27.9660 | 22.8333 |
| A1-shift2 — 2-frame mistiming (exp 191) | 24.7572 | 23.6906 | 28.2625 | 26.5612 |
| **A-est r0** — inferred gate | **24.3226** | 30.2506 | 28.1764 | 23.9002 |
| **A-est r1** — inferred gate | **25.0257** | 30.1549 | 28.1163 | 23.8595 |

Deltas on the decisive `event_return` endpoint:

```
A1-LOCAL − A0′  = +1.0496   (the recorded positive, reproduced exactly)
A1-shift2 − A0′ = −2.3860   (the recorded mistiming control, reproduced)
A-est    − A0′  = −2.4690   ← THIS RESULT
A-est − A1-LOCAL = −3.5188
```

## 2. Verdict — T2 FAILS, and the failure is specific

**Gating on the inferred program is 2.47 dB WORSE than not gating at
all** — essentially indistinguishable from the 2-frame mistiming arm's
−2.39 dB.

**But the timing was not wrong.** The estimated gaps are
`[5.166666666666648, 9.166666666666634]` against the oracle's
`[5.166666666666666, 9.166666666666666]` — identical to ~1e-14. **This
is a pure MEMBERSHIP failure**, and it lands at the same magnitude that
a 2-frame timing error does.

The per-region signature says the same thing and rules out a global
regression:

* `event_episode1` **30.20**, which BEATS A0′'s 29.60 by +0.60 and sits
  near the oracle's 30.84 — while the object is continuously present the
  gate is working;
* `ghost_gap` **23.88**, close to the oracle's 22.83 and far from A0′'s
  29.02 — the absence is being rendered, i.e. the gate fires;
* `ordinary_all` **28.15**, between A0′ and the oracle — no
  ordinary-region catastrophe;
* **only `event_return` collapses.**

**So the gate is on, at the right times, helping during presence, and
destroying the return.**

## 3. The mechanism — two candidates, both pointing the same way, attribution is INFERENCE

**The membership is simultaneously too coarse and too incomplete.**

* **Too coarse (over-gating).** Recorded before this result from the
  seeding log: the estimated program gates **336 rows in 2 families**
  where the oracle gated ~84 in 8. Two voxel cells enclose ~3.8× the
  oracle sphere's volume, so background and ground rows inside those
  cells are driven to exact absence during the gap.
* **Too incomplete (partial gating).** T1 gated only **2 of 8**
  event-overlapping groups, so part of the object is gated and part is
  not. The ungated part keeps the ordinary temporal marginal and is
  therefore trained across presence AND absence, while the gated part
  carries a clean episodic program. At the return the two parts
  superpose inconsistently.

**The decisive observation is the ordering**, which neither mechanism
alone would predict and both jointly explain:

```
fully gated 28.19  >  NOT gated 27.14  >>  PARTIALLY gated 24.67
```

**Partial membership is worse than BOTH full gating and no gating.**
That is the membership analogue of the measured timing result, and it is
new: the project had established that imprecise TIMING has negative
value, and this establishes that imprecise MEMBERSHIP does too, at
comparable magnitude.

This cell cannot separate the two mechanisms. A per-row membership
precision/recall figure against the authored sphere would, and is not
part of this cell.

## 4. Replicate spread — this arm is noisier than the family

r0 24.3226 against r1 25.0257: a spread of **0.703 dB**, against the
recorded LRV3 same-arm spread of **0.09-0.17 dB**. The A-est arm is
**~4-8× less stable** than the family, which is itself consistent with a
partially-gated configuration being sensitive to which rows land inside
the cells. The −2.47 dB mean is ~3.5× that spread, so the sign is not in
doubt, but any magnitude quoted from this arm carries a much wider band
than the family's usual one.

Both cells are **REPLICATES at seed 6666, not two seeds** — see
[[seed-threading-defect-2026-08-23]]. The whole-frame training figures
(28.3407 / 28.2850) differ by 0.056 dB, inside the family band; the
0.703 dB spread is specific to the event-return endpoint.

## 5. What this establishes, and what it does NOT

**Established.** An inferred episode program with EXACT boundaries and
imprecise membership is worse than no gate at all on the endpoint the
method exists to improve. **T1's positive therefore does NOT carry
downstream on its own.** Recovering boundaries is necessary and is now
demonstrated; it is not sufficient.

**NOT established, and the distinction matters.** This does not show that
non-oracle gating cannot work. It shows that **voxel-cell membership at
8³ over the cloud's bounding box does not work**, for reasons diagnosed
above. The estimator's boundary output is exact and reusable; only the
membership half is refuted.

**NOT established:** which of the two mechanisms dominates.

**Unchanged:** the oracle-gated positive (+1.0496 dB) and the mistiming
control (−2.39 dB) both reproduced exactly from the recorded artifacts in
this run, so the comparators are sound and the evaluator is unchanged.

## 6. Consequence for the paper's claim boundary

The localized-presence positive still rests on **authored** boundaries
AND **authored** membership. This block demonstrated the first can be
inferred exactly; it has now measured that inferring the second by the
obvious spatial route makes the mechanism harmful rather than merely
weaker.

**The pre-identified next step is membership, not timing** — a
per-primitive membership estimator, or a finer/adaptive partition, scored
by per-row precision and recall against the authored sphere BEFORE any
retraining. That ordering is exactly the one that made T1 cheap and
decisive, and it should be reused: measure the membership instrument
against ground truth first, and retrain only if it is precise enough.

## 7. Bookkeeping

Claims consumed: `lrv3_a_est_r0` r0 (245), `lrv3_a_est_r1` r0 (246),
`lrv3_eval_a_est_r0` r0 (249), `lrv3_eval_a_est_r1` r0 (250). Local
artifacts pulled to the session scratchpad; recorded comparators read
from `data/synthetic/lrv3_results/local/{a0p_r1,a1l_r1,a1s2}/`.
