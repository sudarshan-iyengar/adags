# B2 DC-edit falsification on the authored fixture — the edit has nothing to give (2026-08-20)

EXPLORATORY. Lane 4 of the 2026-08-20 block: before any 300-frame B2
search, determine whether the current directional donor DC appearance
edit ([[ccr-method-2026-08-20]] §3.2) can help when proposal ambiguity
is removed. Frozen falsification rule, fixed before any output:

> If a non-vacuous oracle-correct appearance edit does not improve
> reserved reconstruction, stop scaling the current B2 appearance
> operator to N3V.

## 1. Setup (all primary-verified)

* Substrate: experiment **209** (`lrv3_b1_packets` r0, commit `4d15fcf`,
  dgx, admitted V100 image) — LRV3 + the EXACT N3V ladder B1 operator
  (7 events, 121 packets, 4,668 rows relocated, 149,800 rows,
  `chkpnt6000.pth`, `best_val/psnr` 28.59).
* Tool: `scripts/falsify_b2_edit.py` (commit `67c77bf`, 21 CPU tests).
  Row sets from AUTHORED ground truth (oracle sphere + μ±2σ operational
  support windows — never called exact support): donor D = 3,722
  episode-1 object rows; recipient R = 3,912 return-local object rows;
  spanning rows excluded (229); wrong-identity pool 29,351 → Dw = 3,722
  descriptor-closest off-object rows; no-op split D_a/D_b =
  1,853/1,869. Links prespecified: L1 oracle-correct D→R, L2
  wrong-identity Dw→R, L3 same-identity no-op D_a→D_b. The edit is the
  UNCHANGED `dc` pointer redirect; the admission rule is the unchanged
  `mean + 3·SE < 0` + per-side ≤ 0 on reserved units (parity-reserved
  at training, never trained on).
* Two amendments to slot ALLOCATION only, both decided before any
  reconstruction render: links share reserved units (each link scores
  against the same identity base, never an accumulating admitted state;
  LRV3's 3-frame return window holds only 12 reserved units), and L3 is
  scored on (W1, W1) where its rows render. Experiment **212** r0 is
  the preserved INVALID_SLOTS run that motivated the second fix (an
  identical-window interaction in `pick_confirmation_slot`); its gate
  refused BEFORE any render, so no outcome was observed pre-amendment.

## 2. Anti-vacuity — established before any delta

| quantity | value |
|---|---|
| L1 pre-edit DC distance (mean / max) | **0.0464** / 0.766 |
| L3 same-surface floor (mean) | 0.0325 |
| L2 wrong-identity distance (mean) | 0.706 |
| comparative gate (L1 > L3 strictly) | **PASS** |
| L1 recipient rows changed | 3,912 (100%) |
| reserved units: W1 / WR | 120 / 12 |

The oracle pair is genuinely non-vacuous — but note the margin: the
return-local rows' trained DC sits only 1.4× the same-surface floor from
the donor's, against 21× for wrong identity. That number is the result
in miniature.

## 3. RESULT (experiment 213, `lrv3_falsify_b2_dc` r1, COMPLETED)

Paired per-unit L1-loss deltas (negative = improvement); event_return =
pooled PSNR over the ground-truth object mask, 4 held-out cameras,
frames 57–59 (report-only, selects nothing):

| link | raw slot Δ (mean ± SE) | all-WR-reserved Δ | event_return (base → edited) | certificate |
|---|---:|---:|---:|---|
| **L1 oracle-correct** | **+2.67e-6 ± 1.53e-6** | +3.50e-6 | 27.218 → 27.226 (**+0.008 dB**) | REJECTED at pooled-rule |
| L2 wrong-identity | +1.06e-3 ± 4.2e-4 | +2.14e-3 | 27.218 → 19.881 (**−7.34 dB**) | REJECTED at pooled-rule |
| L3 no-op | −3.9e-8 ± 7.7e-8 | −3.9e-11 | 27.218 → 27.218 (0.000) | rejected (≈0) |

Structural sanity, all as designed: L1/L2's donor-side means are exactly
0.0 (recipient rows have no support in W1); L3's deltas are numerical
zero (the machinery manufactures nothing); the certificate REJECTS the
wrong-identity edit that would have cost 7.3 dB held-out.

## 4. Verdict — the falsification rule FIRES

**A non-vacuous, perfectly-linked, oracle-correct DC edit does not
improve reserved reconstruction** (raw mean marginally POSITIVE, i.e.
slightly worse; held-out event gain +0.008 dB ≈ nothing). This is not a
certificate-power question: the sample mean is not even negative, so no
power increase can rescue it. Per the frozen rule: **stop scaling the
current B2 DC appearance operator to N3V.**

Mechanism, measured rather than guessed: LRV3's object returns with
identical appearance, and 48 training view-frames of the 3-frame return
sufficed to train the return-local rows' DC to within 0.046 of the
donor's. The edit substitutes nearly-identical values. Retroactively,
this also explains half of the ladder-round-1 zero-admit result
([[ccr-ladder-round1-results-2026-08-20]]): even perfect proposals
offered the certificate an effect size of ~0 at DC.

## 5. What is and is not established

* **Established:** the current stop-grad DC-pointer edit has ~zero
  effect under the best possible conditions this fixture can produce;
  the certificate discriminates identity correctly (rejects a −7.3 dB
  wrong edit, passes nothing); the pipeline manufactures no spurious
  deltas.
* **NOT established:** that appearance consolidation is dead as a
  concept. The fixture's return is not appearance-STARVED — pre-edit
  distance 0.046 bounds the available headroom at ~nothing. A fixture
  variant where the recipient is genuinely starved (1-frame return,
  fewer views, or authored appearance drift under changed illumination)
  could measure whether ANY appearance-transfer payload has value; and
  payloads that move geometry/opacity/support were never tested here.
* **Consequence for Lane 3's Part B design:** the prior-assisted
  lineage-proposal experiment ([[prior-asset-inventory-2026-08-20]]
  Part B) is now BLOCKED on its stated precondition — a better proposer
  cannot rescue an ineffective edit. The zero-acquisition prior work
  that remains live is the BIRTH-prior role (SEA-RAFT flow for B1).

## 6. Bookkeeping

Claims: `lrv3_falsify_b2_dc` r0 (exp 212, INVALID_SLOTS, preserved) and
r1 (exp 213, COMPLETED) consumed → next free r2. Reports:
`falsify_b2_dc_report{,_r1}.json` in experiment 209's run dir; local
copies under `data/synthetic/lrv3_results/local/falsify/` (gitignored).
Cost: 209 ≈ 1.1 slot-h; 212+213 ≈ 0.3 slot-h.
