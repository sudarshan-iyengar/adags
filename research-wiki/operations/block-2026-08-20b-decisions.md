# Block decisions — 2026-08-20/22 (B0-C, B1-D, B2 falsification, priors)

EXPLORATORY tier throughout. The five mandated end-of-block decisions,
strongest-first, each evidence-labelled. Experiments this block:
208–232 (ledger `agent-control/elgs-apollo/experiment-ledger.jsonl` is
the id/retry authority); ≈ 74 slot-hours total (dominated by the two
300-frame hopper runs: 15.9 + 38.2).

## 1. B0-C is a STABLE and QUALIFIEDLY COMPETITIVE 300-frame substrate — with the schedule, not the substrate, as the finding

Verified from primary artifacts
([[b0c-canonical-300f-results-2026-08-21]]): frozen endpoint **33.251 /
0.9535 / 0.0898** over all 300 held-out frames from ONE model;
per-frame spread 0.73 dB (no catastrophic frame); mild late-sequence
degradation only. STG's published 33.52 is six 50-frame specialists —
unmatched representation/init/schedule, different LPIPS convention; no
SOTA claim. The load-bearing discoveries: (a) both capacity regimes
peak at ~12k iterations (~4.2 presentations/unit) and then LOSE PSNR to
densification churn + late train-view overfit (NOT the cap — corrected
append-only in Appendix B); (b) capacity (2.05M vs 600k) buys
perceptual quality (LPIPS −7.6%) but loses endpoint PSNR at 2.4× cost;
(c) the historical 6k/300f protocol at full raster scores 32.90 — the
quarter-raster era inflated the family ~1.1–1.5 dB.

## 2. B1-D is REJECTED; retain B1

All four frozen gate conditions failed, on both seeds where applicable
([[b1d-donor-mask-result-2026-08-20]]): global paired mean −0.060 with
sign-split at seed-spread magnitude, and — decisive — the event-ray
union NEGATIVE on both seeds (−0.145/−0.313): the donor mask removes
exactly the event-region benefit B1 exists for. Inference recorded:
unrestricted B1 transfers capacity from static background INTO events;
masking donors makes relocation cannibalize its targets. Machinery was
healthy (zero shortfalls/skips). Any inverse-mask rule is a new frozen
spec.

## 3. The current B2 DC appearance edit is FALSIFIED for N3V scaling

Verified from primary artifacts ([[b2-edit-falsification-2026-08-20]]):
a non-vacuous oracle-correct link (pre-edit DC distance 0.046, 1.4× the
same-surface floor; 3,912 rows changed) yields raw reserved delta
**+2.7e-6** (marginally worse) and +0.008 dB held-out event return —
not a power question. Controls behaved exactly as designed
(wrong-identity −7.34 dB, certificate rejects it; no-op ≈ 0). Measured
mechanism: 48 return view-frames trained the recipient's DC to within
0.046 of the donor's — the fixture's return is not appearance-starved.
**Recommendation: replace or abandon the DC appearance edit as the
consolidation payload.** Not established: that consolidation is dead —
payloads moving geometry/opacity/support were never tested, and an
appearance-STARVED fixture variant (1-frame return / authored drift)
could revisit appearance transfer.

## 4. Priors: SEA-RAFT flow exists for all six N3V scenes; the BIRTH-prior role is testable first, with zero acquisition

Verified by inventory ([[prior-asset-inventory-2026-08-20]]): dense
SEA-RAFT flow, 20 cams × 299 pairs per scene, already in the
`MotionPriorCache.get_track_flow` layout with per-pixel validity;
CoTracker3 + DA3 weights resident. Long-range tracks exist only for
DiVa-360 (builder hard-requires its conventions) — the LINEAGE-prior
role is blocked on an N3V adapter AND on decision 3's precondition (a
better proposer cannot rescue an ineffective edit; the Part-B four-arm
design stays frozen and unrun). **First prior experiment: flow as a B1
birth prior** (flow-consistent velocity/site initialization for
relocated packets) at the 50-frame screen tier.

## 5. The next 300-frame paired comparison — exact design and measured cost

**Pair: B0-R vs B1** (B1-D rejected; consolidation blocked by decision
3), 2 paired seeds, reserved parity in BOTH arms, `cut_roasted_beef`
0–299, cam00 sealed. **Schedule: a NEW frozen spec at 18,000 iterations
(6.3 presentations/unit), densify 500→12,000, lr horizon 18,000, cap
600k, endpoint chkpnt18000** — justified by the measured twin curves
(both arms peak at 12k; 18k captures peak + settle at ~45% of 36k
cost). Disclosed: the schedule choice derives from held-out convergence
curves of the B0-C/B0-C-UNCAP runs (design-from-prior-evidence).
Required before launch: 300-frame packet-birth window freeze (e.g.
events every 500 over [1000, 12000] ≈ 22 events) and a 300-frame
event-ray mask spec (the 0–49 masks are dev-only and do not transfer).
**Measured cost: ≈ 8 h/cell on H100 at the capped regime → 4 training
cells ≈ 32 slot-h + ≈ 3 slot-h evals ≈ 35 slot-h.** Not recommended: a
300-frame B2 ladder (decision 3), any six-scene benchmark, or scaling
B1-D (decision 2).

## Bonus decisions this block (user-authorized additions)

* **Route-init screen** ([[route-init-screen-2026-08-20]]): the
  init-order defect's forced 4.0 HELPED by +0.50 dB over the intended
  neutral 0.0 (which leaves 24.7% of routes uncertain at 6k); the
  replicate check passed at **0.018 dB**, empirically validating the
  training-path-inert assumption behind this block's comparator reuse.
  The canonical family's explicit 4.0 now rests on measurement. No
  300-frame 0.0 arm.
* **B0-C-UNCAP** (Appendix B of the results page): the capacity
  question answered as above.

## Anomalies and consumed-by-error indices (preserved, never reused)

* `b0c_uncap_eval6k` r0: submission hung between claim and
  `det e create` (verified: no experiment reached the master); r1 ran
  as exp 221.
* `lrv3_falsify_b2_dc` r0 (exp 212): INVALID_SLOTS by the frozen
  allocation — the gate refused BEFORE any render; allocation amended
  (links share reserved units; identical-window direct allocation) and
  r1 (exp 213) completed. Both preserved.
* Config header prose of four copied configs was stale (BOM-silent
  regex failure) — values always parsed-diff-verified; corrected with
  byte-identity checks; recorded in
  [[b0c-canonical-300f-2026-08-20]] Appendix A.
