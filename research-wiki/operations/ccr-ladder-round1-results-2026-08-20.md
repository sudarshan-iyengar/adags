# CCR ladder, round 1 — results (2026-08-20)

EXPLORATORY, `evidence_bearing: false` throughout. Method and gates:
[[ccr-method-2026-08-20]] (frozen before any cell output). Segment and
event-ray masks: [[ccr-segment-selection-2026-08-20]] (frozen before any
cell output). All cells: `cut_roasted_beef` frames 0-49, 1352x1014,
cam00 sealed, 6000 iterations, batch 2, 600k cap, reserved parity ON in
every arm (713 of 950 training units; NOT comparable to experiments
181/194 which trained on all 950 — see the §6a caveat in the 2026-08-20
handover and RESULT note 4 below), admitted image, pool dgx. Exactly 2
seeds per arm, fixed in advance.

## 1. Cells

| cell | exp | commit | outcome |
|---|---|---|---|
| B0 s0 | 196 | `e70ef44` | COMPLETED |
| B0 s1 | 199 | `22b2dd6` | COMPLETED |
| B1 s0 | 197 | `22b2dd6` | COMPLETED — 7 relocation events, 637 packets, 19,555 rows relocated |
| B1 s1 | 200 | `a798949` | COMPLETED — 7 events, 677 packets, 19,568 rows |
| B2-DC pass s0 | 202 | `454ecc6` | COMPLETED — **0 edges admitted** |
| B2-DC pass s1 | 206 | `454ecc6` | COMPLETED — **0 edges admitted** |
| val evals | 201/203/204/205 | | COMPLETED (pooled+clamped+LPIPS) |

Commits differ only by files outside the training path of these configs
(wiki pages, the consolidation script, the event tool); the executing
config pair is byte-fixed and the packet-birth/renderer code is identical
across 196-206.

## 2. Global metrics (val protocol: pooled PSNR over clamped renders, cam00)

| arm | seed 0 | seed 1 | paired delta vs B0 |
|---|---|---|---|
| B0 | 34.0742 / 0.95932 / 0.08123 | 33.8068 / 0.95934 / 0.08104 | — |
| B1 | 33.8037 / 0.95971 / 0.08144 | 34.0987 / 0.95906 / 0.08139 | **−0.271 / +0.292 → mean +0.011** |
| B2-DC | ≡ B1 | ≡ B1 | 0 vs B1, by construction |

Primitive counts: 599,396 / 599,448 / 599,406 / 599,470 — the 600k cap
binds in every arm; capacity matched.

**Reading:** the observation-born relocation operator is GLOBALLY
NEUTRAL — the paired mean is +0.011 dB and the per-seed deltas flip sign
at ±0.28, which is also the measured B0 seed spread (34.074 vs 33.807).
Any single-seed claim in either direction would have been wrong.

## 3. Event-ray metrics (frozen masks; 8-bit saved-render basis)

Pooled PSNR over the predefined revealed-surface regions:

| region | B0 s0 | B1 s0 | Δs0 | B0 s1 | B1 s1 | Δs1 | paired mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A hand-press reveal | 37.415 | 37.505 | +0.091 | 37.750 | 37.847 | +0.097 | **+0.094** |
| B knife-stroke reveal | 36.209 | 36.279 | +0.070 | 36.295 | 36.205 | −0.090 | −0.010 |
| C tongs-band reveal | 27.473 | 27.552 | +0.078 | 27.545 | 27.979 | +0.433 | +0.256 |
| **union** | 31.953 | 32.030 | **+0.077** | 32.066 | 32.411 | **+0.345** | **+0.211** |
| complement | 33.952 | 33.614 | −0.339 | 33.604 | 34.010 | +0.407 | +0.034 |

**Reading:** the event-region union improves under B1 on BOTH seeds
(+0.077, +0.345), and region A — the cleanest occlude-reveal cycle — is
+0.09 on both seeds almost exactly. The complement flips sign (neutral).
So the operator moves quality INTO the diagnosed regions with a
consistent direction, at globally neutral cost. This is a screening
observation on one frozen segment: region B flips sign, the magnitudes
overlap the B0 seed spread within regions, and no promotion rule fires.

## 4. B2-DC: zero certified consolidations, replicated

The first-ever executions of the CCR pass (experiments 202/206):

| funnel stage | seed 0 | seed 1 |
|---|---:|---:|
| eligible packets (≥4 rows post-medoid-gate) | 304 | 361 |
| temporally disjoint pairs | 2,161 | 4,449 |
| mutual-nearest proposals evaluated | 47 | 74 |
| stage-1 screen survivors (mean Δ<0) | 10 | 10 |
| confirmation attempted | 5 | 6 |
| slot-unconfirmable | 5 | 4 |
| **admitted** | **0** | **0** |
| joint veto | never reached | never reached |
| search cost | 37 s | 51 s |

Per the frozen gate: **B2-DC ≡ B1 exactly, and the finding is "no
certified opportunity on this segment"** — the fail-safe design doing
precisely what it was built to do. The funnel separates the causes:

* proposals EXIST (47/74 mutual-nearest pairs) and 10 per seed improved
  reconstruction on the stratified screen;
* ~half the screen survivors could not be allocated a confirmation slot:
  packet temporal supports are ~6 frames (sigma_t = 1.5 frames), and the
  per-side reserved-unit requirement is hard to satisfy inside them —
  a MACHINERY limitation, not evidence about consolidation;
* the other half failed the reserved-unit rule (mean + 3·SE < 0 with
  per-side non-degradation at 16 units) — a deliberately strict,
  low-power certificate, disclosed as such in the frozen spec;
* additionally, stop-gradient DC-only reuse has intrinsically small
  per-pair upside — the hostile review predicted this possibility when
  recommending the DC arm.

**Forbidden by this result:** any claim that certified consolidation
improves N3V reconstruction. **Not established:** that consolidation
cannot help — the pass never got certified material to work with, and
the funnel points at the two specific bottlenecks (slot allocation for
narrow supports; certificate power). Loosening either is a NEW frozen
spec, never a post-hoc adjustment.

## 5. Gate application (frozen ccr-method §4)

* B2-DC promotion: **NOT MET** (paired mean 0 < +0.30; zero admits).
* B1-vs-B0 screening: globally neutral (+0.011 paired mean); consistent
  event-union direction (+0.077/+0.345, both seeds positive) reported
  as a directional datum, not a win.
* Per-seed non-catastrophe: vacuous for B2 (≡ B1).
* Control arms (shuffled / easy-negative / placebo): **NOT RUN** — with
  zero admitted edges there is no admitted set for the shuffled arm to
  match and no Claim-B utility to control; they become mandatory the
  first time any pass admits > 0 edges.

## 6. What round 2 should change (pre-identified, from the funnel, not from tuning on outcomes)

1. **Wider packet supports or slot-side relaxation as a NEW spec**: at
   sigma_t = 1.5 frames the confirmation-slot requirement starves half
   the survivors. Either widen packet temporal sigma at birth, or
   re-specify the slot rule for narrow supports (with its own review).
2. **Give consolidation material**: B1's packets are born late
   ([1000, 4000]) with fresh DC — the same surface rarely produces TWO
   temporally disjoint, well-trained packets inside 50 frames. A longer
   window (300 frames) or an earlier/heavier birth schedule raises the
   opportunity denominator, as does the synthetic fixture (where Claim A
   identity scoring is defined and still untested).
3. **The B1 operator itself is worth one more round**: the event-region
   direction is consistent and region A replicates at +0.09; a variant
   that avoids the complement cost at seed 0 (e.g. donor selection
   excluding static-region rows) could turn "neutral globally, better
   at events" into a clean win. Any such change is a new one-variable
   cell.

## 7. Cost

Ladder cells 195-206: preflight 0.6 + 4 trainings ~2.6 each + 2 passes
~0.15 + 4 evals ~0.15 ≈ **12 slot-hours**; whole block (184-206)
≈ 17-18 slot-hours, all dgx, all exploratory.
