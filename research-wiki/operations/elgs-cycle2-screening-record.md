# EL-GS Cycle-2 Screening Record (IN PROGRESS)

> **APPEND-ONLY SUPERSESSION NOTE (2026-08-13).** Two rows of the screening
> table below were produced from conversions with a VERIFIED image-substrate
> defect — `writing_2` and `xylophone` drew frames from `segmented_ngp`
> (1280x720) against calibration declaring 1160x550. See
> [[operations/elgs-substrate-defect-2026-08-13]].
>
> - **writing_2's row (union 50, coverage 0.845, 84 true-absences) is
>   INVALIDATED PENDING REMEASUREMENT.** Because writing_2 is the SOLE
>   eligible candidate, the FORMAL SELECTION OUTCOME below — "exactly ONE
>   candidate qualifies ... DRY WITHIN THE FROZEN BUDGET" — depends
>   entirely on that one row and is likewise **CORRECTED OUTCOME
>   UNKNOWN**.
> - **xylophone's row (occl 1,031, 0/0/0, coverage 0.577, NO) is
>   INVALIDATED PENDING REMEASUREMENT.** If a corrected xylophone were
>   eligible, the eligible count becomes 2 and the frozen checkpoint
>   autonomy condition (">= 2 eligible") flips from NOT MET to MET — a
>   recorded decision, not a cosmetic row.
> - The other 18 rows are unaffected (all 24 clean conversions verified by
>   decoded-dimension measurement).
> - The R2-prime verification, evaluator alignment, and prereg review
>   chain are unaffected — this defect is upstream of the evaluator, in
>   the converter.
> - All rows and outcomes below are PRESERVED unchanged as the original
>   record. Remeasurement will reapply the UNCHANGED frozen eligibility
>   predicate.
>
> **REMEASUREMENT COMPLETE (2026-08-14).** Both rows were remeasured on the
> corrected substrate under the UNCHANGED frozen eligibility predicate
> (union >= 12 AND coverage >= 0.5):
> **writing_2 union 50 -> 1, coverage 0.845 -> 0.924 => NOT ELIGIBLE**;
> **xylophone union 0 -> 0, occl 1,031 -> 2,899, coverage 0.577 -> 0.779
> => NOT ELIGIBLE.** Coverage IMPROVED in both cases (corrected
> registration tracks better); the absence/return supply collapsed.
> **TRANCHE 1 THEREFORE CONTAINS ZERO ELIGIBLE CANDIDATES.** The DRY
> outcome below STANDS (fewer than 3 eligible either way) and the
> checkpoint autonomy condition (>= 2 eligible) remains NOT MET — xylophone
> does NOT flip it, so the post-tranche decision remains the user's.
> Independent recomputation agreed exactly. Result:
> [[operations/elgs-substrate-remeasurement-result]].

Started 2026-08-12. Governing frozen protocol:
`configs/elgs/prereg_m1_cycle2_screen_v1.json` REVISION 2 at
`d546400` — fresh-context SIGNED (chain below). Plan:
[[operations/elgs-cycle2-experiment-plan]]; design:
[[operations/elgs-cycle2-continuation-design]]; the cycle-1 negative
this responds to: [[operations/elgs-m1-census-result]].

## Prereg review chain (all pre-data; no candidate downloaded before SIGN)

1. Hostile refine of the continuation design: SOUND WITH REPAIRS
   (blocking {1,2,3,5,6,7,8,9,10,14} + notes) — all applied at
   `552b048`.
2. Sign-off review of prereg revision 1 (`8c2c696`): REJECTED
   (pre-data, repairable) — all ten design repairs embodied with no
   drop or weakening, twelve readings faithful, disclosed arithmetic
   independently verified; four residual freedoms B1–B4 + notes.
3. Revision 2 (`d546400`): B1 (R2′ z>0 participation totalized),
   B2 (nearest-rank angular floor; evaluator switched from linear
   interpolation in the same commit), B3 (exactly-3 selection,
   sub-3 ⇒ DRY), B4 (tranche-2 composition, all-screened selection
   scope, checkpoint 'continue' semantics), N1–N3/N6 folded.
4. Scoped re-review: **SIGNED** — semantic no-other-change verified
   exhaustively (exactly 10 differing leaves, all in the allowed
   repair set; floors/thresholds/pool/queue/readings byte-invariant);
   B1–B4 judged closed with no new ambiguity; B2 code conformance
   spot-checked at `d546400`. EXPLICIT BOUNDARY: the signature does
   not cover code conformance of the R2′ evaluator path
   (`r2prime_holds` B1/N2) — a separate implementation verification
   must complete before screening statistics are treated as
   protocol-valid (running in parallel with acquisition).

## Evaluator alignment

Aligned at `ca9f0cb` (twelve readings: R3 transition rule, R9
transition tally, union predicate primary/R2′/single-cam-diagnostic,
halves attribution under the frozen boundary conventions, angular
floor) + `d546400` (B2 nearest-rank). 16 oracle tests in
`tests/test_elgs_m1_census.py`; full suite green (652, only the 3
pre-existing env failures).

## R2′ implementation verification (the signature's boundary) — CLOSED

- First pass (fresh-context, statement-by-statement, 13-row
  conformance table): **DEVIATES** — two findings, both caught
  BEFORE any screening statistic was computed. D2 (material): the
  anchor search covered only the 4-frame termination-detection
  prefix of the re-appearance run, not the frozen MAXIMAL run —
  candidate-differential on the gate-bearing statistic, and it
  explains part of cycle-1's 23-vs-30 primary-return spread
  (cycle 1 stays FINAL; the concordance diagnostic will surface the
  divergence). D1 (narrow): the in-domain filter tested continuous
  coordinates where R8 binds the round-half-up pixel (half-pixel
  boundary band wrongly discarded). The pairwise-separation
  semantics were adjudicated CONFORMS under the unique monotone
  reading.
- Repairs at `1d8f3b0` (full-run extension with same-S closure +
  return_run_end; rounded-pixel domain test) with pinning tests.
- Re-verification: **CONFORMS — screening statistics computed at
  `1d8f3b0` are protocol-valid.** Empirical pinning: both new tests
  were run against the PRE-repair evaluator via module injection
  and fail at exactly the predicted assertions; both pass at
  `1d8f3b0`; 18/18 module suite. Disclosed enumeration convention
  (recorded, not a deviation): frames inside a candidate's
  terminating re-appearance run are not rescanned as potential
  same-identity window starts — the greedy per-identity scan is
  the signed enumeration. Ten non-blocking test recommendations
  carried in the verification record.

## Acquisition mechanics

Cycle-1 precedent: per-sequence `processed_data` zips from the
official public Dropbox share, per-file links collected via the
browser UI (direct folder-path URL guesses serve the JS app shell —
re-verified 2026-08-12 against the already-acquired unlock.zip, no
cycle-2 candidate touched pre-SIGN), detached Determined CPU
download tasks, sha256 manifests, read-only raw trees, zips
retained.

## Tranche-1 execution log

Acquisition: chess pilot validated the path (25.0 GB, sha
`082ad68a…`, 18 min; the read-only seal on the cycle-1 tree had to
be reopened for the zips dir — chmod, no data touched); remaining
19 in four parallel batch tasks (local det streams cap at 10 min
and exit 255 while the server tasks continue — benign, the
filesystem monitor is the real signal).

### Screening table (screened half; gate-bearing statistic = union returns)

| seq | window | occl | true-abs | primary | union | diag | coverage | ret_undef | eligible (≥12 union, ≥0.5 cov) |
|---|---|---|---|---|---|---|---|---|---|
| chess | 0–352 | 7,896 | 0 | 0 | 0 | 0 | 0.936 | 0 | NO |
| maracas | 0–134 | 16,874 | 1 | 0 | 1 | 1 | 0.917 | 0 | NO |
| tambourine | 0–127 | 11,065 | 18 | 3 | 9 | 16 | 0.815 | 0 | NO |
| pour_tea | 0–225 | 12,101 | 73 | 1 | 3 | 18 | 0.591 | 26 | NO |
| jenga | 0–272 | 9,698 | 0 | 0 | 0 | 0 | 0.896 | 0 | NO |
| pan | 0–114 | 12,818 | 11 | 0 | 0 | 1 | 0.853 | 0 | NO |
| soda | 0–171 | 15,468 | 1 | 0 | 0 | 0 | 0.774 | 0 | NO |
| piano | 0–354 | 3,346 | 0 | 0 | 0 | 0 | 0.999 | 0 | NO |
| tea | 0–164 | 18,944 | 13 | 3 | 3 | 7 | 0.743 | 7 | NO |
| put_candy | 0–233 | 15,889 | 18 | 4 | 4 | 15 | 0.507 | 11 | NO |
| writing_1 | 0–367 | 5,476 | 0 | 0 | 0 | 0 | 0.918 | 0 | NO |
| put_fruit | 0–162 | 9,820 | 4 | 2 | 2 | 3 | 0.887 | 1 | NO |
| **writing_2** | 0–239 | 7,315 | 84 | **49** | **50** | 62 | 0.845 | 8 | **YES** |
| kindle | 0–292 | 12,444 | 0 | 0 | 0 | 0 | 0.779 | 0 | NO |
| xylophone | 0–306 | 1,031 | 0 | 0 | 0 | 0 | 0.577 | 0 | NO |
| keyboard_mouse | 0–176 | 3,675 | 0 | 0 | 0 | 0 | 0.853 | 0 | NO |
| poker | 0–267 | 16,376 | 109 | 4 | 10 | 32 | **0.382** | 44 | NO (coverage) |
| slice_apple | 0–233 | 6,241 | 4 | 2 | 2 | 3 | 0.731 | 1 | NO |
| scissor | 0–561 | 35,254 | 343 | 20 | **75** | 254 | **0.441** | 212 | NO (coverage) |
| music_box | 0–2867 | 16,090 | 0 | 0 | 0 | 0 | 1.000 | 0 | NO |

## FORMAL SELECTION OUTCOME (frozen algorithm, prereg revision 2)

Tranche 1 fully screened (20/20, every row from a sealed
protocol-valid census at `1d8f3b0`+). Step 1 eligibility (union ≥ 12
AND coverage ≥ 0.5): exactly ONE candidate qualifies — writing_2
(union 50, coverage 0.845). Fewer than 3 eligible ⇒ steps 3–4
skipped ⇒ **DRY WITHIN THE FROZEN BUDGET** (frozen B3 rule). The
pooled 72-union threshold is also unreachable from eligible +
near-eligible candidates (50 + 9 + 4 = 63). The checkpoint autonomy
condition (qualifying subset exists OR ≥ 2 eligible) is NOT met —
the post-tranche decision belongs to the USER per the frozen
protocol. No cycle-2 gate was run (no selected subset exists).

Screening spend: ≈ 2–3 GPU-h total (20 tracker runs, mostly ≤ 3 min
each with music_box ≈ 1 h; 20 CPU censuses on hopper slots; exact
wall-times in the experiment ledger and Determined) of the 6 GPU-h
ceiling; ≈ 350 GB acquired zips + extractions (33 TB free); every
per-candidate seal set (tracks + census + manifests + realized
windows) retained per the verifiability policy.

What the tranche established POSITIVELY: the dev pool is massively
rich in the OTHER three gated event classes — ~240k pooled occlusion
opportunities, ~700 true-absence candidates, healthy coverage in
17/20 sequences. G13's occlusion/absence program (BIRTH, FISSION,
TRUNCATE, censored evidence) is data-rich in DiVa-360; ONLY G14's
same-object-return supply (REACTIVATE) is scarce — concentrated in
one well-tracked sequence (writing_2: 84 absences / 50 union returns
in HALF a sequence) and two tracker-illegible ones (scissor union
75 at coverage 0.441; poker at 0.382).

Findings visible before music_box completed:
- R2′ discriminates as designed everywhere: union recovers censored
  returns (scissor 20→75, tambourine 3→9, poker 4→10) while staying
  far below the invalid single-camera diagnostic (scissor 254,
  poker 32).
- Only writing_2 is eligible (union 50, cov 0.845 — writing +
  erasing produces genuine leave-and-return events with a
  well-tracked pen). The two other return-rich candidates fail the
  per-sequence coverage floor for the principled reason it exists:
  scissor (union 75, cov 0.441) and poker (union 10, cov 0.382)
  are event-rich but tracker-illegible.
- With at most 2 eligible possible (music_box pending), the frozen
  exactly-3 rule yields DRY WITHIN THE FROZEN BUDGET regardless of
  music_box; the pooled 72-union threshold is also unreachable from
  eligible candidates (50 + 9 + 4 = 63 hypothetically). The
  checkpoint's autonomy condition (>= 2 eligible) is NOT met — the
  post-tranche decision is the user's by frozen design.

- chess (exp 15 tracks: 512 seeds, 7,001 tracks, reproj median
  0.75 px, consensus 99.9%; exp 16 census): the STRONGEST
  name-prior candidate has ZERO true-absence candidates in its
  screened half — pieces never vanish from all containing cameras
  simultaneously; occlusion supply is huge and coverage excellent,
  so this is event content, not pipeline health. The cycle-1
  lesson (names do not predict event content) validated by the
  first measured row.
