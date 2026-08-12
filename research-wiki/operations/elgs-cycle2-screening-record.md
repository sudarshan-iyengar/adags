# EL-GS Cycle-2 Screening Record (IN PROGRESS)

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

| seq | window | occl | true-abs | returns primary | returns union | single-cam diag | coverage | ret_undef | eligible (≥12 union, ≥0.5 cov) |
|---|---|---|---|---|---|---|---|---|---|
| chess | 0–352 | 7,896 | 0 | 0 | 0 | 0 | 0.936 | 0 | NO |
| maracas | 0–134 | 16,874 | 1 | 0 | 1 | 1 | 0.917 | 0 | NO |
| tambourine | 0–127 | 11,065 | 18 | 3 | 9 | 16 | 0.815 | 0 | NO (union 9 < 12) |
| pour_tea | 0–225 | 12,101 | 73 | 1 | 3 | 18 | 0.591 | 26 | NO |

R2′ real-data behavior (first four rows): the union recovers
censored returns the primary misses (tambourine 3→9, pour_tea 1→3,
maracas 0→1) while staying FAR below the invalid single-camera form
(diagnostics 16–18 where union admits 3–9) — the two-camera
angular-separation requirement discriminates as the sign-off
reviewer intended. pour_tea's profile (73 absences, ret_undef 26,
single-diag 18, union 3) is the censoring regime characterized in
cycle 1, measured live.

- chess (exp 15 tracks: 512 seeds, 7,001 tracks, reproj median
  0.75 px, consensus 99.9%; exp 16 census): the STRONGEST
  name-prior candidate has ZERO true-absence candidates in its
  screened half — pieces never vanish from all containing cameras
  simultaneously; occlusion supply is huge and coverage excellent,
  so this is event content, not pipeline health. The cycle-1
  lesson (names do not predict event content) validated by the
  first measured row.
