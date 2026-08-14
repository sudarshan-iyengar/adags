# EL-GS Exhaustive DiVa-360 Screening — Exact Scope and Cost (proposal; NOT authorized)

Status: **PROPOSAL AWAITING USER APPROVAL.** Nothing in this page is
authorized, and no acquisition, conversion, tracking, or census for any
unscreened candidate has been performed. Authority chain: user directive
2026-08-14 (measurement closure first; exhaustive screening after) ->
this scope -> a user approval -> a separate preregistration.

The user is committed to completing the remaining DiVa-360 candidate pool
even where the expected yield is low. This page does not argue against
that. It costs it exactly and states which parts depend on the
measurement-closure outcome.

## 1. Frozen pool state (recovered from the frozen prereg, not from prose)

`configs/elgs/prereg_m1_cycle2_screen_v1.json` revision 2 (SIGNED) fixes
the pool: 41 short + 8 long candidates after excluding the five cycle-1
assignments (battery, flip_book, unlock, peel_apple, pour_salt).
Tranche 1 (20) is fully screened. **21 short + 8 long remain.**

**TRANCHE 2 — the frozen `tranche_2_rule` is "the next 20 unscreened
candidates in `candidates_short` list order, then `candidates_long` list
order". Applied mechanically, tranche 2 is exactly, in order:**

1. blue_car   2. bunny   3. clock   4. dog   5. drum   6. horse
7. hour_glass   8. k1_double_punch   9. k1_hand_stand   10. k1_push_up
11. penguin   12. plasma_ball   13. plasma_ball_clip   14. red_car
15. stirling   16. tornado   17. trex   18. truck   19. wall_e   20. wolf

**COMPLETION BATCH — everything after tranche 2 (9 candidates):**
world_globe (short), then the 8 long: chess_long, crochet, jenga_long,
legos, origami, painting, puzzle, rubiks_cube.

NOTE: the candidate list circulated informally (k1_double_punch,
k1_hand_stand, k1_push_up, blue_car, bunny, dog, horse, penguin, trex,
wolf, wall_e) is a correct but PARTIAL subset. It omits clock, drum,
hour_glass, plasma_ball, plasma_ball_clip, red_car, stirling, tornado,
truck and world_globe. The list above is the frozen one.

**Halt rule (frozen, unchanged):** a tranche is screened IN FULL
regardless of interim qualification; selection happens only over a fully
screened tranche; after any continuation, selection operates over ALL
screened candidates (union of tranches). Queue order therefore affects
only pool composition, never selection within the pool. **Candidates are
never reordered, never dropped for low prognosis, and screening never
stops early on a favourable result.**

**Recommendation on batching:** the completion batch should be a
SEPARATELY PREREGISTERED batch, for two reasons that are cost and
governance facts, not prognosis: (i) the 8 long sequences breach the
frozen `extracted_storage_gb_max` of 400 GB on their own (§4) and
therefore require an explicit ceiling decision; (ii) their duration-scaled
tracker cost is an order of magnitude above the short candidates and uses
the disclosed chunked path. Splitting is a budget boundary, not a
selection device — the full pool is still screened.

## 2. The decisive cost fact: what an instrument correction actually requires

Owner-verified from the sealed artifacts: **the tracks artifact stores
per-report `v`** (`{frame, time, is_miss, v, x, y}`) plus per-frame
consensus points with `n_cam` and `reproj_rms`, and per-seed/per-camera
`fb_rms_px` and `reproj_rms_px` diagnostics.

Consequently:

- **Any change to the visibility threshold, the association rule, the
  absence or return predicate, the applicable camera set, or the
  component-eligibility floor is a CENSUS-LEVEL (CPU) change and needs NO
  new tracking.** Re-evaluating all 20 tranche-1 sequences under such a
  corrected instrument costs about **1 CPU-hour and 0 GPU-hours**.
- **Re-tracking is required only if the QUERY CONSTRUCTION changes** — for
  example the `NEVER_QUERIED` fix (querying every camera whose frustum
  contains the seed rather than only cameras where the seed is
  mask-positive at the query frame), or a change to seeding
  (`hull_resolution`, `max_seeds`), the tracker backend, or the weights.
  That is the expensive branch.
- **Conversions are reusable in every branch.** All 27 sealed conversions
  are substrate-correct (24 originally clean, 3 superseded by
  `_fix79ae5b7`); owner-decoded dimensions match declared calibration in
  every in-scope case.

## 3. Scope, by measurement-closure outcome

| Outcome | Tranche-1 re-evaluation | Tracks reusable | New tracking | Instrument version for the screen |
|---|---|---|---|---|
| Status 1 (adequate) | none | yes | no | unchanged, `1d8f3b0`+ |
| Status 2 (defect), census-level correction | **yes — 20 census reruns, CPU only** | yes | no | corrected, new prereg |
| Status 2 (defect), query-construction correction | **yes — 20 re-tracks + 20 censuses** | no | yes | corrected, new prereg |
| Status 3 (partially confounded) | optional; conservative sub-statistic recomputed CPU-only | yes | no | unchanged + a disclosed conservative reading |
| Status 4/5 (not identifiable / unresolved) | none | yes | no | unchanged, with a disclosed non-identifiability boundary; screening measures candidate-opportunity upper bounds ONLY |

In every branch the screen itself is the same mechanical work; only the
reduction changes.

## 4. Costed scope (measured-anchored; every figure states its basis)

**Measured basis.** Tranche 1: 20 sequences, 15,272 full frames,
half-window tracking, 26 tracking cameras, 512 seeds — **~2-3 GPU-h actual**
(the frozen formula `0.06 + 0.10*(n/1000)` halved predicts only 1.36, so
the formula UNDERESTIMATES by roughly 2x and the measured figure is used
below). Censuses are CPU: ~134-146 s typical, 1,268 s for music_box.
Acquired zips: 25 sequences, 291 GB, median **16.86 GB per 1,000 full
frames** (range 9.25-35.44; long sequences compress better — music_box
10.57).

| Work item | Conversions | Tracker runs | Census reductions | Downloads | Persistent storage | GPU-h | CPU-h |
|---|---:|---:|---:|---|---|---:|---:|
| **A. Tranche-1 re-evaluation, census-level** | 0 (reuse) | 0 | 20 | 0 | ~20 MB | **0** | **~1** |
| **A'. Tranche-1 re-evaluation, re-tracking** | 0 (reuse) | 20 | 20 | 0 | ~+150 GB tracks | **3-4** | ~1 |
| **B. Tranche 2 (20 short)** | 20 | 20 | 20 | ~250 GB (est.) | ~250 GB zips + ~250 GB extracted | **2-3** | ~1 |
| **C. Completion: world_globe** | 1 | 1 | 1 | ~10 GB (est.) | ~20 GB | **~0.15** | ~0.05 |
| **D. Completion: 8 LONG** | 8 | 8 | 8 | **~3-4 TB (derived; see D-note)** | **~6-8 TB** | **~50-70** | ~5 |
| **Total, screen only (B+C+D)** | 29 | 29 | 29 | **~3.3-4.3 TB** | **~6.5-8.5 TB** | **~53-73** | **~6** |
| **Total incl. re-evaluation (worst branch A'+B+C+D)** | 29 | 49 | 49 | same | + ~150 GB | **~56-77** | **~7** |

**D-note — the long batch dominates the entire scope and its size is
UNVERIFIED.** No primary source publishes per-sequence frame counts: the
official GitHub page, the project page and the arXiv abstract were all
checked and none carries a table. The figures above are DERIVED from the
paper's "17.4 M image frames" across 54 sequences, netting out the 23
sequences whose frame counts are measured (17,600 full frames total, mean
765, median 480, max 5,736 for music_box) and the 23 unscreened short
candidates at that measured mean. The residual implies **roughly
37,000-49,000 frames per long sequence** — 6-8x music_box, which is
already the longest short-list candidate — giving ~6-9 GPU-h and
~390-515 GB EACH.

This is an ORDER-OF-MAGNITUDE BOUND, not a figure: "17.4M image frames"
is not defined per-camera in any primary source and may count raw
and/or all 53 cameras rather than the ~41 in the shipped splits. An
earlier estimate in this page put item D at 8-20 GPU-h and 0.6-1.3 TB;
that estimate assumed 3,000-12,000 frames per long sequence and is now
superseded as too low.

**Governance consequence — acquire ONE long sequence first as a cost
pilot.** Tranche 1 used exactly this pattern (chess was the acquisition
pilot that validated the download path before the other 19 ran in
parallel). Screening a single long candidate in frozen queue order
(`chess_long`) measures the real frame count, archive size, chunked-tracker
wall time and extracted footprint, and converts every figure in row D from
derived to measured — for roughly 1/8th of the risk. **The remaining seven
should not be committed until that pilot returns.**

Estimates for the 29 unscreened candidates assume 300-900 full frames for
short candidates and 3,000-12,000 for long ones. **UNVERIFIED**: the
official DiVa-360 GitHub page publishes no per-sequence frame ranges, and
none of the 29 has been downloaded, so exact frame counts and archive
sizes are unknown until acquisition. The ranges above are derived from the
25 measured sequences.

**Storage governance flag.** The frozen `extracted_storage_gb_max` is
**400 GB**. Item D breaches it by roughly **15-20x** on the derived
estimate. Apollo currently has **31.1 TiB free** and DiVa-360 occupies
594 GiB, so the physical headroom exists even at 8 TB, but the ceiling is
a frozen protocol constant and **raising it is a user decision that must be
recorded before item D begins**, not an implementation detail. Items A-C
fit inside 400 GB with deletion-after-seal.

**GPU-hour governance flag.** The frozen `screening_gpu_hours_max` is
**6 GPU-h**. B+C fits (~2-3). **A' and D do not** — D alone is roughly
**10x the entire frozen screening ceiling**. Any branch including the long
batch or a re-tracking re-evaluation requires an explicit ceiling decision.
Note the frozen per-candidate cost formula (`0.06 + 0.10*(n/1000)`, halved)
underestimates measured tranche-1 spend by about 2x and should be replaced
with a measured coefficient in any new preregistration rather than carried
forward.

## 5. Hardware, concurrency, provenance

- **Hardware policy (unchanged):** all census cells on `hopper`; pool
  switches are a new ledger entry, never silent. Cluster measured
  2026-08-14: hopper 3x H100 (2 free), dgx 8x V100 (all free).
- **Maximum useful concurrency:** 11 slots exist. Screening is
  embarrassingly parallel per candidate, but taking every slot starves the
  cluster for other users. **Proposed: 6 concurrent**, which puts tranche 2
  at roughly half a day wall including downloads. Census reductions are
  CPU-bound and can run on either pool.
- **Provenance and verification (unchanged, frozen):** every
  evidence-bearing cell goes through `submit_apollo` with an exact PUSHED
  commit, isolated git-archive context materialization, the digest-pinned
  image `sha256:a2877f26...`, content-hashed configs, O_EXCL claim,
  O_APPEND ledger line, and a terminal-state + artifact-inventory audit.
  Scheduler completion is never scientific completion. Monitoring now uses
  the tracked, tested `scripts/det_monitor.py` (commit `dfb0245`).
- **Seal set retained per candidate (unchanged):** half-window tracks
  artifact + controls, screening census JSON, all manifests, realized-window
  record. Only extracted frame images may be deleted after seal
  verification; raw zips always retained; every deletion ledgered.
- **Independent recomputation plan:** as for every prior cycle — a
  fresh-context worker reduces the gate-bearing statistics from PRIMARY
  inputs using its own reduction written from the frozen text alone,
  never reading the evaluator, its tests, its outputs, or any result page
  until its own results are sealed. For a 29-candidate screen the
  recomputation binds at the SELECTION step (the ranking) and at any gate,
  not at every per-candidate row.
- **Resource governance:** projected GPU-h recorded in every manifest;
  running total checked against the ceiling before each submission;
  exceeding a ceiling requires a user decision and is never silent.
  Estimated-vs-actual GPU-h, CPU-h, storage added, network transferred,
  hardware pool, failed/superseded work, and evidence produced per job are
  reported per tranche.

## 6. Expected zero-yield fraction — a DISCLOSED PROGNOSIS, not a target

Tranche 1 returned **0 of 20 eligible** under the frozen predicate
(>= 12 union returns AND >= 0.5 coverage). One-sided 95% Clopper-Pearson
upper bound on the per-sequence eligibility rate: **0.139**. Point
prediction for the remaining 29 candidates: **0 eligible**; 95% upper
bound: **<= 4**.

The informative decomposition, however, is that **returns are not absent —
eligibility is.** 10 of 20 tranche-1 sequences produced at least one union
return. Exactly one produced >= 12 (scissor, union 75) and it failed the
**coverage** floor at 0.441. The binding constraint is the JOINT predicate,
and its coverage limb is the same quantity the measurement diagnostic
suspects is mechanically coupled to the absence statistic through the
shared `v >= 0.5` threshold. **If the diagnostic returns Status 2 or 3,
then "event-rich but tracker-illegible" may be an instrument artifact
rather than a scene property, and scissor and poker are the sequences most
affected.** That possibility is a reason to complete the screen with a
trustworthy instrument, not a reason to expect a pass.

A large zero-yield fraction is an expected and scientifically useful
outcome. Zero-yield sequences are results, not failed jobs. Name-based
priors have already been falsified twice in this program (chess, the
strongest name-prior candidate, returned zero true-absence candidates;
writing_1 and writing_2 are the same activity type with opposite apparent
outcomes, and writing_2's apparent outcome was a substrate defect) and are
not a valid selection instrument.

## 7. What this page does NOT authorize

No acquisition, no conversion, no tracking, no census, no ceiling change,
and no new preregistration. The post-tranche-1 decision remains the user's
under the frozen cycle-2 checkpoint, whose autonomy condition (a
qualifying subset exists OR >= 2 candidates cleared eligibility) is **NOT
met** — corrected, zero candidates cleared.
