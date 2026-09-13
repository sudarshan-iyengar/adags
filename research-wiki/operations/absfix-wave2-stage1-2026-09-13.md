---
title: "Wave-2 stage 1 (2026-09-13) — prefixes, T1, votes, draws, S2 masks and preconditions on the confirmatory scenes; no gated cell"
date: 2026-09-13
evidence_bearing: false
---

# Wave-2 stage 1 (2026-09-13) — Leonardo

EXPLORATORY, `evidence_bearing: false`. Stage 1 of spec v2.0.0
([[absfix-wave2-spec-v2-2026-09-11]] §13.9(d)) as authorised by the user
on 2026-09-13: the eight ungated 6k prefixes on the `_v3` derived roots,
T1 per prefix, the construction-derived vote per prefix with the
row-weight dump, the sham draws, SAM2 propagation with one bounded
re-seed, the S2 vote per prefix, and both preconditions. **No gated cell
was trained.** Every number is tied to a job id or a file; the machine
record is `research-wiki/assets/absfix-stage1-record.json` (table:
`absfix-stage1-record.md`), the seed boxes `absfix-seed-boxes.json`, and
the spec page §13.9–13.17 carries the append-only decisions.

## 1. What ran, and what went wrong on the way

| step | outcome |
|---|---|
| S2 clicks | 118 placed by the primary from box-free sheets (§13.10); run 1 kept 19/20, 19/20, 19/19; one bounded re-seed on 4 cameras (§13.11); run 2: cam19 fixed on both scenes, sear cam09 → countertop and crb cam17 → label, both kept by rule (§13.14); final masks 20/20, 20/20, 19/19 |
| prefixes | 8/8 COMPLETED at ~2 h each; three sear_steak prefixes hung silently at start-up (empty run dirs for 3 h 45, jobs 57485060/68/69), cancelled and resubmitted (57504087/132/146), then healthy |
| T1 | 8/8; pooled windows [60,89] ×4, [60,90] ×2, [60,93] (flame s3), **[57,106] (sear s2)**; flame s0's T1 hung twice at start-up (224 min and 12 min at zero bytes, both on or after node lrdn0136) before completing on the third submission |
| construction-derived votes | first seeded from each prefix's T1 program on the new scenes; flame s3's vote admitted ZERO rows (T1 window late); re-seeded from authored per-scene boxes (§13.15/13.16), 8/8 COMPLETED: 6,582–6,648 (flame) and 7,434–7,676 (sear) members, LOCO Jaccard ≥ 0.998 |
| sham draws | A and B on 8/8 (count-matched, zero overlap, seeds 1000+s / 2000+s); **L refused on 8/8**: eligible local mass is 0.37–0.41 of the truth mass against a ±10% band (§13.13) |
| S2 votes | flame 4/4: 6,434 / 6,450 / 6,515 / 6,497 members; **sear 0/4: every vote refuses on cam09** ("no DEVA id met the mass bar", the countertop mask kept under 13.9(b)); calibration 4/4 |
| membership precondition | flame 4/4 PASS (precision 0.9998–1.0000, recall 0.9775–0.9800); calibration 4/4 PASS (precision 0.9996–1.0000, recall 0.988–0.990, IoU 0.94); sear: no S2 program, hence no membership precondition |
| T1 precondition (pooled reading, §13.15) | flame s0/s1/s2 PASS, **flame s3 FAIL** (pooled [60,93], onset error 4, IoU 0.882); calibration 4/4 PASS on the pooled reading; sear: computed in T1-only mode after the combined job was cancelled by the expected S2 refusal (see §13.17 of the spec for the values) |

Three instrument lessons, carried as method: (i) a job that shows RUNNING
with a zero-byte log for longer than its siblings' whole runtime is hung,
and the cure is cancel-and-resubmit, not waiting (a warden did this for
four jobs; two of four stalls were on node lrdn0136); (ii) a vote must
not be seeded from an estimator whose failure it is supposed to survive
(the T1-seeded flame s3 vote admitted zero rows; the authored box fixed
it without touching T1); (iii) a combined precondition job inherits
every upstream refusal, so preconditions that do not depend on each
other must be separate jobs.

## 2. What the preconditions decide before any score

* **Claim A** (G against U and the shams) is exercisable on both
  confirmatory scenes: 4 complete G programs per scene, A and B draws
  on every prefix. The L draw is not constructible as frozen; the
  user's disposition is pending (§13.13).
* **Claim B** (the fully estimated arm) is DESIGN_WITHOUT_POWER on
  BOTH confirmatory scenes under §11.5 as frozen: on sear_steak the S2
  instrument is not admitted (the vote refuses on cam09); on flame_steak
  prefix 3 fails the T1 precondition, and Claim B requires all four
  prefixes. The calibration scene admits the instrument on all four
  prefixes. The GESTMEM arm is still trained where its program exists
  (flame 4, calibration 4) and reported descriptively.
* The T1 windows themselves are a result worth keeping: on real footage
  with an authored absence, blind T1 recovers the authored window to
  within one frame on 6 of 8 confirmatory prefixes and fails visibly on
  2 (flame s3 late by 4 at the onset; sear s2 [57,106]).

## 3. Cost

GPU: 8 prefixes (~2.1 h) + 3 hung prefixes (3.75 h each, no output) +
8 T1 (~2 h) + 2 hung T1 (3.9 h) + votes/S2 votes/eligibility (~0.1–0.2 h
each, ~30 jobs) + SAM2 runs (~0.5 h) ≈ 63 A100-h, of which ≈ 19 h were
lost to start-up stalls. CPU: draws, preconditions, census, collector.
