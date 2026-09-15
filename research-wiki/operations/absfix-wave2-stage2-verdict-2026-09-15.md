---
title: "Wave-2 stage 2 verdict (2026-09-15): Claim A NOT_MET on both confirmatory scenes under the frozen 0.50 dB all-pairs rule (P2 return met everywhere, P1 ghost core short on 1 of 4 flame pairs and on sham contrasts by 0.08–0.27 dB); Claim B DESIGN_WITHOUT_POWER as pre-decided; calibration scene CLAIM_CONDITIONS_MET on both; STG baseline descriptive; Codex arithmetic audit folded"
date: 2026-09-15
evidence_bearing: true
---

# Wave-2 stage 2 verdict (2026-09-15)

Spec: [[absfix-wave2-spec-v2-2026-09-11]] §11 (binding), amendments
§13.21–13.30. Setup and go: §13.22–13.23. Repairs made before any pair
value existed: §13.24 (U scoring), §13.25 (Slurm state, two timeouts),
§13.26–13.30 (five reducer/collector plumbing corrections). This page
is written from the reducer output of the sixth reduction attempt and
the primary inputs it names; nothing here comes from a table that was
read before the montages were viewed.

## 0. Verdict of record

| claim | flame_steak | sear_steak | verdict | cut_roasted_beef (calibration, reported not claimed) |
|---|---|---|---|---|
| **Claim A** — G beats U and every wrong-membership sham by ≥ 0.50 dB on P1 (ghost core) AND P2 (return) in every one of 4 pairs, mechanism-exercised set | NOT_MET | NOT_MET | **NOT_MET** | CLAIM_CONDITIONS_MET |
| **Claim B** — the fully estimated gate (S2 membership + T1 gap) under the same rule | DESIGN_WITHOUT_POWER (T1 not admitted on prefix 3; 3 of 4 pairs) | DESIGN_WITHOUT_POWER (S2 refused on every prefix; 0 of 4) | **DESIGN_WITHOUT_POWER** | CLAIM_CONDITIONS_MET |

The verdict rule (`VERDICT_RULE`, precedence DWP > MET > PARTIAL >
NOT_MET) and the floor (0.50 dB, fixed, `PAIRED_MIN_EFFECT_DB`) are
the frozen ones; the operative set is `mechanism_exercised`, and every
one of the 104 cells is in it (all 12 prefixes passed the per-cell
precondition once the reducer read the extractor's proof where it is
written, §13.30). No blocking error, no warning.

**What NOT_MET means here, read from the pairs and not from the label.**
On the return window P2 the gate beats U and all three wrong-membership
shams and the timing sham in **8 of 8 confirmatory pairs**, by +1.7 to
+3.2 dB over U. On the ghost core P1 the gate beats U in 8 of 8 pairs
(+0.44 to +2.8 dB) but one flame_steak pair, prefix 1, gains +0.44 dB,
0.06 dB under the floor; and against the wrong-membership shams the
gate's margin on P1 falls under 0.50 dB on six of the 24 confirmatory
sham pairs (flame prefix 1 on all three shams, +0.23 to +0.40; flame
prefix 3 vs A, +0.43; flame prefix 2 vs L, +0.25; sear prefix 2 vs L,
+0.42). Under the all-pairs rule each of those is sufficient on its own.
The shams never beat G: the smallest margin is +0.23 dB (flame prefix 1
vs L). So on real footage the membership-specific share of the ghost-core
benefit is small on flame_steak (the wrong-membership arms remove
roughly as much ghost as U already lacks), large on sear_steak (+0.4 to
+2.7 dB) and largest on the calibration scene (+2.7 to +5.2 dB); the
return benefit is large and membership-specific everywhere.

## 1. Per-pair tables (from `stage2_paired.json`, sha256 `ed5b9c5d…`)

Each cell is the G−(other) difference in pooled window PSNR (dB), with
the two pooled values in brackets; P1 = `roi:core` frames 63–87
(pixel-weighted), P2 = `BOTTLE_absence_gap` frames 92–99. "**<floor**"
marks a pair under 0.50 dB.

#### CLAIM_A, flame_steak (confirmatory): NOT_MET

| contrast | endpoint | prefix 0 | prefix 1 | prefix 2 | prefix 3 | pairs over 0.50 dB |
|---|---|---:|---:|---:|---:|---|
| G-U | P1 | +2.41 (28.86 vs 26.45) | +0.44 (28.01 vs 27.57) **<floor** | +0.60 (28.56 vs 27.96) | +0.58 (27.89 vs 27.30) | 4/4 |
| G-U | P2 | +2.50 (32.68 vs 30.18) | +2.10 (32.53 vs 30.43) | +1.87 (32.42 vs 30.55) | +3.17 (29.66 vs 26.48) | 4/4 all over |
| G-GWRONGMEM_A | P1 | +2.41 (28.86 vs 26.45) | +0.40 (28.01 vs 27.62) **<floor** | +0.57 (28.56 vs 27.98) | +0.43 (27.89 vs 27.46) **<floor** | 4/4 |
| G-GWRONGMEM_A | P2 | +2.42 (32.68 vs 30.26) | +2.08 (32.53 vs 30.45) | +1.80 (32.42 vs 30.62) | +3.14 (29.66 vs 26.52) | 4/4 all over |
| G-GWRONGMEM_B | P1 | +2.45 (28.86 vs 26.42) | +0.35 (28.01 vs 27.67) **<floor** | +0.58 (28.56 vs 27.97) | +0.62 (27.89 vs 27.27) | 4/4 |
| G-GWRONGMEM_B | P2 | +2.43 (32.68 vs 30.25) | +2.03 (32.53 vs 30.49) | +1.75 (32.42 vs 30.67) | +3.12 (29.66 vs 26.54) | 4/4 all over |
| G-GWRONGMEM_L | P1 | +1.91 (28.86 vs 26.95) | +0.23 (28.01 vs 27.78) **<floor** | +0.25 (28.56 vs 28.31) **<floor** | +0.62 (27.89 vs 27.26) | 4/4 |
| G-GWRONGMEM_L | P2 | +2.27 (32.68 vs 30.41) | +2.10 (32.53 vs 30.43) | +1.70 (32.42 vs 30.73) | +2.90 (29.66 vs 26.75) | 4/4 all over |
| G-GMIS | P1 | +3.56 (28.86 vs 25.30) | +2.80 (28.01 vs 25.22) | +2.68 (28.56 vs 25.88) | +1.66 (27.89 vs 26.23) | 4/4 all over |
| G-GMIS | P2 | +2.45 (32.68 vs 30.23) | +3.82 (32.53 vs 28.70) | +3.66 (32.42 vs 28.76) | +6.23 (29.66 vs 23.42) | 4/4 all over |

Reasons recorded by the reducer: G-U on P1 does not exceed 0.50 dB in flame_steak:1; G-GWRONGMEM_A on P1 does not exceed 0.50 dB in flame_steak:1, flame_steak:3; G-GWRONGMEM_B on P1 does not exceed 0.50 dB in flame_steak:1; G-GWRONGMEM_L on P1 does not exceed 0.50 dB in flame_steak:1, flame_steak:2

#### CLAIM_A, sear_steak (confirmatory): NOT_MET

| contrast | endpoint | prefix 0 | prefix 1 | prefix 2 | prefix 3 | pairs over 0.50 dB |
|---|---|---:|---:|---:|---:|---|
| G-U | P1 | +2.82 (30.57 vs 27.75) | +0.92 (29.93 vs 29.01) | +0.95 (29.93 vs 28.98) | +2.39 (31.36 vs 28.96) | 4/4 all over |
| G-U | P2 | +2.46 (32.22 vs 29.76) | +2.47 (33.00 vs 30.53) | +1.69 (32.05 vs 30.36) | +1.88 (31.86 vs 29.98) | 4/4 all over |
| G-GWRONGMEM_A | P1 | +2.65 (30.57 vs 27.91) | +0.87 (29.93 vs 29.06) | +0.94 (29.93 vs 28.99) | +2.08 (31.36 vs 29.27) | 4/4 all over |
| G-GWRONGMEM_A | P2 | +2.44 (32.22 vs 29.77) | +2.43 (33.00 vs 30.57) | +1.66 (32.05 vs 30.39) | +1.88 (31.86 vs 29.98) | 4/4 all over |
| G-GWRONGMEM_B | P1 | +2.46 (30.57 vs 28.11) | +0.74 (29.93 vs 29.19) | +0.95 (29.93 vs 28.98) | +2.28 (31.36 vs 29.07) | 4/4 all over |
| G-GWRONGMEM_B | P2 | +2.43 (32.22 vs 29.78) | +2.40 (33.00 vs 30.60) | +1.72 (32.05 vs 30.33) | +1.99 (31.86 vs 29.87) | 4/4 all over |
| G-GWRONGMEM_L | P1 | +2.42 (30.57 vs 28.15) | +0.76 (29.93 vs 29.17) | +0.42 (29.93 vs 29.51) **<floor** | +1.86 (31.36 vs 29.50) | 4/4 |
| G-GWRONGMEM_L | P2 | +2.33 (32.22 vs 29.89) | +2.39 (33.00 vs 30.61) | +1.50 (32.05 vs 30.55) | +1.84 (31.86 vs 30.02) | 4/4 all over |
| G-GMIS | P1 | +5.45 (30.57 vs 25.12) | +2.83 (29.93 vs 27.10) | +3.01 (29.93 vs 26.92) | +4.66 (31.36 vs 26.70) | 4/4 all over |
| G-GMIS | P2 | +2.71 (32.22 vs 29.51) | +3.39 (33.00 vs 29.62) | +2.63 (32.05 vs 29.42) | +2.73 (31.86 vs 29.14) | 4/4 all over |

Reasons recorded by the reducer: G-GWRONGMEM_L on P1 does not exceed 0.50 dB in sear_steak:2

#### CLAIM_A, cut_roasted_beef (calibration): CLAIM_CONDITIONS_MET

| contrast | endpoint | prefix 0 | prefix 1 | prefix 2 | prefix 3 | pairs over 0.50 dB |
|---|---|---:|---:|---:|---:|---|
| G-U | P1 | +3.16 (34.24 vs 31.08) | +4.92 (36.28 vs 31.36) | +4.24 (35.02 vs 30.77) | +4.07 (34.98 vs 30.91) | 4/4 all over |
| G-U | P2 | +3.24 (32.34 vs 29.11) | +3.04 (32.10 vs 29.06) | +2.61 (32.61 vs 30.00) | +2.68 (32.58 vs 29.90) | 4/4 all over |
| G-GWRONGMEM_A | P1 | +3.17 (34.24 vs 31.07) | +4.89 (36.28 vs 31.39) | +4.35 (35.02 vs 30.67) | +4.02 (34.98 vs 30.96) | 4/4 all over |
| G-GWRONGMEM_A | P2 | +3.24 (32.34 vs 29.10) | +3.05 (32.10 vs 29.05) | +2.52 (32.61 vs 30.09) | +2.61 (32.58 vs 29.97) | 4/4 all over |
| G-GWRONGMEM_B | P1 | +3.00 (34.24 vs 31.24) | +4.78 (36.28 vs 31.50) | +3.90 (35.02 vs 31.12) | +4.08 (34.98 vs 30.90) | 4/4 all over |
| G-GWRONGMEM_B | P2 | +3.25 (32.34 vs 29.09) | +3.04 (32.10 vs 29.05) | +2.64 (32.61 vs 29.97) | +2.60 (32.58 vs 29.98) | 4/4 all over |
| G-GWRONGMEM_L | P1 | +2.71 (34.24 vs 31.53) | +5.19 (36.28 vs 31.09) | +4.12 (35.02 vs 30.90) | +3.83 (34.98 vs 31.15) | 4/4 all over |
| G-GWRONGMEM_L | P2 | +2.97 (32.34 vs 29.37) | +2.92 (32.10 vs 29.18) | +2.43 (32.61 vs 30.18) | +2.65 (32.58 vs 29.93) | 4/4 all over |
| G-GMIS | P1 | +7.53 (34.24 vs 26.71) | +9.16 (36.28 vs 27.12) | +10.22 (35.02 vs 24.80) | +8.86 (34.98 vs 26.13) | 4/4 all over |
| G-GMIS | P2 | +4.98 (32.34 vs 27.36) | +5.14 (32.10 vs 26.96) | +4.34 (32.61 vs 28.27) | +3.92 (32.58 vs 28.67) | 4/4 all over |

#### CLAIM_B, flame_steak (confirmatory): DESIGN_WITHOUT_POWER

| contrast | endpoint | prefix 0 | prefix 1 | prefix 2 | prefix 3 | pairs over 0.50 dB |
|---|---|---:|---:|---:|---:|---|
| GESTMEM-U | P1 | +2.52 (28.97 vs 26.45) | +0.46 (28.04 vs 27.57) **<floor** | +0.56 (28.52 vs 27.96) | — | 3/4 |
| GESTMEM-U | P2 | +2.37 (32.55 vs 30.18) | +2.14 (32.57 vs 30.43) | +1.93 (32.49 vs 30.55) | — | 3/4 all over |
| GESTMEM-GWRONGMEM_A | P1 | +2.52 (28.97 vs 26.45) | +0.42 (28.04 vs 27.62) **<floor** | +0.53 (28.52 vs 27.98) | — | 3/4 |
| GESTMEM-GWRONGMEM_A | P2 | +2.29 (32.55 vs 30.26) | +2.13 (32.57 vs 30.45) | +1.86 (32.49 vs 30.62) | — | 3/4 all over |
| GESTMEM-GWRONGMEM_L | P1 | +2.01 (28.97 vs 26.95) | +0.26 (28.04 vs 27.78) **<floor** | +0.21 (28.52 vs 28.31) **<floor** | — | 3/4 |
| GESTMEM-GWRONGMEM_L | P2 | +2.14 (32.55 vs 30.41) | +2.14 (32.57 vs 30.43) | +1.76 (32.49 vs 30.73) | — | 3/4 all over |

Reasons recorded by the reducer: visibility-gap instrument not admitted on flame_steak prefix 3: offset error 4 frames > 3; flame_steak: 3 prefixes admitted of the 4 required; GESTMEM-U on P1: 3 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-U on P2: 3 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-GWRONGMEM_A on P1: 3 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-GWRONGMEM_A on P2: 3 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-GWRONGMEM_L on P1: 3 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-GWRONGMEM_L on P2: 3 complete pairs of the 4 required (mechanism-exercised set)

#### CLAIM_B, sear_steak (confirmatory): DESIGN_WITHOUT_POWER

| contrast | endpoint | prefix 0 | prefix 1 | prefix 2 | prefix 3 | pairs over 0.50 dB |
|---|---|---:|---:|---:|---:|---|
| GESTMEM-U | P1 | — | — | — | — | 0/4 |
| GESTMEM-U | P2 | — | — | — | — | 0/4 |
| GESTMEM-GWRONGMEM_A | P1 | — | — | — | — | 0/4 |
| GESTMEM-GWRONGMEM_A | P2 | — | — | — | — | 0/4 |
| GESTMEM-GWRONGMEM_L | P1 | — | — | — | — | 0/4 |
| GESTMEM-GWRONGMEM_L | P2 | — | — | — | — | 0/4 |

Reasons recorded by the reducer: membership instrument not admitted on sear_steak prefix 0: membership precondition: the manifest names no sidecar; membership instrument not admitted on sear_steak prefix 1: membership precondition: the manifest names no sidecar; membership instrument not admitted on sear_steak prefix 2: membership precondition: the manifest names no sidecar; membership instrument not admitted on sear_steak prefix 3: membership precondition: the manifest names no sidecar; sear_steak: 0 prefixes admitted of the 4 required; GESTMEM-U on P1: 0 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-U on P2: 0 complete pairs of the 4 required (mechanism-exercised set); GESTMEM-GWRONGMEM_A on P1: 0 complete pairs of the 4 required (mechanism-exercised set)

#### CLAIM_B, cut_roasted_beef (calibration): CLAIM_CONDITIONS_MET

| contrast | endpoint | prefix 0 | prefix 1 | prefix 2 | prefix 3 | pairs over 0.50 dB |
|---|---|---:|---:|---:|---:|---|
| GESTMEM-U | P1 | +3.04 (34.13 vs 31.08) | +4.88 (36.24 vs 31.36) | +4.01 (34.79 vs 30.77) | +3.99 (34.91 vs 30.91) | 4/4 all over |
| GESTMEM-U | P2 | +3.18 (32.29 vs 29.11) | +3.01 (32.07 vs 29.06) | +2.70 (32.70 vs 30.00) | +2.56 (32.46 vs 29.90) | 4/4 all over |
| GESTMEM-GWRONGMEM_A | P1 | +3.05 (34.13 vs 31.07) | +4.85 (36.24 vs 31.39) | +4.12 (34.79 vs 30.67) | +3.94 (34.91 vs 30.96) | 4/4 all over |
| GESTMEM-GWRONGMEM_A | P2 | +3.19 (32.29 vs 29.10) | +3.02 (32.07 vs 29.05) | +2.61 (32.70 vs 30.09) | +2.48 (32.46 vs 29.97) | 4/4 all over |
| GESTMEM-GWRONGMEM_L | P1 | +2.59 (34.13 vs 31.53) | +5.15 (36.24 vs 31.09) | +3.89 (34.79 vs 30.90) | +3.75 (34.91 vs 31.15) | 4/4 all over |
| GESTMEM-GWRONGMEM_L | P2 | +2.92 (32.29 vs 29.37) | +2.89 (32.07 vs 29.18) | +2.52 (32.70 vs 30.18) | +2.53 (32.46 vs 29.93) | 4/4 all over |


## 2. Descriptive contrasts and floors (no claim)

* **Timing shams cost the most.** G−GMIS on P1 is +1.7 to +3.6 dB on
  flame and +2.8 to +5.4 dB on sear; G−GONES +1.7 to +3.8 dB on flame
  and +2.9 to +5.5 dB on sear; the within-prefix |U−GONES| floor the
  reducer reports is 2.31 dB median on P1 and 1.32 dB on P2 (12
  prefixes). GONES is below U on P1 in 12 of 12 prefixes and on P2 in
  11 of 12 (flame prefix 0 is the exception, +0.08 dB): a gate placed
  in the wrong window is worse than no gate, as in wave 1.
* **Control window C1 (frames 230–259, object present):** G−U is
  −0.14 to +0.03 dB on flame and −0.12 to +0.05 dB on sear (median
  −0.02 over the eight); the wave-1 cost of −0.16..−0.28 dB is not reproduced at that
  size on the confirmatory scenes; on the calibration scene it is
  −0.29 to −0.15 dB, as in wave 1.
* **Pre-gap H1 (frames 30–57):** G−U +0.6 to +1.4 dB on flame, +0.6
  to +1.1 dB on sear, +0.9 to +1.3 dB on cut_roasted_beef — the gated
  rows learn the pre-gap object better, as in wave 1.
* **Whole frame H2:** G−U within ±0.06 dB everywhere.
* **Prefix 3 of flame_steak is a weaker prefix for every arm** (whole
  frame 29.9 dB against 32.0–33.1 on the other three); its pairs are
  paired within the prefix and are not outliers in the differences.
* **GEST equals G to within 0.21 dB on P1 and P2 on every admitted
  prefix**; on the
  two prefixes where T1's pooled interval fails the bound (flame 3,
  offset error 4 frames; sear 2) GEST's return window collapses
  (P2 23.6 and 22.6 dB), the expected signature of a late program.
* **Paired sizing (reducer, δ = 0.30 dB, 12 pairs):** sd of the paired
  G−U difference 1.59 dB on P1 (n₂ = 269 pairs) and 0.52 dB on P2
  (n₂ = 30); feasibility stop on P1.

### 2.1 External baseline X (SpacetimeGaussians, descriptive only, outside the reducer)

STG pinned at 427abfc5, full published schedule (30k iterations, batch
2, scored at 25k), six 50-frame segment models per (scene, seed), cam00
held out, on the same `_v3` derived scenes, profiled with the same
masks and ROIs and pooled with the reducer's own functions
(`stg_desc.py`, jobs 57769509–57769763, four `f_box_profile.json`).

| scene, seed | P1 core 63–87 | P2 return 92–99 | H1 pre 30–57 | C1 control 230–259 |
|---|---:|---:|---:|---:|
| flame_steak s0 | 35.17 | 30.59 | 31.93 | 33.73 |
| flame_steak s1 | 35.35 | 30.96 | 32.88 | 33.92 |
| sear_steak s0 | 34.71 | 30.47 | 32.34 | 33.02 |
| sear_steak s1 | 35.70 | 30.58 | 32.56 | 33.48 |
| our U (range over 4 prefixes) | 26.4–29.0 | 26.5–30.6 | 29.6–32.5 | 29.2–33.2 |
| our G (range over 4 prefixes) | 27.9–31.4 | 29.7–33.0 | 30.2–33.6 | 29.2–33.2 |

Read plainly: **STG, which carries a per-Gaussian temporal opacity and
no gate, renders the ghost core 3.4–7.5 dB better than our gated arm** on
both confirmatory scenes. On the seven prefixes other than the weak
flame prefix 3 its C1 is only +0.4 to +1.2 dB above ours, so this is
not a whole-frame quality gap (against flame prefix 3 every STG margin
widens by the 3–4 dB that prefix is weaker overall: C1 +4.6 to +4.8);
on the return window, on those seven prefixes, STG sits between our U
and G (−0.1 to +0.8 dB against U, 1.3–2.5 dB below G), and against
flame prefix 3 it is +4.1 to +4.5 dB above U and +0.9 to +1.3 dB above
G.
Confounds, stated: STG trains six independent 50-frame models per scene
at 25k iterations against our single 300-frame continuation at 12k; it
is a different representation, schedule and capacity; no pairing and no
floor apply. This is the strongest descriptive fact of the stage and
it bears directly on the paper thesis: the exact-zero presence the gate
is built for is reachable by a temporal-opacity primitive on this
fixture without any membership or window inference, at least in the
gap core; on the return window STG is below our gated arm on seven of
eight prefixes and above it only on the weak flame prefix 3.

## 3. Preconditions, admissions and the mechanism-exercised set

* Scene admission: all three admitted. Membership instrument (S2):
  pass on flame 0–3 and cut_roasted_beef 0–3 (precision 1.00, recall
  0.978–0.990, cam15 mask IoU 0.93–0.94 at f50/f95); no record on
  sear_steak (S2 vote refused on cam09, §13.16) → Claim B sear DWP.
  T1 instrument: pass on 10 of 12 prefixes; FAIL on flame 3 (pooled
  interval [60,93], offset error 4 > 3) and sear 2 → Claim B flame DWP
  (3 of 4 pairs); GEST legs of those prefixes read accordingly.
* Sham programs: A/B count-matched, overlap 0, on 12/12; L
  1,100 rows, overlap 0, mass ratio 0.936–1.000 on 12/12
  (`WRONGMEM_L_RULE`, §13.29).
* Per-cell precondition (§11.4): every gated cell passes; gated rows at
  12k 6,3xx–8,3xx on the object-sized arms and as low as 1,036 on the
  1,100-row L arm (the ≥ 1,000 clause held with a margin of 36 rows on
  flame prefix 3), in-box counts as recounted; A/B in-box
  52–89 with the in-box clause exempt (decision 2); reserved units
  1,425 of 5,700 on cut_roasted_beef and 1,500 of 6,000 on the
  confirmatory scenes, equal within every prefix; `program_family_match`
  proven on every gated cell.
* Path cross-check: `pass = true`, `max_abs_diff = 0` on 3,254 leaves
  on 76 of 76 gated cells; the 8 U cells scored by `--val` (§13.24,
  single path).

## 4. What ran (ids in `agent-control/realdata/jobs/ledger.txt`)

* Cells: 84 jobs 57769768–57770106 at frozen commit F `dc7b301e…`;
  74 gated cells complete (Slurm FAILED by the template's last line
  under `set -e`, §13.25), 8 U cells re-scored by `uscore` jobs
  57781496–57781506 (§13.24), 2 timeouts scored by `gscore` jobs
  57828687 / 57828689 (§13.25). Warden pid 2385710: no hang, no
  resubmission.
* STG: 63 jobs 57769509–57769763, all COMPLETED.
* Reduction: six attempts; attempts 1–5 admitted no pair (§13.26–13.30);
  attempt 6 at commit `7702164a…`: manifest `943d0da5…` (104 cells, 104
  complete; calibration GMIS/GONES read from `wave1_cells/<tag>/`),
  264 reducer inputs hashed (`09779619…`), output `stage2_paired.json`
  `ed5b9c5d…`, spec JSON `e58a9281…`, reducer `61df7cf7…`. Montages
  (12) rendered after the attempt-3 output was hashed and viewed before
  any table was read; QC found no instrument defect
  (`research-wiki/assets/absfix-stage2-<scene>-p<S>-montage.jpg`).
* Assets committed with this page: `absfix-stage2-paired.json`,
  `absfix-stage2-report.md` (the reducer's own markdown),
  `absfix-stage2-manifest.json`, `absfix-stage2-reducer-inputs.sha256`,
  the 12 montages.

## 5. Evidence boundary

Real-footage scenes with an authored (SA4D-edited) absence of one
object on 20–21 cameras, four 6k prefixes per scene, one seed per
prefix, 12k continuation; construction-derived membership for Claim A;
the U arm scored through `main.py --val` and every gated arm through
`eval_n3v_gated.py --restore_state`, the two paths proved identical on
76 gated cells. The floor is a fixed 0.50 dB, not a measured replicate
floor (the within-prefix |U−GONES| floor is 2.3 dB on P1). The claim
that fails is the all-pairs conjunction; no pair has G below U or below
a sham on either endpoint. Nothing here speaks to physical absence, to
scenes with real exits, or to estimated membership on the confirmatory
scenes (Claim B is DWP, not tested).

## 6. Corrections to earlier records (append-only)

* §13.21's expectation "the wave-1 control-window cost of −0.16..−0.28
  dB" reproduces on the calibration scene only; on the confirmatory
  scenes the C1 difference is within ±0.14 dB.
* The §13.22 statement that the U arm would be scored by the evaluator
  in fresh mode was wrong (§13.24); U is scored by `--val`.
* The reducer, as frozen at F, could not have reduced this stage
  (§13.26–13.30). Each correction is a plumbing fix tested against the
  shipped spec; none touched a threshold, a window, a crop or a program.

## 7. Not decided here

Whether the paper thesis survives NOT_MET under this rule, how the STG
comparison enters it, and whether a second seed or a relaxed conjunction
is worth its cost are the user's decisions; the declared intentions of
§13.21 (GESTMEM-S3, product gate, free boundaries, graded membership)
remain declared, none started.

## 8. Audit

A fresh-context Codex pass (gpt-5.6-sol, reasoning high) checked every
number in sections 0, 2, 2.1 and 3 against the endpoint table of the
reducer output and returned ten arithmetic corrections, all applied
above: six (not five) sham pairs under the floor; sear membership share
+0.4 to +2.7; G−GONES stated with its own ranges; GONES vs U stated per
prefix (12 of 12 on P1, 11 of 12 on P2); the C1 median −0.02; the flame
H2 range 32.0–33.1; GEST vs G within 0.21 dB; and the three STG
comparisons that had silently excluded flame prefix 3 now state the
seven-prefix range and that prefix separately. Membership, T1 and
operational figures come from stage-1 records and job logs and were not
checkable by that pass.
