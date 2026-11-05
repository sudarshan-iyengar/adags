# Moving vs view-dependent inside 3D foundation models — overnight lane (2026-10-25)

Evidence boundary: exploratory-to-claim-grade on UNTOUCHED public data (N3V, HouseCat6D, PhoCaL, DAVIS-2016
train/val, DAVIS-2017 val reserved split). N3V same-instant clips re-order published frames (the principal's own
proposed control); N3V is used as a control instrument, never as a headline benchmark. Full append-only log with
observed / explanation / alternative / next test per entry: `idea-stage/vdm-probe-2026-10-25/FINDINGS.md`
(E1–E28 + decisions, failed runs kept). Advisor plan + Codex review: `idea-stage/vdm-probe-2026-10-25/ADVISOR_PLAN_1*.md`.
Code: `idea-stage/vdm-probe-2026-10-25/code/`. Leonardo (scratch, 40-day purge): `/leonardo_scratch/large/userexternal/siyengar/vdm/`.

## Question

Training-free motion detectors built on 3D foundation models (VGGT4D on VGGT, Easi3R on DUSt3R/MonST3R, Dyna3 on
Depth Anything 3) flag content that disagrees across views. Static view-dependent content (glass, reflections)
should then be a false positive, and synchronised N3V cameras (same instant = viewpoint only; one camera = time
only) should separate the two causes inside the model.

## Durable conclusions

1. **The three detectors fail differently** (controls: FROZEN = one image repeated; SAME-INSTANT = N3V cameras at
   one instant; FIXED-CAMERA = one N3V camera over time):
   - VGGT4D measures DISTINCTIVENESS, not disagreement. Its Gram components average q_p·q_k over all window tokens
     (= q_p·mean(q) for the mean terms); its map is reproduced from one repeated image; it flags 69–80 % of hands
     at a single instant and 17.6 % of static glass from a fixed camera. DAVIS-2016 val stage-1 JM: frozen 52.2,
     matched 13-frame real window 53.1, shuffled full sequence 55.5, ordered 58.5 — at matched window length,
     motion is worth < 1 JM of its score.
   - Easi3R is structurally zero without disagreement but pure viewpoint change triggers it: same instant it
     flags 30–60 % of every object (glass not more than opaque); in static hand-held scenes 11–13 % of the
     background. "View disagreement read as motion" HOLDS for Easi3R, as parallax/occlusion around objects.
   - Dyna3's best-match score (re-implemented; MODIFIED backbone DA3-GIANT, paper uses DA3-BASE) passes both
     controls but detects motion weakly (fixed-camera AUC 0.69).
2. **VGGT4D's captured Q/K are post-RoPE.** Comparing keys at different image positions without inverting VGGT's
   2-D RoPE partly measures pixel displacement (Spearman with displacement 0.69 -> 0.38 after inversion on a
   static scene). Any correspondence-level probe of VGGT attention must invert RoPE (exact inverse in
   `code/run_cd_probe.py`, self-tested to 5e-7).
3. **Glass inside VGGT looks like background** in keys (corresponded disagreement), confidence and depth
   consistency (glass vs rest of frame 0.33–0.52): no training-free transparency read-out from VGGT. Pairwise
   models differ: DUSt3R / MonST3R confidence ranks glass vs rest at 0.68 / 0.71 (Depth Anything 3: 0.62).
4. **Corresponded disagreement (CD)** — model-own geometry, RoPE inverted, reference frame duplicated and
   excluded, median over partners — has no objectness prior (objects vs background 0.48–0.54) and does not fire on
   hands at a single instant (0.32–0.44). Band = layers 2–7, chosen on DAVIS-2016 TRAIN.
5. **Certifying VGGT4D's clusters with CD** (per-clip parallax-conditioned robust z): the operating point frozen
   before DAVIS val (monotone, z > 1 in >= 75 % of frames) meets the zero-motion bars (refined static FP glass /
   opaque / metal / bg 1.4 / 0.2 / 0.0 / 0.0 % vs VGGT4D 37.8 / 31.5 / 6.3 / 0.6; N3V same-instant hands 2.3 % vs
   69.4 %) but costs 13.3 JM on DAVIS-2016 val (47.1 vs 60.3; TRAIN cost was 3.9) -> Outcome C. Collapse is on
   large dominant movers and dynamic water (TRAIN diagnosis: separability falls with object area, Spearman -0.65;
   water inverts it). The full val curve shows low-threshold re-threshold rules matching/beating VGGT4D
   (any_2 60.2, mean_0.5 60.7 stage-1 vs 58.5) with 2–7x fewer object false positives; a train-only "Pareto" rule
   (pers_1_0.25) was declared before touching the reserved split. RESERVED SPLIT (10 DAVIS-2017 val sequences,
   read once): VGGT4D refined 41.70 vs certified 37.57 (-4.1; judo +30, shooting +14 / lab-coat -47, india -20):
   the method does not improve DAVIS on untouched data. Status: Outcome C — the durable contribution is the
   diagnosis + controls; certification is a false-positive / accuracy trade-off, not a SOTA method.
6. **Instrument lessons**: (a) a control that changes one thing at a time (frozen / same-instant / fixed-camera)
   separated three mechanisms that DAVIS cannot see; (b) every probe of attention across positions must handle
   positional encoding; (c) per-frame min-max + Otsu guarantees "detections" on static input; (d) a threshold
   rule chosen on TRAIN can fail on val by 3x the train cost — keep an untouched split.

## Open

Whether VGGT's camera follows large movers (the proposed collapse mechanism); a size/ordering-robust certificate;
the 1.8-JM gap between our VGGT4D reproduction (60.34) and the published 62.12; GMOS/MOS-I compatibility;
Trust3R-protocol comparison for the DUSt3R/MonST3R confidence cue; MASt3R and pi3.
