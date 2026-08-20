# CCR ladder segment selection (2026-08-20)

Frozen BEFORE any ladder cell output existed (worker completed 01:45Z;
the first ladder training cell, experiment 196, was submitted 02:00Z and
had produced no output at freeze time). Selection used cam00 IMAGES ONLY
-- no model, no render; cam00 remains sealed for every metric. The frozen
screening window is FRAMES 0-49 (the STG-protocol-matched default), which
the inspection confirms contains qualifying hand/knife occlusion-and-
reveal events. The event-ray mask spec below is the predefined mask set
the gates reference.

---
# Lane D frozen screening-segment selection â€” cut_roasted_beef (N3V)

Date: 2026-08-20. Selection performed BEFORE any model output exists.
Evidence: cam00 PNG frames only (1352x1014, `images/cam00_%04d.png` on
`apollo:/apollo/users/sri/proj_adags/data/n3v/cut_roasted_beef/images/`),
55 frames pulled read-only via rclone (stride 10 over 0-290, then stride 5
around 0-49 / 125-175 / 195-225, then stride 2-5 inside 0-49; ~60 MB total).
**This selection uses cam00 IMAGES only â€” no model, no reconstruction, no
other camera was pulled.** All bboxes below are approximate pixel rects
`[x0,y0,x1,y1]` in the native 1352x1014 cam00 raster.

## Scene content (cam00 view)

Chef at a counter, slicing a roasted beef piece on a wooden board with a
white cutting mat, bottom-center of frame. Active manipulation region:
board `[610,840,890,1014]`; beef pile `[655,880,790,965]`; a static seared
piece sits on the mat at `[785,935,865,995]`. Frames 0-130: continuous
knife slicing (right hand holds knife + tongs; left hand steadies the
meat). ~130-160: knife set down on board, then picked up again; ~150-250:
scraping/arranging slices with knife (left hand) + tongs (right hand);
~250-260: both tools withdrawn, pile fully revealed; 260-299: further
knife arranging.

## Does frames 0-49 qualify? YES

Window 0-49 contains repeated hand/utensil occlusion-and-return events on
the beef surface, verified at stride 2-5 (frames 0,2,5,8,10,12,15,17,20,
22,25,27,30,32,35,38,40,42,45,47):

1. **Left-hand press-and-lift cycles on the pile top.** The left hand
   presses the meat (frames ~0-8), lifts clear (~10-17), presses again
   (~20-27), lifts (~30-38), presses (~45-49). Each press occludes the
   pile-top surface at `[655,780,745,880]` (hand+wrist, ~7-9k px of which
   ~5k px is meat/board surface); that surface is revealed again within
   ~10-15 frames. At least 2 full occlude->reveal cycles inside the window.
2. **Knife-blade slicing strokes.** The blade + right hand chop at the
   pile's right edge continuously; the blade band `[730,845,805,950]`
   (~5-7k px) occludes a strip of the steak each downstroke and lifts/tilts
   between strokes (blade visibly lifted at f35, f38; down at f32, f40+);
   stroke period ~5-10 frames. The blade contact point drifts rightward
   ~730->800 px across the window, so strips occluded early are revealed
   and freshly-cut slice surface appears behind the passing blade.
3. **Right forearm + tongs rocking over the board.** The forearm/tongs
   assembly `[800,690,940,880]` (~20-25k px) swings with each stroke,
   occluding and re-revealing the mat/board/backdrop behind it.

Motion is fast (chopping at full manual rate); the events are small-to-
medium area but numerous and strictly within-window. **0-49 qualifies; per
protocol it is the recommended frozen window.**

## Ranking (top 3 candidate 50-frame windows)

### 1. F=0 (frames 0-49) â€” RECOMMENDED (protocol-matched default; qualifies)
- Events: E1 left-hand press/lift on pile top (>=2 cycles); E2 knife-stroke
  occlusion of steak strip every ~5-10 frames with rightward sweep;
  E3 forearm/tongs rocking over board.
- Event regions / coverage: pile-top `[655,780,745,880]` ~5-7k px meat
  surface per hand cycle; blade strip `[730,845,805,950]` ~5-7k px;
  forearm band `[800,690,940,880]` ~20-25k px board/backdrop.

### 2. F=130 (frames 130-179) â€” strongest single occlusion magnitude
- Events: knife put down/picked up from board (~130-150: an object leaves
  the hand, rests at `[605,870,790,920]`, re-enters manipulation); then
  large hand+knife scraping passes OVER the pile (~150-179), occluding up
  to `[640,840,790,960]` (~15-20k px of meat) per pass with reveal between
  passes. Faster, larger-area occlusions than 0-49.
- Runner-up only because 0-49 qualifies and is protocol-matched.

### 3. F=195 (frames 195-244) â€” tongs manipulation + clean full reveal
- Events: tongs tips grip/move beef pieces (~205-230, occluder
  `[770,850,845,955]`); knife in left hand hovers/arranges
  (`[640,850,725,945]`); by ~250 both tools withdraw leaving the
  rearranged pile fully revealed (largest end-of-window reveal, ~10k px).

## Event-ray mask spec for the RECOMMENDED window (F=0, frames 0-49, cam00)

All rects `[x0,y0,x1,y1]` in the 1352x1014 cam00 raster. Frame indices are
window-relative = absolute here (window starts at 0).

**Event A â€” left-hand press/lift on beef pile top (2 full cycles + 1 partial)**
- (i) Occluder (hand+wrist) during occlusion:
  bbox `[650,775,750,885]`; occluded intervals: frames 0-9, 19-28, 44-49.
- (ii) Revealed surface after reveal (pile top + board edge under hand):
  bbox `[655,845,745,905]` (~5.4k px); revealed intervals: frames 10-18,
  29-43.

**Event B â€” knife-blade stroke occlusion of steak strip (periodic, ~5-10-frame cycle)**
- (i) Occluder (blade + right-hand knuckles) during down-stroke:
  bbox `[725,760,810,955]`; down-contact observed at frames 0-2, 5-8,
  12, 17, 22-27, 32, 40-42, 45-47 (blade lifted/tilted ~34-39 and briefly
  between listed contacts).
- (ii) Revealed surface after reveal (steak cut face + fresh slice tops
  left of the receding blade): bbox `[700,880,790,955]` (~6.8k px);
  revealed progressively as contact point sweeps x~730->800 over the
  window; fully visible whenever blade is lifted (e.g. frames 34-39).

**Event C â€” right forearm + tongs band over board/mat (secondary, large area)**
- (i) Occluder: bbox `[795,690,945,885]`; oscillates with every stroke
  (same phase as Event B).
- (ii) Revealed surface: white mat + board strip `[795,845,890,955]`
  (~10k px) revealed at each blade lift.

Suggested event-ray mask = union of (ii) rects: `[655,845,745,905]` +
`[700,880,790,955]` + `[795,845,890,955]`, ~20k px total (~1.5% of frame),
evaluated on frames where the corresponding occluder has withdrawn.

## Provenance / audit

- Data: read-only rclone pulls from
  `apollo:/apollo/users/sri/proj_adags/data/n3v/cut_roasted_beef/images/`;
  no repo modification, no job submission, ~60 MB pulled (55 PNGs), all in
  scratchpad `crb_frames/`.
- Frames inspected visually (Read tool): 0,2,5,8,10,12,15,17,20,22,25,27,
  30,32,35,38,40,42,45,47,50,60,...,290 (stride 10) plus 125-175 and
  195-225 at stride 10/5.
- Limitations: bboxes hand-estimated from rendered 1352x1014 frames
  (+-15 px); intra-cycle timing between sampled frames interpolated from
  stride-2-5 sampling; no cross-camera check performed (cam00 only, by
  design, since cam00 is the held-out evaluation view and selection
  precedes training).

