# RESULT AND CLOSURE — the ImViD event census on `scene6_puppy`, and why the lane is closed (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. Records the outcome of the census
built under [[imvid-tile-scout-freeze-2026-08-25]] (v1) and
[[imvid-tile-scout-v2-per-tile-2026-08-25]] (v2), the ground-truth audit that
corrected it, and the decision to stop investing in it.

## 1. THE DECISIVE OUTCOME

**Two real occlude-and-return events exist in `scene6_puppy`, they were found
by a human watching the video in about an hour, and the automated census
missed them until it was rebuilt.** After the rebuild it surfaces them, but it
still cannot support a recall claim, and **it was never on the critical path
to the paper.**

| event | departs | absent | returns | gap |
|---|---|---|---|---|
| **A** (`cam12`) | ~src 210 | 240 - 600 | 620 - 630 | ~400 f = **6.7 s** |
| **B** (`cam11`, `cam12`) | ~src 5060 | 5080 - 5430 | 5440 - 5460 | ~370 f = **6.2 s** |

Both gaps are **18-20x** the frozen `W_gap = 20`.

## 2. WHY v1 MISSED THEM — a structural certainty, not a measurement

v1 reduced each frame to `tile_max = max over 144 tiles`, then ran the
two-gate changepoint detector on that single scalar.

* **`cam12`, the clearest view in the rig, produced SIX candidates in the whole
  5,936-frame take and ZERO in any window covering either event.**
* Recomputing the shipped detector on all six relevant `cam12` windows, the
  `median + 3*MAD` threshold **exceeds the signal's own maximum in every one**
  (e.g. **73.99 against 72.19**). Zero candidates was guaranteed before any
  data arrived.
* **Cause:** `tile_max` is monopolised by the loudest region — a person's
  legs/torso at 40-83 grey levels continuously — which sets **both** the median
  and the MAD. **A quiet tile can never move a maximum it does not win.**
* **The object was never small.** The puppy covers **~68,000 proxy px = ~13% of
  the raster**, ~66x v1's declared detection scale, with tile means of
  **41-65 grey levels** against a 2.0 floor.
* **v1's absolute floor was inert**: 119-144 of 144 tiles exceeded it every
  frame, and **95.4%** of v1's candidates would also have cleared the
  whole-frame floor — the tile pass fired uniquely on **4.6%** of its own
  output.

**And v1's own precondition was vacuous in exactly the way this project's
standing rule warns about.** Its fixture was a **constant** background plus a
patch, so the `tile_max` series had **median 0 and MAD 0** and the relative
threshold was **exactly 0.0000**. It proved the max-over-tiles mechanism ran;
it never exercised the gate that decides on real footage.

## 3. WHAT THE REBUILD FIXED, AND WHAT IT DID NOT

v2 ran the same two-gate detector **independently on each of the 144 per-tile
signals**, so each tile's threshold comes from its own history. Tile size,
`k_mad`, window length and proxy rate unchanged.

| | v1 (max-over-tiles) | v2 (per-tile) |
|---|---:|---:|
| per-camera candidates | 560 | **2,402** |
| clusters | 275 | **341** |
| at `C_min >= 3` | 76 | **206** |
| **`cam12` candidates** | **6** | **90** |

Measured noise-derived floor: **2.538 grey levels** over 112,320 tile series —
landing almost exactly where v1's arbitrary 2.0 sat. **v1's floor was never
wrong in value; a whole-frame maximum simply never let a quiet tile reach it.**

### 3.1 Recall passes; silence does not

The frozen P2-RECALL check, written before the census output existed:

* **R1 — finds the events: PASS.** Every audited endpoint is hit on every named
  camera. `cam12` A: `[190,210]` and `[610,620,630,640]`; `cam11` B:
  `[5050,5060]` and `[5440,5450,5470]`; `cam12` B: `[5050,5070]` and
  `[5450,5460,5470]`.
* **R2 — silent through the absence: FAIL.** 2 violations inside event A's gap
  and 13 inside event B's.

**R2's failure is a genuine methodological finding, and it is about the
INSTRUMENT rather than the detector's competence:**

> **A windowed changepoint detector templated on its own median cannot be
> silent through an absence longer than its own window.** In a mid-absence
> window the *absence* becomes the template, and any other moving content —
> a person, wind in foliage, shifting sun — fires against it.

The absences are 370-400 frames against a 300-frame window, so every such
window is mid-absence. The gap firings are mostly in **different tiles** from
the endpoints (for event A the entire gap firing is one 3-tile component
against 76 endpoint tiles), consistent with other scene content rather than
the object.

**R2 was also badly specified**, and that is recorded rather than quietly
dropped: it asked a *change* detector to be silent across an interval that
genuinely contains change. The correct operational question is not silence but
**where the true events land in the human review queue**.

### 3.2 Ranking, and a claim that was half true

An earlier reading of these results stated the census "surfaces the two known
events at ranks 5/11/13 of 341". **That is the RETURN legs only.** Measured
over all 341 clusters by amplitude:

| leg | src | rank | amplitude | cameras |
|---|---:|---:|---:|---:|
| A return | 630 | **5** | 28.291 | 10 |
| B return | 5450 | **11** | 24.262 | 6 |
| A departure | 230 | **19** | 22.120 | **1** |
| B departure | 5040 | **30** | 20.133 | 8 |

**The departures — which define the episode onsets the method exists to
control — rank 19 and 30, and event A's departure has ONE supporting camera,
below `C_min = 3`.** Any statement about this census must quote all four legs.

Ranking by `n_cameras_supporting` was retired because it **inverted** the
ordering: ~24 of 39 cameras cannot see the subject at all, so camera count
measures how *global* a change is — the opposite of the target property.

## 4. THE GALLERY DEFECT

The human-review gallery built from this census has a context window of
**+/-3 proxy steps = 1.001 s**. The events have **6.2-6.7 s** gaps. It
therefore showed **16%** of an event and **could not display departure and
return in one view at all** — no amount of care by the reviewer would have
made it evaluable. Its reference camera was also chosen by *largest tile
signal*, which in an autumn scene selects whichever camera sees the most wind.

Both are ordinary hyperparameter defects. They are recorded because a reviewer
was asked to classify candidates through them.

## 5. WHY THE LANE IS CLOSED

**The census was not on the critical path.** The paper needs *one or two
admitted event windows*. A human supplies those. An automated census supplies
a **supply statistic** — and v2's own Forbidden list rules that out while the
windowing defect stands, so the instrument cannot deliver the only thing that
would have justified building it.

**It is calibrated by the audit, not validated against it.** Every removal from
the pair search and both spatial-coherence bounds were derived from observing
one known-positive and one known-negative. A detector fitted to two known
events says nothing about events it has not been shown.

**Hand annotation would have been the better call from the start.** One
auditor found both events by eye in about an hour. The census consumed a
detector rewrite, two frozen specs, an implementation critique, and two
adversarial reviews.

**And the deeper error, recorded as method:**

> **Freezing protects CLAIMS, not CODE.** What must be frozen before looking
> are the things that determine what may be concluded — the event definition,
> the A/B/C rule, the membership gate, the scored endpoint. A search tool's
> window length, tile size and amplitude floor are **hyperparameters whose
> output a human adjudicates**; nothing downstream claims anything about them.
> Treating them as scientific commitments produced two frozen specs and two
> adversarial reviews of what should have been a parameter change.

## 6. WHAT MAY AND MAY NOT BE SAID

**Permitted.** That two occlude-and-return events exist in `scene6_puppy` at
the frames in §1, established by **human annotation against the video**. That
an independent automated pass **also surfaced both returns near the top of its
ranking** (ranks 5 and 11 of 341) — as corroboration that the annotation is
not idiosyncratic, costing no further investment and making no recall claim.
That v1's blindness was structural and its cause is measured.

**Forbidden.** Any exhaustive-recall or event-supply claim from this census —
the windowing defect stands and **f900-f4800 was never swept by eye**, so more
events almost certainly exist. Citing "ranks 5/11/13" without the departure
ranks of 19 and 30. Describing R2's failure as a detector failure rather than
an instrument-scope finding. Treating the A/B/C classification as done — **no
candidate has been classified**, and the class-A corroboration packet does not
exist.

## 7. PROVENANCE

| item | value |
|---|---|
| take | `scene6_puppy`, 39 cameras, 5,936 frames @ `60000/1001` |
| proxies | 960x540, stride 10, effective `6000/1001` fps, rel. err **0.0000**, 25.27 GiB, 23,206 objects |
| v1 census | `derived/proxy/scene6_puppy.census.tile.json`, 5,160,212 B |
| v2 census | `derived/proxy/scene6_puppy.census.pertile.json`, 26,020,442 B |
| galleries | `derived/gallery_scene6_puppy` (v1), `derived/gallery_pertile_scene6_puppy` (v2, 60 of 341, 35.5 MiB) |
| commits | `6e029be` (v1 scout), `b3f71b1` (per-tile), `da47f32` (gallery) |
| GPU cost | **zero slots** — every decode, census and gallery cell ran on the zero-slot command path |

**NOT RECORDED:** no A/B/C classification, no class-A corroboration packet, no
near-native synchronization measurement, and no census of `scene1_opera`.
