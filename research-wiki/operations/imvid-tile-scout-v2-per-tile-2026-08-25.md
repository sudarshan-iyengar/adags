# SPEC (FROZEN, v2) — per-tile detection replaces max-over-tiles (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Frozen BEFORE the new detector is
run and before any new candidate is scored.**

**SUPERSEDES the detection reduction of**
[[imvid-tile-scout-freeze-2026-08-25]]. That page is preserved unchanged,
including its measurements, which remain correct about what they measured.
Everything in [[imvid-event-definition-2026-08-24]] — `C_min=3`, `W_pre=15`,
`W_gap=20`, `W_post=15`, the A/B/C classes and the POSITIVE class-A
requirement — is **UNCHANGED and still binding**.

## 1. Why v1 had to be replaced — measured, not argued

v1 reduced each frame to `tile_max = max over 144 tiles`, then ran the
two-gate changepoint detector on that single scalar. A ground-truth audit
established:

* **Two real occlude-and-return events exist in `scene6_puppy`**: departure
  ~src 210 / return ~src 620-630, and departure ~src 5060-5080 / return
  ~src 5440-5460. Both gaps are ~370-400 frames, **18-20x `W_gap`**.
* **`cam12`, the clearest view in the rig, produced 6 candidates in the whole
  5,936-frame take and ZERO in any window covering either event.**
* That zero was a **structural certainty**, not a measurement: recomputing the
  shipped detector on all six relevant `cam12` windows, the
  `median + k_mad*MAD` threshold **exceeds the signal's own maximum in every
  one** (e.g. 73.99 against 72.19).
* **Cause:** `tile_max` is monopolised by the loudest region — a person's
  legs/torso at 40-83 grey levels continuously — which sets **both** the
  median and the MAD. **A quiet tile can never move a maximum it does not
  win.**
* **The object was never small.** The puppy covers **~68,000 proxy px = ~13%
  of the raster**, ~66x v1's declared 32x32 detection scale, with individual
  tile means reaching **41-65 grey levels**.
* **v1's absolute floor of 2.0 is inert**: 119-144 of 144 tiles exceed it in
  every frame, and **95.4%** of v1's candidates would also have cleared the
  whole-frame floor — the tile pass fired uniquely on **4.6%** of its own
  output.

**AND v1'S OWN PRECONDITION WAS VACUOUS, in exactly the way this block's
standing rule warns about.** P1's fixture was a **constant** background plus a
patch, so its `tile_max` series had **median 0 and MAD 0** and the relative
threshold was **exactly 0.0000**. P1 proved the max-over-tiles mechanism was
exercised; it never exercised **the gate that actually decides on real
footage**. The rule — *"every frozen rule needs a frozen precondition
asserting the mechanism it reads was actually exercised"* — was applied to
the wrong mechanism.

## 2. THE CHANGE — one reduction, and the gate follows the tile

**Run the existing two-gate changepoint detector independently on each of the
144 per-tile signals**, not on their maximum.

The relative gate becomes **per tile**: tile `(i,j)`'s threshold is
`median_t(S_ij) + k_mad * 1.4826 * MAD_t(S_ij)`, computed from **that tile's
own** temporal series. A quiet tile therefore gets a quiet threshold, which is
precisely the property v1 lacked.

Frozen, unchanged from v1: **tile size 60 px**, **`k_mad = 3.0`**, **window
300 source frames**, **proxy 960x540**, **stride 10 / 6000-1001 fps**, and the
per-tile statistic `mean over tile of |I_t - median_t(I)|`.

## 3. THE ABSOLUTE FLOOR — declared on noise, NOT on the known events

v1's 2.0 is inert (§1) and must be replaced. **The replacement may not be
chosen by what recovers the two known events** — that is tuning against the
answer.

**FROZEN RULE.** The absolute floor is declared as a **fixed multiple of the
take's own per-tile noise scale**, measured before any candidate is read:

```
floor = F * median over all (camera, window, tile) of
            [ 1.4826 * MAD_t( S_ij ) ]
F = 3.0
```

`F = 3.0` is a **declared judgment**, fixed here, chosen to match the existing
relative gate's `k_mad = 3.0` so the two gates express the same strictness in
different denominators. It is **not** derived from data and is labelled as
such. The measured floor value is recorded in the manifest.

A **sensitivity sweep** over `F` in `{1.5, 2.0, 3.0, 4.5, 6.0}` is reported
alongside, as supplementary information. **The primary reading is at
`F = 3.0` only**; every other point is labelled a sensitivity probe and may
not be reported as the census.

## 4. SPATIAL COHERENCE — a real event is contiguous, a lighting change is not

Per-tile detection raises the false-positive rate by construction (144
signals instead of 1). The discriminator is **measured**: the real events fire
**4-9 contiguous tiles**; illumination artefacts fire **38-75 tiles at once**.

**FROZEN:** a per-camera candidate requires

* at least **3 tiles** firing at the same proxy sample, **face-adjacent** in
  the 16x9 grid (a connected component of size >= 3); and
* **fewer than 33% of all tiles** (48 of 144) firing at that sample.

Both are **declared judgments**. The lower bound admits the smallest real
event observed (4 tiles) with one tile of margin; the upper bound sits below
the smallest observed artefact (38 tiles). **They are stated as bounds derived
from observation of a known-positive and a known-negative, and that is
disclosed** — they are not independent of the audit.

## 5. WHAT IS REMOVED from the primary's pair search, and why

Each of these was the primary's addition, not the frozen spec's:

* **`>=3 SHARED cameras` between the two endpoints — REMOVED.**
  [[imvid-event-definition-2026-08-24]] E1/E3 require `C_min=3` at the pre-gap
  segment and `C_min=3` at the post-gap segment **separately**, and never that
  the two sets intersect. The intersection requirement killed the real event
  B, whose intersection was **{cam11}, 1 < 3**. Each end is now scored
  independently at `C_min=3`.
* **`+/-1 tile argmax co-location` — REMOVED.** It is **anti-correlated with
  the target class**: a global illumination drift has a *stationary* argmax
  and passes trivially, while the real event's argmax moves **4-5 tiles**
  because the puppy exits frame-right and returns from bottom-left. Replaced
  by a **per-tile occupancy** test — high for >= `W_pre`, low for >= `W_gap`,
  high for >= `W_post` — **with no requirement that the two ends share a
  tile**.
* **`fall -> rise` polarity ordering — REMOVED.** Polarity records whether the
  object is the temporal *majority* of its window, not whether it is present:
  above 50% occupancy a departure reads `rise`, below it reads `fall`. The
  real return fired as **`fall`** on cam01/04/11 and **`rise`** on cam11
  within one proxy step. Both orders are searched.
* **The 20..600 frame gap cap — WIDENED to 20..3000.** The upper bound
  silently excluded any absence longer than 10 s without inspection.

## 6. WINDOWING — the remaining known defect, NOT fixed here

Windows are 300 source frames (5.0 s), non-overlapping, each templated on its
**own** temporal median. **Both real absences exceed one window**, so their
endpoints always land in different windows and are measured against different
templates, making their amplitudes non-comparable. Worse, an object present
through one window and absent through the next produces **near-zero residual
in both** and **no changepoint at all** at the boundary.

**This is recorded as a live defect and deliberately not repaired in v2.** The
per-tile change is demonstrated to recover both known events with the window
structure unchanged, so repairing the reduction first is the smaller,
verifiable step. A long-horizon or overlapping-window template is the next
repair and needs its own frozen spec. **No claim of exhaustive recall may be
made while §6 stands.**

## 7. PRECONDITIONS — and this time they exercise the DECIDING gate

* **P1-REL — THE ONE v1 LACKED.** The relative gate must be exercised on a
  **NON-CONSTANT** fixture: a background whose per-tile MAD is strictly
  positive in every tile, plus one loud distractor region, plus a quiet
  injected target. The precondition asserts (a) the distractor does **not**
  raise the target tile's own threshold, and (b) a `max`-over-tiles reduction
  on the same fixture **FAILS** to detect the target while the per-tile
  reduction **succeeds**. **This is the exact failure v1 shipped, made into a
  test.**
* **P2-RECALL — the known-positive check.** The detector must fire on both
  ground-truth events, on `cam11` and `cam12`, within +/-2 proxy samples of
  the audited frames, and must be **silent through the absence interval**.
  **These two events are a RECALL FIXTURE, not a tuning set:** no threshold in
  §3 or §4 may be changed to make them pass. If they fail, the spec fails and
  is rewritten before any census is read.
* **P3 — flat and sub-floor controls** yield zero candidates.
* **P4 — the tiling is exact** (unchanged from v1).
* **P5 — measured floor recorded.** The value of §3's computed floor is in the
  manifest, with the sweep.

## 8. RANKING — camera count is retired as the primary order

**Verified inversion:** ranked by amplitude within the 76 clusters at
`C_min>=3`, the real departure is **rank 3/76** (28.455) and the real return
**rank 17/76** (15.786), while the primary's two picks were **54/76** (5.902)
and **60/76** (5.080). Ranking by `n_cameras_supporting` **promoted the two
weakest-amplitude clusters in the set**.

Cause: only **~15 of 39 cameras** see the subject at all; the rest view sky,
facades, canopy or empty paving. **Camera count measures how GLOBAL a change
is, which is the opposite of the target property.**

**FROZEN:** the census and the gallery rank by **mean amplitude**.
`C_min = 3` is retained as a **corroboration floor**, never as a sort key.

## 9. Permitted and forbidden

**Permitted.** To report the per-tile census, its recall on the two
ground-truth events, and the `F` sensitivity sweep.

**Forbidden.** To change any §3 or §4 threshold after reading a candidate. To
report a sweep point other than `F = 3.0` as the census. To claim exhaustive
recall while §6's windowing defect stands. To rank by camera support. To
re-impose co-location, polarity ordering, or the shared-camera intersection.
To describe the two ground-truth events as an exhaustive event list — they
were found by eye by one auditor over a partial sweep, and **more events
almost certainly exist**, particularly in the unexamined span f900-f4800.

---

## IMPLEMENTATION CRITIQUE (2026-08-25, append-only) — six defects in THIS spec, found while implementing it

The implementer was instructed to implement the spec as frozen and to report
disagreements rather than act on them. It did. All six are accepted by the
primary; none was acted on during implementation.

### C1 (MATERIAL) — the §3 absolute floor is very nearly redundant, for the reason §3 gave as its justification

`detect_changepoints` accepts when `max(amplitude, excess) >= floor`, and the
relative gate already requires `excess > k_mad * 1.4826 * MAD_thistile`.
Setting **`F = k_mad = 3.0`** therefore makes the floor bind **only** on tiles
whose own MAD is *below* the pooled median — i.e. **on exactly the quiet tiles
this redesign exists to hear.**

Measured on the implementer's end-to-end fixture: floor **0.0093** grey levels
while every accepted candidate sat at **6.7**. The floor did no work.

**This is v1's inert-floor defect in a new costume**, and §3's own sentence —
*"so the two gates express the same strictness in different denominators"* —
is precisely **why the second gate cannot add independent information.**

It fails **OPEN** (toward more candidates), and §4's spatial-coherence gate
carries the false-positive load, so it is not unsafe. **But it may not be
described as a second independent screen, and §3's justification is
withdrawn.**

### C2 (MATERIAL) — the F sweep is NOT a sensitivity ladder, because candidate count is not monotone in F

Raising `F` thins the firing set, which can move a proxy sample from
*rejected* (>= 48 tiles firing, §4's cap) to *accepted*. **A higher floor can
therefore yield MORE candidates.** §3 presents the sweep as if it were
monotone. It is not, and any reading of it must not assume ordering.

### C3 (MATERIAL) — §3's pooled median has no guard against degenerate windows

A window holding only **2 proxy samples** gives every tile a MAD of
**identically zero** — with `T = 2` the per-pixel median is the mean, so
`|I_t - median|` is equal at both samples. Those zeros enter the pooled median
and **drag the floor down**.

Measured: on a 12-frame fixture, **12 of 24 tile series were exactly zero for
this reason.**

It does **not** bite `scene6_puppy` (594 proxy samples, ~20 windows of ~30),
but **the rule as frozen would silently produce a near-zero floor on a
differently-sized take.** Now visible rather than silent: the manifest carries
`n_zero_scale_tile_series` and `floor_is_degenerate_zero`.

### C4 (MINOR) — §7's P1-REL is unsatisfiable as literally worded

It specifies *"one quiet injected target"*, while §4 requires a **>= 3-tile
face-adjacent component**. A single-tile target can therefore never produce an
accepted candidate. The implemented fixture spans three adjacent tiles.
**The spec's own precondition contradicted the spec's own acceptance rule.**

### C5 (MINOR) — polarity is undefined for a mixed-polarity component

§4 and §5 do not say what polarity a component carries when its tiles
disagree, while `cluster_candidates` still clusters per polarity. Implemented
as the **majority with a deterministic tie-break toward `rise`**, emitting
`n_tiles_rise`, `n_tiles_fall` and `polarity_is_tied` so nothing is hidden.

### C6 (MINOR) — §8 says "rank by mean amplitude" without defining a per-camera amplitude

Implemented as the **mean over the component's own tiles**, also emitting
`max_tile_amplitude`. **Consequence to note:** if the audit's 28.455 / 15.786
came from a different per-camera reduction, the new ranks will not reproduce
those numbers exactly, and the two must not be tabled against each other
without saying so.

### AND ONE DEFECT OUTSIDE THIS SPEC, found in passing

`scripts/imvid_build_gallery.py` ranks by `n_cameras_supporting` and reads
`window["tile_explanations"]`, neither of which exists in a per-tile census.
Because it uses `.get()` throughout, **nothing crashes — it degrades
silently**, producing exactly the ordering §8 forbids with no spatial overlay.
**Silent degradation is the defect; a crash would have been safer.** Repaired
separately, with schema mismatch made to fail loudly.

### STATUS OF P2-RECALL — THE LOAD-BEARING PRECONDITION IS NOT YET SATISFIED

§7's **P2-RECALL was NOT RUN** and is recorded as not run, not omitted. The
implementer had no proxy frames and no cluster access, so **it is not yet
confirmed that the per-tile detector recovers Event A or Event B at the
audited frames, nor that it is silent through the absences.**

P1-REL is a **synthetic reconstruction** of the cam12 failure mode, not the
real footage. It shows the max-reduction threshold **29.8134** exceeding its
own signal maximum **21.5227** (0 candidates) while the per-tile target tile
sits at threshold **0.7601** against a plateau of **7.5449** (detected), with
the distractor tile's gate at **36.1910 — 47.6x** the target's. That is the
audited signature reproduced, and it is *evidence about the mechanism*, not
evidence about the take.

**No per-tile census result may be read as a recall claim until P2-RECALL is
evaluated against the two ground-truth events by the primary.**
