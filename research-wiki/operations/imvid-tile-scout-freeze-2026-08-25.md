# SPEC (FROZEN) — the region/tile-sensitive event scout for the ImViD census (2026-08-25)

EXPLORATORY, `evidence_bearing: false`. **Frozen BEFORE any Puppy or Opera
candidate has been scored.** The proxy decode was still running when this was
written, and no census output existed.

Supplements, and does not replace, [[imvid-event-definition-2026-08-24]] and
its 2026-08-24 amendment. Every threshold in §2.1 of that spec —
`C_min = 3`, `W_pre = 15`, `W_gap = 20`, `W_post = 15` — the §3 POSITIVE
class-A requirement, and the §5 synchronization rule are **UNCHANGED and
still binding**. Nothing here relaxes any of them.

## 1. Why a new instrument was required

The existing census signal is a **whole-frame mean**, on all three of its
channels:

```
absdiff[1:]     = |diff(stack)|.mean(axis=(1, 2))
template_dist   = deviation.mean(axis=(1, 2))
changed_frac    = (deviation > 8.0).mean(axis=(1, 2))
```

The 2026-08-24 amendment already recorded the consequence — *"Global-mean
signals are blind to small objects"* — but recorded it as a **caveat on a
reading**, not as a bound with a number. It is now measured, and the number
is worse than "blind to small objects" suggests.

**MEASURED.** A synthetic patch of **32 x 32 proxy pixels at 25 grey levels**
on a 960 x 540 proxy raster produces:

| pass | signal at the patch frame | vs its 2.0 grey-level floor | candidates |
|---|---:|---|---:|
| **global** (existing) | **0.049383** | **40x BELOW** | **0** |
| **tile** (this spec) | **7.111111** | 3.6x above | **2** |

**The global pass does not merely rank it low. It cannot see it at all** —
the signal sits a factor of 40 under the instrument's own absolute floor.

**No zero-event claim from the global pass alone may be called exhaustive**,
and this table is the reason.

## 2. THE INSTRUMENT — frozen parameters

Additive to `scripts/imvid_event_proxy.py`. **`frame_signals` is byte-identical
to its pre-change form** (verified: sha256 of the extracted function equal at
`3f25241b3ca803ac`, 26 lines, before and after). The default census path is
unchanged; the tile path is opt-in via `--tile-mode`.

| parameter | frozen value | basis |
|---|---|---|
| tile size | **60 px** square | gives exactly **16 x 9 = 144** tiles at the 960x540 census raster |
| tile absolute floor | **2.0 grey levels** | same absolute units as the existing global floor, now measured on a tile mean |
| `k_mad` | **3.0** | unchanged; the existing robust relative gate is reused, not replaced |
| tile statistic | `mean over tile of \|I_t − median_t(I)\|` | the SAME `template_dist` quantity, per tile instead of per frame |
| detector scalar | **`tile_max[t] = max over tiles`** | see §3 — the maximum is load-bearing |
| retained | `tile_argmax[t]`, and the full `(T, 9, 16)` grid | the grid is the review-gallery heatmap source |

Partial edge tiles are **included**, never padded, dropped, or merged, and
**nothing is area-weighted** — a short edge tile is a mean over fewer pixels
and is still read in grey levels, directly comparable with a full tile.

**Both signals are reported side by side.** Every tile-mode candidate carries
the global `template_dist` at the same frame, so a reader can see exactly how
much sensitivity the tile pass bought. The global signal is never dropped.

## 3. THE MAXIMUM IS THE LOAD-BEARING CHOICE, and it is provable

A **mean** over tiles is algebraically the area-weighted whole-frame mean
again — exactly it, when the tiling is regular. So a tile pass reduced by a
mean is *the very blindness it exists to remove*, wearing a grid.

This is not an argument, it is asserted numerically in the frozen
precondition set: the tile mean on the P1 fixture reproduces the global value
to **0.049382716 ± 1e-09**, and **replacing the max by the mean destroys the
P1 detection (2 candidates → 0)**.

The **144x** sensitivity gain is not a tuned quantity. It is
`frame_area / tile_area = 518400 / 3600` exactly, asserted against that closed
form.

## 4. FROZEN PRECONDITIONS — statements about the SETUP, never about the score

Per the block's standing rule that *a frozen reading rule without a frozen
precondition is how an instrument delivers a clean null it never earned*.
Implemented as `--tile-selftest`; **33/33 pass**, independently re-run by the
primary.

* **P1 — detection at a DECLARED SCALE.** A 32 x 32 proxy-px patch at 25 grey
  levels must be DETECTED by the tile pass and MISSED by the global pass.
  Both halves are asserted. The second half is what proves the new pass is
  not redundant.
* **P2 — flat and sub-floor controls.** A constant window yields **0**
  candidates. A pure-noise window below the floor yields **0**.
* **P3 — finite and non-constant.** All tile statistics finite; `tile_max`
  non-constant where change exists; `argmax` in range.
* **P4 — the tiling is exact.** Every pixel covered exactly once — no gaps,
  no double-counting — verified at 960x540, 130x100, 61x59, 60x60 and 1x1,
  including the partial 10-px edge tile at 130. Tile size 0 and an empty
  raster are refused.
* **P5 — NEUTER.** Replacing the maximum by a mean must destroy P1's
  detection. It does (2 → 0).

## 5. THE DECLARED SCALE IS A JUDGMENT, AND IT IS COUPLED TO THE RASTER

**32 x 32 proxy px at 25 grey levels is a declared judgment**, not a derived
quantity. At the census raster it corresponds to roughly **177 x 177 native
pixels** (5312 / 960 = 5.53x downscale). It is fixed here, before any
candidate exists, so that a scene cannot be admitted or excluded by moving it.

**THE COUPLING IS LOAD-BEARING AND WAS VERY NEARLY MISSED.** The script's own
`DEFAULT_LONG_EDGE` is **480**, which would give a 480x270 proxy, an
**8 x 5 = 40**-tile grid, and a fixed-size real object covering **4x fewer**
proxy pixels. The same declared patch would then read about **1.78** grey
levels — **BELOW the 2.0 floor**, i.e. undetectable, while every test still
passed at 960x540.

**Therefore, binding:** the tile parameters in §2 and the declared scale in
this section are valid **only at a 960 x 540 proxy raster**. A census run on
proxies built at any other `--long-edge` is **INVALID under this spec** and
requires a new declared scale.

The Puppy proxies for this block were built with `--long-edge 960` and the
measured proxy raster is **960x540** — confirmed in the proxy manifest before
the census was run. The tiling itself is exact at any raster; only the
declared detection scale is tied to this one.

## 6. What this instrument still CANNOT do

Everything the 2026-08-24 amendment recorded remains true, and the tile pass
fixes exactly one of the listed limitations — the global-mean blindness. It
does **not** fix any of the others:

* **It is not an event detector and not an instance mask.** Every emitted
  explanation block is labelled `is_instance_mask: false` and
  `kind: detector_signal_explanation`, and that wording is asserted by test
  in the JSON and in both docstrings. **A tile is high because pixels inside
  it departed from the window's own temporal median — which a moving object,
  a moving occluder, a shadow, a lighting change and compression noise all
  produce equally.**
* **A tile box is a TILE extent, not an OBJECT extent.** The emitted field is
  named `pixel_box_is_tile_extent_not_object_extent` so it cannot be
  mis-cited.
* **No occlusion reasoning, no identity, no return-fidelity gate.** It cannot
  assign the A/B/C classes of §3; it proposes candidates for that
  classification and nothing more.
* **Localization is still one proxy step** — at stride 10 that is 166.83 ms.
* **Template contamination is unchanged**: content occluded for more than half
  a window becomes the template and inverts the polarity.
* **Support count is still not proof.** A cluster with 3-of-39 support may be
  one real event or three coincidences, which is exactly why §3 requires
  POSITIVE geometric evidence for class A.
* **It cannot establish absence, synchronization, or exact boundaries.**
* **Higher recall means MORE false positives, by construction.** The tile
  pass is deliberately high-recall; its candidate count is not a count of
  events and must never be reported as one.

## 7. Sampling adequacy — measured, not assumed

At the frozen census settings the requested rate resolves **exactly**:

```
requested 6000/1001 fps -> stride 10 -> effective 6000/1001
        (5.99401 fps, rel err 0.0000%)
one proxy step = 10 source frames = 166.83 ms
```

The frozen `W_gap = 20` frames is **333.67 ms**, so **at least two proxy
samples fall inside any admissible gap, by construction**. Puppy yields 594
proxy frames per camera from 5,936 source frames at `60000/1001`.

This satisfies the *scouting* stage only. It does **not** satisfy
[[imvid-event-definition-2026-08-24]] §5: at 166.83 ms the timing values are
**localization brackets**, 8x coarser than ImViD's stated ~20 ms
synchronization bound. Per the 2026-08-24 amendment, a near-native-rate
measurement on the narrow selected window remains **mandatory** before any
gating cell.

## 8. Permitted and forbidden

**Permitted.** To run the tile pass as a high-recall scouting instrument at
960x540. To report that the global pass misses a declared-scale object by 40x.
To report tile candidates as proposals for human/ground-truth classification.
To use the tile grid as the spatial explanation overlay in the review gallery.

**Forbidden.** To run this spec's parameters at any proxy raster other than
960x540. To move the tile size, the tile floor, `k_mad`, or the declared scale
after seeing a candidate list. To report a tile box as an object extent. To
report a tile candidate as an event, or a candidate count as an event count.
To cite the tile pass as satisfying §5 synchronization. To call any zero-event
result exhaustive without stating the declared scale it is exhaustive
*against*.
