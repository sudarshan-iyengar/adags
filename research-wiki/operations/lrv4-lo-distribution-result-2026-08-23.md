# RESULT — LRV4's null is NOT a threshold artefact: a one-frame return
# destroys temporal localization entirely (2026-08-23, block 2)

EXPLORATORY, `evidence_bearing: false`. Spec, populations and the
four-way interpretation table frozen before any output in
[[lrv4-lo-distribution-diagnostic-spec-2026-08-23]]; nothing there moved.
Cells: Determined **259** (`lrv4_lo_prefilter` r0) and **260**
(`lrv3_lo_prefilter` r0), both commit `1190f58`, pool `dgx`, admitted
V100 image `sha256:70a28e3d…`, both `STATE_COMPLETED`.

## 1. The measurement

`recipient_prefilter` on both fixtures. `frame_dt` 0.16667 s.

| | LRV3 (comparator) | LRV4 (starved) |
|---|---:|---:|
| `recipient_support_lower_min` | 9.30000 | 9.66667 |
| `window_return` | [9.5, 9.8333] | [9.8333, 9.8333] |
| selector recipient rows | 3,912 | **1** |
| `row_sets_sufficient` | true | **false** |
| `matches_selector_recipient_count` | **true** | **true** |
| `P0` intersects return | 77,606 | 73,660 |
| `P1` … and in region | 4,148 | **277** |
| `P2` recipient but for `lo` | **3,925** | **8** |
| `P2` below floor / at-or-above | 13 / 3,912 | 7 / 1 |
| `P2` `lo` q05 / med / q95 | 9.5764 / 9.6894 / 9.8175 | 7.0022 / 9.2072 / 9.6835 |
| **`P2` width q05 / med / q95** | **0.0243 / 0.1086 / 0.2784** | **2.1083 / 2.9789 / 10.4982** |

The self-verification identity holds on **both** fixtures: `P2`'s
at-or-above-floor count equals the selector's recipient count exactly
(3,912 and 1). The diagnostic is reading the selector's own pre-image,
not reimplementing it.

## 2. The frozen table, applied

**Reading (a), THRESHOLD ARTEFACT — REJECTED.** It required all three of:
`|P2|` of order 10³ — it is **8**; the majority of `lo` mass within 2
frames below the floor — there are only 7 such rows in total; and widths
overlapping LRV3's recipients — they are **disjoint**, LRV3's q95 is
0.2784 and LRV4's q05 is 2.1083.

**Reading (b), GENUINELY ABSENT LOCALIZED SUPPLY — FIRES.** `|P2| = 8` is
order 10⁰–10¹, and `P2` widths are **27.4× LRV3's** at the median.

**No threshold rescues LRV4, and that is the decisive part.** Dropping
the floor to admit every one of the 7 below-floor rows yields 8
recipients against LRV3's 3,912 — still nowhere near sufficiency — and
those 8 rows have support widths 20–100× LRV3's recipients, so they are
not the same kind of row. The reading holds for the whole family of
floors, not just the derived one.

## 3. The mechanism, and why WIDTH was the load-bearing column

The spec recorded in advance that support width would decide this, on the
ground that LRV3's recipients were ordinary rows the optimizer **narrowed**
onto a 3-frame return (`recipient_rows_with_packet` is 0 in both
fixtures). That is exactly what happened.

LRV4 does have rows in the object region at the return instant — `P1` is
**277**, not zero. But their median support width is **25.6 seconds**,
essentially the whole 10-second sequence rendered as one broad temporal
lobe, against LRV3's **0.113 s**. So the starved fixture is not missing
content at the return; it is missing **temporally localized** content at
the return.

**Had only `lo` been reported, the 7 below-floor rows would have looked
like reading (a) and the fixture would have been re-run at a lower
floor.** The width column is what makes that wrong, and it was frozen as
required before the numbers existed.

Supporting signature, the in-region drop `P0 → P1`: LRV3 loses 18.7×,
LRV4 loses **265.9×**. Reading (d) — a purely spatial failure — is not
selected, because `P1` is non-empty and its widths, not its count, are
what disqualify it.

## 4. What this establishes, and what it does NOT

**Established.** LRV4's one-recipient screen was **not** an artefact of
the derived selection scalar. With a one-frame return the optimizer never
narrows any row's temporal support onto the return, so **LRV4 cannot host
the payload experiment it was built for at any threshold.** That is a
substantive finding about what observation starvation does to the
representation: it does not produce under-trained localized rows, it
produces no localized rows.

**NOT established, and the distinction is the whole point.** The
mechanism claim from [[payload-headroom-result-2026-08-23]] §4 — that
headroom is a question of observation supply rather than of which tensor
is carried — is **still UNTESTED**. This result does not confirm it and
does not refute it. What the result does is close off LRV4 as the
instrument: a fixture starved hard enough to remove observation supply is
also starved hard enough to remove the recipient row set, so this
particular design cannot separate the two.

**Unchanged:** the LRV3 payload negative, the permutation control's
overturning of the identity attribution, and the representation-only
pivot as the recommendation on the record.

## 5. Consequence for the consolidation lane

A fixture that tests observation supply must starve the **views** while
preserving a temporally localized return — for example fewer training
cameras observing an unchanged 3-frame return, rather than fewer return
frames. Per [[lrv3-fixture-hazards-2026-08-23]] §2 the frozen generator
exposes no `--n-cameras` knob and `N_CAMERAS`/`TEST_CAMERAS` are module
constants, so that variant is a NEW named fixture plus a tool-guard
relaxation — a new frozen specification, not a parameter change. **It is
not proposed here**, and nothing in this block authorizes it.

## 6. Bookkeeping

Claims consumed: `lrv4_lo_prefilter` r0 (259), `lrv3_lo_prefilter` r0
(260). Reports written to `payload_headroom_report_prefilter.json` in
each substrate's run dir, deliberately a NEW filename so the recorded
reports — including LRV4's at sha256 `972136ae…` — are preserved. No
threshold, floor, gate, link, map, ratio or decision rule was changed;
`row_sets_sufficient` is still false for LRV4 and true for LRV3.
Measured cost ≈0.1 slot-h for the pair, against 0.4 projected.
