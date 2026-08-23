# SPEC (FROZEN) — the LRV4 recipient pre-filter `lo` distribution
# (2026-08-23, block 2)

EXPLORATORY, `evidence_bearing: false`. Frozen BEFORE any output exists.
Authority: [[lrv4-starved-fixture-result-2026-08-23]] §5, which
pre-identified exactly this diagnostic, required it to change no decision
rule, and estimated it at well under a minute of GPU.

**This spec changes NO threshold, floor, gate, link, map, ratio or
decision rule.** `row_sets_sufficient` keeps its value; LRV4 stays FALSE
whatever the histogram shows. Nothing here reopens the LRV3 negative or
the representation-only pivot.

## 1. The question

Experiment 248 found **ONE** recipient row on LRV4, so
`row_sets_sufficient` is FALSE and the observation-supply mechanism claim
is UNTESTED rather than refuted. Two readings survive and the run could
not separate them:

* **(a) THRESHOLD ARTEFACT** — the derived
  `recipient_support_lower_min` moved 9.3 → 9.6667, and rows do exist
  just below the cut;
* **(b) GENUINELY ABSENT LOCALIZED SUPPLY** — with a one-frame return the
  optimizer never narrows any row's temporal support onto the return, so
  there is no temporally-localized return content to receive anything, at
  any threshold.

## 2. What is measured

A purely additive report field, `report["recipient_prefilter"]`, produced
by `falsify_b2_edit.recipient_prefilter_diagnostic`. Three **nested**
populations, so a temporal loss and a spatial loss are separable:

| population | definition |
|---|---|
| `P0_intersects_return` | `hits_wr` |
| `P1_intersects_return_in_region` | `hits_wr & inside_ret` |
| `P2_recipient_but_for_lo` | `hits_wr & inside_ret & ~spanning` |

`P2` is the EXACT pre-image of `recipient_mask`
(`falsify_b2_edit.py:454-455`) with the `lo >= floor` conjunct removed.
For each population: `n`, quantiles of `lo`, of the **support width
`hi - lo`**, and of the support centre; a frozen frame-offset histogram
of `(lo - floor) / frame_dt`; and `n_below_floor` /
`n_at_or_above_floor`.

**Support width is load-bearing, not decoration.** LRV3's recipients were
ordinary rows the optimizer NARROWED onto a 3-frame return
(`recipient_rows_with_packet` is 0 in both fixtures). A large `P2` of
very wide rows that merely reach the return instant would look like
reading (a) on `lo` alone while actually being reading (b).

**Self-verification.** `P2.n_at_or_above_floor` must equal
`sets["recipient"].numel()` or the tool raises `ContractError`. That
identity is what proves the block reads the selector's own pre-image
rather than being a second implementation of it. It is exact, not
approximate: every population conjoins `inside_ret`, which makes the
selector's `(inside_ep1 | inside_ret)` disjunction true, so `spanning`
reduces to `hits_w1 & hits_wr` there.

`LO_HIST_FRAME_EDGES` is a frozen reporting grid. **The edges are NOT
candidate thresholds** and the code asserts nothing on them.

## 3. Interpretation — FIXED BEFORE ANY OUTPUT EXISTS

| reading | requires | consequence |
|---|---|---|
| **(a) threshold artefact** | ALL THREE: `\|P2\|` of order 10³ (within ~4× of LRV3's 3,912); the majority of its `lo` mass within 2 frames below the floor; and widths overlapping LRV3's recipient widths | LRV4's null is an artefact of the derived scalar and the mechanism claim stays UNTESTED. **This does NOT authorize moving the floor.** A real test needs a principled rule prespecified and applied to BOTH fixtures, never fitted to this histogram. |
| **(b) absent localized supply** | `\|P2\|` of order 10⁰–10¹, OR `lo` median more than ~6 frames below the floor with widths far above LRV3's | A one-frame return produces no temporally-localized return content. LRV4 cannot host the payload experiment at any threshold — itself a substantive finding about what starvation does to the representation. |
| **(c) mixed** (named in advance so it cannot be collapsed into (a)) | mass just below the floor BUT widths far above LRV3's | Content near the return exists but is not the kind of row LRV3's recipients were. Neither (a) nor (b); a new fixture question. |
| **(d) spatial, not temporal** | `\|P0\|` large and `\|P1\|` tiny | The oracle-region condition at t = 9.8333 is what fails, not the `lo` cut. |

## 4. One fact already derivable with ZERO compute

The admissible recipient `lo` band is `[floor, window_return[1]]`:

| fixture | band | width |
|---|---|---:|
| LRV3 | `[9.3, 9.8333]` | **3.20 frames** |
| LRV4 | `[9.6667, 9.8333]` | **1.00 frame** |

The band narrowed **3.20×**. At equal `lo` density LRV4 would have shown
≈1,222 recipients; it showed **1**. **The band narrowing under-explains
the collapse by roughly three orders of magnitude**, which is precisely
why the density is the measurement rather than something to be argued
about. This favours (b) but does not establish it — the density
assumption is exactly what the diagnostic tests.

## 5. Cells

Two cells, both `scripts/payload_headroom.py`, pool `dgx`,
`--exploratory`, 0.2 projected slot-h each. **Both are required**: "rows
pile up just below the cut" is meaningless without the LRV3 distribution
from the same code, so LRV3 is the comparator, not an extra.

| cell | checkpoint |
|---|---|
| `lrv4_lo_prefilter` | exp 247 `.../20260823T001524Z_lrv4_b1_packets_0_bbf1c4f/chkpnt6000.pth` |
| `lrv3_lo_prefilter` | exp 209 `.../20260820T155019Z_lrv3_b1_packets_0_4d15fcf/chkpnt6000.pth` |

`--out_report` uses a NEW filename (`payload_headroom_report_prefilter.json`)
so the recorded reports — including LRV4's at sha256 `972136ae…` — are
preserved rather than overwritten.

## 6. Cost and termination

≈0.4 slot-h total. The measurement ends when both reports exist and the
§3 table has been applied. **No further LRV4 runs are authorized by this
spec under any outcome**, and in particular no re-run at a different
floor.

## 7. Implementation provenance

`recipient_prefilter_diagnostic` and its two helpers live in
`scripts/falsify_b2_edit.py` beside `build_row_sets` (importable and
CPU-testable without the renderer, matching every other numeric in that
module). `scripts/payload_headroom.py` calls it once and adds one report
key. Seven CPU tests cover the selector-pre-image identity, that
below-floor rows are counted and not selected, population nesting, the
empty-set all-`None` convention, the frozen grid, and two fail-closed
paths. `tests.test_falsify_b2_edit` 74/74 and
`tests.test_payload_headroom` 46/46 pass in the CPU venv.
