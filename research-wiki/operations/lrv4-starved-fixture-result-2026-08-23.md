# RESULT — the LRV4 test DID NOT EXECUTE VALIDLY; the mechanism claim
# remains UNTESTED (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Prediction and decision rule
frozen before any result in [[lrv4-starved-fixture-spec-2026-08-23]].
Cells: substrate = Determined experiment **247** (`lrv4_b1_packets` r0,
commit `bbf1c4f`, dgx, admitted V100 image, default seed 6666 matching
LRV3's experiment 209), COMPLETED; screen = experiment **248**
(`lrv4_payload_headroom` r0), COMPLETED. Report sha256
`972136aecdd2832d1d132d82a5a242dcd8ab87b139d932c2e63f69c8a56f180c`.

**Neither branch of the frozen decision rule fires. The claim that
headroom is a question of observation supply is neither confirmed nor
refuted.**

## 1. The integrity check PASSED, and the substrate is healthy

`protocol.fixture` reads exactly what the frozen spec §6 requires:
`scene_id LRV4`, `return_frames [59]`, `window_return
[9.8333, 9.8333]`, `probe_times [2.5, 9.8333]`,
`scalars_source: derived`. **It does NOT read 9.6**, so the run did not
silently select rows against LRV3's window, which was the specific
failure mode that spec anticipated.

The substrate trained normally: `best_val/psnr` **28.393** over 149,316
primitives, against LRV3's 28.59 over 149,800 — the two fixtures differ
by two frames, and their substrates match to ~0.2 dB, as expected.

## 2. Why the screen is INVALID: one recipient row

| row set | LRV3 (exp 233) | LRV4 (exp 248) |
|---|---:|---:|
| donor rows | 3,722 | 4,697 |
| **recipient rows** | **3,912** | **1** |
| wrong-identity pool | 29,351 | 29,256 |
| no-op donor / recipient halves | 1,853 / 1,869 | 2,396 / 2,301 |
| spanning rows excluded | 229 | 273 |
| **`row_sets_sufficient`** | **true** | **FALSE** |

**The donor side is healthy and the recipient side collapsed to a single
row.** Every headroom statistic on the recipient side is therefore a
one-pair statistic and means nothing.

## 3. The near-miss worth recording, because it is the reason the guard exists

With one pair, the report's `_features_dc` headroom ratio reads
**4.995** — comfortably above the frozen screening floor of 2.0, and
higher than any ratio LRV3 produced. **Read without checking the pair
count, that number would have been reported as a spectacular
confirmation of the mechanism claim.**

It was caught because the tool emits `row_sets_sufficient: False` and
because the per-link `pairs` count (1) is carried alongside every ratio.
**A ratio without its n is not a measurement**, and this is the concrete
instance of that in this project's record.

## 4. The diagnosed mechanism, and the two readings it does not separate

Recipients are selected as rows whose operational support intersects the
return window AND whose support LOWER bound satisfies
`lo >= recipient_support_lower_min`. That scalar is now derived from the
fixture rather than hardcoded: LRV3's frozen value is **9.3**; LRV4's
derived value is **9.6667**, one frame before its single return instant.

Packet birth stamps relocated rows with `t_sigma_frames: 1.5`, so a row
born at frame 59 (t = 9.8333) has operational support
`mu ± 2σ = 9.8333 ± 0.5 = [9.3333, 10.3333]`, i.e. **`lo = 9.3333 <
9.6667` — excluded by construction.** Consistent with that,
`recipient_rows_with_packet` is **0** in BOTH fixtures: the recipients
LRV3 found were never packet-birth rows, they were ordinary rows whose
temporal support the optimizer narrowed onto a 3-frame return.

**Two readings remain, and this run cannot separate them:**

* **(a) a threshold artifact** — the derived `lower_min` moved from 9.3
  to 9.6667, tightening by ~2.2 frames, and simply cut off rows that do
  exist near the return;
* **(b) the scientifically interesting reading** — with only ONE return
  frame the optimizer never narrows any row's temporal support onto the
  return at all, so the starved fixture has no temporally-localized
  return content to be a recipient.

If (b) holds it is a substantive finding about what starvation does to
the representation — but it would ALSO mean the fixture cannot host the
payload experiment it was built for, since there is nothing to transfer
INTO.

## 5. What is NOT done, and deliberately

**No threshold was changed after seeing this null.** Re-running LRV4 with
LRV3's 9.3 would probably produce recipients and would look like a fix,
but choosing a selection scalar because it yields a non-empty set — after
observing that the principled one does not — is exactly the post-hoc
adjustment this project forbids.

**The pre-identified diagnostic, for a NEW frozen spec:** report the
DISTRIBUTION of the support lower bound `lo` over rows whose support
intersects the return window, before any `lower_min` cut. That is a pure
diagnostic which changes no decision rule, and it separates (a) from (b)
directly — if rows cluster just below the cut, it is (a); if there are no
rows near the return at any threshold, it is (b). It costs well under a
minute of GPU.

## 6. Status of the mechanism claim

**UNTESTED.** [[payload-headroom-result-2026-08-23]] §4's inference —
that headroom is a question of observation supply rather than of which
tensor is carried — stands exactly as it did: an inference, labelled as
one, with its falsification test built, executed, and returning an
invalid instrument rather than an answer.

The LRV3 negative is untouched. The representation-only pivot remains the
recommendation on the record, and this result neither strengthens nor
weakens it.

## 7. Bookkeeping

Claims consumed: `lrv4_b1_packets` r0 (experiment 247) and
`lrv4_payload_headroom` r0 (experiment 248). The LRV4 fixture is
generated, byte-verified (1,444 files / 83,602,158 bytes identical local
and on Apollo), committed as configs, and remains valid for any future
recipient-selection spec. Its held-out return supply is 18,978
pixel-times against LRV3's 56,934 — exactly one third, as designed.
