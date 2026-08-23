# RESULT — no replacement payload has oracle-correct headroom on LRV3
# (2026-08-23)

EXPLORATORY, `evidence_bearing: false`. Design frozen before output in
[[payload-headroom-spec-2026-08-23]]. **CORRECTED — the original text
here read "nothing there moved", which is FALSE in three respects; see
§6 C1.** Commit, pool, image and seed below are LEDGER claims, not
report-verifiable ones (§6 C9). Cell:
Determined experiment **233** (`lrv3_payload_headroom` r0, commit
`2ba6a62`, pool `dgx`, admitted V100 image
`sha256:70a28e3d…`, seed 0), COMPLETED. Report
`payload_headroom_report.json` in experiment 209's run dir, sha256
`2708eb250a8195a50a338af097e6ffc52e60777c0292d9601eab981802165012`,
schema `adags-payload-headroom-v1`, **`renders_performed: 0`**.

## 1. The screen reproduces the recorded DC falsification EXACTLY

The DC row is carried as a control precisely so this can be checked.
Against [[b2-edit-falsification-2026-08-20]] §2:

| quantity | recorded 2026-08-20 | measured here |
|---|---:|---:|
| L1 oracle-correct DC distance (mean) | 0.0464 | **0.046438** |
| L3 same-surface floor (mean) | 0.0325 | **0.032503** |
| L2 wrong-identity distance (mean) | 0.706 | **0.705927** |
| donor rows | 3,722 | **3,722** |
| recipient rows | 3,912 | **3,912** |
| spanning rows excluded | 229 | **229** |
| wrong-identity pool | 29,351 | **29,351** |

The row sets are byte-identical because they are IMPORTED from
`falsify_b2_edit.build_row_sets` rather than reimplemented. Every number
below is therefore on the same footing as the recorded DC result.

## 2. The measurement (dc_primary map; `headroom = L1/L3`, `discrimination = L2/L3`)

| tensor | space | L1 mean | L3 floor | L2 wrong | **headroom** | discrimination |
|---|---|---:|---:|---:|---:|---:|
| `_features_dc` *(falsified control)* | raw | 0.046438 | 0.032503 | 0.705927 | **1.429** | 21.72 |
| `_opacity` | activated | 0.240105 | 0.196262 | 0.582064 | **1.223** | 2.97 |
| `_opacity` | raw logit | 7.180769 | 5.711843 | 10.048712 | **1.257** | 1.76 |
| `_scaling_t` | activated | 0.330925 | 0.428751 | 0.469926 | **0.772** | 1.10 |
| `_scaling_t` | raw log | 1.936163 | 1.597236 | 2.797251 | **1.212** | 1.75 |
| `_xyz` | raw | 0.183719 | 0.168241 | 1.759687 | **1.092** | 10.46 |
| `_scaling` | activated | 0.018138 | 0.017516 | 0.026819 | **1.036** | 1.53 |
| `_rotation` | geodesic° | 80.34° | 68.52° | 72.28° | **1.172** | 1.05 |
| `_t` | raw | 6.221777 | 1.286162 | 7.323632 | 4.837 | 5.69 |

Payload-native row maps (matching rows in each tensor's own space rather
than by DC) do not rescue anything: `_opacity` reads **0.301** activated
and **1.919** raw, `_scaling_t` **0.026**, `_scaling` **0.754**, `_xyz`
**1.065**.

## 3. Verdict — the screen is NEGATIVE for every usable payload

**Not one transferable quantity clears the frozen screening rule
(`headroom ≥ 2.0` AND `discrimination ≥ 5.0`).** Every candidate sits
between **0.77 and 1.43** times the same-identity floor — that is, the
oracle-correct donor's value is about as far from the recipient's as two
rows of the SAME surface in the SAME episode are from each other, and in
two cases (`_scaling_t` activated at 0.772, `_scaling` raw at 0.966) it
is CLOSER than that floor.

**`_opacity` — CORRECTED.** The original text here claimed it has LESS
headroom than the falsified DC payload (1.223 against 1.429 activated).
**That is FALSE in the raw-logit space the edit actually operates on,
where opacity reads 1.9187 under its native map against DC's 1.4287 —
34% MORE (§6 C4).** What actually defeats opacity is (i) its
discrimination ratio failing the 5.0 floor by roughly 2× under EVERY map
(max 2.9657), so base opacity barely distinguishes a wrong-identity donor
from a same-surface one, and (ii) the ABSOLUTE distance behind that
ratio being 0.0297 logits (7.51e-05 activated) — rendered-negligible.
The frozen rule is scale-free and cannot see that difference.

**`_xyz` confirms the vacuity prediction empirically.** It DISCRIMINATES
identity well (10.46, and 35.40 under its native map — position tells you
which surface you are on) but has **no headroom at all** (1.092). That is
exactly the signature predicted from source inspection in
[[lrv3-fixture-hazards-2026-08-23]] §1: the object returns at the
identical world pose, so the donor's position is already the recipient's
position.

### `_t` passes the rule and is EXCLUDED — a degenerate pass, and the screen exposing it is the point

`_t` is the temporal mean. Donor rows live in episode 1 and recipient
rows in the return, so their `_t` differ by ~6.2 — **that difference IS
the designed episode separation**, not recoverable information.
Transferring it would move the recipient's temporal centre back into
episode 1 and thereby DELETE the return the payload exists to improve.

The numbers say the same thing independently: the wrong-identity link's
`_t` distance is **7.324** against the oracle-correct **6.222**, only 18%
apart, so `_t` carries almost no identity signal. Its apparent
discrimination of 5.69 is an artifact of a small same-episode floor
(1.286), not of the link being informative. **`_t` is not a payload.**

## 4. What this establishes, and what it does not

**Established.** On LRV3, with proposal ambiguity fully removed by an
oracle-correct link, **no payload — appearance, opacity, temporal
support, position, extent or orientation — has material headroom.** The
2026-08-20 DC falsification therefore generalizes: it was not evidence
that appearance is the wrong payload, it was evidence about **this
fixture**. This was pre-registered as the predicted outcome in
[[payload-headroom-spec-2026-08-23]] §6 before the cell ran.

**The mechanism, and it is now measured rather than argued.** LRV3's
returning surface is identical to the departing one in pose, colour and
texture, and its return is observed by 3 frames × 16 training cameras =
48 view-frames. The recipient rows are therefore wrong about NOTHING.
A consolidation payload can only recover what the recipient failed to
learn, so the headroom question is a question about **observation
supply**, not about which tensor is carried.

**NOT established.** That consolidation is dead as a concept. What is
refuted is that any payload can be demonstrated on an
observation-SUFFICIENT fixture. An observation-STARVED fixture — where
the returning surface is genuinely under-determined — remains the
untested case, and is now the design that the per-tensor numbers here
should inform rather than a design chosen blind.

**Also not established:** anything about N3V. No real-data claim is
licensed by this cell, and per the frozen rule no N3V B2 follows a
negative.

---

## 5. THE EDIT — the opacity payload is FALSIFIED, and more strongly than DC

The frozen spec required the edit to run REGARDLESS of the screen, so a
measured reconstruction delta could be set against a measured headroom
bound. Cell: Determined experiment **236** (`lrv3_falsify_opacity` r0,
commit `2ba6a62`, dgx, admitted V100 image, seed 0), COMPLETED. Report
`falsify_opacity_report.json` in experiment 209's run dir, sha256
`b5617f9cd8f12b03a2b0a51aea6bda3ad8717590b6b3709e93a8c73d8b1d61e1`.

**Anti-vacuity, established before any delta was read:** the tool's own
comparative gate PASSES — L1 pre-edit opacity distance **7.1808**
strictly exceeds the L3 same-surface floor **5.7118**, all 3,912
recipient rows change, and every link's reserved slot is satisfiable
(120 units in W1, 12 in WR, 8 per side used). The edit is non-vacuous by
the same rule the DC experiment used.

| link | reserved slot Δ (mean ± SE) | held-out `event_return` | certificate |
|---|---:|---:|---|
| **L1 oracle-correct** | **+6.588e-05 ± 3.237e-05** | 27.2181 → 26.0275 (**−1.1906 dB**) | REJECTED |
| L2 wrong-identity | +5.095e-04 ± 2.341e-04 | 27.2181 → 21.1718 (**−6.0463 dB**) | REJECTED |
| L3 same-identity no-op | +3.600e-05 ± 6.407e-06 | 27.2181 → 27.2284 (+0.0103 dB) | rejected |

**The oracle-correct opacity edit is ACTIVELY HARMFUL.** Its reserved
loss delta is POSITIVE — worse, not merely non-negative — and it costs
**1.19 dB** of held-out event return. Against the frozen promotion gate:
condition 1 (negative reserved loss) FAILS, condition 2 (≥ +0.5 dB)
FAILS by 1.69 dB, condition 3 (wrong identity rejected and harmful)
PASSES. **The payload is falsified**, and unlike DC — which was
indistinguishable from nothing at +0.008 dB — opacity does measurable
damage.

The certificate behaved exactly as designed throughout: it rejected all
three links, and it correctly identified the wrong-identity edit that
would have cost 6.05 dB.

### An instrument finding: the L3 placebo does NOT transfer across payloads

Frozen gate condition 4 requires the no-op to be NUMERICAL ZERO. **It is
not here** (+3.600e-05, +0.0103 dB), and the reason is structural rather
than a fault.

L3 maps donor rows to other donor rows of the SAME surface, paired by the
**nearest-DC** row map. For the DC payload that is a genuine placebo by
construction — a row redirected to its nearest-DC neighbour receives
almost its own value, which is why DC's L3 read ≈1e-11. For opacity it
is not: DC-nearness does not imply opacity-nearness, and two rows of the
same surface differ in activated opacity by 0.196 on average. So under a
nearest-DC map, L3 is a **same-surface, different-row opacity transfer**,
not a placebo.

That is still informative — it bounds the cost of moving opacity between
same-surface rows at ≈+0.01 dB, which isolates the −1.19 dB as a property
of the CROSS-EPISODE link rather than of opacity transfer as such. But
**gate condition 4 is inapplicable as written for this payload**, and is
recorded as inapplicable rather than reported as passed or failed. Any
future non-appearance payload needs a payload-native placebo (a
nearest-in-its-own-space map) if the numerical-zero condition is to mean
anything. Recorded as a required amendment to the certificate's control
set, not applied retroactively.

## 6. CORRECTIONS from a fresh-context adversarial review (append-only, 2026-08-23)

A fresh reviewer with no prior involvement recomputed all 24
(tensor × space × map) ratio pairs from the link means, re-verified both
report sha256s, and re-derived every certificate figure. **Everything it
could check numerically it confirmed** — all 24 ratios agree to 1e-9
relative, both hashes verify byte-exactly, each `pooled_mean` equals the
mean of its `side_means`, and the arithmetic "FAILS by 1.69 dB" is
exact. Its verdict on the headline was **STANDS WITH QUALIFICATIONS**.
The qualifications are real and are recorded here rather than absorbed
silently.

**C1 — "nothing there moved" (§ opening) is FALSE in three respects.**
The spec did move: (a) `_t`'s exclusion is an outcome the frozen §6 did
not enumerate; (b) gate condition 4 was reclassified as inapplicable;
and (c) **the spec's frozen no-pass consequence was dropped** — §5
required "record that consolidation currently has no useful payload, and
recommend the representation-only pivot", and this page recommended an
observation-starved fixture instead without recording the pivot.
**Recorded now, as the spec required: on the evidence of this block,
consolidation has no useful payload, and the representation-only pivot
is the recommendation the frozen rule calls for.** The starved-fixture
test is a pre-identified ALTERNATIVE, not a substitute for that
recommendation, and is pursued as a test of this page's own mechanism
claim rather than as a rescue.

**C2 — a mandatory control was NOT run, and its absence was undisclosed.**
Spec §4 lists `non_pointer_state_hash` unchanged across install→clear as
mandatory. `non_pointer_state_hash` is called only from
`scripts/consolidate_packets.py`; it appears nowhere in
`scripts/falsify_b2_edit.py`, and no such key exists in the report. The
"no parameter tensor was written" guarantee is therefore **unestablished
for the opacity arm**. The identical base PSNR across all three links
(27.21810902385908, three times) is weak corroboration only.

**C3 — the "isolates" claim is WITHDRAWN.** §5 stated that L3 isolates
the −1.19 dB as a property of the cross-episode link. It does not. L3
edits DONOR rows, selected with support ending at ≤ 5.0 s, while
`event_return` is measured on frames 57-59 = [9.5, 9.833] s — so **L3's
edited rows are largely invisible to the metric**, and the two links
differ in two variables at once. The comparison cannot attribute.
**The discriminating control that was NOT run is a within-recipient
random permutation** — the same 3,912 recipient rows, same marginal
opacity distribution, identity destroyed — which alone separates "the
opacity payload carries nothing" from "any opacity reshuffle over these
rows costs ~1 dB regardless of identity". Both remain consistent with
the evidence. Note also that a monotone damage-versus-magnitude curve
fits both L1 (0.240 activated → −1.19 dB) and L2 (0.582 → −6.05 dB), so
L2's correct rejection does not discriminate the two hypotheses either.

**C4 — "`_opacity` has LESS headroom than the falsified DC payload" is
FALSE in the space the edit operates on.** The redirect targets the RAW
LOGIT, and in raw-logit space under opacity's own native map `_opacity`
reads **1.9187 — 34% MORE than DC's 1.4287**, and only 4.1% short of the
frozen 2.0 floor. That number is printed in §2 and then contradicted by
the §3 sentence. **The correct argument, which is stronger:** opacity's
discrimination fails 5.0 by roughly 2× under EVERY map (max 2.9657), and
the absolute distance behind that 1.9187 ratio is **0.0297 logits**
(payload-native activated L1 mean **7.51e-05**, median 1.09e-07) — a
rendered-negligible difference. **The frozen screening rule is
scale-free and therefore cannot distinguish "1.92× of 0.03 logits" from
"1.92× of 3 logits". That is a genuine methodological weakness of the
rule**, and it produced both this misleading near-miss and the `_t`
false positive.

**C5 — `_t` is re-labelled an UNANTICIPATED THIRD OUTCOME.** The
exclusion is substantively correct but was framed retrospectively as
"the screen exposing it is the point". The decisive argument does not
need any measured PSNR: recipients are selected by `lo >= 9.3` and
donors by `hi <= 5.0`, so **copying a donor's `_t` into a recipient makes
that recipient satisfy the DONOR's membership predicate and removes the
row from the window the metric evaluates. That is deletion, not
transfer.** Two supporting numbers this page under-used: under the
payload-native map `_t`'s L2/L1 is **1.018 — 1.8% apart**, not the 18%
quoted; and `_scaling_t`, the gauge-invariant partner that survives a
time shift, has no headroom at all. **Required amendment to the rule:**
a frozen non-degeneracy precondition — a transferability check that a
payload cannot change a row's set membership, plus an absolute-magnitude
floor. Recorded as needed, not applied retroactively. A defect in the
SPEC is also recorded: §4 pre-committed `_opacity` as the edit target
unconditionally, so §3's "PREFERRED" selection clause was dead text at
freeze time and no screen outcome could have redirected any action.

**C6 — the conclusion is NOT threshold-independent.** A sweep shows both
thresholds do independent work (HR binds on `_xyz` and `_features_dc`;
DR binds on `_opacity` and `_rotation`), but at **(HR ≥ 1.5, DR ≥ 2.0)
`_opacity` passes while the falsified DC control still does not.** So a
modest, non-absurd reweighting admits a tensor. The conclusion survives
**empirically** — because opacity was edited anyway and failed at
−1.19 dB — not by threshold-independence, and that contingency is stated
here rather than left to look like robustness.

**C7 — `_rotation` was omitted from the payload-native paragraph, and it
is the third-largest foothold.** Its raw payload-native headroom is
**1.4892, above DC's 1.4287**. It is defeated by discrimination 1.9782
and, decisively, by the fact that under `dc_primary` its oracle-correct
link is **FARTHER than the wrong-identity link** (geodesic 80.34° >
72.28° > 68.52° floor): orientation carries anti-correlated identity
information on this fixture.

**C8 — the scope claim overreaches in two specific ways.**
(a) **`_features_rest` was never screened**, so "appearance" in this
result means degree-0 SH ONLY; view-dependent appearance is unmeasured.
(b) The metric covers ONE payload FORM — per-row L2 between two existing
tensor values under a nearest-row map. It is structurally blind to
joint/structured payloads (e.g. a RELATIVE support schedule rather than
absolute `_t`/`_scaling_t` values), to residual or variance-reducing
payloads that regularize rather than replace, and to capacity-freeing
payloads whose benefit is budget rather than value. **The evidence does
not rule out a payload helping on this fixture through a mechanism this
metric cannot see.** §4's "NOT established" paragraph conceded only the
observation-starvation axis.

**C9 — label corrections.** "The mechanism, and it is now measured
rather than argued" overstates: what is MEASURED is L1 ≈ L3 per tensor;
that this is *because of observation supply* is INFERENCE, and it is the
inference LRV4 exists to test. Separately, the commit, pool, image digest
and seed in the opening are **ledger claims, not report-verifiable
ones** — neither JSON contains a `commit`, `image`, `digest` or `seed`
key — and were presented alongside report-verifiable figures without
that distinction.

**C10 — smaller inaccuracies.** The "0.77 to 1.43" span holds for
`dc_primary` only; across payload-native the span is **0.026 to 1.919**
excluding `_t`. "Two cases below the floor" is two under `dc_primary`;
**five** sub-1.0 entries exist across both maps. §2's table omits
`_scaling` raw (0.9656 / 2.6822) and both `_rotation` raw and activated.
The exp-236 JSON carries a **naming defect** — fields named
`pre_edit_dc_distance_*` hold the OPACITY-logit distance and
`anti_vacuity.rule` still says "DC distance", although the gate was
correctly evaluated in opacity space (clearing at 1.2572×, LESS than the
DC arm's 1.4287×). Finally, "the return region contains no packet at
all" overstates `recipient_rows_with_packet: 0`; rows in the return
region outside the oracle sphere or the `lo >= 9.3` cut are not covered.

**C11 — L3 is significantly non-zero, and that is useful.** Its slot
delta is **+3.5997e-05 ± 6.4074e-06 = 5.62 SE from zero** — not
small-but-noisy. What it legitimately establishes is a **measurement
floor of ≈0.01 dB on `event_return`**, against which the −1.19 dB is
**~115× the floor and certainly not noise**.

**C12 — the DC reproduction is a PLUMBING check, not a footing
guarantee.** §1's "every number below is therefore on the same footing"
is wrong for a reason not stated there: for DC the `dc_primary` map IS
its own native map, so DC's 1.4287 is a best-case ratio, while every
other tensor's `dc_primary` column comes from a FOREIGN map. The
payload-native column is the like-for-like comparison — and there both
`_opacity` (1.9187) and `_rotation` (1.4892) exceed DC. The exact
agreement was guaranteed absent a defect (imported row sets, same
checkpoint, same oracle file, zero renders); it proves determinism and
no code drift, which is worth having, and nothing more.

## 7. THE PERMUTATION CONTROL RAN — and it OVERTURNS the attribution (append-only)

Correction **C3** named a within-recipient permutation as the only run
that could separate "the opacity payload carries nothing" (H1) from "any
opacity reshuffle over these rows costs ~1 dB regardless of identity"
(H2). **It was implemented and run.** Cell: Determined experiment **244**
(`lrv3_falsify_opacity_l4` r0, commit `662b3e2`, dgx, admitted V100
image), COMPLETED. Report
`falsify_opacity_l4_report.json`, sha256
`082e5c9737051a3157534d577cdb8e290018037f1d1aae27936d342ff9a4218e`.

**L4** permutes the recipient set within itself — identity destroyed,
temporal window preserved. One-hop is preserved by a row-index-parity
partition rather than bypassed, so only the edited half (1,917 of 3,912
rows, **49.0% of L1's edit volume**) is redirected. Permutation seed
7717, permutation sha256 `f59e01ab…`.

| link | pre-edit distance (logits) | reserved slot Δ | held-out `event_return` Δ | rows edited |
|---|---:|---:|---:|---:|
| L3 same-identity no-op | 5.7118 | +3.600e-05 | **+0.0103** | 1,869 (donors) |
| **L4 recipient permutation** | **7.1376** | +3.957e-05 | **−0.9685** | **1,917** |
| L1 oracle-correct | 7.1808 | +6.588e-05 | **−1.1906** | 3,912 |
| L2 wrong-identity | 10.0487 | +5.095e-04 | **−6.0463** | 3,912 |

**H2 is CONFIRMED and H1 is refuted AS AN ATTRIBUTION.** L4's pre-edit
distance (7.1376) is within **0.6%** of L1's (7.1808), and it costs
**−0.97 dB while editing HALF as many rows**. Per row edited, the
identity-destroying permutation is MORE damaging than the oracle-correct
link. **The −1.19 dB is therefore not a property of the cross-episode
identity link. It is a property of moving opacity by that much at all.**

**The three links whose edited rows the metric can actually see are
monotone in edit magnitude**, exactly as the reviewer predicted a
confounded design would be:

```
7.1376 → −0.97 dB   (identity destroyed)
7.1808 → −1.19 dB   (identity oracle-correct)
10.0487 → −6.05 dB  (identity wrong)
```

L3 sits off this curve at 5.7118 → +0.01 dB because it edits DONOR rows
whose support ends before the scored frames — the reason C3 withdrew it
as an attribution control in the first place.

### What this changes, and what it strengthens

**Weakened:** §5's characterization "actively harmful" was right about
the SIGN and wrong about the CAUSE. The damage carries no information
about identity correctness. Any reading of the −1.19 dB as evidence that
the oracle link is *wrong about identity* is withdrawn.

**Strengthened, and this is the more important direction:** the payload
negative is now stronger, not weaker. For opacity on this fixture,
held-out damage is a monotone function of how far the value is moved,
**independent of whether the move is correct**. There is therefore **no
regime in which redirecting opacity could help** — a correct link and a
random permutation of the same magnitude are indistinguishable in their
effect, and both are destructive. That is a cleaner refutation of the
payload than a null would have been.

**Unchanged:** the certificate rejected all four links, so nothing was
admitted and no promotion gate is in play. The frozen conclusion — no
useful payload, representation-only pivot recommended — stands and is
reinforced.

**Method note worth carrying:** the adversarial review identified this
control, it cost well under a minute of GPU, and it overturned a causal
claim in the page it reviewed. A control that separates *magnitude* from
*correctness* should be standard in any future edit experiment, not an
afterthought recovered by review.

## 8. Bookkeeping and an incidental observation

Claims consumed: `lrv3_payload_headroom` r0 (experiment 233) and
`lrv3_falsify_opacity` r0 (experiment 236). Cost ≈ 0.2 + 0.3 slot-h. Input hashes recorded in the report:
`configs/lrv3/b1_packets.yaml` sha256 `9085d3bd…` (matching experiment
209's manifest), `configs/lrv3/oracle_correct.json` sha256 `4d7d7d84…`.

Incidental, recorded because it bears on the ladder's zero-admit result:
of 149,800 rows the B1 packet column marks **2,255 rows across 103
packets**, and **zero recipient rows carry a packet id** — the return
region contains no packet at all on this fixture. That is consistent
with, and independent of, the round-1 funnel finding that the same
surface rarely produces two temporally disjoint well-trained packets.
