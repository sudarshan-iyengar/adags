# A1 RESULT — the coverage bounding pair: a corrected instrument moves scissor and poker far above the floor

Date: 2026-08-18. Governing frozen design:
[[elgs-coverage-bounding-pair-design]], committed at `dad3360` BEFORE the
reducer existed. **DIAGNOSTIC ONLY.** No eligibility verdict, floor, census
figure or gate is changed by anything on this page, and G-OA's FAIL is not
reopened.

Run: Determined experiment **154**, cell `a1_coverage_bounding_pair` r0,
commit `ce4a81f`, digest-pinned image
`sha256:70a28e3d46a4768d595bda67328ddaacd18ae3af98ae863fc3f7303c159bb683`,
pool `dgx`, `evidence_bearing: false`, wall 8.0 min (350 s of reduction
across four sequences). Artifact
`runs/elgs/a1_coverage_bounding_pair_r0/bounding_pair.json`
(sha256 `0e1af8178b00af2860c1a019f40d9df374b67be939bec47bdc393eaebf7469c8`).

**Pool switch, recorded rather than silent.** `AGENTS.md` puts census and
diagnostic cells on `hopper`. All three `hopper` H100 slots were held by
another user (two `archaeology_pattern_similarity_training` trials and one
command) at submission time, so this CPU-bound reduction ran on `dgx`. Same
class of scheduling decision as M-2's, and with the same reasoning: the
reduction is deterministic and reads sealed artifacts, so the pool cannot
influence the result.

## 1. Contract checks — ALL PASSED, so the run is not void

| check | outcome |
|---|---|
| reading (i) numerator == sealed census `components_covered`, per sequence | **EXACT, all four** |
| denominator == sealed census `components_total`, per sequence | **EXACT, all four** |
| `coverage(i) <= coverage(iii) <= coverage(ii)` | holds, all four |
| unreadable masks | 0 |
| conversion names as named in the design | yes |
| `visible_duplicate_keys` | **0 in all four** — the last-wins dict semantics never had to arbitrate |

Reading (i) reproducing the sealed numerator exactly, on four sequences, is
the load-bearing check: it shows this reducer implements the frozen
instrument's own admission rule and that the two other readings differ from
it only by the rule change under study.

| sequence | sealed census sha256 (12) | covered/total |
|---|---|---|
| `poker_screen_w0_267` | `19f96fdc0b5c` | 3,237 / 8,464 |
| `scissor_screen_w0_561` | `9bd79e4dbfcd` | 8,147 / 18,478 |
| `put_candy_screen_w0_233` | `00bb9043401a` | 6,284 / 12,401 |
| `pour_tea_screen_w0_225` | `6af8ff6c6206` | 8,125 / 13,752 |

## 2. The bounding pair

26 tracking cameras per sequence. Denominator identical across readings.

| sequence | (i) frozen | (iii) anchor-agreeing | (ii) any-report | class |
|---|---:|---:|---:|---|
| `poker` | 0.38244 | **0.79631** | 0.83365 | **indeterminate** |
| `scissor` | 0.44090 | **0.85209** | 0.91590 | **indeterminate** |
| `put_candy` | 0.50673 | 0.72680 | 0.86146 | **eligible** |
| `pour_tea` | 0.59082 | 0.71030 | 0.79501 | **eligible** |

Sensitivity readings (design section 4):

| sequence | (iii) transposed anchor | (iii) last-defined anchor |
|---|---:|---:|
| `poker` | 0.48523 | 0.82502 |
| `scissor` | 0.49724 | 0.87174 |
| `put_candy` | 0.53205 | 0.76962 |
| `pour_tea` | 0.61584 | 0.74026 |

### The headline

**The absence-diagnostic page's prediction is confirmed, and by a wide
margin.** [[elgs-absence-diagnostic-result]] argued that coverage is
instrument-dependent and that under a corrected instrument the low-coverage
sequences would move UPWARD toward the 0.5 floor. Under the
component-membership admission rule the M-2 finding actually supports:

* **scissor 0.441 -> 0.852**, upper bound 0.916;
* **poker 0.382 -> 0.796**, upper bound 0.834.

Both clear the 0.5 floor by a margin larger than the distance they
previously fell short of it. The two sequences that failed cycle-2 on
coverage did so because of a `v >= 0.5` gate that M-2 showed carries no
confidence information, not because their foreground was untracked.

### The full 20-sequence classification

Reading (i) is available for all 20 from the sealed censuses. By the design's
monotonicity argument, reading (i) at or above 0.5 forces reading (iii) at or
above 0.5, so 16 sequences are `eligible` without any new computation:

| class | sequences |
|---|---|
| `eligible` (measured) | `put_candy` 0.727, `pour_tea` 0.710 |
| `eligible` (`by_monotonicity`, reading (i) >= 0.5) | `music_box` 1.000, `piano` 0.998, `chess` 0.936, `writing_2` 0.924, `maracas` 0.917, `writing_1` 0.918, `jenga` 0.896, `put_fruit` 0.887, `pan` 0.853, `keyboard_mouse` 0.853, `tambourine` 0.815, `xylophone` 0.779 (corrected conversion), `kindle` 0.779, `soda` 0.774, `tea` 0.743, `slice_apple` 0.731 |
| `indeterminate` | `scissor`, `poker` |
| `ineligible` | **none** |

Readings (ii) and (iii) were NOT computed for the 16 monotonicity rows; the
largest omission is `music_box` (2,867 frames), named in the design so the
omission stays visible. Pooled over the four computed sequences: (i) 0.48579,
(iii) 0.77721, (ii) 0.85876.

## 3. Why scissor and poker are `indeterminate`, and why the rule was not changed

Both have primary reading (iii) far above 0.5 — 0.852 and 0.796 — and both
are demoted because the **transposed-anchor sensitivity** lands just below
the floor at 0.497 and 0.485. The design's section 4 rule is explicit: if
either sensitivity crosses 0.5, the sequence is `indeterminate` "regardless
of its primary value". That rule was frozen and committed before the reducer
existed and **it has not been touched**.

**The diagnosis, stated separately from the class.** The transposed reading
is not a rival convention; it behaves as a NULL. Its admission rate is a
fraction of the primary's:

| sequence | invisible reports admitted, primary anchor | transposed | ratio |
|---|---:|---:|---:|
| `scissor` | 2,243,413 | 203,944 | **11.0x** |
| `put_candy` | 604,891 | 99,477 | 6.1x |
| `poker` | 799,112 | 133,763 | 6.0x |
| `pour_tea` | 300,453 | 67,038 | 4.5x |

Two plausible conventions applied to the same data would admit at comparable
rates. A factor of 4.5–11 says the row-major reading finds structure the
mirrored one does not, which is what the section 4 derivation predicted
before the numbers: the census prereg's own `frustum_containment` bounds the
first projected coordinate by `W - 1`, so the first coordinate IS the column
and the transposed lookup is simply wrong. What the transposed reading
measures is the rate at which a mirrored pixel lands in the same large mask
blob by coincidence — a decoy, and one that happens to sit near 0.5.

**Nevertheless the class stands at `indeterminate`.** Changing an outcome
rule after seeing the outcome is the one move this project's discipline
forbids outright, and the demotion is in the conservative direction. A future
preregistration may legitimately replace the transposed variant with a
designed convention probe; that would be a new instrument with its own
freeze, not an amendment of this one.

## 4. What follows for the audit population — and it matters more than the classes

Under the mandatory correction that `E_select = E_eligible` initially, and
that `indeterminate` sequences cannot enter a winning subset without a
separately frozen and signed adjudication rule, **scissor and poker are
excluded from `E_select`.**

That removes 452 of the 597 candidate windows — 75.7% of the population —
and it is the single fact that makes the stage-1 kill rule decidable at all.
See [[elgs-audit-prereg-2026-08-18]] section on power: with scissor, poker
and pour_tea all inside `E_select`, no possible stage-1 outcome can fire the
kill; with scissor and poker excluded, 2,580 of the 7,000 possible outcomes
fire it. The coverage classification therefore decides whether the audit is
worth running, which is not a relationship anyone designed and is worth
stating plainly.

## 5. Independent recomputation — NOT DONE ~~(SUPERSEDED by §8 the same day; preserved because it was true when written)~~

The design's section 8 requires reading (iii) to be recomputed by a
fresh-context worker from the frozen text alone, sealed before the primary
result is disclosed. **It did not happen.** Four separate attempts to launch
a fresh-context reducer failed immediately on transient API `529 Overloaded`
errors, and no independent implementation exists.

Consequences, stated rather than glossed:

* every figure on this page rests on ONE implementation;
* the section 1 contract checks are unaffected — they compare against sealed
  census artifacts produced by a DIFFERENT program (`build_m1_census.py`), so
  reading (i) is externally corroborated even without an independent reducer.
  Readings (ii) and (iii) are NOT;
* the specific risk that independence would have addressed is a shared
  misreading of the frozen text, which is exactly the class of error the M-2
  independent reduction caught in the form of an unstated axis convention;
* the recomputation remains OPEN and should be run before any figure here is
  used to define a population. Because the primary result is now known, a
  later independent run cannot claim the M-2 sealing order, and that
  weakening must be recorded with it.

Termination per the design is therefore NOT reached: classes exist for all
20 sequences and the contract checks passed, but section 8 is outstanding.

## 6. What this does not establish

* **Physical presence or absence.** A `v == 0` report agreeing with its
  anchor means the tracker's reported pixel and its own consensus point fall
  in the same mask component. Nothing here observes the scene. The frozen
  M1-A0b audit sample (73 windows) remains emitted and unrun.
* **That scissor or poker should be admitted to evidence use.** They are
  `indeterminate`, and admission needs a fresh preregistration under a
  corrected instrument in any case.
* **That the corrected instrument is CORRECT.** It is better motivated than
  the `v >= 0.5` gate M-2 falsified, and it is not validated. Reading (ii)
  admits any report at all, including grossly drifted ones, which is why it
  is labelled an upper bound rather than a candidate rule.
* **Anything about the occlusion supply** (239,545), untouched here.
* **G-OA's FAIL**, not reopened.

## 7. Consumed claim indices

```
a1_coverage_bounding_pair   r0 (exp 154)   -> next free r1
```

---

## 8. CROSS-CHECK COMPLETE — the two reductions agree on every cell, and §5 is superseded

Determined experiment **161** (`a1_indep_recompute` r0, commit `8898b0d`,
admitted image, `dgx`) COMPLETED. Artifact
`runs/elgs/a1_indep_recompute_r0/indep_bounding_pair.json`
(sha256 `32ca8f512ee1b058e9432ce4f1d58a6cb893a31f31958caca0912b95252d3ef1`).

`scripts/indep_coverage_recompute.py` was written by a fresh-context worker
from the frozen design text alone, forbidden from reading the primary reducer,
its tests, this page, or any `bounding_pair.json`. It shares **no code** with
the primary: it labels components with `scipy.ndimage.label` rather than
`cv2.connectedComponentsWithStats`, and imports none of the primary's
functions.

### Every comparable quantity, both reductions

| sequence | denominator | (i) frozen | (ii) any-report | (iii) anchor | (iii) transposed | (iii) last-defined | class |
|---|---:|---:|---:|---:|---:|---:|---|
| `poker` | 8,464 | 3,237 | 7,056 | 6,740 | 4,107 | 6,983 | indeterminate |
| `scissor` | 18,478 | 8,147 | 16,924 | 15,745 | 9,188 | 16,108 | indeterminate |
| `put_candy` | 12,401 | 6,284 | 10,683 | 9,013 | 6,598 | 9,544 | eligible |
| `pour_tea` | 13,752 | 8,125 | 10,933 | 9,768 | 8,469 | 10,180 | eligible |

**These are the values from BOTH reducers.** 28 compared cells — four
denominators, twenty numerators, four classes — and **zero differences**. The
independent reducer's own contract status is `ok` on all four sequences.

That is the stronger outcome, and it is worth naming what it rules out: the
worker's report listed nine "residual definitional freedom" items, three of
which could plausibly have produced a divergence — it treats a transposed
anchor falling outside a non-square mask array as out-of-domain, it reads
`v == 0` as exact float equality rather than `v < 0.5`, and it does **not**
clamp the report pixel into the mask raster (the primary does, in order to
reproduce `build_association` exactly). None of the three moved a single
integer on this data, which localises the agreement rather than merely
asserting it.

### §5 is SUPERSEDED, but its caveat about sealing order is NOT

§5 recorded that the independent recomputation had not happened and that every
figure rested on one implementation. **That is now false and the section is
superseded.** What survives from it, and must travel with this result:

* **the sealing order is weaker than M-2's.** M-2's independent reduction was
  sealed BEFORE the primary returned. Here the primary result already existed
  when the independent implementation was written, because four earlier launch
  attempts died on transient API errors. The worker never saw the primary
  result, the primary code, or this page, so **independence of the reduction
  holds**; the property that the independent side committed first does not;
* agreement confirms the **REDUCTION, not the inputs.** Both reducers read the
  same tracks and the same masks. If those were wrong, both would be wrong
  together — which is exactly how the 2026-08-13 substrate defect survived an
  "exact" recomputation. The guard against that is §1's contract checks against
  the sealed census artifacts, produced by a **third** program
  (`build_m1_census.py`), and they passed on all four sequences.

**The design's termination condition (§10) is now met:** classes exist for all
20 sequences, the contract checks passed, and the independent recomputation
agrees.
