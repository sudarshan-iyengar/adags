# FROZEN EXPLORATORY SPEC — corrected LOCALIZED episodic discriminator (LRV3 A1-LOCAL) (2026-08-20)

Status: **FROZEN before any cell produced an output.** EXPLORATORY,
`evidence_bearing: false` on every cell. Extends, and does not reopen,
[[lrv1-oracle-headroom-spec-2026-08-19]]: the 2026-08-19 negative
(`D1 = -5.2316 dB`) stands as recorded. This spec asks the follow-up
question that result licensed in its §15.6.

## 1. The question

> Once the representation swap is LOCAL (non-oracle rows keep the ordinary
> temporal marginal), the oracle is PER-PRIMITIVE, presence gates the TOTAL
> routed contribution, and routing pins are off — does correctly localized
> K=2 episodic presence reconstruct the LRV3 return better than the
> temporal substrate?

## 2. The four wiring defects repaired (all primary-verified in code, 2026-08-20)

1. **Global swap** — under `elgs_enable` the renderer replaced the temporal
   marginal with EL-GS presence for EVERY row (`gaussian_renderer/__init__.py`),
   deleting the learnable temporal lobe on ~149k rows to gain episodes on ~780.
2. **Voxel-cell oracle** — `seed_families` made a whole cell oracle if ANY
   point fell in the region (`elgs/trainer_hooks.py`), gating background off
   with the object (`ghost_gap` −6.56 dB).
3. **Static-twin leak** — presence multiplied only the dynamic branch;
   under soft routing every row also renders through a static twin at
   `base_opacity * static_probability`, which presence never touched. "Exact
   absence" was therefore false in the implemented representation.
4. **Route-init ordering** — `create_from_pcd` materialized `_route_logit`
   from the constructor default **4.0** before `training_setup` read the
   YAML, and `_ensure_route_and_motion_tensors` preserves correctly sized
   tensors. **Every fresh run in this repository trained from route logit
   4.0 (p_dyn ≈ 0.982) regardless of its YAML `route_logit_init`.** This
   corrects, append-only, the statement in
   [[lrv1-oracle-headroom-spec-2026-08-19]] §15.7 that the pinned oracle
   rows were "frozen at `route_logit_init: 0.0`" — they were frozen at the
   materialized 4.0. The pins' zero-gradient mechanics and everything else
   in that section stand.

Repairs landed at commit `a46580b` behind `elgs_local_presence` /
`elgs_routing_pins_enabled` (defaults preserve all historical semantics;
18 new tests in `tests/test_elgs_local_presence.py`, including anti-vacuity
tests that reproduce the old defect shapes). The route-init repair changes
fresh-run materialization ONLY where a YAML value ≠ 4.0 is now honored.

## 3. The cells

| cell | config | substrate | notes |
|---|---|---|---|
| **A0′** `lrv3_a0_prime` | `configs/lrv3/a0_local_control.yaml` | ADAGS temporal, `elgs_reserved_parity: true` | `route_logit_init: 4.0` EXPLICIT = the de-facto init experiment 177 actually ran. A0′ is a pure replicate of A0 modulo run nondeterminism, so it doubles as this scene's FIRST same-arm spread measurement. |
| **A1-LOCAL** `lrv3_a1_local_perprim_totalgate` | `configs/lrv3/a1_local.yaml` | EL-GS localized | K=2 correct oracle (`configs/lrv3/oracle_correct.json`, boundaries frozen `elgs_a_lr: 0.0`), per-primitive membership (~84 gated rows at seeding, 8 families ≤ preregistered 512 cap), total gate, pins off, rounds off. All other rows are UNASSIGNED (−1) and render as pure substrate. |

Dataset, event, boundaries, training units, reservation parity, schedule,
capacity policy/cap, seed 0, evaluator, masks and raster are unchanged from
the 2026-08-19 matrix. Image: the admitted digest `sha256:70a28e3d…`. Pool
`dgx`. The old A0 (experiment 177) is NOT reused as the comparator because
the route-init repair makes bit-identity a matter of argument rather than
proof; A0′ removes the argument for ~1.5 GPU-h and buys the spread number.

## 4. Decision rules — frozen before any output

Let all metrics be pooled PSNR from `scripts/eval_lrv1_event.py` on the
6000-iteration checkpoint.

* **Success**: `event_return(A1-LOCAL) >= event_return(A0′) + 0.5 dB`.
* **Locality admission** (all required for a positive to count):
  `ordinary_all`, `event_episode1`, and `ghost_gap` each within 0.5 dB of
  A0′.
* **Representation/substrate failure**: `event_episode1` or `ghost_gap`
  more than 1.0 dB below A0′ → stop and diagnose; do not tune.
* If close but non-positive: report "close, no further tuning"; a single
  replicate is the only authorized follow-up.
* **Small-mistiming control** (±1–2 frame shifted schedule) is authorized
  ONLY if the success rule fires; the prior maximally-wrong A2 schedule is
  not rerun.
* A0′ vs A0 (experiment 177): agreement within the admitted image's
  expected nondeterminism (prior expectation ~3.3e-4 dB from the N3V
  smoke, never measured on this scene) validates comparability with the
  2026-08-19 matrix; material disagreement is reported, not smoothed, and
  A0′ remains the comparator either way.

## 5. Cost

A0′ ~1.5 GPU-h; A1-LOCAL ~2.6 GPU-h projected (the presence gather now
loops over ~9 unique family ids instead of 512, so it should run faster
than the 2026-08-19 A1; measured rate to be recorded). Evals ~0.2 GPU-h
each. All `dgx`, all exploratory.

## 6. What this cannot establish

Everything in [[lrv1-oracle-headroom-spec-2026-08-19]] §9 carries over:
no statement about real-data event supply, evidence mechanisms, capacity
allocation, SOTA placement, or transfer. A positive here licenses exactly
one thing: the representation question moves from "wired to lose" to
"worth a real-data test", and the observation-born-lineage lane (Lane C/D)
gains a mechanism-level datum.

---

## RESULT (2026-08-20, append-only) — the corrected localized cell BEATS the temporal control, and absence renders EXACTLY zero

Every rule applied below was frozen in §4 before any cell produced an
output. Cells: A0′ = experiment 184 (`lrv3_a0_prime` r0), A1-LOCAL =
experiment 185 (`lrv3_a1_local_perprim_totalgate` r0), both commit
`b7952b0`, admitted image, dgx, 6000 iterations, COMPLETED in ~65 min
each (the 8-family presence gather removed the old ~4x EL-GS slowdown).
Scored by experiments 189/190 (r0) and re-scored with the per-frame
ghost diagnostic by 192/193 (r1, commit `1400b24`; region values agree).

### The numbers (pooled PSNR, 240 held-out views)

| region | **A0′** | **A1-LOCAL** | delta |
|---|---:|---:|---:|
| **`event_return`** | **27.1432** | **28.1928** | **+1.0496** |
| `event_episode1` | 29.6022 | 30.8449 | +1.2427 |
| `ghost_gap` | 29.0248 | 22.8333 | −6.1915 |
| `ordinary_return` | 28.2876 | 28.1735 | −0.1141 |
| `ordinary_all` | 28.3515 | 27.9660 | −0.3856 |
| `whole_frame` | 28.3703 | 28.0082 | −0.3621 |
| whole-frame SSIM | 0.92008 | 0.91989 | −0.0002 |
| primitives | 149,794 | **148,668** | −1,126 |

Per return frame (57/58/59): A0′ 26.75 / 29.94 / 28.28; A1-LOCAL
28.76 / 29.26 / 29.41 — **+2.0 dB on the hardest (first-return) frame**.
Provenance verified in the eval payload: `families 8`, `K {2:8}`,
`local_presence true`, `routing_pins_enabled false`, `gated_rows 6005`
(grown from 84 by clone/split inheritance), `a_lr 0.0`.

### The frozen rules, applied

* **Success rule: FIRES.** +1.0496 dB ≥ the 0.5 dB floor, at matched
  realized capacity (A1-LOCAL runs 1,126 primitives FEWER).
* **Locality admission:** `ordinary_all` −0.39 (within 0.5) PASS;
  `event_episode1` +1.24 (an improvement; the bound existed to catch
  degradation) PASS.
* **`ghost_gap` −6.19 → the diagnose branch fires**, and the per-frame
  diagnostic (below) resolves it in favor of the representation.

### The ghost diagnosis — EXACT absence, plus the two designed ramp frames

`ghost_gap_psnr_by_frame` (experiments 192/193):

* **A1-LOCAL renders frames 34–54 at literally infinite PSNR — exact
  zero error in the vacated footprint.** The total gate produces true
  absence; the temporal control can never do this (A0′ leaks Gaussian-
  tail energy at 38–48 dB on every gap frame).
* A1-LOCAL's entire pooled ghost deficit comes from frames **30 and 56 —
  the two DESIGNED smoothstep ramp frames at presence 0.5** (13.92 and
  13.47 dB), disclosed in the 2026-08-19 ANCHORS note before any cell of
  either block ran, plus small residuals on 31–33/55.
* **Consequence, an append-only correction to
  [[lrv1-oracle-headroom-spec-2026-08-19]] §15.3/§15.7 (M3):** the old
  A1's `ghost_gap` −6.56 dB was attributed to the voxel-cell oracle
  gating background off with the object. The per-primitive cell removes
  that mechanism entirely and ghost_gap barely moves (22.41 → 22.83), so
  the dominant cause was always the ramp frames, not the voxel oracle.
  The −5.23 dB headline of that page is untouched; only the ghost
  attribution is corrected.

### What this establishes and what it does not

**Established:** on the admitted LRV3 event, correctly localized
per-primitive K=2 episodic presence with a total-opacity gate
reconstructs the return **1.05 dB better** than the temporal substrate,
improves the continuously-present object region, renders exact absence,
and costs 0.39 dB on ordinary regions — at slightly lower realized
capacity. Against the 2026-08-19 global-swap A1 (22.04 dB at the
return), the localized wiring is **+6.15 dB**: the entire prior negative
was the wiring, exactly as its §15.6 held open.

**Not established:** that timing precision matters (the small-mistiming
control, experiment 191 `lrv3_a1_local_shift2` r0, was authorized by the
fired success rule and is running); anything about real data, event
supply, or evidence mechanisms (the oracle is authored); transfer beyond
this fixture. The `ordinary_all` −0.39 dB cost is real and unexplained —
candidate mechanisms (the ~6k gated rows' contribution to shared
surfaces, or plain run variation at this scene's same-arm spread, see
below) are not separated.

**Same-arm spread, measured for the first time on this scene:** A0′ vs
the 2026-08-19 A0 (experiment 177) is a pure replicate modulo
nondeterminism and one inert-code commit; the regions differ by 0.09 to
0.17 dB. The +1.05 dB success margin is ~6–8x that spread. The N3V-smoke
figure of 3.3e-4 dB does NOT transfer to this configuration.

---

## RESULT 2 (2026-08-20, append-only) — the small-mistiming control: 2 frames of timing error is worse than no gate at all

Authorized by the fired success rule; experiment 191
(`lrv3_a1_local_shift2` r0, commit `1400b24`), scored by 198. The cell is
byte-identical to A1-LOCAL except the K=2 program is translated 2 frames
EARLY (believed absence frames 28–54; per-frame presence verified on the
production path before submission; return frames 57–59 at exactly 1.0, so
the control CAN render the return). Disclosed: a translation preserves gap
length and total presence but moves 0.333 s of duration from episode 1 to
episode 2 (only a reflection preserves the multiset).

| region (pooled PSNR) | A0′ (no gate) | A1-LOCAL (correct) | **A1-SHIFT2 (2 early)** |
|---|---:|---:|---:|
| `event_return` | 27.1432 | 28.1928 | **24.7572** |
| `event_episode1` | 29.6022 | 30.8449 | 23.6906 |
| `ghost_gap` | 29.0248 | 22.8333 | 26.5612 |
| `ordinary_all` | 28.3515 | 27.9660 | 28.2625 |
| primitives | 149,794 | 148,668 | 149,825 |

```
A1-LOCAL − A1-SHIFT2 = +3.44 dB at the return
A1-SHIFT2 − A0′      = −2.39 dB at the return
```

**A 2-frame timing error does not merely erase the +1.05 dB gain — it
lands 2.39 dB BELOW the ungated substrate.** The ordering is now
measured across four cells of one matrix: correct gate (+1.05) > no gate
(0) > 2-frames-early gate (−2.39) >> maximally wrong gate (−17.16,
2026-08-19 A2). **Timing precision — not gate existence — is what
matters**, the exact distinction REVIEW ROUND 1's finding B2 said the
old A2 could not make.

The mechanism is legible in the per-frame profiles: SHIFT2's ghost
profile is clean (~46 dB) through the frames both programs call absent,
then collapses to **24.95 / 20.03 / 19.63 dB on frames 54–56**, where
the shifted gate asserts presence against true absence — real ghosting,
not a ramp artifact. Its return profile suffers most at first-return
frame 57 (23.14 vs A1-LOCAL's 28.76): the object rows spent frames 55–56
being supervised toward transparency against empty ground truth.
`event_episode1`'s −5.9 dB is concentrated where the shifted gate zeroes
the object on true-present frames 28–29.

**Limits:** one direction (early), one magnitude (2 frames), one seed —
this is not a dose-response curve. And the asymmetric-duration disclosure
above applies. What it licenses: any real-data hard-gating mechanism
needs frame-accurate boundaries; roughly-placed gates are NEGATIVE value
on this event class. Noted for the method lane: CCR's consolidation
deliberately leaves temporal support untouched (no gating), so it is
structurally immune to this failure mode; the constraint binds only a
future explicit-gate limb.
