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
