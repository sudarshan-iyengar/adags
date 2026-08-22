# LRV3 fixture — verified hazards that bound what any payload or timing
# experiment on it can claim (2026-08-23)

Append-only. Every item below was verified against primary source this
block and is recorded because each one can silently invalidate an
experiment built on this fixture. Nothing here retracts a prior result.

## 1. The returning object is IDENTICAL in pose, colour and texture — this bounds EVERY payload, not just DC

`EVENT_SPHERE_CENTRE = (0.70, 0.10, 0.35)` and
`EVENT_SPHERE_RADIUS = 0.20` are module constants
(`scripts/build_synthetic_reveal_scene.py:82-83`) applied identically
wherever `event_present` is true — the ray-trace at `:194` and the
albedo/normal assignment at `:221-223` — with **no time argument**.
Colour `(0.95, 0.45, 0.10)` and texture `stripe_band` are likewise
constants, and the lighting is static.

**Consequence, and it generalizes the 2026-08-20 DC falsification:** on
LRV3 the returning surface differs from the departing surface in
*nothing*. A geometry/pose payload transferred across the oracle link is
therefore an **identity transform — vacuous by fixture construction**,
not merely low-headroom. The DC payload's measured 1.4×-above-floor
pre-edit distance ([[b2-edit-falsification-2026-08-20]] §2) is not a
property peculiar to appearance; it is what this fixture does to any
transferable quantity.

The remaining way a payload could have headroom on a same-pose,
same-appearance return is if the recipient rows are **observation-
starved** — under-trained because too few views/frames observed the
return. With 3 return frames × 16 training cameras = 48 view-frames,
LRV3's return is not starved. **The headroom question is therefore a
question about observation supply, not about which tensor is
transferred.** That reframing is what the 2026-08-23 headroom screen was
built to measure rather than assume.

## 2. Fixture variants are blocked by frozen constants and by a tool guard

* `scripts/build_synthetic_reveal_scene.py` exposes exactly **six** CLI
  knobs (`:271-284`): `--out`, `--num-init-pts`, `--seed`, `--scene-id`,
  `--first-return-frame`, `--ground-half-extent`.
* `--first-return-frame` is hard-capped at **57** by an explicit
  admissibility check that raises `SystemExit` (`:290-292`), because
  episode 2 must clear `floor_len`. With `N_FRAMES = 60`, the shortest
  reachable return is therefore **3 frames**. A 1-frame return needs
  `f0 = 59` and is refused.
* `N_CAMERAS = 20` and `TEST_CAMERAS = (2, 7, 12, 17)` are module
  constants (`:41-42`); every camera renders every frame (`:308-326`).
  There is **no per-episode camera subsetting and no `--n-cameras`**.
* Independently, `scripts/falsify_b2_edit.py:596-600` refuses any fixture
  whose `return_frames` differ from `(57, 58, 59)`.

**Consequence:** an observation-starved variant cannot be produced by
re-invoking the frozen generator. It requires a NEW named fixture with a
declared different scene identity plus a relaxation of the tool guard —
each a new frozen specification, not a parameter change.

## 3. A 3-frame prose error in the LRV3 config headers — DO NOT "fix" it

`configs/lrv3/a1_local.yaml:38-41` and the headers of
`configs/lrv3/{a0,a1,a2}.yaml` state the object is *"genuinely absent
30-53, present again 54-59"*. **That is the LRV2 timing and it is wrong
for LRV3.** The fixture's own `data/synthetic/lrv3/event_spec.json`
records `episode_1 [0,29]`, `gap [30,56]`, `episode_2 [57,59]`, and
`return_frames [57,58,59]`; `configs/lrv3/oracle_correct.json` and
`falsify_b2_edit.RETURN_FRAMES` agree with the spec. Every EXECUTABLE
artifact is correct; only the prose is stale.

The hazard is specific and serious: the measured mistiming control found
a **2-frame** boundary error costs −2.39 dB, i.e. is worse than not
gating at all ([[lrv3-local-presence-corrected-cell-2026-08-20]]).
Anyone building a boundary estimator from the YAML header would target a
boundary **3 frames** wrong and would be inside the known-harmful
regime.

**These comments must NOT be edited.** A YAML config is hashed as RAW
BYTES by `canonical_config_hash` (`scripts/submit_apollo.py`, the
`.yaml` branch — YAML has no canonical form in this repo), so changing a
comment changes the config's content hash and breaks comparator identity
for every recorded run that used the file. The defect is corrected HERE,
in the durable record, and left in place in the file.

## 4. `non_pointer_state_hash` blind spots (recorded, partially repaired)

`scene/appearance_edit.py`'s `non_pointer_state_hash` covered
`_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation, _t,
_scaling_t, _route_logit, _motion_lora_coeff` and omitted
`_rotation_r`, `_motion_v`, `_motion_a`, `_motion_lora_basis`, every
`_motion_scaffold_*`, every `static_*`, and `_packet_ids`. A pure
pointer redirect never trips it, so no recorded result is affected — but
any payload that WRITES a parameter tensor would pass the "B2 is
byte-identical to B1 outside the pointer column" invariant blind. This
block extends the hash toward those tensors; the remaining omissions are
recorded here so the invariant's true scope is never overstated.

## 5. A units asymmetry in "temporal support" — unresolved, disclosed

The renderer and the CUDA kernel consume `exp(_scaling_t)` as a
**variance** (`marginal_t = exp(-0.5·dt²/σ)`,
`gaussian_renderer/__init__.py`, `cuda_rasterizer/forward.cu:432-435`),
while `falsify_b2_edit.effective_support` treats `exp(_scaling_t)` as a
**standard deviation** (`mu ± 2·exp(_scaling_t)`). The falsification
tool already labels its interval "an operational matching interval at a
declared cutoff, NOT exact temporal support", so no recorded claim
overstates it. **Any future support-state payload inherits this
ambiguity and must resolve it explicitly in its own frozen spec before
measuring anything.** Whether the asymmetry is intentional is
UNVERIFIED.

## 6. Infinite PSNR is not handled by the LRV evaluator

`scripts/eval_lrv1_event.py` appends `inf` directly into its per-frame
list and averages with `np.mean`, so `psnr_mean_per_frame` becomes
`Infinity` whenever the total gate renders exact absence — visible in a
real artifact (`ghost_gap` in the A1-LOCAL result). `psnr_pooled` stays
finite because pooling sums MSE first, which is why the pooled figure is
the designated primary. Side effect: the emitted JSON contains the bare
token `Infinity` and is therefore not strict RFC-8259. Recorded, not
repaired — the exact-absence result depends on this behaviour being
understood rather than silently "fixed".
