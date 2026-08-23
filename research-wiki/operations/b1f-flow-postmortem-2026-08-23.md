# POST-MORTEM — the flow null is EXPLAINED, and two premises on the
# record are corrected (2026-08-23, block 2)

EXPLORATORY, `evidence_bearing: false`. **Zero GPU-hours** — artifact
forensics only, from the six terminal arms of
[[b1f-flow-screen-result-2026-08-23]] (237/238 plain B1, 239/240 B1-F,
254/255 B1-X). Nothing here reopens the rejection; the prior stays
REJECTED and the honest reading stays "everything is noise".

## 1. VERDICT: EXPLAINED

**The measurement channel could not have carried the effect regardless of
how good the control was.** The substrate's own trajectory-divergence
floor is established by densification 250–500 iterations **before** the
first birth event, and it is the same size as — or larger than —
anything the intervention subsequently produces.

Plain B1 and B1-F differ by exactly two config lines
(`packet_birth_flow_init`, `packet_birth_flow_source`; verified by
`diff`), same commit, same digest-pinned image, same seed, same source.
They therefore execute identical numerical code until the first birth at
iteration 1000. Relative RMS difference of the per-iteration training L1:

| window | 1–250 | 251–500 | **501–750** | 751–999 | 1001–1250 | 3001–4000 | 5001–6000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1-F vs B1-X, s0 | 0.0001 | 0.0003 | **0.0886** | **0.1100** | 0.0937 | 0.0780 | 0.0958 |
| B1-F vs B1-X, s1 | 0.0001 | 0.0008 | **0.1289** | **0.2364** | 0.0608 | 0.2504 | 0.2231 |

Iteration 1 is bit-identical across all three arms of a seed. Divergence
appears at iteration 2–3 at the 1e-8 level — float nondeterminism, not an
RNG difference, which would be O(1) immediately. At `densify_from_iter:
500` the divergence jumps **1400×** (1.28e-5 → 1.80e-2): a threshold
comparison flips, a clone/split decision differs, and by iteration 999
the point counts already differ by ~140.

**For the decisive B1-F vs B1-X pair the PRE-intervention separation
(0.089–0.236) is statistically indistinguishable from the
POST-intervention separation (0.061–0.315), on both seeds.** Endpoint
renders corroborate: mean |Δ| between arms (1.76–2.45 8-bit levels) is
the same size as between seeds of the same arm (1.79).

## 2. METHOD, and it generalizes past flow

**At this protocol the substrate is chaotic: identical code, identical
seed and identical config separate to ~0.09–0.45 relative RMS training
loss before iteration 1000, and densification is the amplifier.** The
block's measured "0.635 dB global / 0.341 dB event union" seed spread
therefore **understates** the resolution floor, because *same-seed
same-code* runs separate comparably.

**Any 6k/50-frame comparison in this family must state a same-code
replicate floor, not only a seed spread.** This does not retract the
0.635/0.341 figures; it says they are a lower bound on what two runs can
resolve.

## 3. CORRECTION — the camera-swap control is only ~84% decorrelated

The frozen swap rule takes the next camera id, and consecutive N3V ids
are physically adjacent: median baseline **0.44 world units** against a
~5-unit scene distance, and **13 of 14** swap events read the adjacent
camera (the exception is a cam20→cam01 wrap).

Measured against the correct field at the same pixel index — which is
what the code reads:

| correct → swapped | β = Σ⟨f_c,f_s⟩/Σ|f_c|² |
|---|---:|
| cam13_0035 → cam14_0035 | 0.345 |
| cam02_0004 → cam03_0004 | 0.214 |
| cam16_0025 → cam17_0025 | 0.020 |
| cam20_0044 → cam01_0044 (wrap) | 0.061 |
| *spatially shuffled (content-free)* | **0.005–0.011** |

**The "wrong flow" arm delivers ~16% of the correct velocity along the
correct direction — 16–30× more than a content-free control.** It is
well matched on magnitude (sites 39,162/39,141; valid 96.84%/97.57%;
mean event speed 0.0189/0.0208) but only partially decorrelated in
direction.

**Consequence, stated precisely:** gate 3 measures at most ~84% of the
flow-information contrast. Un-attenuating the −0.0952 dB gives ≈ −0.11
dB, still 3× inside the 0.341 dB union seed spread — **so the REJECTION
STANDS**, but gate 3's FAIL should not be cited as decisive. A properly
decorrelated control would be a temporal shuffle or the within-recipient
spatial permutation this block already carries as METHOD.

## 4. CORRECTION — the birth-gating "dynamic mask" is a residual quantile,
## not a motion mask

**Verified against primary storage and source.**
`data/n3v/cut_roasted_beef/motion_priors/` contains exactly one
subdirectory, `masks/`, and it contains exactly 300 files, all
`cam00_*` — **the HELD-OUT camera**. There is no `seg/`,
`dynamic_masks/` or `foreground_masks/`.

`utils/motion_prior_utils.py:401-421` looks for a mask file by
`camera.image_name`, then a panoptic seg, and only then falls through to
the residual branch. Both ladder configs set
`dynamic_mask_from_residual: true`, `dynamic_mask_residual_quantile:
0.85`, `dynamic_mask_dilate: 2`.

**So for every TRAINING camera the "dynamic mask" gating birth sites is
the top-15% photometric-residual pixels of the current render**, with
sites then sampled ∝ residual². The site rule is motion-agnostic by
construction on this scene.

**This falsifies a premise on the record.**
`scene/packet_birth_flow.py`'s opening rationale — that birth sites are
"already restricted by the dynamic mask … so a flow-gated site rule would
be largely redundant" — **does not hold on `cut_roasted_beef`**.
Flow-gated site selection was rejected on inspection using a premise that
is false for this scene. That does not resurrect it: the §1 finding says
the channel cannot resolve the effect either way. Recorded because the
premise is cited elsewhere and is wrong.

Consistent with this, birth sites are only mildly motion-enriched:
**1.910%** of B1-F birth sites exceed their own view's 99th-percentile
flow magnitude against **0.986%** for B1-X and 1.000% at chance
(χ² = 117, 1 dof) — a real ~1.9× enrichment that the swap destroys
exactly, but sub-pixel in absolute terms (birth-site flow ≈ 0.08–0.24
px/frame, INFERRED via a depth assumption).

## 5. The remaining questions, answered or bounded

* **Newborn survival / washout — PARTIAL.** Injected coefficient norms
  (0.019–0.076) are 40–100% of the population mean at the same
  iterations, so the initialization is not negligible in parameter space;
  at iteration 5751 both flow arms sit +3.7%/+4.9% above plain B1 at seed
  1, a persistent trace that **does not distinguish correct from wrong
  flow** (B1-X ends higher). Surviving packet rows at 6k are 9,133–10,249
  across all six arms with no arm-specific attrition. **Per-row survival
  of the initialized velocity direction is NOT RECOVERABLE** without
  `_motion_lora_coeff` from the 536 MB per-arm checkpoints — present, not
  absent, and deliberately not pulled.
* **Early post-birth effects — none detectable.** There is no held-out
  evaluation before iteration 6000 (`test/psnr` has exactly one point).
  The 20-iteration post-birth B1-F−B1 delta is negative at 14/14 events,
  but against a null built from the pre-intervention window all 28
  observations sit at |z| ≤ 2.22 and 25/28 at |z| < 1.1.
* **Mask insensitivity — NO, and this is checked independently.** The
  frozen 0-49 union covers 0.3298% of the scored volume and receives
  0.239–0.447% of the squared arm-difference energy — proportional, not
  blind. Decisively, `test/dynamic_mask_psnr`, computed on cam00's real
  precomputed mask, reproduces the same null at the same size (B1-F − B1
  = −0.097 / −0.008). **Mask curation does not explain the result.**

## 6. Scope

This concerns **flow-derived velocity initialization for relocated
packets at the 50-frame, 6k protocol** only. "EXPLAINED" means the
measured null is fully accounted for; it does **not** establish that the
effect is zero, and it does not bear on persistent rendered-flow
supervision or multi-view 3D trajectory initialization, neither of which
was tested.

**Incidental, recorded not chased:** `test/track_flow_l1` is 0.06124
identically across all six trained models — an inert metric worth its own
look.
