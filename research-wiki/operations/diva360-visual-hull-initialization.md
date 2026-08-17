# FROZEN — temporal-union visual-hull initialization (2026-08-16)

**Status: FROZEN BEFORE THE INITIALIZER WAS RUN.** Recorded and committed
prior to building a single hull, per the directive's rule for this lane
("Before examining its outcome, freeze: the temporal keyframe-selection
rule; training cameras used; mask threshold; hull-agreement rule;
sampling rule; point count; random seed; failure behavior").

Implementation: `scripts/build_visual_hull_points.py`. EXPLORATORY
throughout; nothing here is claim-grade.

## Why this lane exists

The converter synthesizes `points3d.ply` by sampling uniformly inside the
union of every camera frustum, and its own docstring calls that volume
"a coarse smoke-test volume, NOT a claim-grade initialization". Measured
extent about +/-6.5 world units against scissor content at about +/-1.2.

Experiment 84 pruned **20,000 -> 3,398 -> 3,254** points by iteration 990
before densification recovered: roughly 84% of the initial cloud
destroyed as empty space.

The frozen sweep already tested the naive reading of that — S1 put
200,000 seeds in the SAME volume and **lost 3.976 dB**
([[operations/diva360-scissor-sweep-matrix-v1]]). That result sharpens
the diagnosis rather than weakening it: filling a wrong volume more
densely is actively harmful, so the defect is the VOLUME, not the seed
count. The sweep matrix said so in advance and deliberately excluded a
content-aware initializer as "a new component ... this sweep is not the
place to debut one". This is that component, run on its own.

## The frozen specification

| element | frozen value | note |
|---|---|---|
| temporal keyframes | **8** | `np.linspace(0, N-1, 8)` rounded, over the common frame-index list |
| keyframe rule | by POSITION only | never by content, motion, or any hull outcome |
| cameras | **all 35 TRAINING cameras** | see below |
| mask threshold | **> 127** | `build_elgs_tracks.load_mask`, unchanged |
| hull min observers | **3** | `TracksConfig.hull_min_observers`, unchanged |
| hull mask agreement | **0.9** | `TracksConfig.hull_mask_agreement`, unchanged |
| voxel grid | **96^3** | `TracksConfig.hull_resolution`, unchanged |
| grid bounds | **0.5 x camera-ring radius**, centred on the camera-centre centroid | `hull_bounds_scale`, unchanged |
| temporal rule | **UNION** over keyframes | a voxel qualifies if it is in the hull at >= 1 keyframe |
| erosion | **NONE** | see below |
| point count | **exactly 20,000** | MATCHED to the existing initialization |
| sampling | stride `(i * n_union) // num_points`, then uniform jitter inside the voxel cell | see below |
| seed | **0** | the converter's own |
| colours | uniform random uint8, same RNG stream | see below |

### Cameras — what is and is not excluded

All 35 training cameras are used. This deliberately does NOT apply the
extra `cid % 4 == 0` reservation that
`build_elgs_tracks.load_temporal_scene` imposes (which would leave 26).
That reservation governs what the TRACKER may observe in the evidence
lane; an initializer that feeds training may use every camera training
already sees.

The six OFFICIAL held-out cameras (`0, 16, 17, 33, 43, 44`) are a
different matter and are excluded BY CONSTRUCTION, not by a hardcoded
list: the script reads `transforms_test.json` and REFUSES if any camera
id appears in both splits. No held-out image, mask or pose can reach the
initialization, and no held-out metric is consulted anywhere in this
lane.

### No erosion — the one deliberate departure from `build_hull_seeds`

`build_elgs_tracks.build_hull_seeds` keeps only the hull SURFACE (it
erodes and takes `hull & ~eroded`) and caps at 512 seeds, because a
tracker wants surface points to track. An initializer wants the VOLUME:
Gaussians must exist inside the object, not only on its silhouette
boundary. The erosion step is therefore not applied, and the cap does not
apply either. Everything else — the projection, the mask lookup, the
observer/agreement test, the grid construction — is the same code.

### Sampling and jitter

Points are assigned to union voxels by the deterministic stride
`(i * n_union) // num_points`, which spreads over the voxel list when
there are more voxels than points and repeats each voxel evenly when
there are fewer, so the rule is total and needs no branch.

Each point is then jittered uniformly inside its voxel cell. This is not
cosmetic: `scene/gaussian_model.py::create_from_pcd` sets the initial
per-Gaussian scale from `distCUDA2`, the nearest-neighbour distance, so
an unjittered lattice would hand every Gaussian the same degenerate
initial scale.

### Colours are held identical on purpose

Colours are drawn uniform-random uint8 from the same RNG, which is the
converter's own rule. Content-aware colour (sampling the observing
cameras' pixels at each point) is deliberately NOT done here: it would be
a SECOND change and this comparison must isolate initialization
GEOMETRY. Colours are not inert — they seed the SH DC term through
`RGB2SH` — so keeping them identically distributed to the baseline's is
precisely what makes a difference attributable to geometry.

## Failure behaviour (all fail-closed)

A camera present in both splits; a missing mask file; an EMPTY hull at
ANY keyframe; an empty union; a degenerate rig (coincident camera
centres); a keyframe count exceeding the common index set; a non-empty
output directory; an output directory inside the git repository; a
non-positive point count. Each raises `ContractError` and writes nothing.

## The comparison, also frozen

Against the existing 20,000-point initialization with **the same scene,
the same split, the same training budget, the same point cap, the same
losses and the same evaluation**, changing nothing else. The baseline is
experiment 84's configuration; the hull cell differs only in which
`points3d.ply` the scene directory points at, and the scene directory is
otherwise a set of relative symlinks to the same images, masks and
transforms.

To be reported: initial spatial bounds, initial mask agreement,
per-keyframe hull voxel counts, union occupancy, points per union voxel,
early pruning, point-count evolution, convergence speed, final metrics,
and qualitative failure modes.

## Companion: the FROZEN per-frame Gaussian oracle

**FROZEN BEFORE ANY ORACLE TRAINING RAN.** Implementation:
`scripts/make_diva360_frame_slice.py`.

The oracle asks one question: how well can THIS renderer and THIS
initialization fit a scissor frame when the temporal representation is
removed entirely? Each selected frame becomes an independent static
multi-view problem — 35 training views of one instant, 6 official
held-out views of the same instant.

| element | frozen value |
|---|---|
| frames | **8**, `np.linspace(0, N-1, 8)` rounded over the sorted frame indices |
| resulting indices | **0, 80, 160, 240, 320, 400, 480, 560** |
| cameras | untouched: whatever the splits already declare (35 train / 6 held-out per slice, verified) |
| initialization | **visual hull**, inherited by symlinking the hull scene's `points3d.ply` |
| renderer / config | experiment 84's `diva360_scissor_bench30.yaml`, UNCHANGED |
| metrics | the official conventions (`--official-metrics`) |

The frame rule is deliberately the SAME rule the hull keyframes use, so
the oracle frames and the hull keyframes COINCIDE and the two lanes
cannot drift apart. `time` is not renormalized on a slice: one timestamp
makes any rescaling arbitrary, so a slice differs from its source in
exactly one way — which frames are present.

**Why the config is unchanged rather than tuned down.** The directive
requires "the same renderer", and reusing experiment 84's config
verbatim is the only way to guarantee that no part of a gap is a
renderer or schedule difference. It also makes the oracle a genuine
upper bound rather than a like-for-like run: 6000 iterations at batch 4
is ~686 epochs over a 35-image slice against ~4.9 epochs over the
4,935-image window, so the oracle is optimized two orders of magnitude
harder per image. That asymmetry is the point.

**Ordering, and why the first frame is not a choice.** The eight are run
in frozen index order, starting at frame 0, and frame 0 is run first
because it is first — not because of anything observed about it. Its
hull is the smallest of the eight (2,301 voxels against ~7,000-8,000 for
the rest), so it is if anything the least favourable starting point.
Per-frame wall cost is MEASURED on it before committing to the remaining
seven, rather than estimated.

### Interpretation, fixed in advance

* strong per-frame results with weak dynamic results ⇒ the temporal
  representation or the motion optimization is implicated;
* weak per-frame results ⇒ initialization, rendering, training or
  evaluation is implicated BEFORE any temporal question.

Frame selection is never revisited after seeing a result. If fewer than
eight frames complete, the completed subset is reported as a subset with
its size stated, and the frames are NOT re-chosen.

---

## RESULTS (appended after all runs completed; everything above unchanged)

EXPLORATORY throughout. Metrics are `main.py`'s internal conventions
(float PSNR, 3DGS SSIM) on the six OFFICIAL held-out cameras — the same
convention and split as experiment 84, so the comparisons below are
like-for-like even though none of them is comparable to a published row
(see [[operations/diva360-protocol-parity-audit]]).

### 1. The visual hull HELPED — experiment 104

Experiment 104, commit `a824ae3`, dgx/V100, config
`diva360_scissor_bench30.yaml` UNCHANGED, differing from experiment 84
in `--source_path` alone.

| | best PSNR | best SSIM | best iter | points |
|---|---:|---:|---:|---:|
| **hull init** (exp 104) | **22.4553** | **0.92022** | 6000 | 416,599 |
| frustum init (exp 84) | 21.3705 | 0.90698 | 5000 | 338,528 |

At the SAME iteration, which is the cleanest reading because it removes
the differing best-iteration:

| iteration 6000 | PSNR | SSIM | points |
|---|---:|---:|---:|
| hull | **22.4553** | **0.92022** | 416,599 |
| frustum | 21.3079 | 0.90488 | 507,178 |
| delta | **+1.147 dB** | **+0.0153** | **-90,579** |

**+1.15 dB with 90,579 FEWER points.** The initialization diagnosis was
right and the fix works. Two supporting observations:

* The catastrophic early prune is GONE. Experiment 84 collapsed
  20,000 -> 3,398 -> 3,254 by iteration 990, destroying ~84% of its
  cloud as empty space; the hull run grew monotonically (46,244 by 1430,
  131,979 by 3500, 258,811 by 4890). Seeding 2.6% of the frustum
  volume — the measured ratio — is what removed it.
* The hull run's best is at 6000, i.e. it was STILL IMPROVING at the
  end, where experiment 84 peaked at 5000 and declined. So this is a
  lower bound on what the initialization buys, not a converged value.

This does NOT retract the S1 finding
([[operations/diva360-scissor-sweep-matrix-v1]]): 200,000 seeds in the
frustum volume lost 3.976 dB. Both results say the same thing — the
defect was the VOLUME. Filling a wrong volume more densely hurt; seeding
the right volume at the SAME 20,000 points helped.

### 2. The per-frame oracle — all eight frames

Experiments 105-112, one per frozen frame, identical commit
(`fb4c349`), config hash (`4a254b3555…`), archive hash, image, pool,
seed and slots; only run dir and `--source_path` differ. All on
dgx/V100, none on hopper.

| frame | best PSNR | best SSIM | best iter | points at best | final PSNR | final points |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27.4982 | 0.95968 | 2000 | 37,202 | 27.2817 | 70,542 |
| 80 | 25.1066 | 0.93515 | 2000 | 48,791 | 24.9659 | 100,344 |
| 160 | 25.1331 | 0.93870 | 2000 | 48,879 | 24.8008 | 95,852 |
| 240 | 25.8909 | 0.94216 | 2000 | 47,738 | 25.8152 | 94,719 |
| 320 | 25.7729 | 0.94652 | 2000 | 42,826 | 25.6661 | 80,426 |
| 400 | 26.4760 | 0.95258 | 3000 | 58,651 | 26.4047 | 82,717 |
| 480 | 24.9357 | 0.93984 | 3000 | 65,095 | 24.7756 | 94,153 |
| 560 | 25.0116 | 0.94006 | 3000 | 71,644 | 24.9201 | 111,237 |
| **mean** | **25.7281** | **0.94434** | | | | |
| median | 25.4530 | | | | | |
| excl. frame 0 | 25.4753 | 0.94214 | | | | |

**Frame 0 is an outlier and the freeze predicted the reason.** It scores
27.50 against 24.94-26.48 for the rest, and it carries the smallest hull
of the eight (2,301 voxels against ~7,000-8,000). The freeze called it
"least favourable" — that was right about the HULL and backwards about
the ORACLE: least content is the easiest frame to fit. The mean
excluding it (25.475) is reported alongside the mean including it
(25.728) so the conclusion does not rest on which is used. It does not.

### 3. The decomposition — the gap is now quantified end to end

Every step below uses the same renderer, the same 35/6 split and the
same metric convention, and the last step holds initialization constant
at the hull:

```
frustum init, dynamic      21.308      (exp 84 @ 6000)
hull init, dynamic         22.455      +1.147   <- initialization
hull init, PER-FRAME       25.728      +3.273   <- temporal representation
                                                   (mean of 8)
published PF I-NGP         25.346
published MixVoxels        25.090
```

**The per-frame oracle reaches published parity** — above PF I-NGP on
PSNR (+0.38) and level on SSIM (0.9443 vs 0.944) — and still does so
when frame 0 is excluded (25.475). Per the interpretation frozen before
any oracle ran, this is decisively the first branch: **the renderer, the
hull initialization, the training loop and the evaluation are
collectively capable of published-parity scissor quality at a single
instant, so the residual ~3.3 dB is NOT in that machinery.**

**The confound, stated plainly.** The oracle is not only free of the
temporal representation, it is optimized far harder per image: 6000
iterations at batch 4 over a 35-image slice is ~686 epochs against ~4.9
epochs over the 4,935-image window. So "+3.273" attributes to *temporal
representation OR motion optimization OR per-image optimization
exposure*, and this experiment cannot separate the third from the first
two. Experiment 101 — the unchanged 15k continuation at frozen capacity
— is the direct test of that third limb, and it is exactly why that lane
matters now: if 10,000 further iterations at fixed capacity buy little,
exposure is not the explanation and the temporal representation is.

### 4. Capacity overshoot, now seen in ten runs

Every one of the eight oracle frames PEAKED before its final iteration —
five at 2000, three at 3000 — and every one DECLINED by 6000 while its
point count roughly doubled (37k-72k at best, 70k-111k at the end).
Experiment 84 showed the same shape at a different scale, and the frozen
sweep's S2/S3 cells found it on the capacity axis. This is a
densification-schedule problem on this scene that is INDEPENDENT of the
temporal question and independent of the initialization, and it is now
observed under every combination of the two tested so far.

Note that it did NOT appear in the hull dynamic run (exp 104 peaked at
6000), which is consistent with the overshoot being driven by how much
capacity is added relative to how much the model still needs, rather
than by an absolute point count.

## Recorded in advance: what a negative result would and would not mean

If the hull initialization does not help, that does NOT restore the
frustum volume as a good prior — S1 already showed the frustum volume is
bad. It would instead say that this model recovers from a bad
initialization through densification, which is a different and useful
finding. Neither outcome is a claim about EL-GS; this lane is
photometric and exploratory.
