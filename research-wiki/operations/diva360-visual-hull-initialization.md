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

## Recorded in advance: what a negative result would and would not mean

If the hull initialization does not help, that does NOT restore the
frustum volume as a good prior — S1 already showed the frustum volume is
bad. It would instead say that this model recovers from a bad
initialization through densification, which is a different and useful
finding. Neither outcome is a claim about EL-GS; this lane is
photometric and exploratory.
