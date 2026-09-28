# Two design questions about the synthetic absence-return suite (ARS), for independent review

Reviewer: please act as a critical co-thinker. Challenge the analysis and the recommendation, check the
facts you can check in the workspace (D:\adags) or on the web, and say plainly where I am wrong or
where a different choice is better. Mark anything you cannot verify.

## Context (verified from the project record unless marked)

- Project: a NeurIPS 2026 workshop paper (PhysWorldAI; 8 pages; the principal wants it done ~Oct 15;
  non-archival window Sep 29 - Oct 29). The method ("CER", implemented on Ex4DGS, NeurIPS 2024) gives
  4D Gaussian primitives an exact-zero presence while an object is truly gone from the world and restores
  the same primitives when it returns. Baselines: Ex4DGS base, Multi4D (ECCV 2026), SpacetimeGaussians
  (STG, CVPR 2024), each at its own published regime, one common evaluator.
- Active plan: `research-wiki/operations/ars-fast-path-plan-2026-09-23.md` (rev 3). Its ideal-design
  spec is `research-wiki/operations/absence-return-suite-and-baseline-lanes-plan-2026-09-23.md` (rev 2).
- Evidence hierarchy (principal, 2026-09-23): the synthetic suite ARS is PRIMARY evidence; three real
  N3V fixtures (a wine bottle digitally removed for frames 60-89 by an SA4D teacher render) are a
  SECONDARY realism check. Workshop bar reference: PersistGS (CVPRW 2026) used three synthetic
  rigid-ball scenes, 5 train + 2 test cameras at 512x512.
- Schedule: wave 1 (9 scenes = 3 layouts x {same-pose return, moved return, pass-through}) is meant to
  be frozen ~Sep 25-26, baselines trained over the weekend, "proof readout" Mon Sep 28. Wave 2 (7 more
  event types on one layout) frozen by Sep 29; "no new datasets after Sep 30"; data freeze Oct 4.
  We are already ~1 day behind.
- Compute: ~15,500 A100 GPU-h allocation on Leonardo; 1-GPU jobs queue slowly. The whole plan uses ~2%.

### What the three baselines can load (verified on Leonardo 2026-09-23)
- Ex4DGS loaders: Colmap, Technicolor, Neural3DVideo. Reads intrinsics/extrinsics from `colmap_0`
  (a sparse COLMAP model of frame 0 with known poses), hardcodes near/far 0.01/300, keyframes every
  10 frames, trains 300 frames.
- Multi4D: dynerf, technicolor, nerfies loaders. Static branch initialised from `fused.ply`, a DENSE
  COLMAP cloud of frame 0 (4DGaussians `colmap.sh llff` recipe).
- STG: Colmap, Technicolor, Immersive, Colmapmv. Trains six independent 50-frame models; per-frame
  sparse COLMAP points.
- None loads Blender / NeRF-synthetic format. Published configs are for N3V ("dynerf"/"n3d"):
  ~20 forward-facing cameras, 300 frames at 30 fps, 1352x1014, cam00 held out.

### The current ARS renderer (built this week, `scripts/build_absence_suite_scene.py`, uncommitted)
- Fully synthetic; only camera matrices come from the real N3V rigs (`poses_bounds.npy` of three
  scenes). Analytic ray casting of spheres, oriented boxes, capped cylinders, planes; procedural
  object-frame value-noise textures; Lambertian + ambient, one directional light, NO shadows; 2x2
  supersampling; float64; 8-bit output. Frozen per-scene JSON specs (renderer-independent: rig ref,
  objects, shapes, textures, analytic trajectories, presence intervals, clock, windows).
- Measured properties: fast (identity pass of 20 cams x 300 frames in ~30 s; full colour ~1 s per
  frame per core, ~0.5 MB PNG after calming the wall); byte-deterministic; the cached/hull-limited fast
  path is byte-identical to a brute-force render (tested); exact ground truth: front-most identity per
  pixel, target-alone masks, and the property that the scene differs from its "base world" (static +
  ordinary dynamics) ONLY inside the event objects' silhouettes (+1 px) - checked per frame.
- Scene content now: a room with a calm wall, a window with a blind, a poster, a wall shelf, a table
  with props, a human-proportioned "cook" built from boxes/cylinders/spheres (sways, two-joint arms
  moving over the back of the table), a rotating object; the event target (bottle / carton / mug)
  placed where the real fixture's bottle sits in cam00 (footprint ~5,000 px).
- 8 checks pass on the bottle layout (visibility, absence, move, pass-through coverage, collision and
  contact, pixel-denominators, no-leak, fast==brute). COLMAP admission (each baseline's own
  preprocessing on the rendered frames) is being run now.

### Blender facts
- Not installed locally; no `bpy` wheel for the local Python 3.12 (bpy 4.x wheels need Python 3.11);
  local GPU: RTX 4080 SUPER 16 GB (OptiX-capable). Leonardo A100s (CUDA, no RT cores); login node has
  network. Installing Blender (~350 MB from blender.org) needs the principal's explicit OK.
- UNVERIFIED estimates: Cycles/OptiX on the 4080S at 1352x1014, 64-128 spp, no denoiser, ~0.3-1.5 s
  per frame -> 6,000 frames per scene -> ~0.5-2.5 h per scene; EEVEE ~0.1-0.3 s per frame.
  Engineering ~1.5-3 days (rig import with exact pinhole/pixel-centre convention; scene assembly from
  the JSON spec or new asset scenes; animation baking; object-index and target-alone passes; render
  scripting; a cross-check of Blender's object-index pass against the analytic identity buffer on
  identical geometry; re-running every check).
- Kubric-5D (bpy 3.4, GSO assets, HDRI backgrounds) was examined 2026-09-23: its published scenes have
  no absence/return events, ship no masks, and use a 16-camera 15 m sphere rig at 576x384, 60 frames,
  24 fps - outside all three baselines' regimes. It would be a generator, not a dataset.

## Question 1: is it useful and feasible to build these scenes in Blender (3D) and extract the ground truth from there?

My analysis.
- Feasible: yes, technically. Masks come from object-index passes (not anti-aliased, i.e. same kind
  as our pixel-centre identity) and from per-object holdout renders for target-alone masks; depth from
  Z passes; poses we author. The rig import is the main correctness risk and can be validated exactly
  by rendering the current analytic scene geometry in Blender and comparing identity buffers.
- What Blender adds that matters for THIS claim:
  1. Cast shadows. Removing an object removes its shadow; the counterfactual background in the gap
     is the unshadowed table. A presence gate that only switches off the target's primitives leaves
     the shadow baked into the table primitives, so P1 (the gap endpoint) becomes harder and more
     realistic. This is the single most absence-relevant realism feature. (Rev 2 had analytic hard
     shadows in the spec; rev 3 deferred them, ~0.5 day in the analytic renderer.)
  2. Specular/transparent materials (the real target is a glass bottle), soft light, GI, real
     scanned assets and a rigged human: visual realism and the "toy scenes" objection.
- What Blender costs:
  1. Time: ~1.5-3 engineering days plus render time, on a schedule already a day behind. Switching
     now moves the proof readout from Sep 28 to ~Oct 1-2 and collides with "no new datasets after
     Sep 30"; wave 2 would slip past the freeze.
  2. Exactness: Monte Carlo noise (the gap GT and all static regions carry sampling noise; a denoiser
     is non-local and leaks around the target); global illumination makes an absence change pixels
     outside silhouette+shadow, so the "differs only inside the support" check can no longer be exact
     unless lighting is restricted to direct light (+ shadows). Determinism across GPU/driver versions
     is not guaranteed, so frames must be archived rather than regenerated.
  3. New failure modes (asset licences, human rig licences, render farm scripting) at the worst time.
- The specs are renderer-independent, so a Blender re-render later is a re-render, not a redesign.

Recommendation (mine): do NOT switch wave 1 to Blender. Keep the analytic renderer for the proof
(exact, fast, verified). Add analytic hard cast shadows (the one realism feature that bears on
absence) either to wave 1 now if the principal accepts ~0.5 day, or to wave 2. Offer a Blender
re-render of a SUBSET (e.g. the same-pose return on one layout, with shadows, glass and a real
human asset) as a robustness appendix only after the proof, and only with the download approved.
Plan the full Blender/Kubric-style generator for the later self-built dataset / CVPR-ICCV paper.

## Question 2: is it useful or necessary that the synthetic scenes look like the real N3V scenes? Why not any plausible synthetic setting that showcases what we test?

My analysis: "look like N3V" has two separable meanings.
- (a) Capture-protocol likeness: an N3V-format rig (~20 forward-facing cameras, the same pose
  convention and depth range, 300 frames at 30 fps, 1352x1014, cam00 held out). This IS necessary:
  it is the only way all three baselines run with their published configs and no loader code, and
  their scale-dependent hyperparameters (densification thresholds relative to camera extent, position
  learning rates scaled by scene extent, Ex4DGS's fixed near/far and 10-frame keyframes, STG's
  50-frame segments) stay in the regime they were tuned for. A different rig (e.g. a surround ring)
  would need new configs, which invites "you mis-configured the baselines".
- (b) Content likeness (a kitchen, a cook, a window): NOT necessary for the method. The principal is
  right that the goal is to show CER's benefit under controlled, exactly-known events; any plausible
  (not implausible) scene that supports every event type works. Content matters only through:
  (i) event geometry: a support surface for same-pose/moved returns, a lane for pass-through, a
  cover for the occlusion control, space for a moving target; (ii) scene statistics that keep
  baselines in their tuned regime (texture density, fraction of dynamic pixels, depth range) - this is
  also why the high-frequency wall was a problem; (iii) comparability with the real fixtures: same rig,
  same image location and size of the target, same event windows, so the realism check compares like
  with like; (iv) reviewer plausibility.
- So: keep (a) strictly; keep the content generic and plausible (a tabletop room), and stop spending
  effort on resembling the N3V kitchen beyond cheap dressing. The current scenes already are a generic
  tabletop room with light N3V flavour; the cook matters because N3V's dynamics are a person working
  behind a table, and "ordinary dynamics" must exist for the do-no-harm endpoints.

## What I would like from you
1. Is any fact above wrong or misleading? Check the baseline loader claims and the scale-dependent
   hyperparameter claims against the code if you can (Leonardo is not reachable from your sandbox;
   use your knowledge of the public repos and mark it as such).
2. Is the recommendation on Blender right for this schedule and this claim? What is the strongest
   argument for switching now? What would you do instead?
3. On question 2: is there a plausible non-N3V-looking setting that would test CER BETTER (e.g. more
   decisive for the return, the gap, or the controls), while keeping the capture protocol?
4. What is the biggest risk you see in the current synthetic design that neither question covers?
Answer in <= 900 words, most important points first, with a clear verdict per question.
