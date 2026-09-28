## Verdict

MATERIAL DEFECT

## Confirmed defects

- **Q2 verdict: the N3V camera geometry is not “necessary.”** The memo conflates file/capture protocol with the physical rig. Multi4D consumes LLFF-style `poses_bounds.npy`; STG consumes calibrated COLMAP cameras. Neither requires cameras copied from an N3V scene. A different calibrated rig can retain 20 cameras, 300 frames, resolution, frame rate, scale, and held-out-view policy without loader changes. Evidence: [Multi4D data instructions](https://github.com/BatFaceWayne/Multi4D) and [STG data instructions](https://github.com/oppo-us-research/SpacetimeGaussians). Fix: call the real rig a defensible control choice, not a loader requirement.

- **The Ex4DGS near plane is misstated.** The reviewed config says `"near": 4.0`, `"far": 300`, not `0.01/300` ([config](/D:/adags/refine-logs/review-2026-09-21/src/ex4dgs/n3v_base.json:16)); the code audit says those become CUDA `min_depth` and `max_depth` ([code map](/D:/adags/research-wiki/operations/ex4dgs-code-map-2026-09-19.md:145)). `0.01` may be a camera projection z-near, but it is not the effective Gaussian culling bound. Fix: distinguish projection clipping from training/rasterization culling and verify every ARS surface lies beyond 4 model units.

- **“Each at its published regime” is false for STG.** STG’s documented experiment trains the **first 50 frames**; six independent models tiled over 300 frames are a project adaptation, not its published regime ([official STG repository](https://github.com/oppo-us-research/SpacetimeGaussians), [workspace protocol record](/D:/adags/research-wiki/operations/stg-n3v-protocol-parity-2026-08-19.md:41)). Fix: describe this as six segment-wise published-configuration models and predeclare boundary handling and metric aggregation.

- **The Blender mask argument is overconfident, especially with glass.** Blender documents that Index passes are not anti-aliased and are governed by an opacity threshold; this does not establish equivalence to the analytic renderer’s pixel-centre identity rule. Blender recommends Cryptomatte where transparency and multiple objects per pixel matter ([Blender render-pass manual](https://docs.blender.org/manual/en/4.5/render/layers/passes.html)). Target-alone masks also do not identify refracted, reflected, or illumination-mediated influence. Fix: retain an explicit opaque geometric ID pass and separately define radiometric support using paired renders with and without the target.

- **The renderer’s aggregate checker fails open.** It computes `all_pass` only over applicable checks that happen to be present: `if k in res` ([checker](/D:/adags/scripts/build_absence_suite_scene.py:1205)). A missing applicable check can therefore yield a pass. Fix: require every applicable key to exist and equal `pass: true`, while marking intentionally partial runs separately.

## Suspected issues

- **Q1 verdict: do not switch wave 1 to Blender, but also do not put shadows on its critical path.** The memo calls shadows absence-relevant, while the ideal plan declares lighting changes out of scope ([ideal plan](/D:/adags/research-wiki/operations/absence-return-suite-and-baseline-lanes-plan-2026-09-23.md:128)). Shadows may test relighting or baked appearance rather than primitive presence. Settle this with a post-proof paired shadow/no-shadow arm and explicit shadow-support attribution.

- COLMAP admission may be more consequential than the Blender decision. Procedural textures and repeated analytic geometry can produce poor or biased SfM even when images look plausible. The pending preprocessing runs must report registered cameras, reprojection error, point coverage, scale, and failures—not merely success/failure.

## Unverified claims

- I found no preserved `checks.json`, rendered ARS output, timing record, or manifest under `D:\adags`. Therefore the “8 checks pass,” full-scene byte determinism, 30-second identity render, one-second colour render, and full-resolution fast-versus-brute claims are unverified. The committed test only compares two reduced-resolution camera/frame pairs ([test](/D:/adags/tests/test_build_absence_suite_scene.py:98)). Tests could not be executed in this read-only environment because Python could not create a temporary directory.

- COLMAP admission is explicitly still running, so baseline loadability of ARS is not established.

- The Blender/Cycles and EEVEE timings, 1.5–3-day engineering estimate, and analytic-shadow half-day estimate have no benchmark evidence.

- Blender was absent from `PATH` and `bpy` was unavailable to Python 3.12 here, but that is not an exhaustive installation search.

## Missing

- The largest omitted risk is **pseudoreplication**. The plan itself says the repetitions are “repeated trainings, not seeds” ([active plan](/D:/adags/research-wiki/operations/ars-fast-path-plan-2026-09-23.md:174)). Training repeats are not independent scenes; inference should use layout/scene as the unit or be explicitly limited to these fixed scenes.

- The Ex4DGS reproduction is recorded as only partial—33.14 versus 33.73 published ([reproduction record](/D:/adags/research-wiki/operations/ex4dgs-n3v-leonardo-reproduction-2026-09-19.md:3)). That qualification belongs beside any “published regime” claim.

- A better non-N3V-looking design is a calibrated conveyor/test-cell: asymmetric textured objects, two distinct return pedestals, a textured revealed backplane, and an independent occluder. It separates identity restoration, gap reconstruction, moved return, and occlusion controls while retaining the accepted file and temporal protocol.

Review conducted independently; no ARIS skill was used.