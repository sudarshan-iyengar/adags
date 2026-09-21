# Orchestrator notes on Fable round 2 (2026-09-21)

Independent checks of the load-bearing new facts in `fable_round2.md`.

## Verified — Fable was right

1. **Panoptic Sports is on Leonardo and was never censused.** Path
   `/leonardo_work/EUHPC_D21_034/proj_adags/data/panopticsports`, 716 MB.
   Six sequences — `basketball, boxes, football, juggle, softball, tennis` —
   each with **31 cameras** in `ims/`, **150 frames** per camera, and
   `seg/` for **27 cameras**. This is the Dynamic 3D Gaussians (Luiten et al.)
   subset. It clears the ≥ 8-camera bar by a wide margin.
   **Not verified:** whether any sequence contains an all-camera exit (a ball
   leaving the capture volume, a box carried out) rather than momentary
   occlusion; and whether `seg/` is instance-level or a single foreground mask.
   Both are the scout Fable proposes, and neither is a fact yet.

2. **Mask-Benchmark covers all three fixture scenes.**
   https://huggingface.co/datasets/yunjinli/Mask-Benchmark/tree/main/Neu3D-Mask
   lists exactly `coffee_martini, cook_spinach, cut_roasted_beef, flame_steak,
   sear_steak`, 1,500 files in total — **300 per scene**, consistent with one
   held-out camera over all 300 frames. So the crb semantic reproduction gate is
   possible, and so are flame and sear. There is also a 595 MB
   `Mask-Benchmark.zip` at the root.
   **Not verified:** that the 300 are cam00; the PNG resolution; and whether the
   536×536 JPEGs on the dataset card are these masks or a preview rendering.

## Consequence worth stating

Because the authored-fixture frames **outside [60,89] are byte-identical to the
originals** (already on the record: the derived scenes were verified
byte-for-byte outside the edit), Mask-Benchmark's original-scene GT can score
**the return window** of the fixtures directly — conditional on camera and frame
coverage. It cannot score the gap itself, where the object was removed. That
turns "the fixtures need their own masks" from a blocker into a scoped problem:
only the gap window needs fixture-specific masks, and there the GT is simply
"object absent".
