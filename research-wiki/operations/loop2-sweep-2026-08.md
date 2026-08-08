# Loop-2 Literature/Dataset Sweep — 2026-08-08 (durable findings)

Three verified sweeps under the user-relaxed constraints (external
priors allowed; any public dataset; per-scene optimization fixed).
Every claim verified on primary pages during the run. Extends
[[operations/repr-sweep-2026-08]].

## F — Event-dense datasets (verified shortlist)

Primary recommendation: **DiVa-360** (2307.16897, CVPR24 Highlight):
53-cam TRUE 360° surround, table-scale, 25 dedicated hand-object
interaction sequences + 21 object-centric + 8 long-duration, 120fps,
fg/bg masks, MIT license, open; NO published dynamic-GS baselines
(comparability gap AND opportunity — first pass must establish
4DGS/STG-class baselines). Secondary: Ego-Exo4D cooking (2311.18259;
ego + 4 exo; domain continuity; stress tier). Metric-validation only:
HOT3D (2411.19167) and Aria Digital Twin (2306.06362) — industry-grade
6DoF object-pose GT across pickup→occlude→putdown→reappear cycles but
narrow headset rigs. Also verified: ParaHome (70 cams, manipulation,
no GS record), CMU Panoptic (surround, people-crossing occlusion),
Technicolor (CEM-4DGS occlusion-error precedent), ENeRF-Outdoor,
DNA-Rendering, HiFi4G. Ruled out for per-scene GS: BEHAVE (4 cams),
HOI4D/TUM/Bonn (monocular), ActorsHQ (no object events), Nymeria.

## G — Prior-supervised landscape: three unoccupied cells (verified)

1. TRACKER VISIBILITY STATES → representation presence/identity:
   UNOCCUPIED. MoSca 2405.17421 and Shape-of-Motion 2407.13764
   verified: visibility used ONLY as loss masks/interpolation gates for
   motion; TrackerSplat 2604.02586: geometric initialization only;
   2606.23212 visibility is internal/render-derived. CoTracker3
   2410.11831: visibility is a first-class BCE-supervised output
   (tracker side mature; consumption side empty).
2. MEASUREMENT-MODEL EXISTENCE INFERENCE in differentiable rendering:
   NEAR-UNOCCUPIED. Sole close work: Consistent Instance Field
   2512.14126 (factored P(E=1|x,t)·P(K|E) from DEVA, per-scene,
   differentiable — but segmentation-consistency framing, no temporal
   existence dynamics, no birth/death, no reactivation, no tracks).
   No PMBM/JPDA-style association-with-existence found in
   differentiable rendering.
3. AMODAL-SUPERVISED persistent hidden state in per-scene dynamic GS:
   UNOCCUPIED as a pairing (TACO/Amodal-SAM are 2D feeders; GenMOJO/
   Lift4D are diffusion completion decoupled from amodal masks;
   PersistGS is a physics process model).
Occupied context: mask-tracker identity at loss/feature level (GenMOJO
one-hot labels; SA4D identity field; Director 2604.01678 differentiable
8D features w/ KNN smoothing through occlusion); CubifyGS discrete
DINO retrieval; depth/flow representation moves mature (Mode-GS,
MoDGS, GaussianFlow, SplatFlow, MotionGS).

## H — Layered/object-complete/hidden state under relaxed capture

The conjunction (self-inferred occlusion order + persistent hidden
appearance/geometry + identity, per-scene) remains UNOCCUPIED even with
surround/ego capture + priors. Key verifications: ST-NeRF 2104.14786
(SIGGRAPH21, 16-cam) does NOT maintain persistent hidden-layer state
(per-entity always-live fields + per-ray compositing; deeper claims
absent) and has NO GS successor (Free360 = static). PersistGS
2606.03479: rigid-body physics permanence on a genuine surround rig
with held-out-camera occlusion eval (+2.46 dB vs kinematic; 0.19 dB
from GT bound) — RIGID-ONLY. 4DPM 2512.16564: general rigid-primitive
permanence, MONOCULAR. Driving scene graphs: node+SE(3) trajectory
occupied (Ost 2011.10379 → OmniRe 2408.16760, AD-GS 2507.12137) but
hidden-interval state is EXTERNALLY SUPPLIED (OmniRe: AV perception-log
track IDs) or parametrically bridged (AD-GS spline; one appear + one
disappear mask). Egocentric dynamic GS: no hidden-state mechanism
anywhere; CVPR26 EgoVis workshop paper 2604.23803 confirms the
evaluation gap. GASPACHO 2503.09342: template-bound HOI, heavy (not
total) occlusion. Gaussian Object Carver 2412.02075: static completion.
OPEN slices: non-rigid permanence through occlusion on surround
(food/cloth/hands); layered persistent hidden state on surround;
egocentric hidden-state dynamic GS; self-inferred identity in
compositional per-scene reconstruction.

## Consequence

These three sweeps triangulated the Loop-2 design: the two unoccupied
inference cells (G.1, G.2) + the open application domains (H) + an
event-dense benchmark with mask GT and pose-GT metric validation (F)
produced EL-GS ([[operations/elgs-method]]).
