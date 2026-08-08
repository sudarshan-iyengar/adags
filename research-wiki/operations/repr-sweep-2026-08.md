# Representation-Level Literature Sweep — 2026-08-08 (durable findings)

Five parallel verified sweeps (dynamic-GS representation families;
layered/amodal/visibility state; temporal parameterization; Feb-Aug 2026
recency; adjacent-field state semantics) plus nine paper deep-dives,
executed during the representation-first method-discovery run. Every
mechanism claim below was verified against arXiv abs/HTML pages, the
arXiv API, or code repositories during the run — none is
search-snippet-only. Extends (does not replace)
[[operations/sota-sweep-2026-08]]. Companion pages:
[[operations/lgs-method]], [[operations/lgs-review-history]],
[[operations/rejected-representations-2026-08]].

## Families synthesis (25 papers, 8 families)

Element semantics across time per family, with the shared conclusion:
1. Deformation-field (4DGS-Wu 2310.08528, Deformable-3DGS 2309.13101,
   Grid4D 2410.20815): canonical Gaussians + smooth warp; identity
   inescapably persistent; no birth/death/topology/occlusion state.
2. Explicit 4D (RealTime4DGS 2310.10642, STG 2312.16812, FreeTimeGS
   2506.05348, Ex4DGS 2410.15629): unimodal temporal presence windows +
   low-order trajectories; identity weak/local ("flipbook").
3. Trajectory/track (Dynamic3DG 2308.09713 — color/opacity FROZEN over
   time; Shape of Motion 2407.13764; MoSca 2405.17421): strongest
   identity, rigidity priors suppress events; no occlusion state.
4. Hybrid static/dynamic (Hybrid 3D-4DGS 2505.13215 — venue unconfirmed;
   Swift4D 2503.12307): routing over an existing family.
5. Superpoint/part/basis (SC-GS, D-MiSo, PaMoSplat 2605.10307 TCSVT26,
   MoE-GS 2510.19210 ICLR26 — pixel-space expert routing; SE(3)-B-spline
   GS 2603.25058 CVPR26 — most locally adaptive explicit motion).
6. Mesh/surfel (GauSTAR 2501.10283 — bind/unbind + re-create; AT-GS
   2411.06602; GaMeS): discrete state changes exist but identity is
   traded away exactly at topology change.
7. Scaffold/anchor (Scaffold-GS 2312.00109; 4D-Scaffold-GS 2411.17044
   AAAI26): bin-quantized presence, no occlusion variable.
8. Streaming (3DGStream 2403.01444, HiCoM 2411.07541, V3 2409.13648):
   only family with lifecycle ops; REAPPEARANCE ALWAYS CREATES NEW
   GAUSSIANS — no reactivation of stored identity anywhere.

Capabilities NO family expresses directly (sweep-verified):
(1) causal per-element occlusion/visibility state; (2) identity-
preserving reappearance (no method revives a stored identity);
(3) discontinuous scene events as first-class queryable state.

## Layered/amodal/visibility verdict

"Layered/occlusion-ordered dynamic GS with persistent hidden-surface
state" remains UNOCCUPIED as a conjunction, bounded by near-misses:
CLOTH-HUGS 2604.15875 (3 fixed ordered avatar layers; per-frame
compositing rule; deep-dive verified NO temporal hidden state);
PersistGS 2606.03479 (physics pose memory, single rigid object);
CIF 2512.14126 (CVPR26; occupancy × SEMANTIC identity; deep-dive:
γ=π·p factorization, DEVA-supervised, calibration factor m_i^k);
LayerPano3D (static panoramas); CPSL 2511.14927 (2.5D per-frame);
RT-Splatting 2605.18263 (CVPR26; geometric occupancy σ vs optical
opacity α — "presence ≠ visibility" mechanism shape, for transparent
materials; code verified); NVGS (visibility for culling); GauSTAR
(re-creates). Frontal-rig constraint (P03 lesson) argues against the
ORDER half; the MEMORY half was the open target.

## Temporal parameterization taxonomy (occupied → unoccupied)

Occupied presence shapes: Gaussian bump; super-Gaussian
exp(−(|t−μ|/σ)^β) (4D-Scaffold, motivated for sudden appearance);
flat-top with independent asymmetric Gaussian edges (Ex4DGS — earliest,
single contiguous interval); boxcar+tails + lifespan-damped velocity
(SharpTimeGS); learned smooth opacity table (TOGS 2403.19586);
scene-global segment+residual decomposition (CTRL-GS 2505.18306 CVPRW25,
deep-dive verified: quantized-time trick, boundaries shared by ALL
Gaussians); per-frame residual chains (QUEEN); periodic POSITION with
single-bump opacity (PVG 2311.18561 IJCV26 — name notwithstanding).
TOM-GS 2607.22717 (Jul 2026): presence-ONLY encoding (static geometry,
all dynamics via per-Gaussian temporal opacity) — deep-dive verified:
exactly one (μτ,στ) pair per primitive, no mixture/multi-window/
periodic form anywhere; −12.14 dB without temporal opacity.

VERIFIED UNOCCUPIED (as of 2026-08-08): multi-interval/disjoint/
reactivating presence per primitive ("no primitive can come back");
monotone/latched presence; per-primitive changepoints as first-class
parameters; exact compact-support absence; periodic temporal opacity.

## 2026 recency sweep (2602-2608)

Representation-level 2026 entries verified: GP-4DGS 2604.02915 (CVPR26,
GP motion posterior), GraphiXS 2601.19843 (SIGGRAPH26,
distribution-valued primitives), Director 2604.01678 (semantic instance
features), TOM-GS 2607.22717, Grassmannian Splatting 2607.10489 (rank-2
spacetime surfels), RetimeGS 2603.13783 (CVPR26, continuous-time),
SPIN-4DGS 2607.12362 (ICLR26, implicit spacetime attribute field for
fast motion), PersistGS, MoPe 2606.29237 (RSS26WS, persistent Bayesian
dynamicness log-odds), CubifyGS 2606.28720 (IROS26, reusable object
assets — deep-dive verified: DINO retrieval, frozen templates,
non-differentiable ray-cast presence state machine; asset-library
ablation −4.52 dB), R5DGS, MeGAS, DR-GS. None touch multi-episode
presence or content-tied reactivation.

## Adjacent-field state semantics (verified; cited as ancestry)

H.264/HEVC long-term reference frames (explicitly pinned references
surviving occlusion; Lagrangian D+λR mode decision = reuse-vs-recreate
arbitration); ElasticFusion active/inactive surfel maps (frozen-not-
deleted; reactivation on re-observation; IJRR 2016); deep shadow maps
(transmittance as stored piecewise function; SIGGRAPH 2000); CoTracker3
2410.11831 (joint position/visibility/confidence track state; occluded
tracks propagated via neighbors); TSA 2606.13714 (deep-dive verified:
one activation scalar α gates BOTH slot-state update (convex blend) and
decoder attention (log-bias); diagnoses existence-vs-visibility
conflation; 2D slots, no 3D/rendering); FLIP/PIC reseeding; holdout
mattes.

## Wiki corrections recorded this run (explicit, not silent)

1. arXiv 2606.23212 ("Temporally Aware Densification", IISc) internally
   names its method VAD (Visibility-Aware Densification); the query pack
   and [[operations/sota-sweep-2026-08]] call it "TAD-GS". Distinct from
   VAD-GS 2510.09364 (Zhang, urban). Cross-references should
   disambiguate: 2606.23212 = Sandu et al. temporally-aware
   densification; [[papers/sandu2026_temporally_aware_densification]].
2. [[papers/kang2025_cec_4dgs]] ("CEC-4DGS"): the paper's actual title
   is "Clustered Error Correction with Grouped 4D Gaussian Splatting"
   (arXiv 2511.16112) and the official repository is named CEM-4DGS.
   The mechanism description in the STAR-GS record stands; the
   name/repo detail is corrected here rather than silently rewritten.

## Deep-dived paper pages added this run

[[papers/hou2025_ctrl_gs]], [[papers/shi2026_rt_splatting]],
[[papers/lisowski2026_tom_gs]], [[papers/wu2026_consistent_instance_field]],
[[papers/mubashshira2026_cloth_hugs]], [[papers/zhang2026_se3_bspline_gs]],
[[papers/xiao2026_mope]], [[papers/ren2026_cubifygs]],
[[papers/nguyen2026_tsa]] — all status deep-dived, full-text/code
verified, with preserved Relevance-to-ADAGS sections.
