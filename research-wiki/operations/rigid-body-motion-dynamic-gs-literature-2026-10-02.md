# Rigid-body and affine motion in dynamic / 4D Gaussian splatting: literature deep dive (2026-10-02)

Asked by the principal: what has been done with rigid-body motion (shape preserved; only rotation
and translation change) and affine-transform guidance in dynamic 3DGS / 4DGS, and does it help
reconstruction. Four parallel sweeps (≈150 web searches, every arXiv ID fetched on its abs or HTML
page; mechanism details read through a summariser, so re-check before citing). Exploratory
literature record, no experiment. Follow-ups: the refinement in
`refine-logs/rigid-return-2026-10-02/` and the novelty check recorded at the end of this page.

## 1. The short answer

Rigid structure is **heavily used** in dynamic GS, in five families. What is **not** done is
using rigidity to keep an object's **identity** across a full absence from every camera and to
re-attach the same primitives at a **new pose** on return.

| family | what rigidity does there | representative work (arXiv) |
|---|---|---|
| A. shared / low-rank SE(3) motion bases, skinning | compact motion; soft rigid groups | Shape of Motion 2407.13764, MoSca 2405.17421, SC-GS 2312.14937, SP-GS 2406.03697, DynMF 2312.00112, SE(3) B-spline bases 2603.25058, Motion Blender GS 2503.09040, MotionScale 2603.29296, RiGS 2605.23672, TRiGS 2604.00538 |
| B. local rigidity / ARAP / isometry losses | regulariser on neighbourhoods | Dynamic 3D Gaussians 2308.09713 (Panoptic), SC-GS (rotation fitted by SVD inside ARAP), MoSca, Inst4DGS 2603.18402 |
| C. per-object canonical set + per-frame SE(3) (scene graphs) | object-level motion prior | NSG 2011.10379, Street Gaussians 2401.01339, HUGS 2403.12722, 4DGF 2406.03175, OmniRe 2408.16760, PaMoSplat 2605.10307 (N3V, Panoptic), R5DGS 2605.25909, MoGaF 2602.21668 |
| D. articulated parts (joints, screws) | part-level rigid motion between observed states | PARIS 2308.07391, ArtGS 2502.19459, SplArt 2506.03594, ScrewSplat 2508.02146, GaussianArt 2508.14891, FreeArtGS 2603.22102, template-free articulated GS 2412.05570 |
| E. rigid-body physics | trajectory through unobserved frames | PersistGS 2606.03479 (differentiable rigid-body simulation through full-multiview occlusion), PMGS 2508.02660, PhysGaussian 2311.12198 (continuum, per-Gaussian affine F) |

Rigid tracking and re-identification **outside** 4DGS view synthesis: BundleSDF 2303.14158,
Point2Pose 2604.10415 (object fully leaves and re-enters one RGB-D view; SVD + RANSAC
registration), RRTrack 2607.23669, CubifyGS 2606.28720 (stored Gaussian assets retrieved and
rigidly re-posed after rearrangement; RGB-D SLAM, discrete before/after), 3DGS-CD 2411.03706,
LTGS 2510.09881 (opens a NEW template for a reappearing object), 4D Primitive-Mâché 2512.16564
(rigid primitives replayed while invisible; no re-entry).

### 1b. Rigid-part DISCOVERY from motion in dynamic scenes (added after the principal's question)

"Identify the rigid bodies, then give each an SE(3)" is an established line, older than GS:

| work | setting | how rigid parts are found | motion model |
|---|---|---|---|
| Watch It Move 2112.11347 (CVPR 2022) | multiview video | unsupervised: points static relative to each other = one part; relative motion = joint | ellipsoid parts + joints |
| MovingParts 2303.05703 (CVPR 2023) | monocular dynamic NeRF | particles sharing a motion pattern | per-part rigid (Lagrangian) |
| SP-GS 2406.03697 (ICML 2024) | monocular | learned soft superpoint association | rigid per superpoint |
| template-free articulated GS 2412.05570 (NeurIPS 2024) | multiview | superpoints merged when relative pose stays constant; spanning-tree skeleton | kinematic, LBS |
| SpeeDe3DGS 2506.07917 (CVPR 2026) | monocular | trajectory-similarity clustering; Umeyama/Procrustes init | hard per-group SE(3) |
| MotionScale 2603.29296 (CVPR 2026) | monocular | k-means on 3D trajectories, split by HDBSCAN when inconsistent | per-cluster SE(3) + local bases |
| MoGaF 2602.21668 (CVPR 2026) | monocular | masks with rigidity labels + feature region growing | shared SE(3) per rigid group |
| PaMoSplat 2605.10307 (TCSVT) | multiview (N3V, Panoptic) | SAM masks lifted, graph clustering; IDs fixed at t = 0 | per-part SE(3), learnable rigidity |
| Inst4DGS 2603.18402 | multiview (N3V, Panoptic) | SAM 3 + Sinkhorn cross-view matching | per-instance SE(3) scaffolds |

Articulated-object counterparts (several observed states of one object, not a dynamic scene):
Part²GS 2506.17212 (part-aware transforms + contact / velocity-consistency / vector-field and
repel-point constraints), PARIS, ArtGS, SplArt, ScrewSplat, GaussianArt, REArtGS(++).

## 2. Affine transforms specifically

- **No paper fits a per-group 12-DoF affine to Gaussian motion and projects it onto SE(3)/Sim(3)
  (polar decomposition, Procrustes, Kabsch) as a training-time guidance target** (eight query
  formulations). Nearest: SC-GS (SVD rotation inside a local ARAP loss — which IS a local
  Procrustes projection, so "Procrustes rigidity" at neighbourhood scale is occupied);
  SpeeDe3DGS 2506.07917 (Umeyama initialises per-group SE(3), groups from trajectory similarity);
  Gaussians-to-Life 2411.19233 (per-Gaussian weighted Kabsch Sim(3) used as the motion itself,
  text-to-animation); SurFhead 2410.11682 (polar decomposition of per-triangle affines, head
  avatars); PhysGaussian (affine deformation gradient from simulation).
- **Several methods are affine without saying so:** a linear blend of SE(3) matrices (RiGS,
  GaussianArt's soft phase) leaves SE(3). Shape of Motion blends 6D parameters; MoSca, SE(3)
  B-spline, Inst4DGS and Motion Blender use dual-quaternion blending to stay on SE(3).
- Reading: "fit affine, project to rigid" is a sound **estimator** (closed-form, cheap) but at
  the regulariser level it is a re-description of ARAP/Procrustes. It is novel only where it
  solves a problem the regulariser cannot, e.g. registering a returning object to its stored self.

## 3. Does the rigid prior help reconstruction? (reported numbers, as read)

| work | rigid component | reported gain |
|---|---|---|
| SC-GS | ARAP | ≈ +0.69 dB, D-NeRF |
| RiGS | rigid primitive type | −0.32 dB when removed (NVIDIA) |
| PaMoSplat | per-part SE(3) warm start, learnable rigidity | +1.69 dB over Dynamic 3D Gaussians on Panoptic (29.83 vs 28.14) |
| SE(3) B-spline bases | continuous SE(3) bases | mPSNR 20.17 vs 19.33 (MoSca), iPhone |
| SpeeDe3DGS | hard per-group SE(3) | mainly speed (13.7× render); LPIPS slightly worse |
| PersistGS | rigid-body simulation through occlusion | +2.46 dB over constant velocity; 0.19 dB below the true trajectory (3 synthetic ball scenes) |
| Inst4DGS | per-instance SE(3) scaffolds | N3V 30.88 vs TRASE 29.29 (own protocol; not comparable to 33.x numbers) |

Gains are moderate (0.3–1.7 dB) on ordinary motion, and are mostly about compactness and
regularisation in monocular settings. The large gains occur where observations are missing
(PersistGS) — consistent with this project's record that the event windows, not whole frames,
are where a structural prior matters.

## 4. Gaps searched for and not found (with the sweep's queries on the agents' reports)

1. Re-identifying a returning object by registering it to its stored canonical Gaussian set and
   **resuming the same primitives** in a continuous-time multiview 4DGS (CubifyGS and Point2Pose
   are the nearest; neither is 4D view synthesis).
2. A **resume-vs-respawn** measurement (does the returning content reuse pre-departure rows?).
3. Rigid groups combined with multi-interval presence (absent, then present).
4. Rigid-group discovery from motion evaluated on multiview N3V (motion-clustering methods are
   all monocular; N3V rigid methods use masks or fixed regions).
5. Constant-twist / screw interpolation across a gap using the return as an endpoint (PersistGS
   calls endpoint interpolation a "non-causal" baseline).
6. A real-footage multiview benchmark for absence and return (only PersistGS's 3 synthetic ball
   scenes; MemoBench 2606.27537 targets video generators).

## 5. Relevance to this project

The ARS wave-1 record already shows the opening: on the **moved-return** scenes the base Ex4DGS
return window is 19.3–23.9 dB against 23.0–32.8 dB on same-pose (plan log, wave-1 table), and
the CER spec states its after-return memory gate may fail there because memory rows "may not
follow the target to its new pose" (cer-ex4dgs-spec §gates). The 2026-09-23 handoff already
reserved an **oracle episode-2 transform diagnostic** for exactly this case. A rigid re-pose of
the memory cohort (one SE(3) per return) is the estimated version of that diagnostic. Every ARS
target (bottle, carton, mug) is rigid, and the renderer stores the true per-frame SE(3), so an
oracle-pose arm and a wrong-pose sham are free.

Closest prior to cite for any such claim: PersistGS (identity given, gap bridged, no re-ID),
CubifyGS (re-ID + rigid re-pose of stored Gaussians, RGB-D, discrete), Point2Pose (rigid re-ID
after full exit, RGB-D TSDF), 4D Primitive-Mâché (replay through invisibility), LTGS (new
template on reappearance — the opposite design).

## 6. Novelty check of the general idea (Track A), GPT-6.1 Sol xhigh, 2026-10-02

Dossier and raw report: `refine-logs/rigid-return-2026-10-02/NOVELTY_DOSSIER_trackA.md`,
`novelty-trackA-sol.md` (ignored directory; the substance is promoted here).

**Verdict: 2/10, ABANDON "identify rigid bodies, then SE(3)-guide them" as a standalone
contribution.** All four core claims are LOW: rigid discovery from motion (MotionStruct4D, IJCV
2026, quasi-rigid assignment + differentiable weighted Kabsch; template-free articulated GS),
hard or soft per-group SE(3) (SpeeDe3DGS hard; MoGaF Procrustes + soft anchoring), affine →
SE(3) guidance (an estimator variant; Procrustes guidance exists in MoGaF, polar decomposition in
SurFhead), and reconstruction gains (PaMoSplat, MoGaF). Additional occupants found by the check,
not in the sweeps above (IDs from Sol's links, not fetched by me): FreeGave 2506.07865 (multiview
motion groups), TRACE 2508.09811 (motion segmentation on real Panoptic), AiM 2603.02910 (automatic
part count by sequential RANSAC + Kabsch), GaussTwin 2603.05108 (mass/inertia-weighted contact).
Sol also states that an independent gpt-5.6-sol pass agreed with 2/10; that pass is not on the
record here and is unverified.

**Technical correction (affine route).** Affine-first then polar projection (`A = H C⁻¹`, project
to SO(3)) differs in general from direct Kabsch (project `H`) for anisotropic sources; without
correspondences, covariance-matching affines do not recover the rotation at all (round-1 refine
review: a 30° yaw returned as −5.82°). The affine route is an estimator variant at best.

**Candidate openings (Sol's inference, not established absences):** (c) **reuse-vs-new decision
for a returning rigid object** (strongest; competing similar newcomers, false-reuse cost, return
reconstruction, oracle controls); (a) changing body inventory and contact events with identity
preserved (PaMoSplat fixes IDs at t = 0); (b) a physical-identity evaluation on real multiview
capture; (d) a predictive hard-vs-soft rigidity crossover; (e) contact/impulse evidence for
discovery (hard; identifiability); (f) persistent-body count selection with ambiguity.
Suggested positioning: "When should multiview 4D reconstruction reuse a previously observed rigid
object, and when should it instantiate a new one?" — SE(3) registration as supporting machinery.

**Correction (append-only) to §1 and §5 above.** LTGS 2510.09881 is not a "new template on every
reappearance" design: it associates instances across timestamps and shares one template when the
match passes a geometric (Chamfer) consistency check; separate templates arise only under large
geometric change (§4.4). Checked on the arXiv HTML by fetch on 2026-10-02. Also, PersistGS
resumes the same object Gaussians at re-emergence (§3.3), so "resume after invisibility" is
occupied for a given identity; what it does not do is decide identity.

## 7. Track B: what rigidity can do for CER (2026-10-02, refinement in progress)

Refinement record: `refine-logs/rigid-return-2026-10-02/` (round files, raw GPT-6.1 Sol reviews;
score 6.30 → 7.35 → 7.80, all REVISE; round-4 review and the Track-B novelty check were cut off by the
reviewer's usage limit and are rescheduled). Proposal name: Rigid Return Transport (RRT).

**Mechanism (round 3/4):** at CER begin (30k), render the memory cohort's canonical state at
t₀ = A − 1 through ONE SE(3) for t > B (object-frame SH, presence revived), switch off the rows the
substrate grew for the returned object (zero pre-gap contribution, ≥ 0.5 of return contribution in
the target masks), train jointly 30k → 40k. Pose from eigenframe candidates `U_H P D U_Mᵀ` scored by
render-and-compare with an occlusion-preserving silhouette. Scope: stationary returns (ARS R2;
R3 conditional on wave 2).

**Decisive controls:** zero-training Step 0 on the CER R2 30k checkpoints (oracle transport must beat
resume-in-place and presence-revival-only by ≥ 1 dB P2 on 2 of 3 layouts, else stop); hard onset of
the respawned rows (`σ(o)·1[t > B]`); wrong-direction sham; **RRT-copy** (transported copies as
independent rows) to separate parameter SHARING from correct PLACEMENT. ≈ 95 A100-h + < 1 h Step 0.

**Interim novelty reading (Track-B check, interrupted):** LTGS is closer than §1 said (shared template,
rigid transforms, SH rotation, zero-opacity temporal filtering, joint refinement, a comparison against
new Gaussians per timestamp); ChronoFuseGS 2609.31339 (Pacific Graphics 2026, verified) fuses captures
with per-splat persistence. The mechanism is therefore not new in kind; what may be new is the
controlled finding on a continuous multiview temporal substrate (resume / revival / onset / transport /
sham / copy).

**Other rigidity uses for CER, ranked:**
1. RRT on moved/rotated returns (above) — the only one aimed at a measured deficit.
2. Re-identification gate: the pose-fit residual as accept/abstain, tested with the deferred N2
   control (a different object returns). Links Track A's strongest opening (reuse vs new) to CER.
3. Port to Multi4D's persistent bank (the CER-Multi4D port exists; its transient bank paints
   0.88–0.96 of the gap region).
4. Moving return (M1): an object-frame trajectory after B — out of scope now.
5. Not useful for CER: ARAP / SE(3) bases on visible motion (occupied, and not CER's failure);
   physics through the gap (the target is absent, nothing to integrate); rigid-motion membership on the
   real fixtures (the target is static before departure, so motion carries no membership signal).

### 7b. Track-B novelty check, completed (GPT-6.1 Sol xhigh, resumed after the usage limit)

**4.5/10, PROCEED WITH CAUTION as a diagnostic study; the mechanism alone ≈ 2/10.** Likely reviewer
citation: LTGS first (shared matched template, SE(3) + SH rotation, temporal opacity, joint refinement,
new-template ablation), then 3DGS-CD (re-poses pre-optimised object Gaussians from sparse post-change RGB;
RA-L 2025) and CubifyGS (reuse vs cold start compared). Also found: Flow4DGS-SLAM 2604.22339 (CVPR 2026;
ablates insertion on re-occurring dynamic objects), PersistBench 2609.20819 ("Can 4D foundation models
remember?", rendered-output permanence evaluation). The possibly novel part is the FINDING: on one
substrate and checkpoint, a moved-return deficit that survives presence revival and hard onset but responds
to cohort transport, plus a clean sharing contrast. Positioning: "Diagnosing rigid moved returns in
temporal Gaussian splatting". A failed sharing threshold must be worded "did not clear the pre-declared
superiority threshold", not "sharing adds nothing".

RRT-copy must match (i) the full rendered canonical state (effective opacity at t₀ incl. temporal factor,
covariance, object-frame SH; render equality checked over the whole return window), (ii) the optimisation
path (same trainable pose parameterisation, matched optimiser moments/lr/schedule — M rows carry Adam
history, CER's append path starts at zero), (iii) regulariser averaging and pruning lifecycle, and (iv) it
must report the extra stored parameters. A one-step gradient/update equivalence test on a post-return-only
batch is the compact check.

## 8. Second direction: "a few motion models capture the scene" on a static rig (Fable 5.1 → GPT-6.1 Sol, 2026-10-02)

Process (principal's instruction): Fable 5.1 developed the formulation over three rounds; GPT-6.1 Sol checked it
twice. Records: `refine-logs/motion-bodies-2026-10-02/` (fable-round-1-summary.md, formulation-after-fable-r2.md,
fable-round-3.md, sol-check-1.md, sol-check-2.md).

**Formulation reached.** Coverage measured over contribution-weighted DYNAMIC mass (frozen from an independent
baseline), reported as a count–coverage–distortion curve, never one "99%". Representation: labels static / free
(own keyframes) / body k (canonical pose only, shared keyframed SE(3)). Selection: rate–distortion
J = D + λR, proposals by local RANSAC + weighted Kabsch, every complete change ACCEPTED by rendered ΔJ (the J_π δx
footprint proxy only proposes); release of a member only by a displacement probe whose rendered gain pays for its
bits (members are otherwise self-confirming). Estimand: the smallest motion dictionary sufficient for the
observations under a rendering budget — not the number of physical bodies.

**Novelty: the representation is occupied, 2–3/10 (both models).** CEC-4DGS 2511.16112 already does shared
keyframed group motion with canonical members, discovered during Ex4DGS's progressive stage, on N3V (see the
addendum on papers/kang2025_cec_4dgs.md); SpeeDe3DGS (hard group SE(3)), MotionScale (adaptive clusters),
2603.25058 (adaptive SE(3) B-spline count), AiM 2603.02910 (sequential RANSAC + Kabsch), DynMF, Shape of Motion.
Motion-first robust fitting from multiview tracks is a port (multibody SfM with model selection; PaMoSplat flow
warm-start); RANSAC with a rendered consensus is a verification rule, not a contribution.

**Corrections by the second Sol check.**
- "Motion RD is unmeasured" is FALSE as a broad claim: TSA-RDO 2507.17336 §4.2 reports quality–size RD curves on
  N3V/Technicolor with a motion-wavelet ablation; GroupFlow (SpeeDe3DGS App. A.3) sweeps group count vs quality and
  storage; GSCodec Studio 2506.01822 §VI-F compares polynomial vs shared-basis motion. What may remain: controlled
  MOTION-ONLY RD at matched independent dynamic-region quality (population, canonical attributes and temporal
  opacity held fixed; motion bytes and total bytes both reported).
- The presence "RD decision" (reuse vs new): with constant saved bytes, `reuse ⟺ ΔD < λΔR` IS a rendered-error
  threshold, so agreement with a calibrated threshold is algebraic, not a kill; the claim reduces to a storage
  policy. Rate must be incremental; appearance adaptation must be a priced, interval-scoped patch (plus an exposure
  option); comparing reuse against the substrate's own defective respawn rows does not show good selection.
- Event-dependent motion grouping (atoms = body × active interval, merge/split at grasp/release) ≈ 3/10: Özden,
  Schindler & Van Gool multibody SfM handles changing numbers of independent motions; temporal mesh segmentation
  (Arcila et al.) merges and splits rigid segments over time; Relaxed Rigidity 2603.24994 has time/view-varying
  Gaussian groups (regulariser neighbourhoods). Sharper residual: **persistent asset ownership vs temporary motion
  sharing** — objects keep their own canonical payloads but share a trajectory reference during an interval
  (a switching motion-sharing graph), selected on rendered distortion and complete coded bytes. Decisive first test
  uses ORACLE events: fixed bodies with adaptive SE(3) knots and constant-segment coding vs the same with
  shared/copy references vs event-dependent sharing; stop if oracle events do not improve the byte–quality frontier.
  Data: HO-Cap (8 fixed RGB-D + HoloLens; pick-and-place / handover; poses) first; DiVa-360 (53 fixed cameras);
  ARCTIC (8 fixed RGB); DexYCB (8 fixed RGB-D; single pickups). Clips not visually audited.

**Rankings (Sol):** NeurIPS workshop and CVPR 2027 — controlled motion-only RD measurement > presence storage policy
> event-dependent sharing; 2027 main-venue project — event-dependent sharing (if oracle byte savings appear) >
presence > measurement. Fable concurs that no mechanism is open within "few motion models, RANSAC-style" itself.
