# CCR novelty record (2026-08-20)

The primary-source evidence behind [[ccr-method-2026-08-20]] section 1,
in three parts, each produced by a bounded worker during the 2026-08-20
block and reviewed by the primary: (A) the missing-middle mechanism
matrix over the mandatory literature set; (B) the hostile kill-search
over the two shortlisted candidates (16 distinct searches); (C) full-text
verification of the three surviving threats (CubifyGS, ClipGStream, the
Inria EG26 adaptive spatio-temporal method). Provenance caveats are
inside each part; parts A/B lean on WebFetch summarization for some
sources, part C reads the named threats in full (the Inria paper from
the author PDF directly).

---

# PART A — mechanism matrix
# Lane C â€” Mechanism/Novelty Matrix for "Observation-Born Episodic Lineages"

Compiled 2026-08-20. READ-ONLY sweep. Sources: arXiv abstracts/HTML full text,
official project pages, and existing ADAGS paper notes under
`D:\adags\research-wiki\papers\`.

**Hypothesis under pressure (not evaluated here, only mapped):** sparse
observation-born dynamic packets; independent by default; **selective
consolidation** of packets across temporally separated observations into a
shared appearance/identity while retaining episode-local pose/motion;
discontinuous temporal support with exact absence; ordinary primitives keep the
standard global temporal model. Claimed novelty locus = **selective
consolidation + discontinuous identity-sharing**.

**Decomposition used for "occupancy" judgments (5 components):**
- C1 observation-born/observation-local primitive birth
- C2 temporal partitioning / localized temporal support
- C3 discontinuous (multi-interval) support per primitive with exact absence
- C4 identity/appearance sharing across **temporally separated** observations
- C5 **selective consolidation decision** (a mechanism that decides which
  separated packets are the same surface and merges their identity)

---

## Verification status legend

- **[P]** verified from primary text (arXiv HTML/abstract fetched this run)
- **[N]** from an existing ADAGS deep-dive note (itself primary-source derived)
- **[S]** search-result summary only â€” NOT verified against primary text
- Anything marked [S] must not be cited as evidence without a primary read.

---

## Compressed matrix

| # | Work | Birth / init | Canonical vs obs-local | Temporal support shape | Multi separated episodes per primitive? | Identity sharing across separated observations | Motion model | Flow/depth use | Capacity policy | Reactivation semantics | Code | N3V PSNR (protocol) |
|---|------|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Spacetime Gaussians (STG)** [N] | Shared SfM cloud over all timestamps + error/depth-guided ray sampling (up to 3x) | Obs-local in TIME (per-primitive Î¼_t), global in space | 1D Gaussian RBF opacity `Ïƒ_sÂ·exp(-s_t|t-Î¼_t|Â²)`; soft, unimodal | **No** | **None** | deg-3 poly position, deg-1 poly rotation | rendered coarse depth only | clone/split + aggressive prune + guided sampling | none | **Yes** (oppo-us-research/SpacetimeGaussians) | **32.05** / DSSIM1 0.026 / LPIPS 0.044; 6 scenes, 300 frames, 1352Ã—1014, first (center) cam held out |
| 2 | **Ex4DGS** [N][P] | COLMAP (can be first-frame-only); top-Î·%=2 by motion metric converted to dynamic *during training* | Explicit keyframed, obs-local at sparse keyframes (I=10) | **Flat-top two-sided Gaussian**: fade-in, plateau [a_s,a_f], fade-out â€” ONE window | **No** (one onset/offset pair) | **None** | cubic Hermite (pos) + Slerp (rot) between keyframes | no | progressive temporal window; visibility-weighted error backtracking prune | none | **Yes** (juno181/Ex4DGS) | **32.11** / SSIM 0.9422 / LPIPS 0.0478; per-scene: coffee 28.79, cook 33.73, cut 29.29, salmon 33.91, flame_steak 33.69, sear 33.23 |
| 3 | **FreeTimeGS** (CVPR 2025) [P] | **RoMa multi-view matching + triangulation â†’ primitives placed at arbitrary (x,t)**; kNN cross-frame matching for initial velocity | **Fully observation-local in spacetime** (no canonical space) | Gaussian temporal opacity `exp(-Â½((t-Î¼_t)/s)Â²)` â€” explicitly ONE active interval | **No** ("one active interval per primitive") | **None** â€” "no cross-primitive appearance sharing; each Gaussian maintains independent parameters" | explicit linear velocity `Î¼_x + v(t-Î¼_t)`, annealed | RoMa matching for init; no depth | **budget-neutral periodic relocation** every 100 it, score `Î»_gâˆ‡_g + Î»_o Ïƒ`; 4D reg loss Î»=1e-2 | none (relocation overwrites, does not restore own content) | project page zju3dv.github.io/freetimegs (repo not confirmed this run) | **33.19** / DSSIM2 0.013 / LPIPS 0.036; 6 scenes, first 300 frames, 19â€“21 cams, 2704Ã—2028 Ã—0.5 |
| 4 | **FreeTimeGS++** (2605.03337) [P] | **UFM (Unified Flow & Matching) guided, per-frame observation-local**: project 3D pts â†’ sample multi-view flow â†’ back-project velocities; kNN fallback | Observation-local (inherits FreeTimeGS) | **Gated marginalization**: `Ï†_i(t) = g_i + (1-g_i)Â·exp(-(t-Ï„_i)Â²/2(d_i/6)Â²)`, gate `g_i=Ïƒ(Î³gÌƒ_i)` â†’ persistent vs transient role | **No** â€” single Ï„_i, d_i; "no evidence supports sharing identity across disconnected observations" | **None** | **shared neural velocity field** `v_Î¸(x,t)` (hash-grid) replacing per-Gaussian v | UFM optical flow for init | inherits relocation; adds color correction | none | "will release"; GitHub SNU-VGILab/FreeTimeGSPlusPlus exists [S] | DyNeRF **33.51** (FreeTimeGS_ours reproduction 32.62); DSSIM2 0.011; LPIPS 0.056 (worse than 0.033 baseline â€” suspicious, re-verify before citing) |
| 5 | **SharpTimeGS** [N] | COLMAP static (v=0, lifespan â‰ˆ3Ã— seq) + **flow+SAM2 dynamic seeding** with 3D-matched velocity, Ïƒ_t â‰ˆ3 frames | Observation-local per primitive (center time T) | **Flat-top lifespan**: 1 inside \|tâˆ’T\|â‰¤r, Gaussian tail outside â€” ONE window | **No** | **None** | linear velocity damped by lifespan `v/f(Ïƒ_t,r)`, `f=max{1,(Ïƒ_t+r)Â²}` | RAFT-class flow + SAM2 for init | Stage-2 velocity-lifespan-aware score `Î»_eE+Î»_oO+Î»_l(...)`, **fixed count** (clone==prune) | none | **No** (repo is project page only) | ~**33.57** self-reported (per query pack; not re-verified this run) |
| 6 | **MoRel** (2512.09270, CVPR 2026) [P] | Key-frame Anchors (KfAs) at periodic indices, each initialized from a Global Canonical Anchor then optimized locally | **Locally canonical per keyframe anchor** | Per-anchor-point learnable temporal offset + decay speed; bidirectional blending between adjacent KfAs | **No** | **NO â€” "each KfA's Gaussians are independent"**; on-demand loading of 1â€“2 anchors | bidirectional inter-keyframe deformation at anchor level | â€” | feature-variance-guided hierarchical densification (3 levels) | none | project page cmlab-korea.github.io/MoRel | **not reported on N3V** (SelfCapLR, 5 seqs, 3500+ frames; PSNR/SSIM/LPIPS/tOF) |
| 7 | **MAPo** [N] | Inherits E-D3DGS canonical set; **no new birth from observations** | Canonical + deformation MLP | Per-Gaussian temporal **segment** from recursive bisection (max level 3); segments are contiguous partition of [0,T] | **No** â€” one segment per (duplicated) Gaussian; duplicates are separate primitives | Duplicate inherits parent state at split, then **diverges**; no re-sharing, no cross-segment identity | E-D3DGS coarse+fine deformation nets, **cloned per sub-window** | no | recursive temporal bisection + static bypass | none | **Not found** | +0.48 dB over E-D3DGS on MeetRoom; N3V flame_salmon_frag3 29.93â†’30.36 |
| 8 | **TAD-GS** (2606.23212) [N] | Inherits periodic/RBF backbone; multiple temporal centers initialized every 20 frames | Obs-local temporal centers | Temporal RBF `Ïƒ_sÂ·exp(-Ïˆ(t-t_i)Â²)` â€” unimodal | **No** | **None** | K=4 Fourier modes + TOW time warping | flow only for masked metrics | **VAD** visibility-normalized gradient; **TAT** lifespan-scaled threshold | none | **Not found** | 32.42 (their backbone); M-PSNR 24.68 |
| 9a | **Director** (2604.01678, ShanghaiTech+ByteDance) [P] â€” the 2026 "immersive volumetric video" method | Two-layer: static bg uniform sample; **foreground re-initialized per frame** via SEA-RAFT flow + multi-view least-squares triangulation from instance masks | Per-frame observation-local foreground | Per-frame foreground tracking; bg constant | **No** | **YES-BUT (the closest thing that exists): a learnable, view-independent 8-D SEMANTIC/INSTANCE attribute per primitive, kept consistent through occlusion by KL regularization; clones inherit all parent attributes.** This is *instance labelling*, not shared radiance/geometry identity, and it is carried by continuous per-frame tracking, not by re-association across a gap. | per-frame flow-triangulated positions + ARAP | **SEA-RAFT flow + triangulation (heavy)** | clone/prune capped at <5% of Gaussians modified per frame | **No explicit inactiveâ†’reactivated mechanism** ("focus is continuous tracking and cloning") | Not stated; page caiyw2023.github.io/Director | ST-NeRF Basketball 38.912/0.967/0.0463; **N3V not reported in fetched text** |
| 9b | **ATGS** (SIGGRAPH/TOG 2026) [P] | Anchors initialized from **per-view extracted keyframes**, time-conditioned | Anchor-local (Scaffold-style), Gaussians decoded from anchor features | Anchors localize **spatial AND temporal support**; temporal windowing activates only time-relevant anchors | **No** (not described) | Not described; multi-level anchor features = global + local-spatial + local-temporal | anchors query spatio-temporal grids â†’ decoded temporal Gaussians | â€” | not described | none described | Not stated | Long360/MeetRoom/VRU/SelfCap/PKU-DyMVHumans; **no N3V** |
| 10 | **DSD-GS** (2605.30863) [P] | **Feed-forward GS encoder + optical flow, deterministic, COLMAP-free** | Static/dynamic split; init is observation-derived | Not specified in abstract | Not specified | Not described | Not specified in abstract | **yes â€” optical flow drives the decomposition** | static regions excluded from dynamic compute | none described | Not stated | claims SOTA-ish quality, 10-min training, 700+ FPS RTX 5090; **numbers not in abstract** |

### Additional 2026 sweep hits (relevant, not on the mandatory list)

| Work | Why it matters | Occupancy |
|---|---|---|
| **TRiGS** (2604.00538) [P] | **Explicitly names the identity problem**: "primitives are repeatedly eliminated and regenerated to track complex dynamics", degrading "long-term temporal identity". Fix = continuous SE(3) exponential-map motion + hierarchical quadratic-BÃ©zier residuals inside visibility-driven temporal windows; motion-guided relocation recycles low-opacity primitives; fixed 0.5M budget; Gaussian temporal opacity window [Î¼_tâˆ’2s_t, Î¼_t+2s_t]. **N3V PSNR 33.36, LPIPS 0.031** | Occupies C2 + *identity continuity through persistence*; does **NOT** occupy C3/C4/C5 â€” it avoids fragmentation by never fragmenting, not by re-associating fragments |
| **ClipGStream** (2604.13746) [P] | Reference Clip + Source Clips; source clips **inherit anchors** and add residual anchors for new/displaced structure; **static features `f_s` shared across clips**; decoder trained on Clip0 frozen and reused; per-clip independent spatio-temporal fields. N3DV **32.53** (5Ã—300-frame scenes) | Occupies C2 + a *contiguous* form of cross-segment parameter sharing (adjacent clips, always-present static content). Not separated-observation consolidation |
| **4D Primitive-MÃ¢chÃ©** (2512.16564, CVPR 2026 Oral) [N] | Per-primitive SE(3) poses â‡’ any observed primitive replayable at any timestamp; parent-child occlusion extrapolation | Occupies "previously observed content remains queryable while occluded" â€” but via *inherited parent pose*, not via consolidating two separate observations of the same surface. Not a GS/rasterizer method |
| **PersistGS** (2606.03479) [N] | Object permanence through full occlusion via differentiable rigid-body physics | Occupies occluded-interval trajectory supply; single continuous object identity assumed a priori (MV-SAM3D), never re-derived |
| **CTRL-GS** (2505.18306) [N] | Scene-global flow-based temporal segmentation + segment/frame cascaded deformation | Occupies scene-global segmentation only; segments are contiguous; no per-primitive episodes |
| **TOM-GS** (2607.22717) [N] | Presence-only: one Gaussian bump per primitive; explicitly **no** mixture/periodicity/multi-window | Direct negative evidence for C3 in the presence-only family |
| **CEC-4DGS** (2511.16112) [N] | Time-local birth at clustered error sites, temporal opacity peaked at the error frame, motion inherited from nearest group | **Strongest existing occupant of C1** (observation/deficit-born time-local birth); unbudgeted; no consolidation |
| **4D Scaffold-GS** (2411.17044) [N] | Generalized-Gaussian temporal envelope `exp(-(|Î”t|/Ïƒ)^Î²)`; temporal-coverage-weighted anchor growth | C2 only; still unimodal |
| **Adaptive Spatio-Temporal 3DGS for Oscillatory Motion** (Eurographics 2026, INRIA) [P, partial] | **Per-Gaussian keyframe LIST**, adaptively grown by temporal error variance; per-primitive splines. Closest structure to per-primitive changepoints. PDF fetch exceeded size limit â€” *temporal-support topology (single vs multi-interval) and exact-absence behavior NOT verified*. Code: github.com/graphdeco-inria/adaptive-spatio-temporal-gaussians | **OPEN ITEM** â€” must be read before any C3 claim |
| **TrackerSplat** (2604.02586, SIGGRAPH Asia 2025) [P] | Off-the-shelf point tracks triangulated to guide relocation/rotation/scale of Gaussians **before training**. Code: yindaheng98/TrackerSplat | Occupies "tracks inform Gaussian placement". Occlusion/track-loss re-association **not described** in the fetched text |
| **LocalDyGS** (2507.02363, ICCV 2025) [S] | Seeds define local spaces; **static feature shared across all time steps** + dynamic residual field | Sharing is across *contiguous* time within one seed's local space, not across separated episodes |
| **Multi4D** (2606.22197) [N] | static / persistent-dynamic / transient typing; velocity-aware lifting promotes deformation Gaussians into 4D transient primitives | Occupies "ordinary primitives keep global temporal model while transient content gets 4D primitives" â€” i.e. the *hybrid* half of the hypothesis. No consolidation; lifting is one-way |
| **ReAct-GS** (2510.19653) [S] | "Re-activation" is parameter perturbation of *frozen/stalled* primitives in **static** 3DGS optimization â€” an optimizer fix, not temporal reactivation | Name collision only; not a competitor |

---

## Answers

### A. Does ANY published method consolidate temporally separated observation-born primitives into shared-identity lineages?

**No. The 2026-08-08 finding stands, and is now re-verified against 2026 literature.**

Positive evidence for the negative, in the strongest available form:

1. **FreeTimeGS** (primary text): "one active interval per primitive"; **"No
   cross-primitive appearance sharing is described; each Gaussian maintains
   independent parameters."** This is the flagship observation-born spacetime
   method and it explicitly declines C4/C5.
2. **FreeTimeGS++** (primary text): single Ï„_i, d_i; "the paper does not
   discuss primitives spanning multiple separated temporal intervals... **no
   evidence supports sharing identity across disconnected observations.**"
   Its `gated marginalization` gate is a *binary role* (persistent vs
   transient), which is exactly the "ordinary primitives keep the standard
   global temporal model" half of the hypothesis â€” but with no episode set and
   no consolidation.
3. **MoRel** (primary text): **"No identity sharing occurs across segments.
   Each KfA's Gaussians are independent."**
4. **TOM-GS** (note, primary-derived): "no mixture, periodicity, or
   multi-window formulation anywhere in the paper"; repeat appearances are
   handled at the *population* level by distinct primitives.
5. **MAPo**: temporal bisection duplicates a Gaussian into two sub-window
   primitives that then diverge â€” the mechanistic **opposite** of
   consolidation (fission without fusion).
6. **TRiGS**: names the fragmentation-of-identity problem out loud in 2026 and
   solves it by *preventing* fragmentation (continuous SE(3) + relocation),
   not by re-associating separated packets.

Partial/adjacent occupancy that must be disclosed:
- **Director** (ByteDance/ShanghaiTech, 2026) is the only work found carrying a
  **per-primitive, time-invariant identity attribute** (8-D semantic vector,
  KL-regularized, inherited by clones, "identity remains consistent through
  occlusions"). But (a) the shared quantity is a *semantic/instance label*, not
  radiance/geometry; (b) identity is maintained by **continuous per-frame flow
  tracking**, not re-derived across a gap; (c) no inactiveâ†’reactivated
  primitive mechanism exists. It occupies "identity attribute survives
  occlusion", not "separated packets are consolidated".
- **ClipGStream** shares *static* features and a frozen decoder across clips â€”
  sharing across contiguous temporal partitions of always-present content.
- **4DPM / PersistGS** supply poses through occlusion for an object whose
  identity was given a priori.

**C5 (the selective consolidation decision) is unoccupied. C3 (multi-interval
support with exact absence) is unoccupied in everything verified, with one open
item: the INRIA Eurographics-2026 per-Gaussian keyframe-list method must be
read before this is asserted as clean.**

### B. Closest single foil and the one-sentence differentiation

**Closest single foil: FreeTimeGS (CVPR 2025), with FreeTimeGS++ (2605.03337)
as its 2026 successor.** It is the only method that is simultaneously
observation-born in spacetime (RoMa matching + triangulation places primitives
at arbitrary (x,t)), temporally localized, budget-managed by relocation, and
the current N3V quality reference (33.19; ++ reports 33.51).

**Surviving differentiation (one sentence):** *FreeTimeGS gives each
observation-born primitive exactly one temporal lobe and never lets two
primitives share parameters, so a surface observed before and after an
occlusion is reconstructed as two unrelated primitives; the proposal adds a
decision that consolidates such separated packets into one lineage with shared
appearance/identity and episode-local pose, which is a capability no verified
method has.*

Runner-up foils, and why they do not displace it:
- **TRiGS** â€” closest on *identity*, but it preserves identity by persistence
  (never fragmenting), so it cannot handle content that genuinely leaves and
  returns; and it is a fixed-budget relocation method, not observation-born.
- **Director** â€” closest on *identity attributes surviving occlusion*, but the
  shared quantity is semantic labelling on top of per-frame re-initialized
  foreground, driven by continuous flow tracking.
- **CEC-4DGS** â€” closest on *observation/deficit-born time-local birth*, and it
  is the natural B1 accuracy reference, but has zero identity machinery.

### C. Most sensible B1 ingredient (observation-born init WITHOUT consolidation)

**Recommend: FreeTimeGS-style spacetime birth.** Reasons, ranked:

1. **It is the ablation the hypothesis needs.** B1 must be "observation-born
   packets, independent by default" â€” that is *literally* FreeTimeGS's stated
   representation ("one active interval per primitive", "no cross-primitive
   appearance sharing"). Adding consolidation on top then isolates exactly C4+C5
   as the delta, with no confound from the birth mechanism.
2. **Smallest surface area against the existing ADAGS repo.** The pieces are: a
   per-primitive (Î¼_t, s) temporal opacity, a linear velocity term, and a
   relocation step keyed on `Î»_gâˆ‡_g + Î»_o Ïƒ`. Temporal opacity and
   gradient/opacity-scored capacity moves are already the repo's vocabulary
   (LoRA route0 + clone/split/prune + the fixed-count reassignment substrate
   from B01). No MLP decoder, no keyframe spline machinery, no SAM2/flow
   dependency is required for the minimal variant.
3. **It is the strongest published reference point on the target benchmark**
   (33.19 N3V under the standard 300-frame / 1352Ã—1014-equivalent / held-out
   first-camera protocol), so B1's number is directly interpretable.
4. **It is budget-neutral by construction** (relocation, not growth), which
   matches the project's hard G5 requirement for matched-capacity controls â€”
   unlike CEC-4DGS-style unbudgeted birth.

Why not the alternatives:
- **STG-style per-frame triangulation clouds**: the birth mechanism is
  *error/depth-guided ray sampling from the model's own rendered depth*, which
  the project has already shown to be structurally unreliable at exactly the
  disocclusion sites of interest (see CEC-4DGS critique in
  `kang2025_cec_4dgs.md`), and STG's guided sampling contributes only +0.3 dB
  in its own ablation â€” a weak ingredient to build a lane on. Public code is
  excellent, so keep STG as a **baseline**, not the B1 substrate.
- **Ex4DGS-style keyframes**: keyframes are a *motion-representation* choice
  (Hermite/Slerp between sparse samples) with a contiguous flat-top opacity
  window; adopting it imports a spline machinery that is orthogonal to
  consolidation and makes "exact absence between episodes" harder to express
  cleanly. Its real value is as a second baseline (public code, per-scene N3V
  numbers already tabulated above) and as the backbone CEC-4DGS builds on.
- Practical caveat: **FreeTimeGS's official code was not confirmed public in
  this run** (project page only; no repo URL in the fetched HTML). The
  FreeTimeGS++ authors state they reproduced and formalized FreeTimeGS as
  `FreeTimeGS_ours` and a repo `SNU-VGILab/FreeTimeGSPlusPlus` appears in
  search results [S, unverified]. B1 should be implemented from the paper's
  equations (all of which are recovered above) inside the ADAGS repo rather
  than ported.

---

## The 2â€“3 works that most threaten the hypothesis's novelty

1. **FreeTimeGS++ (2605.03337, 2026)** â€” the single biggest threat. Its
   *gated marginalization* already implements "ordinary primitives keep the
   standard global temporal model while transient primitives get localized
   support", as a **learned continuous gate**, and its UFM-guided per-frame
   initialization already implements observation-local birth. Everything in
   the hypothesis *except* C3/C4/C5 is now published. It also raises the
   quality bar (33.51 DyNeRF) and explicitly frames itself as the principled
   account of "the secrets of dynamic GS", which invites reviewers to ask what
   is left.
2. **TRiGS (2604.00538, 2026)** â€” pre-empts the *motivation*. It states the
   identity-fragmentation problem in the same words the hypothesis would use,
   in 2026, and reports **33.36 on N3V**. Any framing of "primitives are
   repeatedly regenerated and lose identity" is now a cited-and-solved problem
   statement; the proposal must argue that persistence-based identity fails
   precisely where content genuinely disappears and returns.
3. **Director (2604.01678, ByteDance/ShanghaiTech, 2026)** â€” pre-empts
   "identity survives occlusion" at the primitive level with a time-invariant
   per-Gaussian identity vector and explicit KL-based consistency through
   occlusion. It occupies the vocabulary of primitive-level identity in
   dynamic volumetric video even though the mechanism (semantic label +
   continuous tracking) is different.

Honourable mention: **the INRIA Eurographics-2026 adaptive spatio-temporal
method** â€” per-Gaussian adaptively-grown keyframe lists for oscillatory
(repeating) motion is the nearest structure in the literature to per-primitive
episodes. **This is the one unresolved item in the sweep** and it should be
read from the PDF before any "multi-interval support is unoccupied" claim is
recorded in the wiki.

---

## Open items / what this sweep did NOT establish

- INRIA `adaptive-spatio-temporal-gaussians` (Eurographics 2026): PDF exceeded
  the fetch size limit; temporal-support topology unverified.
- DSD-GS: only the abstract was accessible; no mechanism detail, no numbers.
- ATGS: project page only; no arXiv id, no N3V numbers, densification and
  cross-anchor sharing not described.
- SharpTimeGS 33.57: carried from the ADAGS query pack, not re-verified here.
- FreeTimeGS++ LPIPS 0.056 vs baseline 0.033 while PSNR improves is internally
  odd; re-read the table before citing.
- One search-engine summary asserted that "some methods support multiple
  separated temporal intervals" without naming any. No primary source was found
  for this and it is treated as unsupported synthesis, not evidence.
- FreeTimeGS official code availability unconfirmed.


---

# PART B — kill-search

# Lane C hostile novelty kill â€” CCR (M1) and Channel-Selective Sharing (M4)

Date: 2026-08-20. Mode: adversarial (find prior art that occupies the mechanism).
Evidence level: abstracts + project pages + search snippets. **Full-text mechanism
was NOT read for CubifyGS, ClipGStream, or the INRIA Eurographics 2026 paper** â€”
those three are the highest-risk unverified items and must be read in full before
any claim is fixed.

## Searches actually run (16 distinct)

1. Gaussian splatting merge primitives trial render acceptance criterion structural operations
2. dynamic GS re-association after occlusion primitive identity reappearance
3. 4D GS parameter sharing across time segments tied appearance SH
4. arXiv 2026 dynamic GS consolidate merge lineage identity reuse primitives
5. INRIA adaptive spatio-temporal 3DGS project page (direct fetch)
6. L2D2-GS 2606.29374 (search + abs fetch)
7. GS merge Gaussians validation render loss accept/reject candidate merge
8. FreeTimeGS spacetime birth lifespan temporal opacity
9. GS SH codebook sharing / VQ / per-band compression
10. GauSTAR topology change re-create surface tracking
11. CTRL-GS dynamic scene decomposition
12. streaming GS anchor reuse across clips / ClipGStream (search + abs fetch)
13. "trial render" OR "counterfactual" acceptance test structural edit GS held-out ray
14. MDL / model selection merging Gaussians parameter tying
15. TrackerSplat / CubifyGS (search + CubifyGS abs fetch)
16. "intermittent"/"episodic"/"discontinuous" temporal support multi-interval;
    amodal/object-permanence reuse after full occlusion; canonical-deformation SH
    tying; R3G ray grouping; MoE-GS/MoDE; SWinGS; Re-Activating Frozen Primitives;
    Learning Stable Canonical Worlds (2606.23027, fetched â€” irrelevant, feed-forward
    multi-view fusion, no temporal identity).

---

## Killers, wounds, misses

### K1. Canonical-space deformable 3DGS family â€” **KILLS the naive form of M1's sharing claim**
Deformable-3DGS (2309.13101), SC-GS (2312.14937), GaGS, MoDE / MoE-GS (2607.08250,
2510.19210), TimeFormer, and essentially every deformation-field method.
**Exact mechanism:** one canonical Gaussian per primitive holds SH *and* opacity;
the time-conditioned deformation MLP emits offsets for position/rotation/scale
only. Appearance is therefore **tied across the entire sequence by construction,
including across every occlusion**. MoDE additionally has multiple deformation
experts "operating on a shared canonical Gaussian representation".
**Verdict: KILLS** any claim of the form "we share appearance between temporally
separated observations of the same surface." That is the field's default, not a
contribution. It **WOUNDS** M1 into: sharing must be *selective, per-pair,
data-accepted, and constructed inside a representation that has no sharing by
default* (observation-local birth). Note the flip side: canonical methods cannot
express exact-zero support or episode-local opacity, so the *union-of-episodes with
exact zero between* half survives.

### K2. CubifyGS â€” Object-Centric 3DGS for Lifelong Dynamic Scene Maintenance (arXiv 2606.28720, IROS 2026) â€” **WOUNDS M1 severely**
**Exact mechanism (abstract-level):** detects object appearance and disappearance;
models movable instances as **reusable Gaussian assets**; scene updates proceed by
**asset retrieval + rigid transformation + explicit pruning rather than
reconstruction from scratch**; event-triggered adaptive optimization focuses
compute on affected regions.
**Why it hurts:** this is "reuse previously trained content when the thing comes
back", i.e. the *same idea* as CCR consolidation, already published â€” at object
granularity, with retrieval as the matcher.
**Verdict: WOUNDS.** M1 must narrow to: (a) **primitive**-level, not instance-level;
(b) **no instance segmentation / no rigid-asset assumption** â€” episode-local
non-rigid pose and motion are retained and only appearance is tied; (c) reuse is
**admitted by a render-based counterfactual test**, not by a retrieval match; (d)
the object is not required to be a movable rigid asset. Anything less and a
reviewer reads CCR as CubifyGS at finer granularity.
**Unverified:** whether CubifyGS gates retrieval by any reconstruction test, and
whether asset reuse re-optimizes SH. Read the full paper.

### K3. PersistGS â€” Differentiable Physics for Object Permanence in 4D GS (arXiv 2606.03479) â€” **WOUNDS (problem-level), MISSES (mechanism)**
**Exact mechanism:** when an object is fully occluded from all training cameras its
Gaussians lose photometric gradient and degrade; PersistGS keeps the **complete**
Gaussian set alive and posed by a differentiable **rigid-body SE(3)** trajectory
through the occlusion, and supervision resumes on re-emergence.
**Verdict: WOUNDS.** Kills "first to preserve a primitive's trained content across
a full occlusion." **MISSES** the mechanism: continuous (never-zero) support, a
physics prior instead of a data-driven decision, no descriptor matching, no
consolidation of *separately born* packets, no acceptance test, rigid bodies only.
CCR's differentiator: it never assumes the object persisted â€” it *tests* whether a
before/after pair should be treated as the same thing, and works on non-rigid
content where no physics prior exists.

### K4. ClipGStream (arXiv 2604.13746, CVPR 2026) â€” **WOUNDS the "structure carried across segments" framing**
**Exact mechanism (abstract-level):** clip-level stream optimization; clip-independent
spatio-temporal fields + residual anchor compensation; **inter-clip inherited anchors
and decoders** maintain structural consistency across clips.
**Verdict: WOUNDS.** Cross-segment parameter inheritance exists. **MISSES** on all
three CCR specifics: clips are **contiguous** (no absence), inheritance is
**unconditional** (initialization/consistency, not a tested tie), and there is no
per-primitive multi-interval support with exact zero.
**Unverified:** whether inherited anchors are *tied* (shared parameters) or merely
initialized. If tied, this becomes the strongest "shared parameters across temporal
segments" citation and M1 must explicitly contrast contiguous-inheritance vs
absence-spanning-consolidation. Read the full paper.

### K5. L2D2-GS (arXiv 2606.29374, Jun 2026, PKU + Xiaomi EV) â€” **WOUNDS the "reconstruction-gain-driven structural decision" claim**
**Exact mechanism:** feed-forward dynamic reconstruction reformulated as iterative
optimize+densify; a **self-supervised densification policy derives explicit reward
signals from global reconstruction gains to guide local densification**; zero-shot,
policy learned **offline**.
**Verdict: WOUNDS.** "Reconstruction gain decides a structural op" is occupied.
**MISSES** on: densification only (never merge/consolidate/tie), amortized offline
policy rather than a live per-decision paired trial render, no held-out-ray split,
no identity semantics. CCR must say "test-time paired counterfactual acceptance",
never "reconstruction-gain-guided structural search".

### K6. 3DGS compression: SH codebooks and per-band assignment â€” **NEAR-KILL for M4**
LightGaussian (2311.17245), CompGS (2311.18159), Compact-3DGS, Reduced-3DGS (INRIA),
ContraGS (2509.03775), RDO-Gaussian (ECCV 2024), 3DGS.zip survey (2407.09510).
**Exact mechanisms:** VQ of SH coefficient vectors under the explicit assumption
that *a subgroup of Gaussians shares appearance*; **separate SH codebook** vs
scale/rotation codebook; **SH band assignment performed once mid-optimization**
(Reduced-3DGS) deciding per-Gaussian how many SH bands to keep; SH-degree
distillation; degree>0 coefficients treated as the compressible block.
**Verdict for M4: WOUNDS to the point of near-occupation.** "Per-SH-band selective
sharing of appearance across primitives, decided during optimization" already exists
in compression, at scale, with rate-distortion criteria. M4's only surviving
differentiators are (i) sharing is between two *specific episodic packets of one
lineage* (identity), not a global codebook, and (ii) each block is admitted by a
**cross-fitted non-inferiority test on held-out rays** rather than an RD objective.
That is a thin margin for a standalone contribution.

### K7. FreeTimeGS (2506.05348) / FreeTimeGS++ (2605.03337) / SharpTimeGS (2602.02989) / STG / 4DGS-native (2412.20720) â€” **MISSES, but owns the word "reuse"**
**Exact mechanism:** primitives born anytime/anywhere, each with a motion function
and a temporal opacity function; explicit lifespan regularization; SharpTimeGS
replaces Gaussian temporal decay with a learnable **flat-top lifespan**. FreeTimeGS
states motion "facilitates the **reuse** of Gaussian primitives along the temporal
dimension".
**Verdict: MISSES.** Support is a **single contiguous** interval (unimodal or
flat-top); "reuse" means a primitive *moves* to cover neighbouring regions, not that
two disjoint packets are identified as one entity. But do not use the word "reuse"
unqualified in the paper â€” say "cross-absence consolidation".

### K8. GauSTAR (CVPR 2025) â€” **MISSES; the clean foil**
Detects topology change from positional gradients + reconstruction error, **unbinds**
Gaussians on changed surfaces and **adds new** ones, re-meshing. Re-create, never
re-associate. Exactly the opposite policy to CCR; keep as the named contrast.

### K9. INRIA "Adaptive Spatio-Temporal 3DGS for Scenes with Oscillatory Motion" (Eurographics 2026, CGF 45(2)) â€” **MISSES on current evidence**
**Exact mechanism:** each Gaussian carries an **associated keyframe list queried by
time** to yield displacement and rotation (per-primitive splines); keyframes are
**added adaptively based on the variance of the temporal error**.
**Verdict: MISSES.** Project page shows no multi-interval / exact-zero support, no
appearance sharing across segments, no merge/consolidation â€” the keyframe list is
adaptive temporal *resolution* of a continuous trajectory, not multi-interval
*presence*. **WOUNDS** the weaker framing "per-primitive adaptive temporal
structure", which is now occupied. **Unverified:** whether a keyframe list can encode
zero opacity spans. Read the CGF paper before claiming multi-interval support is new.

### K10. TrackerSplat (SIGGRAPH Asia 2025, 2604.02586) â€” **MISSES; mandatory citation**
Off-the-shelf 2D point tracks triangulated across views to *pre-update* Gaussians
before refinement; reduces fading/recoloring under large motion. Tracks drive
geometry initialization, never identity consolidation or parameter tying. Cite it
because CCR's descriptor-matching proposer will otherwise be read as track-driven.

### K11. Others checked, all MISS
- **CTRL-GS (2505.18306)**: cascaded video/segment/frame temporal residues â€” a
  hierarchical shared base + residual, weakly wounds M4's block decomposition framing.
- **R3G (2603.24994)** and **SpeeDe3DGS motion grouping (2506.07917)**: grouping
  shares **motion**, not appearance.
- **SWinGS (ECCV 2024)**: separate model per sliding window, canonical representation
  *changes* per window (adjacent windows share only a frame) â€” the anti-CCR.
- **ReAct-GS (2510.19653)**: "re-activating frozen primitives" is parameter
  perturbation of stalled Gaussians in **static** densification. Name collision only.
- **Director (2604.01678)**: instance-consistent semantic identity via MLLM masks â€”
  identity as a *semantic label*, not a parameter tie; no render-based acceptance.
- **Dynamic 3D Gaussians (Luiten)**: persistent identity by construction, no absence.
- **CausalSplat (2608.11150)**, **Learning Stable Canonical Worlds (2606.23027)**:
  irrelevant on inspection.

### K12. Not found â€” the genuinely unoccupied cell
No paper was found in which a **per-decision paired counterfactual trial render**
(render with the tie, render without the tie, compare held-out-ray error) **accepts
or rejects a structural parameter tie at test time**, with **exact restoration of
independence on rejection**. L2D2-GS is the nearest (offline amortized reward,
densification only). This is consistent with the project's prior LGS/EL-GS finding
and survives this round.

---

## (1) Verdict for M1: **NARROWED** (not CLEAR, not OCCUPIED)

Surviving one-sentence novelty claim:

> Starting from an observation-local spacetime-birth representation that has no
> cross-time primitive identity by construction, we recover identity **across an
> absence at the primitive level** by consolidating temporally disjoint packets into
> a lineage that ties **only trained appearance** while keeping per-episode pose,
> motion and opacity with exact-zero support in between â€” and we admit each tie only
> when a **paired counterfactual trial render reduces held-out-ray error**, with
> rejection restoring exact independence.

Load-bearing words that must all stay: *primitive-level*, *across an absence*,
*only appearance*, *exact-zero support*, *paired counterfactual trial render*,
*held-out ray*, *exact restoration on rejection*. Drop any one and a specific paper
takes the cell: drop "only appearance / observation-local" â†’ canonical deformable
3DGS; drop "primitive-level / non-rigid" â†’ CubifyGS; drop "across an absence" â†’
ClipGStream; drop "test-time paired trial render" â†’ L2D2-GS; drop "identity" â†’
SH-codebook compression.

## (2) Verdict for M4: **NARROWED to the point of not being a standalone contribution**

Surviving claim, honestly stated:

> The identity tie is admitted **per appearance block** (DC, then successive SH
> orders) by cross-fitted non-inferiority on held-out rays, so a lineage can share
> low-frequency colour while keeping view-dependent terms episode-private.

Recommendation: **demote M4 to an ablation/refinement of M1, not a second claim.**
Per-SH-band selective treatment during optimization is occupied by compression
(Reduced-3DGS band assignment; LightGaussian/CompGS/Compact-3DGS SH codebooks
explicitly premised on subgroups sharing appearance). Presented standalone, a hostile
reviewer will call it "VQ-style per-band SH sharing with a hypothesis test attached".
Presented as the granularity ablation of M1's tie, it is defensible and it
strengthens M1 by showing the tie is not all-or-nothing.

## (3) The three works a reviewer will cite against the paper

1. **CubifyGS â€” Object-Centric 3DGS for Lifelong Dynamic Scene Maintenance (arXiv
   2606.28720, IROS 2026).**
   *Pre-emptive sentence:* "CubifyGS reuses whole instances as rigid Gaussian assets
   retrieved and re-posed after disappearance, which presupposes an instance
   segmentation and a rigid-body model; CCR makes reuse a per-primitive, non-rigid
   decision that requires no instance labels and is admitted only when a
   counterfactual trial render shows the shared appearance actually lowers held-out
   reconstruction error."

2. **PersistGS â€” Differentiable Physics for Object Permanence in 4D GS (arXiv
   2606.03479).**
   *Pre-emptive sentence:* "PersistGS *assumes* permanence and carries content
   through occlusion with a rigid-body physics prior and continuous support; CCR
   makes permanence a hypothesis that is tested against held-out rays, permits exact
   zero support during the absence, and therefore applies to deformable content for
   which no physics prior is available."

3. **The canonical-deformation family (Deformable-3DGS 2309.13101 / SC-GS 2312.14937,
   with MoDE 2607.08250 as its 2026 form).**
   *Pre-emptive sentence:* "In canonical-deformation models appearance is tied across
   all time unconditionally and absence cannot be represented at all; CCR begins from
   an observation-local representation with *no* tie, and adds ties one at a time only
   where reconstruction evidence supports them â€” making cross-time appearance sharing
   a measured, revocable decision rather than an architectural assumption."

Runner-up citations to pre-empt in related work: **L2D2-GS** (reconstruction-gain
reward for structural ops), **ClipGStream** (inter-clip inherited anchors),
**FreeTimeGS/FreeTimeGS++** (the substrate and the word "reuse"), **GauSTAR**
(re-create foil), **Reduced-3DGS / LightGaussian** (M4's per-band precedent).

## Residual risk

Three highest-risk items were assessed from abstracts/project pages only:
**CubifyGS**, **ClipGStream**, and the **INRIA Eurographics 2026** paper. If
CubifyGS gates asset retrieval by a rendering test, or if ClipGStream's inherited
anchors are genuinely *tied* rather than initialized, or if the INRIA keyframe list
admits zero-opacity spans, the corresponding narrowing above tightens further. Read
all three in full before fixing the claim.


---

# PART C — full-text threat verification

# Lane C full-text verification â€” 2026-08-20

Claim under test ("the tie claim"): from an observation-local birth representation, recover identity at the
primitive level across an absence by tying ONLY appearance between temporally disjoint primitives,
episode-local pose/motion/opacity, exact-zero support in between, each tie accepted only by a paired
counterfactual trial render lowering held-out-ray error, rejection restoring exact independence.

Sources read: arXiv HTML full text for papers 1-2 (via WebFetch on arxiv.org/html/...); author-version PDF
full text for paper 3 (downloaded from repo-sam.inria.fr, text-extracted with pypdf; local copy
`adaptive_st.pdf` / `adaptive_st.txt` in this scratchpad).

---

## 1. CubifyGS (arXiv 2606.28720, IROS 2026)

Full title: "CubifyGS: Object-Centric 3D Gaussian Splatting for Lifelong Dynamic Scene Maintenance"
(Ren, Yang, Liu, Gao, Tang, Lai, Yang, Fu). Robotics lifelong mapping under RIGID object rearrangement.
Movable instances are modeled as "reusable Gaussian assets"; maps are updated by "asset retrieval, rigid
transformation, and explicit pruning rather than reconstruction from scratch" (abstract).

(a) Retrieval gating: detection/feature-matching driven, NO reconstruction-quality test.
   Sec. III-C2: "we compute the cosine similarity between the current object's visual feature and the
   multi-view feature banks of all assets"; "If the maximum similarity exceeds a threshold tau_match, we
   retrieve A*". DINO-feature cosine similarity vs threshold; no trial render, no photometric error
   criterion, no held-out check gates whether reuse happens. (Photometric error appears only AFTER
   retrieval, to estimate the rigid pose: "we freeze the Gaussian parameters and optimize the relative
   camera pose T by minimizing the photometric error" â€” pose refinement, not an accept/reject gate on
   the identity tie itself.)

(b) Granularity: whole-object assets only. Sec. III-A2: "Instantiating an asset A_k as a scene object O_i
   involves transforming its Gaussian parameters to world space." No per-primitive matching or reuse.

(c) Appearance after reuse: instantiated (copied into world space) and subsequently modified â€” the
   asset's Gaussian parameters are transformed to world space at instantiation, frozen during pose
   estimation, then refined by the event-triggered "focus-driven" optimization inside the dynamic ROI.
   No statement that the new scene instance and the earlier instance continue to share one trained
   tensor; the tie is via the asset bank, at object level, and post-reuse optimization diverges the copy.

(d) Temporal discontinuity: YES, explicit exact absence via pruning. Sec. III-C1: "Upon identifying a
   'Vanished' object ... we trigger the pruning mechanism. This explicitly removes all Gaussian
   primitives confined within the object's bounding box." Zero contribution between disappearance and
   reappearance.

(e) Continued optimization on the reused asset: YES. Sec. III-C2 reformulates object pose estimation
   "as a differentiable camera tracking problem" (gradient-based), and the event-triggered adaptive
   (focus-driven) optimization continues gradient descent on the reused asset within the scene.

Verdict vs the tie claim: WOUNDS. CubifyGS occupies "reuse previously trained Gaussian content across
an explicit absence with exact-zero support in between, in one scene" â€” but at OBJECT granularity, for
RIGID rearranged instances, with reuse gated by a DINO-similarity threshold (heuristic detection), the
whole asset (not appearance-only) copied and then re-optimized, and no counterfactual trial-render /
held-out-error acceptance and no rejection-restores-independence semantics. The claim survives narrowed
to: per-PRIMITIVE ties, appearance-ONLY sharing with episode-local pose/motion/opacity, non-rigid
content, and above all the paired counterfactual trial-render acceptance test with exact-independence
rejection. Any novelty text must cite CubifyGS as the object-level occupancy of "reuse across absence."

---

## 2. ClipGStream (arXiv 2604.13746, CVPR 2026)

Full title: "ClipGStream: Clip-Stream Gaussian Splatting for Any Length and Any Motion Multi-View
Dynamic Scene Reconstruction" (Liang, Wu, Wang, Yang, Zheng, Xiong, Wang, Yan, Gao, Wang).

(a) Tied vs re-initialized: inherited anchors/decoder are FROZEN shared parameters â€” neither continuing
   training nor re-initialized. Sec. 3.2.2: "the anchors A0 and their static features f_s,0 are inherited
   from the Reference Clip and kept fixed during the training of all subsequent clips"; "the decoder d
   trained on the Reference Clip is reused by all Source Clip[s] and remains frozen during their
   optimization." So the tensors are literally the same objects across clips, but they stop learning
   after clip 0; per-clip dynamics live in clip-independent spatio-temporal fields plus residual anchors.

(b) Across-gap consolidation: none in the pairwise sense. Sec. 3.2: "The first clip Clip0 serves as the
   Reference Clip, and the remaining clips Clip_n, n in [1..N-1], are treated as Source Clips." All
   source clips share the single reference-clip anchor set and decoder; there is no clip-i-to-clip-j
   re-identification event, no absence semantics, and no consolidation of two non-adjacent clips'
   independently-born structure. The sharing models PERSISTENT static structure, not reappearance.

(c) Acceptance test: none for inheritance â€” base anchors/decoder are inherited unconditionally. The only
   selection is geometric deduplication of NEW residual anchors (Sec. 3.2.1: "if SDF(q)>0, q is retained
   as a residual anchor ... otherwise discarded"), which filters additions, not inherited ties, and uses
   an SDF geometry test, not a rendering-error criterion.

Verdict vs the tie claim: MISSES. ClipGStream shares appearance-bearing parameters across time segments,
but for continuously present structure, unconditionally, frozen, at anchor-set granularity, with no
absence, no re-identification across a gap, and no acceptance test. It does not occupy any limb of the
claim; it is at most adjacent prior art for "parameter sharing across temporal segments in streaming."

---

## 3. Adaptive Spatio-Temporal 3D Gaussian Splatting for Scenes with Oscillatory Motion
   (Tzathas, Hu, Meuleman, Cordonnier, Drettakis; Inria/UCA; Eurographics 2026, CGF 45(2),
   DOI 10.1111/cgf.70410; author PDF from repo-sam.inria.fr/nerphys/adaptive-spatio-temporal/)

Representation (Sec. 4): canonical 3DGS parameters (center mu, covariance Sigma, opacity o, SH) plus a
per-Gaussian keyframe list storing ONLY "one translation Delta-mu_i_g, and one rotation represented by a
quaternion q_i_g" per keyframe. Between keyframes: linear interpolation for translation, slerp for
rotation (Eq. 6). Outside the keyframe range the displacement is CONSTANT-extrapolated: "Delta-mu_0_g,
t < t_0_g; ... Delta-mu_n_g, t > t_n_g" â€” the Gaussian exists over the entire sequence.

(a) Multi-interval support: NO. The keyframe list is adaptive sampling of ONE continuous trajectory over
   the whole sequence; a Gaussian is never absent. Sec. 2 explicitly positions the method AGAINST
   limited-temporal-support 4D primitives: 4D primitives "have limited temporal support, allowing them
   to fade in and out over time ... They are also redundant, as Gaussians must be duplicated when the
   motion is complex." Separated episodes with absence between are not representable.

(b) Identity/appearance sharing across separated observations: ABSENT. There is no re-detection, asset
   reuse, or tying between different Gaussians at separated times anywhere in the paper. (Trivially, one
   persistent Gaussian keeps its appearance for all t, but there is no across-absence mechanism because
   there is no absence.)

(c) Opacity between keyframes: opacity is NOT keyframed â€” it is a canonical, time-constant scalar; only
   translation and rotation vary with t. Removal is only the standard low-opacity culling (Sec. 3.2).
   Decisive quote, Appendix E (N3V multi-view experiment): "Our method does not explicitly handle
   transient effects like fire that are present in this dataset, in contrast to 4D-Scaffold-GS which
   allows opacity to vary over time."

(d) (context) Temporal densification criterion, Sec. 4.2.2: a keyframe is added when the ratio
   rho_g = sigma(e_g)/mean(e_g) of per-segment area-normalized image error exceeds tau_time ("we split
   all the segments for a Gaussian if the ratio of at least one of its segments exceeds threshold
   tau_time"), new keyframe placed "at the argmax of e_g,I". This is a thresholded error-statistic
   heuristic â€” no counterfactual trial render, no accept/reject on held-out error.

Verdict vs the tie claim: MISSES. Per-Gaussian adaptively-grown keyframe lists parameterize a single
continuous trajectory with time-constant opacity and full-sequence support; no multi-interval presence,
no absence, no cross-episode identity tie, and its adaptive growth is error-variance thresholding, not
trial-render acceptance. It is prior art only for "per-primitive adaptive temporal parameter growth,"
not for episodic support or identity recovery.

---

## Pooled verdict on the tie claim

- CubifyGS: WOUNDS â€” object-level, rigid, detection-gated reuse-across-absence is occupied; the claim
  narrows to per-primitive, appearance-only tying with episode-local pose/motion/opacity and a paired
  counterfactual trial-render acceptance (held-out-ray error decrease; rejection restores exact
  independence), on non-rigid content.
- ClipGStream: MISSES â€” frozen unconditional sharing for persistent structure; no absence, no gate.
- Adaptive Spatio-Temporal 3DGS (EG 2026): MISSES â€” continuous single-trajectory support, constant
  opacity; explicitly does not handle transients; growth is thresholded statistics, not acceptance.

Net: the claim survives with one mandatory narrowing (primitive-level + appearance-only + trial-render
acceptance are the load-bearing differentiators; "reuse across absence" alone is now occupied at object
level by CubifyGS and must not be claimed unqualified).

Caveats: paper-1 and paper-2 quotes were extracted through WebFetch's summarizer over the arXiv HTML full
text (quotes carry section numbers as reported); paper-3 quotes were verified directly against the
extracted author-PDF text. No other papers surveyed, per scope.

