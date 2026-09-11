---
title: "Literature check (2026-09-10) — motion bases, anchor/window methods, absence-and-return datasets, failure-analysis benchmarks, per-primitive presence, event-window metrics"
date: 2026-09-10
evidence_bearing: false
---

# Literature check of 2026-09-10 — promoted from the session record

Purpose: fact-check six claims made in an LLM (Gemini Flash)
conversation about the project's direction, and locate the closest
work for [[paper-thesis-v1-2026-09-11]]. Method: web search of arXiv
and project pages by a subagent on 2026-09-10, roughly 20 queries;
every entry carries its arXiv id; "not found" and "not verified" are
recorded as such. This is a literature record, not a deep read; the
deep-read pages are linked where they exist
([[papers/ramlal2026_persistgs]], [[papers/liao2026_sharptimegs]],
[[papers/ren2026_cubifygs]], [[sa4d-read-2026-08-24]],
[[spacetime-gaussian-grouping-read-2026-08-24]]).

## 1. Closest work to per-primitive presence with an inferred gap and the return of the SAME primitives

**Verdict: the cell is narrowing fast, and one 2026 paper names the
problem verbatim.**

1. **PersistGS** — Ramlal & Zelek, "Differentiable Physics for Object
   Permanence in 4D Gaussian Splatting", arXiv 2606.03479, CVPR 2026
   Workshop on Generative 3D Reconstruction
   ([[papers/ramlal2026_persistgs]]). States our motivation verbatim:
   when an object is invisible from all training cameras its Gaussians
   receive no photometric gradient, are pruned, drift or collapse, and
   the reconstruction must recreate the object from scratch on
   re-emergence. Mechanism: differentiable rigid-body simulation
   (NVIDIA Newton) supplying the SE(3) trajectory during the gap;
   per-object Gaussians plus collision meshes; synthetic scenes; five
   cameras; held-out cameras that DO see the object during the
   occlusion; PSNR and trajectory error; no event-window metric. **Owns
   the problem statement; not the inferred-window, per-primitive,
   trained-with-the-gate mechanism.**
2. **SharpTimeGS** — arXiv 2602.02989, CVPR 2026
   ([[papers/liao2026_sharptimegs]]). Learnable per-primitive lifespan
   with a flat-top visibility profile and lifespan-scaled velocity.
   Single interval; no absence and return. Makes per-primitive temporal
   support standard.
3. **FreeTimeGS** — arXiv 2506.05348, CVPR 2025; **FreeTimeGS++** —
   arXiv 2605.03337. Per-primitive temporal opacity and motion; ++
   analyses per-Gaussian lifetime. Unimodal support; no return of the
   same primitives.
4. **RetimeGS** — arXiv 2603.13783. Primitives appear and disappear via
   short-tailed temporal opacity with frame-pair regularisation.
   Single interval.
5. **CubifyGS** — arXiv 2606.28720, IROS 2026
   ([[papers/ren2026_cubifygs]]). Object-level asset pruning on
   disappearance and rigid re-alignment on reappearance; entity-level,
   rule-based existence state; RGB-D SLAM; own synthetic benchmark.
   Object-asset granularity, not per-primitive.
6. **SA4D** — arXiv 2407.04504 ([[sa4d-read-2026-08-24]]). Temporal
   identity feature field for segmentation and editing; no presence
   inference. It is the fixture's editing tool.
7. **TAD-GS** (arXiv 2606.23212), **VAD-GS**, **CEC-4DGS** (arXiv
   2511.16112): densification statistics; presence not modelled.
8. Name collision: **Re-Activating Frozen Primitives for 3DGS**, arXiv
   2510.19653, is a static-3DGS optimisation mechanism. The paper must
   not use "reactivation".

**Still open:** multi-interval (absent → present) per-primitive support
where the absence window is inferred from training-view evidence and
the SAME primitives resume, trained with the gate, on real multi-view
footage.

## 2. Multi-view (≥ 8 calibrated cameras, video) datasets with true absence and return

**Verdict: none found that documents or annotates full-multiview
disappearance and return.**

| dataset | cameras | rate / length | absence + return? |
|---|---|---|---|
| ImViD (arXiv 2503.14359, CVPR 2025 Highlight) | 46 synchronised (paper); the processed subset used here has 35 | 60 fps, 1–5 min takes | long everyday takes; not annotated. Our own census found two occlude-and-return events in `scene6_puppy` (5,936 frames), unclassified as absence ([[imvid-event-census-result-and-closure-2026-08-25]]) |
| DiVa-360 (arXiv 2307.16897, CVPR 2024) | 53 | 2–3 min sequences | our census: 0 of 597 candidate windows corroborated ([[elgs-absence-diagnostic-result]]) |
| N3V / DyNeRF (arXiv 2103.02597) | 21 (19–20 training) | 30 fps, 300 f | occlusion only; one event in `cut_roasted_beef` ([[crb300-event-mask-curation-2026-08-23]]) |
| DNA-Rendering (arXiv 2307.10173) | 60 | 15 fps | human-centric; props held throughout |
| CMU Panoptic (arXiv 1612.03153) | 31 HD + 480 VGA | long | incidental entries and exits; unannotated; poor for photometric NVS |
| ARCTIC (arXiv 2204.13662) | 8 + 1 ego | 2.1M frames | objects persist |
| Assembly101 (arXiv 2203.14712) | 8 static + 4 ego | — | parts leave and return, but monochrome statics, not NVS-calibrated |
| BEHAVE (arXiv 2204.06950) | 4 | — | fails the camera bar |
| Ego-Exo4D | up to 4 exo | 30–60 fps | fails the camera bar |
| HOI4D (arXiv 2203.01577) | egocentric | — | fails the camera bar |
| ActorsHQ, Technicolor, MeetRoom, Google Immersive, ENeRF-Outdoor | — | — | not verified in this pass; none known for absence events |

Adjacent but the wrong shape: Remove360 (arXiv 2508.11431; pre/post
removal captures, not video), SceneDiff (arXiv 2512.16908; change
detection across visits), MUVOD (arXiv 2507.07519), Charge (arXiv
2512.13639, CVPR 2026; dense multi-view synthetic from an animated
film with segmentation and flow, worth checking for absence events),
MemoBench (arXiv 2606.27537; a disappear-and-reappear world-model
benchmark, not multi-view NVS).

## 3. Low-rank motion bases as a standalone contribution

**Verdict: OCCUPIED.** DynMF (arXiv 2312.00112, ECCV 2024) decomposes
per-point motion into a small learned basis; Shape of Motion (arXiv
2407.13764) uses a compact set of SE(3) motion bases; MoSca (arXiv
2405.17421, CVPR 2025) and SC-GS (arXiv 2312.14937, CVPR 2024) anchor
dense Gaussians to sparse motion scaffolds or control points; 2025–26
continuations include MoDec-GS (arXiv 2501.03714), SMG (arXiv
2608.31023) and motion-trajectory fields (arXiv 2508.07182). 4D-Rotor
GS not separately verified. A motion basis is at most a component.

## 4. "Episodic anchor trajectories with localised Gaussian bundles in temporal sub-windows"

**Verdict: OCCUPIED as a conjunction of standard practice.** Sparse
anchors with SE(3) plus local residuals = SC-GS and MoSca; persistent
per-Gaussian trajectories with local rigidity = Dynamic 3D Gaussians
(arXiv 2308.09713, 3DV 2024); anchor-structured 4D capacity = 4D
Scaffold-GS (arXiv 2411.17044, AAAI 2026); temporal sub-window
optimisation chained by boundary conditions = the streaming line:
3DGStream (arXiv 2403.01444, CVPR 2024), QUEEN (arXiv 2412.04469,
NeurIPS 2024), HiCoM (arXiv 2411.07541, NeurIPS 2024), StreamSTGS
(arXiv 2511.06046). "StreamGS" as a distinct canonical paper: not
found.

## 5. Failure-analysis and non-PSNR benchmark papers for dynamic NVS

**Verdict: no dedicated occlusion/disocclusion failure-analysis paper
for 4DGS was found.** Closest: "Monocular Dynamic View Synthesis: A
Reality Check" (arXiv 2210.13445, NeurIPS 2022; co-visibility-masked
metrics — the load-bearing precedent for our event-window framing);
Charge (arXiv 2512.13639, CVPR 2026; three capture regimes, multi-modal
ground truth); Style4D-Bench (arXiv 2508.19243; temporal coherence and
multi-view consistency protocol, wrong task); a temporally aware IV-PSNR
extension (Applied Sciences 2026, doi 10.3390/app16010274); and a
retrospective standardised dynamic-NVS benchmark (arXiv 2605.12437)
that argues explicit temporal coupling is unnecessary in dense
multi-view, a premise our event-window results speak to directly.

## 6. Event-window metrics on N3V

**Verdict: NOT FOUND.** Masked dynamic-region PSNR / M-SSIM on N3V is
now common (e.g. DSD-GS, arXiv 2605.30863; several 2026 papers), and
background-stability PSNR over the 300 frames of `coffee_martini`
appears in streaming work; no paper found scores a temporal window
around a reveal or occlusion event. Our ghost-core / return-window /
control-window protocol (spec v1.2.0 and v2.0.0) appears unoccupied.

## 7. What this means for the paper (the five bullets, as recorded)

* Drop both Gemini proposals as contributions: low-rank motion bases
  and anchor-plus-window optimisation are each fully occupied, and a
  bundle of two occupied cells is the "several weak ideas" failure
  mode.
* PersistGS is the nearest competitor and must be cited and contrasted
  first: it owns the problem statement (object permanence, vanished
  gradients, recreate from scratch) and solves it with rigid-body
  physics on synthetic scenes with occlusion-observing held-out
  cameras. Our differentiators: the window is INFERRED from
  training-view evidence, the mechanism is PER-PRIMITIVE (not
  per-object-asset, cf. CubifyGS), the gate is TRAINED WITH (our own
  measurement that render-time gating is the opposite of training-time
  gating), and the footage is real.
* No dataset supplies real absence and return at ≥ 8 cameras; that is
  a defensible paper claim, not a gap in the work, and it is why an
  authored counterfactual fixture is necessary. Check ImViD long takes
  and Charge before asserting "none exists".
* The event-window metric is a contribution in itself; frame it against
  the Reality Check's co-visibility masking as its temporal analogue.
* Two threats to pre-empt: SharpTimeGS / FreeTimeGS++ / RetimeGS make
  learned per-primitive lifespans standard, so novelty must rest on
  multi-interval support with an inferred gap, not on temporal opacity;
  and the "reactivation" vocabulary collides with arXiv 2510.19653.

Corrections after the kill-argument review
([[paper-thesis-v1-kill-argument-2026-09-11]]): the ImViD events are
occlude-and-return events in a 5,936-frame take, not adjudicated
absences, and the 15,215-frame figure belongs to the Opera take
([[imvid-acquisition-quota-2026-08-24]]).
