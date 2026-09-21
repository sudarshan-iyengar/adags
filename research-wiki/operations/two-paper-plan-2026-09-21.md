# Two-paper plan — consolidated outcome (2026-09-21)

**Status: plan agreed by both models; nothing run. Decisions in §7 are the
principal's.** Evidence-bearing: no.

Produced by a fixed two-round exchange the principal specified — **Claude Fable
5.1 → gpt-5.6-sol (xhigh) → Fable → Sol** — orchestrated by Claude Opus 5, which
checked every load-bearing fact independently between rounds. Full record, all
tracked: [`assets/two-paper-2026-09-21/`](../assets/two-paper-2026-09-21/)
(`BRIEF.md`, `fable_round1.md`, `sol_round1.md`, `fable_round2.md`,
`sol_round2.md`, `orchestrator_notes_r1..r3.md`).

Both reviewer attestations were checked each round: Sol quoted a named source
line verbatim and a fact only the plan contained, four times, all exact.

---

## 1. The two papers

**Paper W — "Occlusion Is Not Absence".** A diagnostic benchmark plus a
two-sided finding, built almost entirely from the EL-GS record. The lead
contribution combines the **real-occlusion null** (a presence gate is inert,
within 0.1 dB, 4/4) with the **authored-counterfactual positive** (12/12 paired
prefixes over 3 scenes, medians +2.40 dB ghost core / +2.49 dB return). Plus the
event-window protocol and fixtures, estimated-gate feasibility on the admitted
subset, and a qualified STG observation. **Sol: 7/10, weak accept, confidence
4/5.**

**Paper C — persistent object memory on Multi4D.** Roots carry identity,
semantics and canonical appearance behind a presence notch; transient leaves are
disposable. Headline: return-censored 4D segmentation / re-ID. **Sol: 2/10 on
today's evidence; 6/10 if every gate passes.**

## 2. Acceptance odds (both models, reconciled)

| | Fable | Sol |
|---|---|---|
| W, PhysWorldAI 4-page abstract | 60–70% | **65–75%** |
| W, full paper at a CVPR 2027 4D workshop | 55–65% | 55–65% |
| W, smaller archival venue | — | 50–60% |
| C at CVPR, authored-only | ≤10% | would not submit |
| C at CVPR, all gates passed | 25–35% | **30–40%** |
| C at ICCV 2027, 3–5 physical events | 40–50% | 40–55% |
| **probability all CVPR gates pass by Oct 12** | 20–30% | **15–25%** |

**Reading:** W is the dependable paper. C at CVPR is a genuine but minority
chance; C at ICCV 2027 is the likelier home.

## 3. The decisive rules

* **No two certified physical events by October 12 → C goes to ICCV 2027.** Do
  not submit an authored-only object-permanence C to CVPR. Both models agree.
* A benchmark-only CVPR C ("Persistent Root Semantics for 4D Gaussian
  Segmentation") is defensible in principle but only if a large standard win
  already exists by Oct 12: all five Neu3D-Mask scenes, **≥ +3 mean mIoU**,
  positive on ≥ 4/5 scenes, a real advantage over TRASE. v0.3 predicted 0–2
  points. **Do not plan on this rescue.**
* **The single thing most likely to cost the principal a paper** (Sol): failing
  to obey the Oct 5 / Oct 12 stop rule — continuing to scale an authored-only C
  after a gate has failed, which consumes the calendar and starves W.

## 4. Gates for C

| gate | date | must hold |
|---|---|---|
| **A** | Sep 28 | fail-closed semantic train+eval; Mask-Benchmark camera/frame/resolution/target mapping exact; M0 semantics reproduce on crb; converter frame-hashes pass; target represented in FG roots with measurable pre-gap contribution **and** incorrect gap contribution; ≥ 1 physical candidate survives visual review |
| **B** | Oct 5 | M1 > M0 **and** M1 > M2 on return J&F; ≥ 15 pp root-contribution difference; no gap false-positive increase; **two surviving physical candidates in two scenes, one fully certified** |
| **C** | Oct 12 | v0.3 thresholds vs both M0 and M2; Cutie does not erase the advantage; **two physical events fully certified** |

**M2 (episode-split control) must be leak-free**, and exact-zero rendered
presence is not enough. A dormant pre-gap row must also be excluded after
return from FG→TR donor selection, densification/cloning, KNN semantic
smoothing, neighbourhood and regularisation losses, contribution statistics, and
any optimiser update driven by post-return frames. Tombstoning is acceptable if
the row is frozen and removed from every index and operator. The reason:
Multi4D's FG→TR lift **copies** rows and leaves the donor alive
([multi4d-code-map-2026-09-19](multi4d-code-map-2026-09-19.md) §4).

## 5. Paper W — binding conditions

* "**authored counterfactual**" everywhere; never "genuine" or "physical
  absence". Mandatory limitation sentence (Sol): *"Our evidence establishes
  behaviour on segmentation-guided counterfactual removals in captured multiview
  footage; it does not establish automatic discovery of physical scene exit or
  performance on naturally occurring absence-and-return events."*
* **The frozen NOT_MET verdict leads**; the 12/12 pattern follows and is called
  descriptive.
* **Independent mask re-scoring**: SAM 2 masks generated independently, **every
  frame human-audited in P1, P2 and S1**, every fifth in H1/C1; re-run the
  frozen reducer unchanged; report DEVA and SAM 2 side by side. If mask choice
  reverses a scene-level sign, that becomes a central result.
* The 12 prefixes are runs **nested in 3 scenes** — scene-clustered plots, no
  language implying 12 independent scene samples.
* STG only with the unpaired/confounded qualifier; no numerical superiority in
  the abstract.
* Cut the instrument section (W8) from the contribution list; keep it in the
  supplement.
* **Required compute ≈ 15–21 GPU-h** (STG on crb). Free-boundary and
  product-gate runs are optional, not required.

## 6. Dual submission and the W/C boundary

**Verified verbatim** from https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines
(the 2027 page was HTTP 404 on 2026-09-21):

* a publication is "a written work longer than four pages (excluding
  references)" that was peer-reviewed and accepted;
* "peer-reviewed workshop papers _are_ considered as publications if their
  length is more than four pages (excluding references), **even if they do not
  appear in a proceedings**";
* "arXiv preprints ... are _not_ considered as publications";
* substantially similar = "20 percent or more overlap".

**Route for W:** 4-page non-archival abstract to NeurIPS 2026 **PhysWorldAI**
(window Sep 29 – Oct 29, notification Nov 9 — verified) → full W to a CVPR 2027
4D workshop (≈ March 2027, list not yet announced).

**W arXiv — Sol's final adjudication:** not Nov 17 (Fable) and not
automatically Feb 26 (Sol round 1). Post on **Nov 24**, after C's supplementary
is frozen, **only if** the live 2027 guidelines keep the arXiv exemption, C's
main tables are physical/re-ID, and a final overlap audit passes. Otherwise
hold until Feb 26.

**Boundary:** residual overlap ≈ 10–15% (Sol: page accounting is false
precision). Holds **only if** C's main evidence is physical/re-ID and none of
W's photometric result reappears in C — C5's event-window table and the STG
observation are deleted from C entirely, including the supplement. C cites W
once, for fixture provenance.

## 7. Decisions only the principal can make

1. Whether an authored-only C ever goes to CVPR (both models: no; ≈ 10%).
2. The physical-event budget if the zero-acquisition scouts come back empty:
   ImViD puppy transfer from Apollo (size unverified, moving rig,
   occlusion-class events) or a BEHAVE download (MPI non-commercial licence,
   4 cameras).
3. W's full-text exposure: Nov 24 arXiv under Sol's conditions, or hold to
   Feb 26; and whether CVM 2027 (Oct 26, 14 pages) is acceptable for W if C is
   deferred.
4. Whether PhysWorldAI is the right topical home (its CFP lists "3D/4D
   reconstruction ... physically grounded view synthesis").
5. Release licences: fixtures derive from N3V; Mask-Benchmark is CC-BY-NC;
   BEHAVE forbids redistribution.
6. That **C owns the engineer's first week** and W's 8-page text waits until
   after Oct 5.

## 8. This week — the single most important action

**Run the all-camera physical-event census now**, and produce human-reviewable
synchronised contact sheets for every surviving candidate, **before**
implementing the full M1. Order: existing flame/sear DEVA masks (CPU) → DEVA on
the remaining N3V scenes (≈ 6 GPU-h) → Panoptic Sports proxy scout then a full
31-camera audit → stop and choose ICCV if all fail. It is cheap, it retires the
largest uncertainty, and it decides CVPR vs ICCV.

Admission checklist for a certified event: see Sol round 2 §3 — gap **≥ 30
synchronised frames or 1 s, whichever is longer**; ≤ 10 target pixels in any
camera/frame during the gap; **human adjudication of raw RGB in every camera**,
not masks alone; a second method or annotator agreeing; visible in ≥ 2 cameras
for ≥ 10 frames before and after; two annotators agreeing on identity; a
distractor if re-ID is claimed; two events from two different captures.

## 9. Facts established during the exchange

Each checked by the orchestrator, not taken from either model.

* **Compute: ≈ 15,500 GPU-h remain.** Measured from Slurm accounting — 17 jobs,
  20.91 GPU-h, 167.26 billing-h, ratio **8.00**, matching the `saldo` delta
  exactly. The brief said 124k (wrong ×8); Fable's correction said 35k at
  3.5 local-h/GPU-h (also wrong).
* **Panoptic Sports is on Leonardo, never censused**:
  `/leonardo_work/EUHPC_D21_034/proj_adags/data/panopticsports`, 716 MB; six
  sequences (basketball, boxes, football, juggle, softball, tennis), **31
  cameras × 150 frames**, `seg/` for 27. Sol notes the upstream authors describe
  that segmentation as crude foreground/background subtraction, and a 31-camera
  dome makes true all-view invisibility rare — a scout, not a promise.
* **Mask-Benchmark** (https://huggingface.co/datasets/yunjinli/Mask-Benchmark):
  public, ungated, ~985 MB, CC-BY-NC; `Neu3D-Mask/` covers exactly
  `coffee_martini, cook_spinach, cut_roasted_beef, flame_steak, sear_steak`,
  1,500 files = 300 per scene. **All three fixture scenes are covered.**
* **Scoring fixture returns with it is legitimate** — fixture frames outside
  [60,89] are byte-identical to the originals — conditional on exact camera,
  resolution and target mapping, and never for the gap (a multi-object semantic
  mask must not be replaced with all-zero).
* **TRASE is public** (https://github.com/yunjinli/TRASE, MIT, training + eval).
* **Multi4D's `train_semantic.py` is also fail-open**: a missing mask prints
  "Mask not found" and skips the loss.
* **An unverified claim that was nearly carried forward**: Sol's round-1 statement
  that ImViD Opera and Puppy are "fully available" came from Codex's own memory
  store, not this workspace. ImViD is on Apollo, not Leonardo. Treat as
  unverified.
