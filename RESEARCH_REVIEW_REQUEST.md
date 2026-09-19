# Review brief — multi-interval presence on Ex4DGS

**Reviewer model:** `gpt-5.6-sol`, reasoning effort `ultra` (or `xhigh`, never below).
**Prepared:** 2026-09-20. **Nothing has been implemented.** The plan under review is a draft.

---

## 0. Instruction to the reviewer

Act as a senior ML reviewer at NeurIPS/ICML/ECCV level. **Start from the
assumption that this work is broken somewhere; your job is to find where.** Be
adversarial. **Trust nothing the brief tells you — verify it against the
codebase yourself.** Executor notes are not evidence beyond the files they cite.

The authors have deliberately listed their own weaknesses in §7 of the plan.
Treat that list as a floor, not a ceiling: the useful output is the weaknesses
they did *not* list.

Two framing notes, because they change what "good" means here:

1. The substrates under study (Multi4D, ECCV 2026; Ex4DGS, NeurIPS 2024) are
   themselves rigorous, and the ask is explicitly that this work reach the same
   standard of *technical precision* — including small, specific engineering
   decisions that carry real consequences.
2. The authors ask specifically whether any **small technical contributions**
   are real, versus implementation craft being inflated into novelty. They have
   made an honest self-assessment in plan §6; please stress-test it in both
   directions — both "you are overclaiming" and "you are underselling something
   that is actually load-bearing".

---

## 1. How to access the code

Everything is on the CINECA Leonardo cluster. **Read-only inspection is all that
is needed** — do not run training or submit jobs.

```
ssh siyengar@login.leonardo.cineca.it
```

(Windows note for the operator relaying this: use the native
`C:\Windows\System32\OpenSSH\ssh.exe`; Git Bash's own ssh cannot reach the
Windows ssh-agent. The CINECA credential is a 12-hour certificate.)

Roots:

```
$W = /leonardo_work/EUHPC_D36_068/sri/proj_adags

$W/repo/ex4dgs                 Ex4DGS @ 7fac649        <- the substrate to extend
$W/repo/multi4d                Multi4D @ c483e82       <- the rejected alternative, for comparison
$W/repo/adags/research-wiki    the durable record (code maps, reproduction pages, this plan)
$W/runs/ex4dgs                 9 completed runs + the E0 gate
$W/data_staging/ex4dgs         the staged cut_roasted_beef scene
$W/models/ex4dgs_release       the authors' released pretrained model and their own metrics
$W/agent-control/substrates    ledger.txt, sbatch/, collectors, env scripts
```

The files the plan turns on, by absolute path:

```
$W/repo/ex4dgs/utils/interpolations.py            lines 55-61   the entire temporal gate
$W/repo/ex4dgs/scene/c_gaussian_model.py          the model; see :363-366, :874-1017, :1074-1093,
                                                  :1147-1227, :1243-1297, :1330-1358, :513-670, :765-787
$W/repo/ex4dgs/train.py                           the loop, :92-282
$W/repo/ex4dgs/gaussian_renderer/__init__.py      :19-124, note :121 dominent_idxs
$W/repo/ex4dgs/submodules/diff_gaussian_rasterization_df/   the CUDA extension
$W/repo/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu   :278-285
$W/repo/multi4d/diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu  :687-692
```

## 2. Documents to read, in order

All under `$W/repo/adags/research-wiki/`:

1. `operations/ex4dgs-multi-interval-plan-2026-09-20.md` — **the plan under review**
2. `operations/ex4dgs-code-map-2026-09-19.md` — the substrate, with file:line throughout
3. `operations/ex4dgs-n3v-leonardo-reproduction-2026-09-19.md` — reproduction, gates, the 0.59 dB gap, the point-cloud swap
4. `operations/multi4d-code-map-2026-09-19.md` — why the alternative substrate was rejected
5. `operations/multi4d-n3v-leonardo-reproduction-2026-09-18.md` — the other reproduction (PASSED)
6. `query_pack.md` — compressed project history; the 2026-08-24 and 2026-09-15 blocks matter most

Figures (SVG, `research-wiki/assets/`, also attached if the channel allows):

* `ex4dgs-k-interval-change-surface.svg` — what changes for K>1, what does not, the four silent bypasses
* `ex4dgs-iteration-dataflow.svg` — one training iteration
* `lineage-column-hazard-map.svg` — every site a new per-primitive column must survive, by failure mode
* `substrate-temporal-presence-comparison.svg` — α(t) for both substrates and where gradient dies

## 3. Established facts (verify, do not assume)

Each was re-read against the checkout before being recorded. Please confirm or
refute the load-bearing ones.

**Substrate, Ex4DGS**
* The entire temporal gate is `utils/interpolations.py:55-61`, seven lines, in Python.
* CUDA has **no time concept**; Python hands it one already-evaluated scalar opacity per primitive. So K>1 needs **no CUDA change**.
* Presence support is a **single contiguous interval by construction**: the plateau test `(c0-t)(c1-t) < 0` is true on exactly one interval for any `(c0,c1)`.
* α is **never exactly zero** outside the interval; only the per-pixel `alpha < 1/255` test produces true absence.
* **Defect:** `m = (t-mean).min(dim=1)[0]` measures the left tail from `c1`, the far edge, so α jumps discontinuously to 1.0 at `t=c0`.
* The static bank has no temporal opacity, but its positions *do* drift (`_xyz + _xyz_disp·t/duration`).
* `opacity_reset_interval` is accepted from config and **read by nothing**; `reset_opacity` is never called.
* The degenerate-interval prune at `c_gaussian_model.py:1078-1081` computes a `logical_or` and **never assigns it**.
* `cat_tensors_to_optimizer:765-787` silently skips groups absent from `tensors_dict`, carrying the author's own comment `# warning this may skip invaild paramgroups`.
* Seed 0 is hardcoded in `safe_state`; there is no `--seed`. CUDA atomics in the rasterizer backward make bitwise reproducibility impossible regardless.

**Substrate, Multi4D (why it was rejected)**
* Temporal factor is computed **in CUDA** and hard-culled at `marginal_t > 0.05` with an early `return` in **both** `forward.cu:282` and `backward.cu:690` — an absent primitive receives **zero gradient** and cannot learn to re-activate.

**Empirical**
* **E0 gate PASSED essentially exactly.** Rendering the authors' released 40k model through our staged scene reproduces their own shipped metrics to PSNR −0.0016 dB, with per-frame max deviation 0.0130 dB over all 300 held-out frames. The evaluation stack is verified.
* **Trained reproduction is PARTIAL:** n=6, mean 32.673, sd **0.3564**, best 33.143, against a published 33.733. Range across six identical runs is **0.9128 dB**.
* **Point-cloud hypothesis refuted.** Swapping in the authors' own 6,141-point `input.ply` (ours has 8,206) gave mean 32.134, n=3 — 0.539 dB *worse*, pooled p=0.072, 95% CI [−0.063, +1.142]. Data, resolution, metrics and initialisation are all excluded; **the 0.59 dB gap is undiagnosed.**
* The event union in this project's authored fixtures is ~0.34% of pixel-times, so whole-frame PSNR cannot resolve event-region effects (~0.02 dB even for an infinite event-region gain).

## 4. The specific questions

**On soundness of the mechanism**
1. Is the `relu(d_k)` per-interval-distance formulation correct and complete? Any degeneracy, non-differentiable point, or gradient pathology we have missed — particularly at interval edges and when two intervals touch or overlap?
2. Is noisy-OR the right union? Our registered prediction is that `max` collapses to K=1 because a never-argmax interval receives no gradient. Is that reasoning right, and is there a better operator (log-sum-exp, soft-OR with temperature, something else) with a principled justification?
3. Is the sorted projection for identifiability sound, or does it create a discontinuity in the optimisation when two intervals cross?
4. **Is there a fundamentally better way to parameterise multi-interval presence** than K fixed intervals — e.g. a learned step/monotone function, a mixture with learned cardinality, an occupancy MLP in time? What would the reviewer do instead, and is our choice defensible or merely convenient?

**On experimental rigor**
5. Is the arm matrix (A0–A5) sufficient, and is **A4 (contiguity-constrained, matched-parameter)** actually a valid capacity control, or can it be defeated?
6. Are preconditions P1–P4 the right ones, and is P1's 5% threshold defensible or arbitrary? (It is currently arbitrary — we would like a principled alternative.)
7. **Is the C2 identity metric sound?** Lineage recall/precision/respawn via `dominent_idxs`. Is per-pixel dominance the right attribution, given a pixel's colour is a blend over many primitives? Is there a better identity measurement?
8. Does building on a baseline **0.59 dB below the published number, with an undiagnosed cause**, invalidate the work? Is "we compare against our own reproduction" sufficient, and what would a reviewer demand?
9. The absence events are **authored fixtures**, not physical disappearances. An earlier cross-model review returned FAIL on exactly this. Is the claim licensed by authored absence, and if not, what is the minimum that would license it?

**On contribution**
10. Given the self-assessment in plan §6 — that items 1, 2, 4 are craft and only the C2 claim is potentially novel — **is there enough here for a top venue?** If not, what is the smallest addition that would change that?
11. Are we missing a small technical contribution that is actually sitting in the code? The kind of thing the authors of these substrates did — e.g. Multi4D encodes a per-Gaussian dynamic mask in SH channel 2 and a light variance in channel 1, reusing existing capacity rather than adding parameters. Is there an analogous move available here?
12. **What is the strongest argument that this whole direction is not worth doing**, and what is the cheapest experiment that would settle it?

**On what to run first**
13. Given ~1.2 GPU-h per run and a measured sd of 0.3564 dB whole-scene (event-window sd not yet measured), **what is the minimum experiment package with the highest acceptance lift per GPU-hour?** Be specific about arms, n, scenes and endpoints.

## 5. Compute and constraints

* Leonardo, A100 64GB. Ex4DGS: 40k iterations ≈ 56 min, ~1.2 GPU-h per run including render.
* Allocation: ~124k local hours remaining; compute is **not** the binding constraint.
* Scheduling is: `PriorityWeightJobSize = 10000000` with `FavorSmall = no`, so single-GPU jobs are structurally disfavoured and short walltimes are the only lever.
* **Deadline: CVPR 2027, 16 November 2026** — approximately eight weeks.
* Working practice: every substrate change is a small review-gated diff; the user performs the evaluation and analysis.

## 6. Deliverables requested

1. A verdict on **technical soundness** of the mechanism (§4 Q1–Q4), with file:line evidence where you disagree with the code map.
2. A verdict on **scientific rigor** of the design (Q5–Q9), naming any confound we have not controlled.
3. A judgement on **sufficiency for a top venue** (Q10–Q12), and if insufficient, the smallest addition that changes it.
4. A **minimal experiment package** (Q13): arms, n per arm, scenes, endpoints, ordered by acceptance lift per GPU-hour.
5. A **results-to-claims matrix**: what claim is licensed under each outcome of C1 × C2 × A1/A4 controls.
6. A **mock review** (Summary, Strengths, Weaknesses, Questions, Score, Confidence, What Would Move Toward Accept).

Please be brutally honest. A verdict of "this direction is not worth eight
weeks" is a useful and acceptable answer, and is preferred over a polite one.
