# Request to gpt-5.6-sol (xhigh): refine the plan for an 8-page NeurIPS 2026 workshop paper, and find genuine novelty

You are the senior research collaborator for this project. Another AI (Claude
Opus 5) wrote the plan in PART B after reading the project record. The principal
(a PhD student) wants you to **refine and improve it**, not just review it. Spend
your effort on thinking hard, reading the actual code and papers, and finding
**genuinely novel, feasible ideas** the principal can build into this paper by
mid-October 2026.

## 0. File-access attestation (answer first, before anything else)

(a) Can you read files from disk? YES/NO.
(b) Quote VERBATIM line 58 of
`D:\adags\refine-logs\review-2026-09-21\src\ex4dgs\utils\interpolations.py`.
(c) In `D:\adags\research-wiki\assets\absfix-stage2-report.md`, what is the
median of the `G-U` contrast on endpoint `P2` in the `itt` analysis set (4 decimals)?

## 1. What you can use

* **Workspace** `D:\adags` (git repo). Most important:
  * `research-wiki/` â€” the durable record. Start with
    `research-wiki/operations/neurips-workshop-paper-handoff-2026-09-21.md`
    (reproduced in full in PART A), then the files its Â§8 lists, then
    `research-wiki/query_pack.md` and `research-wiki/gap_map.md`.
  * `research-wiki/papers/*.md` â€” paper notes, incl.
    `ramlal2026_persistgs.md`, `liao2026_sharptimegs.md`,
    `ren2026_cubifygs.md`, `wang2026_multi4d.md`, `li2023_spacetime_gaussians.md`.
  * `research-wiki/assets/absfix-stage2-paired.json` (main result â€” derive any
    number from this, never from prose) and `absfix-stage2-report.md`.
  * **Local copies of the two external substrates' source**:
    `refine-logs/review-2026-09-21/src/ex4dgs/` (Ex4DGS @ 7fac649) and
    `refine-logs/review-2026-09-21/src/multi4d/` (Multi4D @ c483e82).
  * ADAGS / EL-GS code: `elgs/`, `scene/`, `scripts/`, `main.py`, `configs/n3v/`.
* **ARIS skills**: all installed at `C:\Users\sucar\.claude\skills\<name>\SKILL.md`.
  Use any that help, reading the full SKILL.md first â€” e.g. `novelty-check`,
  `research-lit`, `idea-creator`, `research-refine`, `experiment-plan`,
  `kill-argument`, `ablation-planner`, `paper-plan`, `arxiv`, `semantic-scholar`,
  `alphaxiv`, `deepxiv`, `exa-search`. You cannot call Claude-side MCP tools;
  where a skill expects one, do the equivalent yourself with the shell and web.
* **Web search**: use it heavily â€” arXiv, OpenReview, CVF open access, project
  pages, GitHub. Verify every paper you cite (title, arXiv id, venue) by
  fetching it; do not rely on memory. Mark anything you could not fetch as
  UNVERIFIED.
* **Leonardo (datasets, run outputs)**: optional, READ-ONLY. From PowerShell only,
  `C:\Windows\System32\OpenSSH\ssh.exe <user>@login.leonardo.cineca.it` may work if
  the principal's 12-h certificate is live (`ssh -G` / an existing config entry
  may give the host alias; if it fails, do not try to fix auth â€” say so and move
  on). Root `/leonardo_work/EUHPC_D36_068/sri/proj_adags`. **Never submit jobs,
  never write, never delete** there.

**Hard rules**: do not edit, create or delete any file anywhere except that your
final answer is captured automatically. No git operations that change state. No
downloads larger than a paper PDF.

## 2. What the principal asked for, in their words (paraphrased faithfully)

> Refine/improve the plan. Think it through very thoroughly and identify
> novelties I can incorporate. Look at papers like PersistGS, which are workshop
> papers, and make sure my work is on a similar level â€” if not much better â€” so
> it can definitely be published and accepted. At minimum I think this means I
> beat the PSNR, SSIM, LPIPS benchmarks (I think if I address the quality in the
> gap, my overall metrics should also go up), and for that I need true novelty
> that is not found in any paper so far, but can be found by combining or
> re-seeing similar approaches in a new way. Use any ARIS skills you want.

Three things in that request you must adjudicate honestly, with arithmetic and
sources, rather than accept:

1. **PersistGS's venue.** Our record says arXiv 2606.03479, **CVPR 2026 Workshop**
   (Generative 3D Reconstruction). The principal remembered it as an ECCV workshop
   paper. Verify. Then assess concretely what bar it sets (scenes, cameras,
   baselines, metrics, depth of evaluation) and what bar a strong NeurIPS 2026
   workshop 8-page paper must clear. Find 3â€“6 other recent 4DGS workshop papers
   (2025â€“2026) to calibrate, and say where this project currently sits relative
   to them.
2. **"If I fix the gap, overall PSNR/SSIM/LPIPS go up."** The record says the event
   union is ~0.34% of pixel-times, so even infinite event-region gain moves
   whole-frame PSNR by ~0.02 dB (query_pack 2026-08-24 block, item 6b). Check that
   arithmetic for these fixtures (the gap core ROI is ~5,100 px of 1352Ã—1014, 25
   frames of 300). Then say what "beating the benchmark" can realistically mean
   here: (i) event-window PSNR/SSIM/LPIPS on the fixtures against published
   methods at their own settings, (ii) standard whole-scene N3V metrics, (iii)
   something else. Be explicit about which are achievable by mid-October and which
   need a better base method, and whether a whole-frame gain is even a sensible
   target for a mechanism that acts on 0.3% of pixels.
3. **"True novelty not found in any paper."** Do a real novelty search (use
   `novelty-check` / `research-lit` style procedure) for each candidate idea you
   propose, and for the ideas already in the plan (presence notch / multi-interval
   presence on published 4DGS substrates; event-window metric; authored
   counterfactual absence fixtures; training-time vs render-time gating sign flip;
   contribution-mass attribution of which primitives return). Known threats
   already on the record: PersistGS 2606.03479, SharpTimeGS 2602.02989,
   FreeTimeGS/++ 2506.05348/2605.03337, RetimeGS 2603.13783, CubifyGS 2606.28720,
   Flow4DGS-SLAM 2604.22339 (GMM temporal opacity), TRiGS 2604.00538 (primitive
   regeneration / long-term identity), MaskGaussian (existence probabilities),
   SA4D 2407.04504. Search beyond them, especially 2026 papers.

## 3. What I need from you (output format)

Write in English, concrete, with file:line and URLs as evidence.

1. **Attestation** (Â§0).
2. **Verdict on the plan** (PART B): what is right, what is wrong, what is missing.
   Separate CONFIRMED problems (with evidence) from SUSPECTED ones.
3. **The bar**: PersistGS's venue and bar; the calibration set; where we sit now.
4. **Benchmark arithmetic** (Â§2 item 2) and your recommendation for what the paper's
   headline quantitative claim should be.
5. **Novel ideas â€” the core of your answer.** Propose 3â€“6 candidate ideas, ranked.
   For each:
   * the mechanism in precise terms (equations where helpful), and exactly where it
     attaches in Ex4DGS / Multi4D / EL-GS code (file:line in the local copies);
   * why it is new: closest prior work found by your search (with URLs) and the
     precise difference;
   * the "similar approaches seen in a new way" angle â€” what existing idea from
     another area (e.g. survival analysis, change-point detection, video object
     segmentation memory, SLAM loop closure/map persistence, keyframe memory in
     tracking, occupancy/free-space reasoning, amodal completion, codec reference
     frames, hidden-state models) it transplants;
   * the expected effect on which metric and window (gap core P1, return P2/S1,
     control C1, whole frame), and whether it could plausibly beat STG in the gap
     while keeping the return advantage;
   * cost in engineering days and GPU-h (1 run: Ex4DGS ~1.2 GPU-h, Multi4D ~2.7,
     STG ~2.5â€“3.5 per 50-frame segment Ã— 6 segments, EL-GS 12k ~2.5â€“4 h);
   * the kill criterion and the non-vacuity precondition;
   * risks, including novelty collisions.
   Specifically consider â€” but do not limit yourself to â€” an idea that gets BOTH
   STG's gap quality AND our gated return advantage (the record shows temporal
   opacity makes the gap easy but weakens the return; the gate does the opposite).
6. **A revised plan**: dated steps from Sep 22 to Oct 15, with gates and kill
   rules, compute, and what the principal must decide. Respect: 8 pages, Oct 15
   done, ~15,500 GPU-h available but wall-clock-limited (Leonardo 1-GPU jobs
   schedule badly; walltime is the only lever; resume is unusable on Ex4DGS and
   Multi4D), the principal reviews every substrate diff, baselines run at their own
   published settings with one common evaluator, the honesty conditions in PART A
   Â§2.
7. **A mock review** of the paper you recommend, as a NeurIPS workshop reviewer:
   score, strengths, weaknesses, what would move it up.
8. **UNVERIFIED list**.

Be willing to disagree with Claude's plan and with the principal's assumptions.
The principal wants the paper accepted; unfounded optimism does not serve that.

======================================================================
# PART A â€” the handoff, in full
(`research-wiki/operations/neurips-workshop-paper-handoff-2026-09-21.md`)
======================================================================

# NeurIPS 2026 workshop paper â€” handoff (2026-09-21, revision 2)

**Read this first in a fresh session.** It is self-contained: the session that
wrote it will not be available to you. Everything below is verified with a cited
source or marked UNVERIFIED.

> **Revision 2 (same day), corrections from the principal to revision 1.**
> Revision 1 said the object-permanence paper was "parked" and that this paper
> must not spend time on it; that was **wrong**. It fixed a working title; that
> was **premature**. It argued for a matched-training-schedule baseline; the
> principal **disagrees**. It left 4 vs 8 pages open; **8 is decided**. All four
> are corrected below. Revision 1 is in git history.

---

## 0. The decision, in the principal's words

> "I would like to put most of my effort into getting a NeurIPS workshop paper
> ... an extremely high quality NeurIPS workshop paper. I don't mind even if it
> exceeds the quality required for a workshop paper by a lot. I want to work hard
> towards this, and I think if my results exceed existing benchmarks and I show
> progress, that's enough to make it through. I want to be done with this paper
> by mid-October."

And, correcting revision 1:

> "I want this workshop paper **at any cost**. If that means part of my Paper C
> ideas that are feasible here should be incorporated and it'll strengthen my
> work within the time frame, then I want that. On the 28th I'm having an
> important discussion with a colleague / senior PhD about a CVPR/ICCV paper, and
> that will end up being something completely novel anyway, so I am not worried
> about ideas for that â€” it will probably be about Gaussian world models. I will
> be going 8 pages, and there could be important things from Multi4D or Ex4DGS
> that can be incorporated into this work. Don't already assume the title ... the
> title can be formed after I get substantial improvements over the benchmark, or
> I am actively able to demonstrate certain things in code after training, so
> let's hold off till then. But I agree that from a getting-a-paper-out-fast
> perspective, we need to start with authored counterfactuals."

**So, concretely:**

* **One paper, 8 pages, very high quality. Done ~Oct 15; submitted by Oct 29.**
* **Nothing is held back for another paper.** The CVPR/ICCV paper will be a
  different, novel project (likely Gaussian world models), so every idea from
  the object-permanence plan ("Paper C"), Multi4D or Ex4DGS that is **feasible by
  mid-October and strengthens this paper** is in scope.
* **No title yet.** It is chosen after the results â€” after substantial
  improvements over baselines, or after something can be *demonstrated* in a
  trained model. Use a neutral placeholder until then.
* **Start from the authored counterfactual fixtures.** They are the fast,
  already-built foundation.

## 1. The venue â€” verified

**Physical World AI: Geometry, Characteristics, and Multimodal Sensing**
("PhysWorldAI"), a **NeurIPS 2026 workshop**, Atlanta.
CFP: https://physworld-org.github.io/physworld.github.io/cfp â€” fetched and
confirmed 2026-09-21.

| item | value |
|---|---|
| **non-archival track window** | **Sept 29 â€“ Oct 29, 2026** |
| **notification** | **Nov 09, 2026** |
| archival track | closed Sep 09, 2026 â€” not available |
| page limit | "Long papers up to 8 pages ... (Reference and Appendix are not included)" â€” **8 pages is decided** |
| topical fit | CFP lists "3D/4D reconstruction ... physically grounded view synthesis" |

**Still to verify in the fresh session:** the style file, anonymity rules, the
exact submission portal.

**Dual submission, for completeness.** Under the CVPR 2026 Author Guidelines
(https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines), a peer-reviewed
workshop paper longer than four pages is a "publication" **even if
non-archival**; arXiv is not; "20 percent or more overlap" is substantially
similar. So an accepted 8-page paper here constrains any *later* main-venue
paper to under 20% overlap. **Given the principal's plan that the next main
paper is a separate, novel project, this is not expected to bind** â€” note it,
don't design around it.

## 2. What the paper starts from â€” on the record

The foundation is the EL-GS presence-gating work on authored counterfactual
full-multiview absence fixtures. **This is evidence, not a fixed framing** â€” the
story and title follow the results.

| finding | status |
|---|---|
| **Occlusion vs absence, same gate.** On a real occlusion the trained gate is within 0.1 dB (Gâˆ’U âˆ’0.07 ghost, +0.05 return; 4/4). On the authored absence it gives **12/12** positive paired prefixes over 3 scenes, **median +2.40 dB ghost core, +2.49 dB return**. Render-time gating on the same fixture: âˆ’5.9 dB ghost, +3.2 dB return â€” a **sign flip**; a gate must be trained with, not applied after. | REC |
| **Fixtures and protocol.** Three real-footage N3V scenes (`cut_roasted_beef`, `flame_steak`, `sear_steak`) with an authored counterfactual absence of one object, frames 60â€“89, all cameras; six frozen event windows; wrong-membership shams A/B/L; timing shams GMIS/GONES; mechanism preconditions. | REC |
| **Estimated gate on the admitted subset.** Blind visibility-gap estimator T1 within one frame on 10/12 prefixes, zero false activations; one-click SAM 2 membership precision â‰¥ 0.9996, recall 0.978â€“0.990 on 8 admitted prefixes; within 0.21 dB of oracle. Not fully estimated: sear refused on membership, flame s3 failed on timing. | REC |
| **STG observation.** STG (per-Gaussian temporal opacity, no gate) renders the gap core 3.4â€“7.5 dB *better* than the gated arm and is 1.3â€“2.5 dB *worse* on the return on 7/8 prefixes. | REC on flame/sear; RUN on crb |

**Reporting conditions from the adversarial review** â€” these are about honesty,
not framing, so they stand:

* Describe the fixtures as "**authored counterfactual**" absence; never "genuine"
  or "physical" absence. Include a limitation sentence to the effect that the
  evidence covers segmentation-guided counterfactual removals in captured
  footage, not automatic discovery of physical exit or naturally occurring
  absence-and-return.
* **The frozen NOT_MET verdict is reported first**; the 12/12 pattern follows,
  labelled descriptive (it was NOT_MET under a pre-registered all-pairs 0.50 dB
  rule).
* The 12 prefixes are runs **nested in 3 scenes** â€” present them scene-clustered.
* **Independent mask re-scoring**: SAM 2 masks generated independently,
  human-audited on every frame of the ghost core, return and settled windows;
  re-run the frozen reducer unchanged; report both mask families.

## 3. Baselines and "exceeding the benchmark" â€” the principal's position

**Revision 1 argued for retraining baselines on a schedule matched to ours. The
principal disagrees, and that is the standing decision:**

> "Different training regimes have different requirements, and what counts is
> the novel view synthesis ability and the results on the metrics."

**So: every baseline runs at its own published or recommended configuration,
and every method is scored under an identical evaluation** â€” same scenes, same
held-out camera, same frames, same event windows, same metric code. This is the
standard practice in novel-view-synthesis papers. Two things still have to be
true for the comparison to be fair, and they are about evaluation, not training:

* **Identical evaluation for every method** â€” one evaluator, one set of masks,
  one convention for PSNR pooling, clamping, SSIM and LPIPS. This project has
  three SSIM/LPIPS conventions across its substrates (Â§6), so this is a real
  task, not a formality.
* **State each baseline's configuration in the paper** (schedule, frame
  segmentation, iterations) so a reader can see what was run. STG's published
  N3V protocol trains 50-frame segments; running it that way is correct under
  this position.

**What the evidence currently says about beating baselines** â€” be precise,
because this is where a reviewer pushes:

* The gated arm is **above STG on the return window** (7/8 prefixes,
  1.3â€“2.5 dB). That comparison is **unpaired** â€” STG and our cells are
  different trainings â€” so it carries that qualifier in the text.
* STG is **3.4â€“7.5 dB better in the gap**.
* **Whole-frame PSNR cannot show this effect**: the event is ~0.34% of
  pixel-times, so even an infinite event-region gain moves whole-frame PSNR by
  ~0.02 dB. Event-window metrics are the endpoint; whole-frame is a do-no-harm
  check.

## 4. Ideas from Paper C, Multi4D and Ex4DGS that could strengthen this paper

**In scope if feasible by mid-October.** The fresh session should evaluate each
for cost against the Oct 15 target and choose â€” this is a menu, not a plan.
Anything that changes substrate code falls under the principal's control model
(Â§7).

**Strong baselines, already reproduced â€” likely the highest-value addition.**

| idea | why it strengthens the paper | cost | status |
|---|---|---|---|
| **Run Multi4D on the three fixtures** at its published schedule | a reproduced ECCV 2026 method (3/3 in its authors' band) as a baseline | ~2.7 GPU-h/run; needs a fixture-to-Multi4D layout converter | Multi4D REC on N3V; fixtures not yet ingested |
| **Run Ex4DGS on the three fixtures** at its published schedule | a NeurIPS 2024 method whose presence is **single-interval by construction** â€” the natural foil for absence-and-return | ~1.2 GPU-h/run; needs a converter (its own preprocessor extracts zero frames as shipped) | Ex4DGS REC (eval stack exact; training 1.06 dB below published, mean) |

**The mechanism on a published substrate â€” the most direct route to "improves
on an existing method".**

| idea | why | cost | notes |
|---|---|---|---|
| **Presence notch on Ex4DGS** | its temporal gate is 7 lines of Python and its CUDA has no time concept, so a multi-interval presence needs **no CUDA change**; "base method vs base method + our presence" on the fixtures is a clean improvement claim | days of engineering + runs | Use the **ordered lifespan-minus-notch** form, not K exchangeable intervals: verified that interior absence frames give gradient to the notch boundaries (`dp/db1 = +3.5e-02` where the discarded design gave exactly 0), and `q = 0` reproduces the single-interval model **exactly**. See `ex4dgs-multi-interval-plan-v0.3-2026-09-21.md` and the round-1 review for why the earlier design failed. |
| **Presence gate on Multi4D's persistent bank** | its persistent-foreground opacity is ordinary PyTorch, so a gate attaches with **no CUDA change** at `gaussian_renderer/__init__.py:468/:198/:314` â€” **`:314` (the mask render) is the silent bypass** | more engineering than Ex4DGS | its *transient* bank hard-culls at `marginal_t > 0.05` in both `forward.cu` and `backward.cu`; do not try to gate that bank |

**Demonstrations "in code after training" â€” what the principal named as a
possible basis for the title.**

| idea | what it demonstrates | cost |
|---|---|---|
| **Contribution-mass attribution** via Ex4DGS's existing auxiliary flow VJP: inject `object_mask Ã— acc` at evaluation and read per-primitive `Î±Â·T` mass | *which* primitives paint the object before and after the gap â€” shows whether the same ones return, blend-aware, **no new parameters, no CUDA change** | evaluation code only |
| **Mask-Benchmark return scoring** (https://huggingface.co/datasets/yunjinli/Mask-Benchmark, public, ~985 MB; `Neu3D-Mask` covers exactly `coffee_martini, cook_spinach, cut_roasted_beef, flame_steak, sear_steak`) | a public segmentation GT for the **return** window â€” legitimate because fixture frames outside [60,89] are byte-identical to the originals | download + converter; verify camera, frame and resolution mapping first; **not** usable for the gap itself |

**Probably too heavy for mid-October â€” note, don't start without a clear
reason:** persistent roots with a lineage column threaded through ~20 mutation
sites; Multi4D's semantic stage (never run here; fail-open in both
`train_semantic.py` and `eval_seg.py`); hashed lineage sketches as a training
loss.

## 5. Traps this project has fallen into â€” do not repeat them

* **Report cohort means, not best-of-n.** A best-of-six was once headlined as the
  gap to a published number.
* **Whole-frame PSNR cannot see a 0.34%-of-pixels event.**
* **Look at the pixels of the scored region on every arm before reading a
  table.** A gate-off evaluation defect was once caught only by a montage.
* **A frozen reading rule needs a frozen precondition that the mechanism was
  exercised** â€” stated about the setup, never the score.
* **A validation that writes into its own reference is not a validation.**
* **Check a knob is actually read before varying it.** Ex4DGS's
  `opacity_reset_interval` is accepted from config and read by nothing; Multi4D's
  `opa_threshold` is dead because `rot_4d=True`.
* **Noise floors are fixed-seed lower bounds**: Multi4D 0.3445 dB spread (n=3),
  Ex4DGS sd 0.3564 dB (n=6). Neither exposes `--seed`; Multi4D's visible
  `setup_seed(6666)` is overwritten to 0 by `safe_state`.
* **A reviewer's upstream-GitHub line numbers have been wrong**; its local
  citations were exact. Re-locate before quoting.
* **Compute is â‰ˆ 15,500 GPU-h**, measured (1 GPU-h = 8 local-h on Leonardo).
  Figures of 124k and 35k have both circulated and are wrong.

## 6. Access and infrastructure

* **Leonardo.** Use the native `C:\Windows\System32\OpenSSH\ssh.exe` (or
  `/c/Windows/System32/OpenSSH/ssh.exe` from Bash); **Git Bash's own `ssh` cannot
  reach the ssh-agent.** The credential is a CINECA **12-hour certificate** only
  the principal can renew (2FA). Chain jobs so they survive a lapse.
* `$W = /leonardo_work/EUHPC_D36_068/sri/proj_adags`. Fixtures:
  `$W/data_derived/absfix/`. Wave-2 runs: `$W/runs/realdata/absfix2/`. STG:
  `$W/runs/stg/`, `$W/agent-control/realdata/stg/`. Multi4D and Ex4DGS:
  `$W/repo/{multi4d,ex4dgs}`, `$W/envs/{multi4d,ex4dgs}`,
  `$W/agent-control/substrates/` (env scripts, sbatch templates, collectors,
  ledger).
* **Scheduling:** Leonardo structurally disfavours 1-GPU jobs
  (`PriorityWeightJobSize` 30Ã— every other term). **Short walltimes are the only
  lever.** Measure a rate on `boost_qos_dbg` (30 min cap) before committing.
* **Metric conventions differ by substrate** â€” unify before comparing: Multi4D
  computes LPIPS on [0,1] inputs **without** `normalize=True` (non-standard) and
  reports two SSIMs; Ex4DGS's vendored LPIPS is standard; EL-GS uses its own
  pooled, clamped evaluator.
* **Codex review:** the Codex MCP is a local shim over `codex exec`. Use
  `sandbox: danger-full-access` (read-only cannot read files here), inline the
  text under review, require a file-access attestation, and recover the reply
  from `~/.codex/sessions/.../*.jsonl` â€” the shim returns only metadata.

## 7. The principal's control model

Stated 2026-09-18 for the external substrates: **every change to Multi4D or
Ex4DGS code is a small, review-gated diff** â€” one diff, the principal reads it,
the principal says "submit", then it runs â€” and **the principal does the
evaluation and analysis**; deliver analysis-ready artefacts, not conclusions.
Reproduction work (running a method unmodified) was autonomous. **Confirm with
the principal whether this also applies to changes in ADAGS's own EL-GS code** â€”
it was never stated either way.

## 8. What to read, in order

1. `research-wiki/operations/two-paper-plan-2026-09-21.md`, then
   `assets/two-paper-2026-09-21/fable_round2.md` and `sol_round2.md` â€” the most
   recent plan and its final review. **Their "Paper W / Paper C" split and the
   "Occlusion Is Not Absence" title are superseded by Â§0 above.**
2. `operations/absfix-wave2-stage2-verdict-2026-09-15.md` and
   `assets/absfix-stage2-report.md` + `absfix-stage2-paired.json` â€” the main
   result. **Generate tables from the JSON; never hand-copy numbers.**
3. `operations/absence-fixture-lane-2026-09-10.md` â€” the fixture, the sign
   flip, the gate-off defect.
4. `operations/realdata-gating-lane-2026-09-09.md` â€” the real-occlusion null.
5. `operations/ex4dgs-multi-interval-plan-v0.3-2026-09-21.md` and its two
   reviews â€” the notch, and why the first design failed.
6. `operations/ex4dgs-code-map-2026-09-19.md`, `multi4d-code-map-2026-09-19.md`
   (with the figures in `assets/`), and the two reproduction pages.
7. `operations/paper-thesis-v1-2026-09-11.md` and its kill-argument;
   `operations/literature-check-absence-return-2026-09-10.md`.
8. `research-wiki/query_pack.md` â€” the whole programme, newest first.

## 9. Timeline to mid-October

| dates | work |
|---|---|
| Sep 22â€“24 | Verify venue details. Read the evidence. **Choose from the Â§4 menu** what fits by Oct 15. Confirm the control model for EL-GS. Build the fixture converters for whichever substrates are chosen. Submit STG on `cut_roasted_beef`. |
| Sep 25â€“Oct 1 | Baselines on the fixtures. Mechanism work if chosen (small reviewed diffs). SAM 2 independent masks and their human audit. **One evaluator, one metric convention, for every method.** |
| Oct 2â€“8 | Results. **Title and framing decided from them.** Figures generated from data. Full draft. |
| Oct 9â€“12 | Adversarial review of the draft; claim audit against the data; citation audit. |
| **Oct 13â€“15** | **Final revision â€” done.** |
| by Oct 29 | Submit to PhysWorldAI (window opens Sep 29). |

**Sep 28:** the principal meets a colleague about the separate CVPR/ICCV paper.
It does not change this paper's scope or timeline.


======================================================================
# PART B â€” Claude's current plan (as presented to the principal)
======================================================================

## Venue facts (verified 2026-09-21 from the CFP page)

NeurIPS 2026 style; double-blind via OpenReview; long papers up to 8 pages;
non-archival window Sep 29 â€“ Oct 29 (portal "will be open soon"); concurrent
submission permitted if other venues' policies allow. Ambiguity: "(Reference and
Appendix are not included)" is attached to the 4-page extended-abstract line;
confirm it also applies to 8-page papers.

## The story the evidence points to (a hypothesis, not a title)

What exists: temporal opacity (STG) renders the authored gap 3.4â€“7.5 dB better
than our gate; our gate returns 1.3â€“2.5 dB better than STG on 7/8 prefixes
(unpaired); on a real occlusion the gate is inert. Missing: what published methods
do on these fixtures. If they also render the gap well but return poorly, the
paper is: (1) an absence-and-return benchmark, (2) the finding that published
methods fail on the return, (3) a presence mechanism that improves a published
method on its own code, (4) a post-training demonstration of which primitives come
back. That is the "beats the benchmark" story. The first two weeks test whether it
holds.

## Menu decisions

Do:
1. Baselines on all three fixtures at their published settings, one evaluator:
   STG on cut_roasted_beef (flame/sear done), Ex4DGS and Multi4D (~1.2 / 2.7
   GPU-h per run, n=3 per scene). Converters are simpler than they look: only
   frames 60â€“89 are edited, so frame 0 and its point cloud are unchanged â€” Ex4DGS's
   `colmap_0` from the reproduction is reusable on crb. Multi4D needs `fused.ply`
   per scene; crb exists; flame/sear unverified.
2. Presence notch on Ex4DGS â€” base vs base+notch, 6 runs per arm per scene
   (~43 GPU-h). Ordered lifespan-minus-notch form, q=0 exact nesting.
3. Contribution-mass attribution via Ex4DGS's auxiliary flow-VJP channel â€”
   which primitives paint the object before and after the gap. Evaluation code
   only; the "demonstrate after training" item.
4. Honesty conditions: independent SAM 2 masks, every P1/P2/S1 frame
   human-audited, frozen reducer re-run unchanged.

Stretch only if Gate 1 shows Multi4D returns badly: gate on Multi4D's persistent
foreground bank.

Skip: Mask-Benchmark return scoring (segmentation not needed for this story),
Multi4D semantic stage, persistent roots / lineage column, lineage sketches,
physical-event census.

## A risk found while reading

Ex4DGS's STATIC bank has no temporal opacity at all
(`get_opacity` = `sigmoid(_opacity)`, `c_gaussian_model.py:355-357`), and
staticâ†’dynamic promotion is triggered by motion (`â€–_xyz_dispâ€–`), not by
disappearance. The wine bottle is static, so in Ex4DGS it probably ghosts
through the gap from the static bank. The notch must therefore also go on the
static bank's opacity (plain sigmoid in Python, no CUDA). Adding it only to the
dynamic bank would be a vacuous treatment. Multi4D more likely rebuilds the
bottle from new transient Gaussians on return (respawn). Which failure each
baseline shows is itself a result; attribution measures it.

## Compute and gates

~100â€“150 GPU-h total (~1% of allocation); wall-clock is the constraint.

| gate | date | condition |
|---|---|---|
| 1 | Sep 26 | Baseline fixture results in. If published methods already return well (P2 â‰¥ our G), the paper stays benchmark+diagnostic and the notch becomes secondary. |
| 2 | Oct 3 | Notch comparison frozen before runs, with a non-vacuity precondition (notch acts on the bottle's Gaussians); results read only after per-arm montages. |
| 3 | Oct 8 | Results in; pick the title; full draft. |
| â€” | Oct 9â€“15 | Adversarial review; claim and citation audits. Done. |

## Open decisions for the principal

1. Does the review-gated-diff control model also cover EL-GS code?
2. Accept the menu above?
3. Who does the ~129-frame SAM 2 mask audit?
4. Certificate renewals during Sep 23â€“26 baseline runs.

