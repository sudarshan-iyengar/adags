# NeurIPS 2026 workshop paper — handoff (2026-09-21, revision 2)

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
> about ideas for that — it will probably be about Gaussian world models. I will
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
* **No title yet.** It is chosen after the results — after substantial
  improvements over baselines, or after something can be *demonstrated* in a
  trained model. Use a neutral placeholder until then.
* **Start from the authored counterfactual fixtures.** They are the fast,
  already-built foundation.

## 1. The venue — verified

**Physical World AI: Geometry, Characteristics, and Multimodal Sensing**
("PhysWorldAI"), a **NeurIPS 2026 workshop**, Atlanta.
CFP: https://physworld-org.github.io/physworld.github.io/cfp — fetched and
confirmed 2026-09-21.

| item | value |
|---|---|
| **non-archival track window** | **Sept 29 – Oct 29, 2026** |
| **notification** | **Nov 09, 2026** |
| archival track | closed Sep 09, 2026 — not available |
| page limit | "Long papers up to 8 pages ... (Reference and Appendix are not included)" — **8 pages is decided** |
| topical fit | CFP lists "3D/4D reconstruction ... physically grounded view synthesis" |

**Still to verify in the fresh session:** the style file, anonymity rules, the
exact submission portal.

**Dual submission, for completeness.** Under the CVPR 2026 Author Guidelines
(https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines), a peer-reviewed
workshop paper longer than four pages is a "publication" **even if
non-archival**; arXiv is not; "20 percent or more overlap" is substantially
similar. So an accepted 8-page paper here constrains any *later* main-venue
paper to under 20% overlap. **Given the principal's plan that the next main
paper is a separate, novel project, this is not expected to bind** — note it,
don't design around it.

## 2. What the paper starts from — on the record

The foundation is the EL-GS presence-gating work on authored counterfactual
full-multiview absence fixtures. **This is evidence, not a fixed framing** — the
story and title follow the results.

| finding | status |
|---|---|
| **Occlusion vs absence, same gate.** On a real occlusion the trained gate is within 0.1 dB (G−U −0.07 ghost, +0.05 return; 4/4). On the authored absence it gives **12/12** positive paired prefixes over 3 scenes, **median +2.40 dB ghost core, +2.49 dB return**. Render-time gating on the same fixture: −5.9 dB ghost, +3.2 dB return — a **sign flip**; a gate must be trained with, not applied after. | REC |
| **Fixtures and protocol.** Three real-footage N3V scenes (`cut_roasted_beef`, `flame_steak`, `sear_steak`) with an authored counterfactual absence of one object, frames 60–89, all cameras; six frozen event windows; wrong-membership shams A/B/L; timing shams GMIS/GONES; mechanism preconditions. | REC |
| **Estimated gate on the admitted subset.** Blind visibility-gap estimator T1 within one frame on 10/12 prefixes, zero false activations; one-click SAM 2 membership precision ≥ 0.9996, recall 0.978–0.990 on 8 admitted prefixes; within 0.21 dB of oracle. Not fully estimated: sear refused on membership, flame s3 failed on timing. | REC |
| **STG observation.** STG (per-Gaussian temporal opacity, no gate) renders the gap core 3.4–7.5 dB *better* than the gated arm and is 1.3–2.5 dB *worse* on the return on 7/8 prefixes. | REC on flame/sear; RUN on crb |

**Reporting conditions from the adversarial review** — these are about honesty,
not framing, so they stand:

* Describe the fixtures as "**authored counterfactual**" absence; never "genuine"
  or "physical" absence. Include a limitation sentence to the effect that the
  evidence covers segmentation-guided counterfactual removals in captured
  footage, not automatic discovery of physical exit or naturally occurring
  absence-and-return.
* **The frozen NOT_MET verdict is reported first**; the 12/12 pattern follows,
  labelled descriptive (it was NOT_MET under a pre-registered all-pairs 0.50 dB
  rule).
* The 12 prefixes are runs **nested in 3 scenes** — present them scene-clustered.
* **Independent mask re-scoring**: SAM 2 masks generated independently,
  human-audited on every frame of the ghost core, return and settled windows;
  re-run the frozen reducer unchanged; report both mask families.

## 3. Baselines and "exceeding the benchmark" — the principal's position

**Revision 1 argued for retraining baselines on a schedule matched to ours. The
principal disagrees, and that is the standing decision:**

> "Different training regimes have different requirements, and what counts is
> the novel view synthesis ability and the results on the metrics."

**So: every baseline runs at its own published or recommended configuration,
and every method is scored under an identical evaluation** — same scenes, same
held-out camera, same frames, same event windows, same metric code. This is the
standard practice in novel-view-synthesis papers. Two things still have to be
true for the comparison to be fair, and they are about evaluation, not training:

* **Identical evaluation for every method** — one evaluator, one set of masks,
  one convention for PSNR pooling, clamping, SSIM and LPIPS. This project has
  three SSIM/LPIPS conventions across its substrates (§6), so this is a real
  task, not a formality.
* **State each baseline's configuration in the paper** (schedule, frame
  segmentation, iterations) so a reader can see what was run. STG's published
  N3V protocol trains 50-frame segments; running it that way is correct under
  this position.

**What the evidence currently says about beating baselines** — be precise,
because this is where a reviewer pushes:

* The gated arm is **above STG on the return window** (7/8 prefixes,
  1.3–2.5 dB). That comparison is **unpaired** — STG and our cells are
  different trainings — so it carries that qualifier in the text.
* STG is **3.4–7.5 dB better in the gap**.
* **Whole-frame PSNR cannot show this effect**: the event is ~0.34% of
  pixel-times, so even an infinite event-region gain moves whole-frame PSNR by
  ~0.02 dB. Event-window metrics are the endpoint; whole-frame is a do-no-harm
  check.

## 4. Ideas from Paper C, Multi4D and Ex4DGS that could strengthen this paper

**In scope if feasible by mid-October.** The fresh session should evaluate each
for cost against the Oct 15 target and choose — this is a menu, not a plan.
Anything that changes substrate code falls under the principal's control model
(§7).

**Strong baselines, already reproduced — likely the highest-value addition.**

| idea | why it strengthens the paper | cost | status |
|---|---|---|---|
| **Run Multi4D on the three fixtures** at its published schedule | a reproduced ECCV 2026 method (3/3 in its authors' band) as a baseline | ~2.7 GPU-h/run; needs a fixture-to-Multi4D layout converter | Multi4D REC on N3V; fixtures not yet ingested |
| **Run Ex4DGS on the three fixtures** at its published schedule | a NeurIPS 2024 method whose presence is **single-interval by construction** — the natural foil for absence-and-return | ~1.2 GPU-h/run; needs a converter (its own preprocessor extracts zero frames as shipped) | Ex4DGS REC (eval stack exact; training 1.06 dB below published, mean) |

**The mechanism on a published substrate — the most direct route to "improves
on an existing method".**

| idea | why | cost | notes |
|---|---|---|---|
| **Presence notch on Ex4DGS** | its temporal gate is 7 lines of Python and its CUDA has no time concept, so a multi-interval presence needs **no CUDA change**; "base method vs base method + our presence" on the fixtures is a clean improvement claim | days of engineering + runs | Use the **ordered lifespan-minus-notch** form, not K exchangeable intervals: verified that interior absence frames give gradient to the notch boundaries (`dp/db1 = +3.5e-02` where the discarded design gave exactly 0), and `q = 0` reproduces the single-interval model **exactly**. See `ex4dgs-multi-interval-plan-v0.3-2026-09-21.md` and the round-1 review for why the earlier design failed. |
| **Presence gate on Multi4D's persistent bank** | its persistent-foreground opacity is ordinary PyTorch, so a gate attaches with **no CUDA change** at `gaussian_renderer/__init__.py:468/:198/:314` — **`:314` (the mask render) is the silent bypass** | more engineering than Ex4DGS | its *transient* bank hard-culls at `marginal_t > 0.05` in both `forward.cu` and `backward.cu`; do not try to gate that bank |

**Demonstrations "in code after training" — what the principal named as a
possible basis for the title.**

| idea | what it demonstrates | cost |
|---|---|---|
| **Contribution-mass attribution** via Ex4DGS's existing auxiliary flow VJP: inject `object_mask × acc` at evaluation and read per-primitive `α·T` mass | *which* primitives paint the object before and after the gap — shows whether the same ones return, blend-aware, **no new parameters, no CUDA change** | evaluation code only |
| **Mask-Benchmark return scoring** (https://huggingface.co/datasets/yunjinli/Mask-Benchmark, public, ~985 MB; `Neu3D-Mask` covers exactly `coffee_martini, cook_spinach, cut_roasted_beef, flame_steak, sear_steak`) | a public segmentation GT for the **return** window — legitimate because fixture frames outside [60,89] are byte-identical to the originals | download + converter; verify camera, frame and resolution mapping first; **not** usable for the gap itself |

**Probably too heavy for mid-October — note, don't start without a clear
reason:** persistent roots with a lineage column threaded through ~20 mutation
sites; Multi4D's semantic stage (never run here; fail-open in both
`train_semantic.py` and `eval_seg.py`); hashed lineage sketches as a training
loss.

## 5. Traps this project has fallen into — do not repeat them

* **Report cohort means, not best-of-n.** A best-of-six was once headlined as the
  gap to a published number.
* **Whole-frame PSNR cannot see a 0.34%-of-pixels event.**
* **Look at the pixels of the scored region on every arm before reading a
  table.** A gate-off evaluation defect was once caught only by a montage.
* **A frozen reading rule needs a frozen precondition that the mechanism was
  exercised** — stated about the setup, never the score.
* **A validation that writes into its own reference is not a validation.**
* **Check a knob is actually read before varying it.** Ex4DGS's
  `opacity_reset_interval` is accepted from config and read by nothing; Multi4D's
  `opa_threshold` is dead because `rot_4d=True`.
* **Noise floors are fixed-seed lower bounds**: Multi4D 0.3445 dB spread (n=3),
  Ex4DGS sd 0.3564 dB (n=6). Neither exposes `--seed`; Multi4D's visible
  `setup_seed(6666)` is overwritten to 0 by `safe_state`.
* **A reviewer's upstream-GitHub line numbers have been wrong**; its local
  citations were exact. Re-locate before quoting.
* **Compute is ≈ 15,500 GPU-h**, measured (1 GPU-h = 8 local-h on Leonardo).
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
  (`PriorityWeightJobSize` 30× every other term). **Short walltimes are the only
  lever.** Measure a rate on `boost_qos_dbg` (30 min cap) before committing.
* **Metric conventions differ by substrate** — unify before comparing: Multi4D
  computes LPIPS on [0,1] inputs **without** `normalize=True` (non-standard) and
  reports two SSIMs; Ex4DGS's vendored LPIPS is standard; EL-GS uses its own
  pooled, clamped evaluator.
* **Codex review:** the Codex MCP is a local shim over `codex exec`. Use
  `sandbox: danger-full-access` (read-only cannot read files here), inline the
  text under review, require a file-access attestation, and recover the reply
  from `~/.codex/sessions/.../*.jsonl` — the shim returns only metadata.

## 7. The principal's control model

Stated 2026-09-18 for the external substrates: **every change to Multi4D or
Ex4DGS code is a small, review-gated diff** — one diff, the principal reads it,
the principal says "submit", then it runs — and **the principal does the
evaluation and analysis**; deliver analysis-ready artefacts, not conclusions.
Reproduction work (running a method unmodified) was autonomous. **Confirm with
the principal whether this also applies to changes in ADAGS's own EL-GS code** —
it was never stated either way.

## 8. What to read, in order

1. `research-wiki/operations/two-paper-plan-2026-09-21.md`, then
   `assets/two-paper-2026-09-21/fable_round2.md` and `sol_round2.md` — the most
   recent plan and its final review. **Their "Paper W / Paper C" split and the
   "Occlusion Is Not Absence" title are superseded by §0 above.**
2. `operations/absfix-wave2-stage2-verdict-2026-09-15.md` and
   `assets/absfix-stage2-report.md` + `absfix-stage2-paired.json` — the main
   result. **Generate tables from the JSON; never hand-copy numbers.**
3. `operations/absence-fixture-lane-2026-09-10.md` — the fixture, the sign
   flip, the gate-off defect.
4. `operations/realdata-gating-lane-2026-09-09.md` — the real-occlusion null.
5. `operations/ex4dgs-multi-interval-plan-v0.3-2026-09-21.md` and its two
   reviews — the notch, and why the first design failed.
6. `operations/ex4dgs-code-map-2026-09-19.md`, `multi4d-code-map-2026-09-19.md`
   (with the figures in `assets/`), and the two reproduction pages.
7. `operations/paper-thesis-v1-2026-09-11.md` and its kill-argument;
   `operations/literature-check-absence-return-2026-09-10.md`.
8. `research-wiki/query_pack.md` — the whole programme, newest first.

## 9. Timeline to mid-October

| dates | work |
|---|---|
| Sep 22–24 | Verify venue details. Read the evidence. **Choose from the §4 menu** what fits by Oct 15. Confirm the control model for EL-GS. Build the fixture converters for whichever substrates are chosen. Submit STG on `cut_roasted_beef`. |
| Sep 25–Oct 1 | Baselines on the fixtures. Mechanism work if chosen (small reviewed diffs). SAM 2 independent masks and their human audit. **One evaluator, one metric convention, for every method.** |
| Oct 2–8 | Results. **Title and framing decided from them.** Figures generated from data. Full draft. |
| Oct 9–12 | Adversarial review of the draft; claim audit against the data; citation audit. |
| **Oct 13–15** | **Final revision — done.** |
| by Oct 29 | Submit to PhysWorldAI (window opens Sep 29). |

**Sep 28:** the principal meets a colleague about the separate CVPR/ICCV paper.
It does not change this paper's scope or timeline.
