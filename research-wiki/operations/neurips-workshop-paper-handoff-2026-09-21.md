# NeurIPS 2026 workshop paper — handoff (2026-09-21)

**Read this first in a fresh session.** It is self-contained: the session that
wrote it will not be available to you. Everything below is either verified with
a cited source or marked UNVERIFIED.

## 0. The decision, in the principal's words

> "I think the NeurIPS workshop paper is a great idea — that would also allow me
> to go to NeurIPS and learn from the incredible tier-1 conference atmosphere. I
> would like to put most of my effort into getting a NeurIPS workshop paper ...
> an extremely high quality NeurIPS workshop paper. I don't mind even if it
> exceeds the quality required for a workshop paper by a lot. I want to work hard
> towards this, and I think if my results exceed existing benchmarks and I show
> progress, that's enough to make it through. **I want to be done with this paper
> by mid-October.**"

So: **one paper, very high quality, finished ~Oct 15, submitted by Oct 29.** The
object-permanence paper ("Paper C") is **parked** by the principal's choice; do
not spend this paper's time on it.

## 1. The venue — verified

**Physical World AI: Geometry, Characteristics, and Multimodal Sensing**
("PhysWorldAI"), a **NeurIPS 2026 workshop**, Atlanta.
CFP: https://physworld-org.github.io/physworld.github.io/cfp — fetched and
confirmed 2026-09-21.

| item | value |
|---|---|
| **non-archival track window** | **Sept 29 – Oct 29, 2026** |
| **notification** | **Nov 09, 2026** |
| archival track | closed Sep 09, 2026 — **not available** |
| page limits | "Long papers up to 8 pages and extended abstracts up to 4 pages (Reference and Appendix are not included)" |
| archival status | "Archival & Non-archival; accepted papers will also be posted on the workshop website" |
| topical fit | CFP lists "3D/4D reconstruction ... physically grounded view synthesis" |

**Still to verify in the fresh session:** the style file, anonymity rules,
whether the long track expects a particular format, and the exact submission
portal.

## 2. The first decision the fresh session must make: 4 pages or 8

The previous plan made this a **4-page extended abstract** — but *only* to
protect a CVPR submission of the other paper. With that paper parked and the
principal wanting maximum quality, **an 8-page long paper is the natural fit.**
The consequence must be decided deliberately, not inherited:

**The rule**, verbatim from the CVPR 2026 Author Guidelines
(https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines; the 2027 page was
HTTP 404 on 2026-09-21):

* "A publication, for the purposes of this policy, is defined to be a written
  work longer than four pages (excluding references) that was submitted for
  review by peers for either acceptance or rejection, and, after review, was
  accepted."
* "peer-reviewed workshop papers _are_ considered as publications if their
  length is more than four pages (excluding references), **even if they do not
  appear in a proceedings**."
* "arXiv preprints ... are _not_ considered as publications."
* Substantially similar = "20 percent or more overlap".

**What this means:** an accepted **8-page** PhysWorldAI paper **is a
publication** even on the non-archival track. That is fine — it only constrains
any *later* CVPR/ICCV/ECCV submission to be **< 20% overlapping**. The parked
object-permanence paper was designed with a boundary that keeps it under that
(see `two-paper-plan-2026-09-21.md` §6). A **4-page** version would not be a
publication at all. **Recommendation from the previous session: go 8 pages**,
and keep the boundary in §6 of the two-paper plan so the parked paper stays
viable.

## 3. The paper

**Working title:** *Occlusion Is Not Absence: Counterfactual Full-Multiview
Absence Fixtures and Training-Time Presence Gating for 4D Gaussian Splatting.*

**Thesis.** A photometric residual cannot tell an object that has *left the
scene* from one that is merely *hidden*: in both cases its pixels vanish from
every training view. At the primitive level the two behave oppositely. A
presence gate is **inert** on a real occlusion and **helps substantially** on an
authored full-multiview absence — and a gate must be *trained with*, not applied
after.

### 3.1 Contributions after two rounds of adversarial review

Final shape agreed by Claude Fable 5.1 and gpt-5.6-sol (xhigh). Status tags:
**REC** = on the record, **RUN** = must be run, **WRITE** = must be written.

| # | contribution | status |
|---|---|---|
| **W-A (lead, Figure 1)** | **Occlusion vs absence, same gate.** On a real occlusion the trained gate is within 0.1 dB (G−U −0.07 P1, +0.05 P2; 4/4); on the authored-counterfactual absence it gives **12/12** positive paired prefixes over 3 scenes, **median +2.40 dB ghost core, +2.49 dB return**. Render-time gating on the same fixture: −5.9 dB ghost, +3.2 dB return — the **sign flip**. | REC |
| **W-B** | **Benchmark / protocol.** Three real-footage N3V scenes with an authored counterfactual full-multiview absence (frames 60–89, all cameras); six frozen event windows; wrong-membership shams A/B/L and timing shams GMIS/GONES; mechanism preconditions. | REC; release packaging WRITE |
| **W-C** | **Estimated-gate feasibility on the admitted subset.** Blind visibility-gap estimator T1 within one frame on 10/12 prefixes, zero false activations, 2 visible failures; one-click SAM 2 membership precision ≥ 0.9996, recall 0.978–0.990 on 8 admitted prefixes; within 0.21 dB of oracle. **Not** "fully estimated" — sear refused on membership, flame s3 failed on timing. | REC |
| **W-D (discussion, qualified)** | **Absence is free, return is hard.** STG (per-Gaussian temporal opacity, no gate) renders the gap core 3.4–7.5 dB *better* than the gated arm but is 1.3–2.5 dB *worse* on the return on 7/8 prefixes. **Unpaired and confounded** (six 50-frame models at 25k vs one 300-frame continuation at 12k). | REC on flame/sear; **RUN** on crb |

Cut from the contribution list (kept in the supplement): the instrument /
vacuity-precondition section.

### 3.2 Binding conditions — from the final review

* Say "**authored counterfactual**" everywhere; never "genuine" or "physical
  absence". Mandatory limitation sentence: *"Our evidence establishes behaviour
  on segmentation-guided counterfactual removals in captured multiview footage;
  it does not establish automatic discovery of physical scene exit or
  performance on naturally occurring absence-and-return events."*
* **The frozen NOT_MET verdict leads**, and the 12/12 pattern follows as
  descriptive. Suggested wording is in `assets/two-paper-2026-09-21/sol_round1.md`
  under "Frozen NOT_MET versus 12/12".
* **Independent mask re-scoring.** Generate SAM 2 masks independently; **human-
  audit every frame of P1, P2 and S1**, every fifth of H1/C1; re-run the frozen
  reducer unchanged; report DEVA and SAM 2 side by side. If mask choice reverses a
  scene-level sign, that becomes a central result.
* The 12 prefixes are runs **nested in 3 scenes**: scene-clustered plots, no
  language implying 12 independent scene samples.
* STG only with the unpaired/confounded qualifier; no numerical superiority
  claim in the abstract.

## 4. The "exceed existing benchmarks" question — handle this honestly

The principal believes exceeding benchmarks is enough. **Be precise about what
the evidence supports**, because this is where a workshop reviewer will push:

* **Where the paper does beat a published method:** on the **return window**,
  the gated arm is above STG on 7/8 prefixes by 1.3–2.5 dB.
* **Where it does not:** in the **gap**, STG is 3.4–7.5 dB *better*. And the
  comparison is **unpaired and confounded** (different schedule, frame count,
  capacity).
* **Whole-frame PSNR is not where this paper wins.** The event union is ~0.34%
  of pixel-times, so even an infinite event-region gain moves whole-frame PSNR by
  ~0.02 dB.

**The highest-leverage way to exceed the workshop bar** is therefore to make the
return-window comparison **clean**: retrain STG (and ideally one more
temporal-opacity baseline) under a *matched* schedule and frame set, paired with
the gated arm, on all three fixtures. That turns W-D from a qualified
observation into a controlled result. Decide early whether it fits by mid-
October; the STG pipeline already exists (`agent-control/realdata/stg/` on
Leonardo).

## 5. Traps this project has fallen into — do not repeat them

* **Report cohort means, not best-of-n.** A best-of-six was once headlined as the
  gap to a published number.
* **Whole-frame PSNR cannot see a 0.34%-of-pixels event.**
* **Look at the pixels of the scored region on every arm before reading a
  table.** A gate-off evaluation defect was once caught only by a montage.
* **A frozen reading rule needs a frozen precondition that the mechanism was
  exercised** — stated about the setup, never the score.
* **A validation that writes into its own reference is not a validation.**
* **Line numbers from a reviewer's upstream-GitHub citations have been wrong;**
  its local citations were exact. Re-locate before quoting.
* **Compute is ≈ 15,500 GPU-h**, measured: 1 GPU-h = 8 local-h on Leonardo.
  (Figures of 124k and 35k have both circulated and are wrong.)

## 6. Access and infrastructure

* **Leonardo.** From Windows, use the native `C:\Windows\System32\OpenSSH\ssh.exe`
  (or `/c/Windows/System32/OpenSSH/ssh.exe` from Bash). **Git Bash's own `ssh`
  cannot reach the ssh-agent.** The credential is a CINECA **12-hour
  certificate**; only the principal can renew it (2FA). Plan job chains to
  survive a lapse.
* `$W = /leonardo_work/EUHPC_D36_068/sri/proj_adags`. Fixtures under
  `$W/data_derived/absfix/`; wave-2 runs under `$W/runs/realdata/absfix2/`; STG
  under `$W/runs/stg/` and `$W/agent-control/realdata/stg/`; ledgers under
  `$W/agent-control/realdata/`.
* **Scheduling:** Leonardo structurally disfavours 1-GPU jobs
  (`PriorityWeightJobSize` 30× every other term). **Short walltimes are the only
  lever**; `boost_qos_dbg` (30 min cap) is where to measure a rate first.
* **Codex review:** the Codex MCP is a local shim over `codex exec`. Use
  `sandbox: danger-full-access` (read-only cannot read files on this box),
  inline the text under review, require a file-access attestation, and recover
  the reply text from `~/.codex/sessions/.../*.jsonl` because the shim returns
  only metadata.

## 7. What to read, in order

1. `research-wiki/operations/two-paper-plan-2026-09-21.md` — consolidated plan.
2. `research-wiki/assets/two-paper-2026-09-21/fable_round2.md` and
   `sol_round2.md` — the final revision and the final review; §2 of Fable's is
   the detailed W plan.
3. `research-wiki/operations/absfix-wave2-stage2-verdict-2026-09-15.md` — the
   main result W rests on.
4. `research-wiki/assets/absfix-stage2-report.md` and `absfix-stage2-paired.json`
   — the paired contrasts behind Table 1. **Generate tables from the JSON; never
   hand-copy numbers.**
5. `research-wiki/operations/absence-fixture-lane-2026-09-10.md` — the fixture,
   the sign flip, the gate-off defect.
6. `research-wiki/operations/realdata-gating-lane-2026-09-09.md` — the
   real-occlusion null.
7. `research-wiki/operations/paper-thesis-v1-2026-09-11.md` and
   `paper-thesis-v1-kill-argument-2026-09-11.md` — the thesis and the review that
   failed it; every kill point's disposition is in `fable_round1.md` §3.7.
8. `research-wiki/operations/literature-check-absence-return-2026-09-10.md` —
   related work and the dataset census.
9. `research-wiki/query_pack.md` — the whole programme, newest first.

## 8. What must be run and written

**RUN** (≈ 15–21 GPU-h required; more only if §4's matched baseline is chosen):
* STG on the `cut_roasted_beef` fixture, 2 seeds × 6 segment models.
* SAM 2 independent masks for the re-scoring (GPU minutes) — the human audit is
  the real cost.
* Optional: matched-schedule STG on all three fixtures (§4) — the single most
  valuable addition if time allows.

**WRITE:** the paper; Figure 1 (occlusion vs absence); fixture-construction
figure (QA montages already exist in `assets/`); per-frame profile plots from
`f_box_profile` files; paired-contrast strip plot from `absfix-stage2-paired.json`;
the supplementary protocol appendix; a release manifest.

## 9. Timeline to mid-October

| dates | work |
|---|---|
| Sep 22–24 | Verify venue details (§1). Decide 4 vs 8 pages (§2). Decide whether the matched-STG comparison (§4) is in scope. Freeze the claim list and table schema. Submit STG-crb. |
| Sep 25–Oct 1 | SAM 2 independent masks; human audit of P1/P2/S1; re-run the frozen reducer. Draft introduction and method. |
| Oct 2–8 | Results and figures generated from JSON. Matched STG runs if in scope. Full draft. |
| Oct 9–12 | Adversarial review of the draft (kill-argument; claim audit against the JSON; citation audit). |
| **Oct 13–15** | **Final revision — done.** |
| by Oct 29 | Submit to PhysWorldAI (window opens Sep 29). |

## 10. Open decisions for the principal

1. **4-page or 8-page** (§2) — recommendation: 8.
2. **Is the matched STG baseline in scope** (§4)? It is the clearest route to
   "exceeds the workshop bar".
3. **arXiv timing** for the full paper, given the parked paper.
4. **Release licences** — the fixtures derive from N3V footage.
5. Whether the principal's control model (review-gated substrate changes; the
   principal does the evaluation and analysis) applies here. It was stated for the
   external substrates Multi4D and Ex4DGS; this paper runs on ADAGS's own EL-GS
   code. Confirm before assuming.
