# Two-paper plan — round 1 (Claude Fable 5.1, 2026-09-21)

Input: `refine-logs/two-paper-2026-09-21/BRIEF.md`. Every number below carries its
source: a wiki page (`[[page]] §n`), a file and line, a Leonardo path read today
(marked `LEO`), or a URL fetched today. Three categories are tagged everywhere:
**REC** = on the record (measured, page cited); **RUN** = must be run (with cost);
**WRITE** = must be written (no compute). Anything I could not verify is marked
**UNVERIFIED** and collected in §9.

Facts carried per BRIEF §5 (each re-checked against its page today): Ex4DGS mean
32.6731 vs 33.7329 = **1.06 dB** gap, best-of-six 33.1426 is selection
([[ex4dgs-n3v-leonardo-reproduction-2026-09-19]] §13); noise floors are fixed-seed
lower bounds — Multi4D 0.3445 dB (n=3, effective seed 0 via `safe_state`,
[[multi4d-n3v-leonardo-reproduction-2026-09-18]] §9, §11), Ex4DGS sd 0.3564 /
range 0.9128 (n=6, §11); event union ≈0.34 % of pixel-times so whole-frame PSNR
is a trap ([[gap_map]] Measurement-Channel Update 2026-08-24: 0.3377 %); STG
renders the authored gap core 3.4–7.5 dB better than the gated arm and sits
1.3–2.5 dB below it on the return on 7 of 8 prefixes
([[absfix-wave2-stage2-verdict-2026-09-15]] §2.1); the fixtures are SA4D-edited
removals and kill-argument P1 is out of scope, not answered
([[paper-thesis-v1-kill-argument-2026-09-11]] §4); Multi4D `eval_seg.py` is
fail-open — re-read today on `$W/repo/multi4d/eval_seg.py`: silent `continue` on
a missing prediction at lines 36–39, full-image pixel accuracy at 14–15, clean
exit on zero matched frames at 47–48 (`LEO`); the round-2 reviewer's upstream
line numbers were wrong (`train_semantic.py` is 550 lines, `LEO wc -l`), the
substance held: `gaussians_transient=None` at `train_semantic.py:98-99`,
`--semantic_dim default=16` at `:517` (`LEO sed`).

---

## 1. Executive decision

**Paper W** is an 8-page workshop paper, *"Occlusion Is Not Absence: Counterfactual
Full-Multiview Absence Fixtures and Training-Time Presence Gating for 4D Gaussian
Splatting"*, built entirely from the EL-GS record: a benchmark/protocol
contribution (three authored-absence N3V scenes, a six-window event metric, sham
and timing controls, vacuity preconditions), one two-sided empirical finding
(a presence gate is inert on a real occlusion and +2.4/+2.5 dB on an authored
absence, and training-time versus render-time gating have opposite signs), a
fully estimated gate (blind visibility-gap estimator + one-click SAM 2 membership)
that matches the oracle to 0.21 dB where admitted, and the descriptive STG
result that absence is free and return is the hard part. Its target is a
**CVPR 2027 workshop (deadlines ≈ March 2027, list not yet announced)** with a
**4-page non-archival extended abstract to the NeurIPS 2026 PhysWorldAI workshop
(window Sep 29 – Oct 29)** as the fast, dual-submission-clean first outing, and
an arXiv posting the day Paper C is submitted. No open, topical, smaller
conference exists between today and March 2027 (WACV/3DV/BMVC/AAAI/ICRA/VR are
all closed; §2). **Paper C** is the object-permanence paper of plan v0.3 on the
Multi4D substrate — persistent roots with a presence notch, disposable leaves,
return-censored segmentation/re-ID as the headline endpoint, episode-split
control M2 as the load-bearing arm — targeted at **CVPR 2027 main (Nov 16 AoE)**
under a **hard go/no-go on Oct 12** (the v0.3 §8 two-week screen must pass on
`cut_roasted_beef`), with **ICCV 2027 (≈ March 2027, UNVERIFIED) as the declared
fallback**. The full v0.3 (physical BEHAVE benchmark, failure atlas, compression,
return-pose residual) is **not achievable by Nov 16** alongside W; the minimum
publishable C (§4.7) is, if and only if the Oct 12 gate passes. They do not
collide because W's contributions are a benchmark and a finding on the EL-GS
substrate with photometric event-window endpoints, while C's are a
representation and a semantic/identity endpoint on Multi4D with new experiments;
W is cited by C as the source of the fixtures and the metric, no table or
figure is shared, the shared text is well under 20 %, and the timing of the two
submissions never puts a > 4-page W inside C's review period (§2.3, §5).

---

## 2. Venue analysis

### 2.1 CVPR 2027 main (Paper C) — verified today

| item | value | source |
|---|---|---|
| paper registration | Nov 10 '26 AoE | https://cvpr.thecvf.com/Conferences/2027/Dates (fetched 2026-09-21) |
| **submission** | **Nov 16 '26 AoE**; CFP: "will not be changed" | same; https://cvpr.thecvf.com/Conferences/2027/CallForPapers |
| supplementary | Nov 23 '26 AoE | Dates page |
| reviews / decisions | Jan 25 '27 / Feb 25 '27 (Dates); CFP states review period "November 16, 2026 – February 22, 2027" — a 3-day inconsistency between the two official pages | both pages |
| page limit | **UNVERIFIED for 2027** (2027 Author Guidelines page is HTTP 404 today); CVPR 2026: "Papers are limited to eight pages, including figures and tables, in the CVPR style. Additional pages containing only cited references are allowed." | https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines |
| contemporaneous cutoff | "Papers that appeared online after September 15th, 2026 will generally be considered 'contemporaneous'" | 2027 CFP |

### 2.2 The dual-submission / prior-publication rule — what is actually posted

The CVPR 2027 CFP carries one sentence and delegates the rest:

> "By submitting to CVPR, authors confirm that no substantially similar paper has been or will be submitted to another conference or workshop during the review period (November 16, 2026 – February 22, 2027)." — https://cvpr.thecvf.com/Conferences/2027/CallForPapers
> "For complete dual submission guidelines and prior work policies, see the Author Guidelines on the conference website." — same page

The linked Author Guidelines page **returns 404 today** (https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines, fetched 2026-09-21). The operative definitions therefore come from the CVPR 2026 Author Guidelines (https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines), quoted verbatim, and the identical text is live for the 2027 cycle at WACV 2027 (https://wacv.thecvf.com/Conferences/2027/AuthorGuides) and 3DV 2027 (https://3dvconf.github.io/2027/author-guidelines/):

1. **Page threshold.** "A publication, for the purposes of this policy, is defined to be a written work longer than four pages (excluding references) that was submitted for review by peers for either acceptance or rejection, and, after review, was accepted."
2. **Non-archival workshops are NOT exempt.** "However, peer-reviewed workshop papers _are_ considered as publications if their length is more than four pages (excluding references), even if they do not appear in a proceedings."
3. **arXiv is exempt.** "arXiv preprints and university technical reports are _not_ considered as publications."
4. **Overlap.** "no publication substantially similar in content (defined as having 20 percent or more overlap) has been or will be registered or submitted to this or another conference, workshop, or journal during the review period."

**Consequences that bind this plan.** (a) The safe answers the brief anticipated are both real: a W of **≤ 4 pages excluding references** is never a "publication", archival or not; a W of **> 4 pages is a publication even at a non-archival workshop**, so it must share **< 20 % material** with C — non-archival status buys nothing. (b) Timing: a > 4-page W accepted **before** Nov 16 is a prior publication (overlap rule applies); a > 4-page W **submitted inside Nov 16 – Feb 22** is a concurrent submission (overlap rule applies); a > 4-page W submitted **after Feb 22/25** (CVPR 2027 workshops, ≈ March) is neither. (c) The Author Guidelines must be re-read the day the 2027 page goes live; if the four-page carve-out is absent from the 2027 text, the fallback is the 20 % rule alone, which §5 satisfies regardless.

### 2.3 Where Paper W can go — verified candidates

| venue | deadline (verified) | pages | archival | status | interaction with C | URL |
|---|---|---|---|---|---|---|
| **NeurIPS 2026 workshop PhysWorldAI, non-archival track** | "Sept 29 – Oct 29, 2026", notification "November 09, 2026" | "Long papers up to 8 pages and extended abstracts up to 4 pages" | non-archival (posted on workshop site) | **open Sep 29**; CFP lists "3D/4D reconstruction … physically grounded view synthesis" | decided before Nov 16; **4-page abstract = not a publication → clean**; 8-page = publication → 20 % rule | https://physworld-org.github.io/physworld.github.io/cfp |
| **CVPR 2027 workshops** | **not announced**; 2026 pattern: proposals Nov 3 2025, paper deadlines Mar 11 (4D World Models, archival) / Apr 2 (4D Vision, non-archival, "up to 8 pages excl. references") | 8 | per workshop | expect list ≈ late Dec 2026, deadlines ≈ March 2027 (**inference, UNVERIFIED**) | after C's decision → no concurrency; C already submitted → W is not prior | https://cvpr.thecvf.com/Conferences/2026/CallForWorkshops ; https://4dvisionworkshop.github.io/ ; https://ivl.cs.brown.edu/4dworldmodels/ |
| ICLR 2027 workshops | proposals Oct 9 2026; list Nov 29; suggested paper deadline Feb 1 2027; notification Feb 26 | per workshop | customarily non-archival (UNVERIFIED 2027) | inside C's review period | needs ≤ 4 pages or < 20 % overlap | https://iclr.cc/Conferences/2027/CallForWorkshops |
| Eurographics 2027 short papers | not posted; 2026 pattern Jan 16, ≤ 4 pages, EG DL | 4 | archival | UNVERIFIED | ≤ 4 pages → not a publication | https://eg2026.github.io/call_for_short_papers/ |
| Eurographics 2027 full papers | Oct 1 2026 | unlimited | CGF, archival | open 10 days; wrong community, no time | publication, 20 % rule, concurrent with C registration | https://eg2027.isti.cnr.it/call-for-papers/ |
| ICLR 2027 main | Sep 25 2026 | 9 | OpenReview | open 4 days; not a workshop fit | publication; decision inside C's window | https://iclr.cc/Conferences/2027/Dates |
| WACV 2027 main / workshops | R2 Aug 28 (closed); workshops: no dynamic-scene fit (MAVU Oct 20 aerial; CALIPOSE TBA) | 8 | yes | closed / poor fit | — | https://wacv.thecvf.com/Conferences/2027/Dates ; /Workshops |
| 3DV 2027 | Aug 28 2026 | 8 | IEEE (UNVERIFIED on 2027 pages) | **closed** | — | https://3dvconf.github.io/2027/call-for-papers/ |
| BMVC 2026 | May 29 2026 | 14 | yes | **closed** | — | https://bmvc2026.bmva.org/calls/call-for-papers/ |
| AAAI-27 / ICRA 2027 / IEEE VR 2027 / SIGGRAPH Asia 2026 TC / ECCV 2026 workshops | Jul 28 / Sep 16 / Aug 31 / Jul 27 / Jul 2026 | — | — | **all closed** | — | agent report, URLs in §9 |
| ICCV 2027 | no CFP; aggregators ≈ Mar 5–7 2027 (**UNVERIFIED**) | — | yes | not open | C's fallback, not W's venue | https://iccv.thecvf.com/Conferences/2027 |
| arXiv | any time | — | not a publication | — | exempt | — |

**Decision.** (i) Submit a **4-page extended abstract of W to PhysWorldAI by Oct 29** (fast outing, provably outside the dual-submission rule; topical fit to be confirmed by the lead against the CFP wording — it is a physical-world-models workshop, and W is a 4D-reconstruction paper with an explicit "physical absence not claimed" scope). (ii) Post the **full 8-page W to arXiv on Nov 16** together with C's submission (arXiv exempt; C cites it as prior work in the third person). (iii) Submit the full W to a **CVPR 2027 workshop in March 2027** (4D Vision successor if it runs; non-archival preferred so it can later be extended). If the lead wants the full W reviewed *before* Nov 16, the only open > 4-page option is the PhysWorldAI 8-page long paper, and then the < 20 % overlap of §5 is the binding constraint — it is satisfied, but the cleaner path is the 4-page abstract.

---

## 3. Paper W

### 3.1 Title and abstract (WRITE)

**Title.** Occlusion Is Not Absence: Counterfactual Full-Multiview Absence Fixtures and Training-Time Presence Gating for 4D Gaussian Splatting.

**Abstract (draft, 190 words).** Dynamic Gaussian representations give each primitive a temporal support learned from photometric residuals, and a residual cannot distinguish an object that has left the scene from one that is merely hidden: in both cases its pixels vanish from every training view. We show the two cases behave oppositely at the primitive level. On a real occlusion in a multi-view cooking capture, switching the object's primitives off costs 3.4 dB inside the occluded box, and training with a presence gate changes nothing measurable, because a primitive with broad temporal support paints the object and its occluder alike. On a genuine full-multiview absence the same gate, trained with, removes a ghost the ungated model paints (+2.4 dB) and restores the return (+2.5 dB) in 12 of 12 paired runs on three scenes, while applying it after training does the opposite (−5.9 dB). Because no public ≥ 8-camera capture supplies documented absence-and-return, we construct counterfactual absences on real footage by segmentation-guided removal, and evaluate with an event-window protocol with timing and membership controls. A blind visibility-gap estimator plus one-click SAM 2 membership reproduces the oracle gate to 0.2 dB. A temporal-opacity baseline renders the empty gap better than any gated model and the return worse: absence is free, return is the problem.

### 3.2 Contribution list, with evidence status

| # | contribution (what a reviewer ticks) | status | evidence |
|---|---|---|---|
| W1 | **Benchmark/protocol:** three real-footage N3V scenes with a counterfactual full-multiview absence of one object (SA4D hull-mode removal inside the segmenter silhouette, 1,720 rows, frames 60–89, all 20–21 cameras, derived scene byte-identical outside the edit), six frozen event windows (P1 core [63,87], P2 return [92,99], S1 [100,109], H1 pre [30,57], H2 whole, C1 control [230,259]), pooled-MSE window PSNR on the held-out camera, count-matched / local-mass-matched wrong-membership shams (A, B, L), timing shams (GMIS, GONES), per-cell mechanism preconditions | **REC** | [[absence-fixture-lane-2026-09-10]] §0–2, §5; [[absfix-wave2-stage2-verdict-2026-09-15]] §1, §3; `configs/n3v/absfix_gate_spec_v2.json`; derived roots `LEO $W/data_derived/absfix/{cut_roasted_beef_absfix_60_89, flame_steak_absfix_60_89_v3, sear_steak_absfix_60_89_v3}` (6,300 frames each) |
| W2 | **Finding, occlusion side:** on the real occlusion (beef under hand/knife, frames 158–187) the render-time gate is −3.39 dB in the box and −0.14 dB over 140–230 on one fixed model; trained with the gate (3,017–3,234 rows bound), every paired contrast is within 0.1 dB (G−U −0.074 P1, +0.050 P2; floor 0.040/0.068) | **REC** | [[realdata-gating-lane-2026-09-09]] §4.1; [[absence-fixture-lane-2026-09-10]] §3, §7A |
| W3 | **Finding, absence side:** trained gate vs ungated, 12 paired prefixes over 3 scenes: P1 median +2.40 (0.44..4.92) 12/12, mean +2.29 [1.10, 3.48]; P2 +2.49 (1.69..3.24) 12/12, mean +2.48 [2.09, 2.86]; H1 +1.14 12/12; H2 +0.01; C1 −0.11 (−0.28..+0.05) | **REC** | `research-wiki/assets/absfix-stage2-report.md` (paired contrasts, itt set); [[absfix-wave2-stage2-verdict-2026-09-15]] §1 |
| W4 | **Sign flip:** on the same fixture the render-time gate on the ungated 6k model costs 5.9 dB on P1 (31.22 → 25.32) and gains +3.2 dB on P2 (27.34 → 30.50); trained with, both windows gain | **REC** (one scene) | [[absence-fixture-lane-2026-09-10]] §5, §7B reading 1 |
| W5 | **Membership specificity:** G − wrong-membership shams on P1 +2.25 (A), +2.37 (B), +1.88 (L) median, 12/12; shams themselves gain only +0.04 / +0.10 / +0.29 over U; scene-dependent share (flame small, sear +0.4–2.7, crb +2.7–5.2); timing shams lose 4.1 dB (P1) and 3.7 dB (P2) | **REC** | report.md paired contrasts; verdict §0, §2 |
| W6 | **Fully estimated gate:** T1 (training views only) recovers the authored window within one frame on 10 of 12 prefixes with zero false activations (LRV3 exact; real occlusion 159/188 vs 158–187; crb 60/90–92), fails visibly on 2 (flame s3 [60,93], sear s2 [57,106]); SAM 2 one-click membership precision ≥ 0.9996, recall 0.978–0.990 on 8/8 admitted prefixes; GESTMEM − G within 0.21 dB (8 pairs) | **REC** | [[paper-thesis-v1-2026-09-11]] §2 table; [[absfix-wave2-stage1-2026-09-13]] §1; verdict §2, §3 |
| W7 | **Absence is free, return is hard:** STG (per-Gaussian temporal opacity, no gate) P1 35.2–35.7 vs our G 27.9–31.4; P2 30.5–31.0 vs G 29.7–33.0, below G on 7/8 prefixes | **REC** flame/sear (2 seeds each); **RUN** for `cut_roasted_beef` (no STG fixture run exists: `LEO ls $W/runs/stg` shows flame/sear only) ≈ 15–21 A100-h | verdict §2.1; [[paper-plan-assessment-2026-09-14]] §0 (STG 30–42 A100-h for two scenes) |
| W8 | **Instrument section (not a headline):** gate-off `--val` defect caught by a montage; four vacuity catches; fail-closed reducer; preconditions on the accepted set | **WRITE** | [[absence-fixture-lane-2026-09-10]] §6; [[gap_map]] Instrument-Validity Updates 2026-08-24 and 2026-09-13 |

What W does **not** claim, stated in the paper: physical absence discovery (P1), absence-vs-occlusion classification (T1 is a visibility-gap estimator, thesis §8.3), zero-error absence on real footage (only exact-zero *presence*; zero-error frames only on synthetic LRV3, 21/27, thesis §8.2), and any improvement over published Multi4D/Ex4DGS/STG whole-frame numbers.

### 3.3 Main-results tables (rows and columns named)

**Table 1 — Paired within-prefix contrasts, three scenes, 12k continuation from four independent 6k prefixes (REC).** Columns: P1 ghost core [63,87] · P2 return [92,99] · S1 settled [100,109] · H1 pre-gap [30,57] · H2 whole frame · C1 control [230,259]; each cell = median (min..max) dB, sign count. Rows, per scene (cut_roasted_beef calibration; flame_steak, sear_steak confirmatory) and pooled (12 pairs, mean with 97.5 % CI):

- G − U · G − GWRONGMEM_A · G − GWRONGMEM_B · G − GWRONGMEM_L · G − GMIS · GONES − U · GWRONGMEM_{A,B,L} − U · G − GEST · G − GESTMEM (8 pairs).
- Pooled values already exist: report.md "Paired contrasts (itt)"; per-scene values: verdict §1 tables. Every cell has a job id in `agent-control/realdata/jobs/ledger.txt` (verdict §4).
- Reported alongside, verbatim from the record: frozen verdict NOT_MET under the all-pairs 0.50 dB rule on both confirmatory scenes (flame prefix 1 G−U P1 +0.44; six sham pairs +0.23..+0.43), Claim B DESIGN_WITHOUT_POWER (verdict §0). The paper reports the frozen verdict and the descriptive pattern, and does not re-declare a rule after the fact.

**Table 2 — Occlusion vs absence, same substrate, same gate (REC).** Rows: real occlusion render-time (−3.39 P1 / +0.21 P2, one model); real occlusion trained, 4 pairs (−0.07 / +0.05); authored absence render-time on ungated 6k (−5.9 / +3.2); authored absence trained, 12 pairs (+2.40 / +2.49). Columns: gap window, return window, control window, n. Sources: realdata §4.1; fixture §7A, §5, report.md.

**Table 3 — Estimators (REC).** Rows: LRV3 synthetic (417 groups, 2 gated, 30/57 exact), real occlusion (1,218, 1 gated, 159/188), crb fixture (1,245, 2 gated, 60 / 90–92), flame s0–s3, sear s0–s3 (pooled windows [60,89]×4, [60,90]×2, [60,93], [57,106]); columns: groups, gated, onset/offset error, false activations. Membership block: SAM 2 precision / recall / mask IoU per admitted prefix (flame 4, crb 4), sear refusal (cam09 countertop) reported as an admission failure. Sources: thesis §2; stage1 §1; verdict §3.

**Table 4 — STG descriptive (REC + RUN).** Rows: flame s0/s1, sear s0/s1, **crb s0/s1 (RUN)**; our U range; our G range. Columns: P1, P2, H1, C1. Confounds stated in the caption verbatim from verdict §2.1 (six 50-frame models at 25k vs one 300-frame continuation at 12k; different representation; no pairing).

**Table 5 — Ablations (§3.4).**

### 3.4 Ablations

| ablation | question it answers | status | cost |
|---|---|---|---|
| timing shams GMIS / GONES | is it the window or the code path? | **REC** (12 prefixes) | — |
| membership shams A / B / L | is it the rows or generic deletion? | **REC** (12 prefixes) | — |
| G-est vs G-oracle | does estimated timing cost anything? | **REC** (12 pairs; two failures shown) | — |
| GESTMEM vs G | does estimated membership cost anything? | **REC** (8 pairs) | — |
| **product gate** (presence multiplies the learned marginal instead of replacing it) | is the −0.1 dB C1 cost the marginal override? | **RUN**: 4 cells on crb prefixes, flag-gated renderer change with verified empty diff when off (spec §13.21 b1) | ≈ 10 A100-h ([[paper-plan-assessment-2026-09-14]] §4.3) + ~1 day engineering |
| **free boundaries** (`elgs_a_lr > 0`, init from T1) | why not learn the boundaries? | **RUN**: 4 cells (spec §13.21 b2) | ≈ 10 A100-h |
| graded membership 25/50/75 % | how does partial membership degrade? | optional **RUN**: 12 cells, calibration only, labelled post-verdict exploratory (spec §13.21 c) | 12 × 2.3–5.9 h ≈ 30–70 A100-h |
| GESTMEM-S3 (SAM 3 text prompt) | does a click-free segmenter admit sear? | optional **RUN**: 8 cells + gated HF access + login-node install (spec §13.21 a; assessment §4.1) | ≈ 20 A100-h + access risk |

Recommended for W: the two 4-cell ablations (≈ 20 A100-h, both answer a question a reviewer will ask); graded membership and S3 only if the W schedule has slack after Oct 5.

### 3.5 Baselines

| baseline | role | code | status |
|---|---|---|---|
| ungated EL-GS substrate U (reserved-parity, `b0c_crb300_12k_rp.yaml`) | paired control; at STG parity on the ungated task (33.5050 vs published 33.52, frames 0–49) | ours | **REC** ([[stg-n3v-protocol-parity-2026-08-19]] App. C) |
| SpacetimeGaussians, pinned 427abfc5, published schedule | descriptive external temporal-opacity baseline | public | **REC** flame/sear; **RUN** crb |
| PersistGS | closest problem statement; contrast in related work only | **no code** (agent verification, https://arxiv.org/abs/2606.03479) | WRITE |
| SharpTimeGS / RetimeGS / FreeTimeGS | per-primitive lifespan family; related work | no code (SharpTimeGS project page has no source; RetimeGS none; FreeTimeGS official training code not found) | WRITE |
| **FreeTimeGS++** | the agent found a public repository (https://github.com/SNU-VGILab/FreeTimeGSPlusPlus) — this **contradicts the record's "no released training code"** ([[project-status-2026-09-16]] §5.2) | **UNVERIFIED** whether it trains on N3V | optional **RUN** if verified: 3 scenes × 2 seeds, cost unknown; otherwise cite as not run |
| CubifyGS, SA4D, TRiGS, Flow4DGS-SLAM | related work | — | WRITE |

### 3.6 Figures

| fig | content | status |
|---|---|---|
| F1 | concept: one 4D primitive shared between object and occluder (occlusion) vs. nothing else to paint (absence); gate trained-with vs applied-after | WRITE (draw) |
| F2 | fixture construction: real frame, SA4D background render, composite, QA montage on two cameras; hull-outlier control (cap floating on v1/v2, gone on v3) | REC assets: `absfix-flame_steak-qa-montage.jpg`, `absfix-sear_steak-qa-montage.jpg`, `absfix-*-cam00-boundary-v2.jpg`; crb `build_id117_60_89/qa_montage.jpg` on Leonardo |
| F3 | per-arm scored-crop montage, frames 40/63/75/87/92/99/109, rows GT/U/U render-gate/G/GEST/shams | REC: 12 tracked montages `assets/absfix-stage2-<scene>-p<S>-montage.jpg` |
| F4 | per-frame box PSNR over frames 30–110 for U, G, GMIS, STG, per scene | **WRITE** from `f_box_profile.{json,csv}` per cell and `$W/runs/stg/*/f_box_profile.csv` (exist) |
| F5 | paired-contrast strip plot (12 pairs × 6 windows × contrasts) | WRITE from `assets/absfix-stage2-paired.json` |
| F6 | T1 activation: gated cells over the cloud, per-camera contribution traces, the two failure prefixes | REC partial: `$W/runs/realdata/t1_viz`; WRITE composite |
| F7 | real-occlusion montage (GT / ungated / gated at 150–200) | REC: `assets/realdata-crb-gate-montage-2026-09-09.jpg` |

### 3.7 Kill-argument points and their disposition in W

| point | severity | disposition in W | status |
|---|---|---|---|
| P1 authored ≠ physical | critical | **Scoped out honestly, in title, abstract and a limitations paragraph**: "counterfactual absence on real footage"; report the supply census (DiVa-360 0/597, ImViD unclassified, Charge 0 scene-level events on 6/8 scenes, N3V occlusion-only, BEHAVE/Ego-Exo4D fail the ≥ 8-camera bar); P2/S1/H1/C1 read real unedited frames, only P1 reads teacher pixels. **A workshop review accepts this; a main-track review would not** — which is one reason W is W. | WRITE; **REC** for the census ([[literature-check-absence-return-2026-09-10]] §2, [[charge-absence-census-2026-09-15]] §4) |
| P2 construction-dependent oracle | major | evaluation ROIs from the held-out camera's DEVA ids (never used in construction); SAM 2 membership is instrument-independent; object and window frozen before any prefix trained; timing shams separate window from rows | **REC** (fixture §2; stage1 §1; spec §11) |
| P3 independent membership | critical | GESTMEM arm with frozen precision/recall precondition; admitted on 8 prefixes (precision ≥ 0.9996), DWP on sear (cam09) and flame s3 (T1) — reported as admission failures, never relaxed | **REC** (stage1; verdict §0) |
| P4 T1 detects invisibility | critical | renamed visibility-gap estimator; the real-occlusion null is the evidence that the gate is inert on occlusion — this becomes contribution W2, not a weakness | **REC** (thesis §8.3; fixture §7A) |
| P5 "exact absence" false | major | wording rule: exact-zero presence + lower ghost-core error; zero-error frames only on LRV3 | WRITE (thesis §8.2) |
| P6 generic-deletion share | critical | primary contrast is G − wrong-membership shams with zero truth overlap, A/B count-matched and L local-mass-matched; the frozen NOT_MET is reported as such | **REC** (verdict §1; spec §13.21 decision 1) |
| P7 replication and baseline | major | three scenes, 12 independent pairs, STG trained on the identical derived scenes and windows | **REC** (2 scenes) + **RUN** (crb STG) |

**Unpublishable-unless-fixed:** none remains for a *workshop* W, provided (i) the title and claims carry the P1 scope, (ii) the frozen NOT_MET verdict and the 1.59 dB paired sd on P1 (n₂ = 269, feasibility stop; report.md) are reported rather than replaced by the pooled CI, and (iii) the STG fact is in the abstract, not the limitations.

### 3.8 What must be run (cost) and what must be written

RUN (total ≈ 45–60 A100-h + queue waits; nothing on the critical path of C):
1. STG on the crb fixture, 2 seeds × 6 segment models, `agent-control/realdata/stg/{prep_derived.py, train_stg.sbatch, render_stg.sbatch}` exist (`LEO ls`) — ≈ 15–21 A100-h.
2. Product gate, 4 cells on the crb prefixes — ≈ 10 A100-h + renderer flag with empty-diff test.
3. Free boundaries, 4 cells — ≈ 10 A100-h.
4. (optional) graded membership 12 cells; GESTMEM-S3 8 cells; FreeTimeGS++ if its code trains.

WRITE: the whole paper (8 pages + supplementary); F1, F4, F5, F6 figures; the reducer output → LaTeX table script (from `absfix-stage2-paired.json` — never hand-copied); the supplementary protocol appendix (spec §11 windows, sham construction, preconditions, hashes); code-release manifest (`docs/paper-path.md` per assessment §6: ~25 load-bearing files, no refactor).

### 3.9 Reviewer attacks and answers (W)

1. *"STG beats you by 3–7 dB in the gap; your gate is worse than a 2024 baseline."* — It is the paper's thesis: exact-zero absence is free for any temporal-opacity primitive (rasterizer cull at α < 1/255; Multi4D hard-culls p(t) < 0.05, [[multi4d-code-map-2026-09-19]] §2); the claim lives on the return, where STG is 1.3–2.5 dB below the gated arm on 7/8 prefixes. The gap-core margin is scene-dependent and the paper says so. Confounds are printed in the caption; STG is descriptive, not paired.
2. *"The fixture is authored; the ghost window's ground truth is a render."* — Scoped (P1). P2/S1/H1/C1 are unedited frames; the return benefit (+2.49 dB, 12/12) is measured on real pixels. Supply census cited.
3. *"Membership is an oracle."* — GESTMEM (SAM 2, one click per camera) matches G within 0.21 dB on 8 admitted prefixes; refusals are reported. Wrong-membership shams cost 1.9–2.4 dB.
4. *"n = 4 per scene, medians, no test."* — 12 independent prefixes (prefix = seed), sign consistency 12/12 on P1 and P2, pooled 97.5 % CI [2.09, 2.86] on P2; the frozen all-pairs 0.50 dB rule is NOT_MET and reported; paired sd 1.59 dB on P1 means the ghost-core effect is not resolvable to 0.30 dB at this n (n₂ = 269) — stated.
5. *"The gate costs −0.1 dB in the control window."* — Product-gate ablation (RUN) tests the marginal-override attribution; C1 cost is within ±0.14 dB on the confirmatory scenes and −0.15..−0.29 on crb (verdict §2).
6. *"Novelty: SharpTimeGS/RetimeGS/FreeTimeGS make per-primitive lifespans standard; PersistGS owns object permanence."* — W claims a benchmark and a finding, not a representation; the event-window protocol has no precedent on N3V ([[literature-check-absence-return-2026-09-10]] §6); PersistGS is synthetic, 5 cameras, physics prior, held-out cameras that see the object during the gap ([[papers/ramlal2026_persistgs]]).
7. *"Why not train FreeTimeGS++?"* — verify the repository first (§3.5); if it trains, add it; if not, say "not run".
8. *"The all-pairs rule failed — the method failed."* — The paper distinguishes the frozen rule (NOT_MET, reported) from the descriptive pattern (no pair has G below U or any sham on either endpoint, verdict §0). DWP is "unresolved", never "negative" (assessment §5.2).

---

## 4. Paper C

### 4.1 Title, abstract, claims (WRITE)

**Title.** Persistent Object Memory for 4D Gaussian Splatting: Object Permanence through Full-Multiview Absence.

**Abstract (draft, 170 words).** Multi-level dynamic Gaussian representations separate a compact persistent bank from a large transient bank, but treat both as render support: when an object leaves every camera, its persistent Gaussians keep no state that survives the gap, and on return the object is re-explained by new transient primitives with no identity. We make the persistent bank an object memory. Each root carries identity, semantics and canonical appearance behind a learned presence function with an exact-zero absence notch; render detail lives in disposable leaves that inherit a root id and may split, die and respawn. We evaluate object permanence directly: semantics are trained on pre-gap masks only and the target is selected once before the gap; the held-out cameras then score first-return segmentation, recovery latency and target–distractor swaps, and the fraction of return contribution carried by pre-gap roots. An episode-split control with identical visibility and parameter budget but fresh identities after the gap isolates persistence from capacity. [Result sentence — MUST BE RUN.] On counterfactual absences in three multi-view scenes, [...]. Standard reconstruction and segmentation quality are unchanged.

**Claims.** C1 (headline): return segmentation/re-ID under censored supervision improves with persistent roots — M1 > M0 and **M1 > M2** — by ≥ 5 J&F points with paired direction positive in ≥ 5 of 6 scene-seed units (v0.3 §8 go/no-go). C2 (mechanism): persistent-root contribution retention on the return ≥ +15 pp over M0 and M2. C3 (secondary, photometric): return-window PSNR ≥ +0.5 dB using W's event-window protocol, cited. C4 (safety): Neu3D-Mask mIoU ≥ −1 point, ordinary-region PSNR ≥ −0.2 dB, no increase in gap false positives, all with the fail-closed evaluator. C5 (cross-substrate, optional): the notch on Ex4DGS reproduces the photometric direction (A2 > A1, A4).

### 4.2 What is on the record for C

| item | status | source |
|---|---|---|
| Multi4D reproduces: 33.4226 / 33.7672 / 33.5383, mean 33.576, band 33.4–34.0 frozen before, paper 34.02; FG 19,933–25,166 / BG 150–153k / TR 198–252k; 2:34–2:45 wall | **REC** | [[multi4d-n3v-leonardo-reproduction-2026-09-18]] §9 |
| fixed-seed spread 0.3445 dB; effective seed 0 via `safe_state` (`train.py:1113`, `utils/general_utils.py:125-127`); no `--seed` | **REC** | same §9, §11; code map §7 |
| Ex4DGS: E0 eval stack −0.0016 dB; mean 32.6731 (n=6) vs 33.7329, sd 0.3564, range 0.9128; point-cloud swap refuted (+0.54 dB the other way, p = 0.07) | **REC** | [[ex4dgs-n3v-leonardo-reproduction-2026-09-19]] §3, §11, §13 |
| attachment points for a presence multiplier on Multi4D FG: `gaussian_renderer/__init__.py:468` (hybrid), `:198` (standalone), `:314` (mask render — the silent bypass), BG `:475/:86`, TR `:419`; gradient with no CUDA change | **REC** (static read, partly verified) | [[multi4d-code-map-2026-09-19]] §5 |
| FG opacity has **no temporal dependence** under the shipped config (`enable_do_deform` False, `arguments/dynerf.py:45`); only TR has a (single, hard-culled) temporal bump | **REC (verified)** | code map §2 |
| lineage column hazards: ~20 grow/shrink paths, `capture()/restore()` positional tuples, TR has no checkpoint at all, ply `'f4'` ids exact < 2²⁴, TR `load_ply` prefix filter swallows `scale*/rot*/f_rest*` | **REC** | code map §6 |
| semantic stage excludes TR; `--semantic_dim` 16; `eval_seg.py` fail-open; TRASE mask store expected at `./data/TRASE/neu3d/<scene>/masks/camXX_XXXX.pt` | **REC (`LEO`)** | `train_semantic.py:98-99, :514-517`; `eval_seg.py:14-48` |
| **the semantic stage has never been run here; no TRASE/Mask-Benchmark data is on Leonardo** (`$W/repo/multi4d/data` absent; no `*trase*` under `data_staging`) | **REC (`LEO`)** | today's probe |
| Multi4D ingests `camNN.mp4 + poses_bounds.npy + fused.ply`, extracts `camNN/images/%04d.png` at 1352×1014 and skips on directory existence; DeGauss `fused.ply` exists for all six scenes (`_fused_ply/*.ply`, 8.6–10.9 MB) and `poses_bounds.npy` for all six | **REC (`LEO`)** | repro §3.1–3.3; today's probe |
| the fixtures are Blender-layout (`transforms_train.json`, `images/camNN_FFFF.png`, 6,300 frames); STG conversion exists (`prep_derived.py` → `colmap_0..250`) | **REC (`LEO`)** | today's probe |
| Leonardo: `PriorityWeightJobSize` 30× everything, no QOS upgrade for 1-GPU jobs, walltime the only lever; resume unusable on both substrates | **REC** | repro §8; §3.6 |
| plan v0.3 mechanism (roots/leaves, notch, hashed lineage sketches, unbiased with ~33 % per-draw error), arms M0–M4, evaluation protocol, go/no-go | **WRITE-level** (nothing implemented) | [[ex4dgs-multi-interval-plan-v0.3-2026-09-21]] §1–4, §8 |
| round-1 kill argument ("identity is bookkeeping"), round-2 reframe | **REC** | review round1 §12; round2 |
| novelty pressure: PersistGS (CVPRW 2026, no code), TRiGS 2604.00538, Flow4DGS-SLAM 2604.22339 (GMM temporal opacity), SharpTimeGS, CubifyGS (IROS 2026), MaskGaussian (static pruning masks) | **REC** | review round1 §10; agent report §6 |

### 4.3 Main-results tables (rows and columns named)

**Table C1 — Return-censored 4D segmentation (headline).** Rows: per scene (crb, flame_steak, sear_steak fixtures) × arm (M0 shipped Multi4D with censored supervision · M1 persistent-root notch · **M2 episode-split control** · M3 M1 + sketch loss · [M4 only if the oracle test says location fails]); n = 6 paired seeds (target; screening at n = 3), mean ± sd and paired direction count. Columns: first-return J&F · J&F at return +1/+5/+10/+30 · recovery latency (frames) · gap false-positive area · target–distractor swap rate · IDF1/HOTA where a distractor exists. **RUN.**

**Table C2 — Root contribution retention.** Rows as C1; columns: fraction of return-window contribution mass from pre-gap roots · from their descendants · from new roots; causal cohort ablation (disable pre-gap roots at fixed parameters → registered event-window change vs repeat-render null band, v0.2 §5 P1). **RUN.**

**Table C3 — Photometric event windows (secondary, W's protocol, cited).** Rows as C1; columns P1, P2, S1, H1, H2, C1 pooled-MSE PSNR on cam00. **RUN** on Multi4D (new experiments; no W table is reused).

**Table C4 — Safety.** Standard Neu3D-Mask mIoU/mAcc (fail-closed evaluator, missing prediction = failure) · whole-frame PSNR/SSIM/LPIPS in Multi4D's own conventions (both SSIMs, LPIPS non-normalised, labelled) · primitive counts per bank · training time. **RUN**; M0 must first reproduce the published semantic number (Table C4 row 0, **RUN**, cost UNVERIFIED).

**Table C5 — Ex4DGS cross-substrate probe (optional).** A1 corrected single support (q = 0) · A2 notch · A4 q frozen · A6 matched budget; 3 fixtures × 3 seeds; columns P1/P2/C1 and contribution retention via the auxiliary-channel attribution. **RUN** ≈ 36 × 1.2 = 43 GPU-h.

**Ablations.** sketch loss on/off and **code-permutation control** on the return side; `--occlusion_aware` render on/off; KNN smoothing on/off; TR-encoded-target check (fraction of target mask explained by TR); notch width τ; oracle-initialised notch vs learned; A0 vs A1 continuity repair on Ex4DGS (quantifies the shipped discontinuity, code map §2).

### 4.4 Baselines

| baseline | role | code | status |
|---|---|---|---|
| Multi4D shipped (M0), censored supervision | primary | public, pinned c483e82 | reconstruction **REC**; semantic **RUN** |
| M2 episode-split | load-bearing control | ours | **RUN** (WRITE first) |
| TRASE (arXiv 2411.19290) | external 4D segmentation baseline named by round 2 | **UNVERIFIED** code | RUN if public, else "not run" |
| SAM 2 video propagation from the pre-gap mask (2D VOS first-mask baseline) | external, identical supervision | public; `scripts/s2_sam2_propagate.py` exists | **RUN** (cheap, CPU/GPU minutes) |
| STG / Ex4DGS / EL-GS gated | photometric-only reference rows | exist | **REC** photometric; no semantics |
| PersistGS | related work only | no code | WRITE |

### 4.5 Figures (C)

F1 roots/leaves formulation with the notch presence function and the sketch channels (WRITE); F2 the censored-return protocol timeline (WRITE); F3 qualitative return: GT mask, M0, M1, M2 at return +1/+10/+30 on held-out cameras (RUN); F4 contribution-mass Sankey pre-gap → return (roots/descendants/new) per arm (RUN); F5 failure atlas strata curves — only if time (RUN); F6 Multi4D-in-the-gap montage: does the persistent bank ghost? (the first thing to render, §4.6).

### 4.6 Feasibility verdict for 16 November — hard-nosed

**The full plan v0.3 is not achievable by Nov 16 with W in parallel.** Reasons, each sourced:

1. **Nothing is implemented** (v0.3 status line). The Multi4D changes are: presence multiplier at five sites including the mask render (code map §5), a lineage column threaded through ~20 mutation paths with three silent-drop classes (code map §6), seed plumbing in `safe_state`, fail-closed `eval_seg.py`, return-censoring in `train_semantic.py`, exact pre-gap selection, occlusion-aware rendering, root/descendant contribution logging, the sketch loss through the three auxiliary channels, and the M2 identity-split. Ex4DGS's own change surface is documented as smaller (seven lines for α(t)) but its four silent bypasses (`densify_and_split:945`, `adjust_temp_opa:1338`, promotion `:1180-1187`, `load_ply:633-643`, code map §6) apply to any K > 1 form.
2. **The semantic stage has never been reproduced and its data is not on the cluster** (`LEO`): the Mask-Benchmark store must be acquired (HF `yunjinli/Mask-Benchmark`, size UNVERIFIED), the TRASE layout built, and the published Neu3D-Mask number reproduced before any semantic number is evidence. This is the first gate and it is pure risk.
3. **The fixtures must be re-ingested** into Multi4D's layout (symlinked `poses_bounds.npy` + `fused.ply`, `camNN/images/` populated from the 6,300 edited frames). Feasible in a day — `prep_derived.py` did the same for STG — but a converter bug here fails open (repro §3.1: the extractor skips on directory existence alone).
4. **The kill-argument's strongest point is unanswered by design**: authored fixtures license fixture claims only (round 1 §9; v0.2 §7); the physical benchmark route in v0.3 §7 uses BEHAVE, which has **4 Kinects** (agent report §6; [[literature-check-absence-return-2026-09-10]] §2 "fails the camera bar"), ≈140 GB, an MPI non-commercial licence forbidding redistribution and requiring face blurring — a CVPR reviewer will ask why a 4-camera RGB-D rig is a 4DGS benchmark. No physical evidence will exist by Nov 16.
5. **The mechanism is unproven** and the round-1 reviewer's position — "not worth an eight-week commitment" until a bounded pilot passes — was accepted into v0.2/v0.3. The v0.3 screen (days 1–14, 120–145 GPU-h) cannot start before the semantic stage reproduces; started Sep 28 it ends ≈ Oct 12.
6. **Statistics.** The headline metric has no measured variance; at n = 3 the go/no-go rules are screening rules (v0.3 §8). A main-track paper needs n ≥ 6 paired seeds per scene and a variance statement, which is the Oct 12 → Nov 1 window if the gate passes.
7. **Calendar arithmetic.** 8 weeks; W needs ≈ 3 weeks of writing; C needs ≈ 3 weeks of writing after results; that leaves ≈ 2 weeks of C results-generation after the gate — enough for the minimum package (§4.7) only.

**Verdict:** C for CVPR 2027 is **conditional**: GO only if all of §4.7's gate criteria hold on Oct 12; otherwise C retargets **ICCV 2027** (CFP not posted; aggregators say ≈ Mar 5–7 2027, UNVERIFIED) with the full v0.3 scope and a physical event if one can be captured or certified by then. The lead's preference for CVPR is honoured by making the gate explicit rather than by assuming it.

### 4.7 Minimum publishable Paper C (the CVPR 2027 package if the gate passes)

Scope: Multi4D only (Ex4DGS optional), three authored fixtures, arms **M0 / M1 / M2** (+ M3 if S2 wins the screen), **n = 6 paired seeds** per scene, headline Table C1 + C2, secondary C3, safety C4, the SAM 2 VOS external baseline, ablations sketch on/off + code permutation + occlusion-aware. Declared out of scope, in the paper: physical absence (cite W's census and the 4-camera BEHAVE bar), compression, FPS, failure atlas. Cost: reconstructions 3 scenes × 3–4 arms × 6 seeds ≈ 54–72 × 2.7 = **146–195 GPU-h**, plus semantic stages (cost UNVERIFIED, expected small), plus the screen 120–145 GPU-h, plus optional Ex4DGS 43 GPU-h: **≈ 350–400 GPU-h total**, against ≈ 35k GPU-h available (§7 note on units).

**Oct 12 gate (all must hold, from v0.3 §8, on crb, paired n = 3):** return J&F ≥ +5 vs both M0 and M2; root contribution ≥ +15 pp; return-region PSNR ≥ +0.5 dB or comparable tracking gain; Neu3D-Mask mIoU ≥ −1; ordinary-region PSNR ≥ −0.2 dB; no increase in gap false positives; paired direction positive ≥ 5/6 units; **and** the two instrument preconditions: M0 semantic reproduces the published crb number (fail-closed), and F6 shows the persistent bank actually ghosts in the gap (if TR paints the background over the roots, the mechanism has nothing to remove and the design is DWP before it starts — the real-occlusion lesson, [[realdata-gating-lane-2026-09-09]] §4.1).

### 4.8 What must be run (C) and what must be written

RUN, in order (all GPU-h at Multi4D 2.7 / Ex4DGS 1.2 per run; queue waits extra):
1. Acquire Mask-Benchmark to the TRASE layout (login node, no compute-node network; size UNVERIFIED); reproduce M0 semantic on crb (≤ 1 day; cost UNVERIFIED).
2. Fixture ingestion converter + M0 reconstruction on the three fixtures (3 × 2.7 = 8 GPU-h) + F6 montage.
3. Oracle test (render-time: select the pre-gap root cohort by hand, ask whether its deformed geometry lands on the returning object; ≈ 0 GPU-h beyond a render).
4. Single-seed screen S0–S5 on crb (6 × 2.7 = 16 GPU-h + semantics).
5. Paired confirmation, 2 scenes × 3 arms × 3 seeds = 18 recon = 48.6 GPU-h (v0.3 §8).
6. If GO: scale to 3 scenes × n = 6 (§4.7), ablations, optional Ex4DGS probe.

WRITE: presence multiplier at the five sites with a flag-off empty-diff test; lineage column with mutation unit tests through clone/split/prune/lift and a `capture()/restore()` round-trip test (the `_packet_ids` lesson, code map §6d); `--seed` in `safe_state`; fail-closed `eval_seg.py`; return-censoring and exact pre-gap selection; occlusion-aware render for evaluation; contribution logging; sketch loss + permutation control; M2 split; the reducer with preregistered rules and preconditions (spec-driven, tested on one real extractor output — the 2026-09-15 lesson); the paper.

### 4.9 Reviewer attacks and answers (C)

1. *"Authored fixtures do not show object permanence."* — Stated scope; the claim is about a representation's behaviour under a certified full-multiview absence with exact ground truth, and W (cited) documents that no public capture supplies the event. Weakest point of C; a CVPR reviewer may reject on it alone. Mitigation available by Nov 16: none beyond honesty. Mitigation by ICCV: a physical capture or a certified event.
2. *"Gaussian identity is bookkeeping; a copied id fakes continuity."* — The claim is about root contribution mass and downstream semantic behaviour under censored supervision, not row survival; M2 has identical visibility and budget and fresh identities; the operational test is return segmentation with the selection frozen before the gap.
3. *"You defined your own metric."* — Standard Neu3D-Mask mIoU/mAcc are reported unchanged (C4) and the fail-open defect in the shipped evaluator is disclosed and fixed; the censored-return protocol is the object-permanence analogue of the co-visibility masking precedent ([[literature-check-absence-return-2026-09-10]] §5).
4. *"n = 3 / 0.3445 dB floor."* — n = 6 paired seeds with real seed plumbing; the floor is reported; photometric endpoints are secondary.
5. *"Multi4D reproduces 0.44 dB under its paper; Ex4DGS 1.06 dB under."* — Stated; all comparisons are paired within-implementation; no claim of improving published numbers.
6. *"TRiGS / Flow4DGS-SLAM / PersistGS / CubifyGS already do this."* — Differentiation: per-root exact-zero presence with disposable leaves on a multi-level substrate; identity supervised through hashed contribution sketches in existing render channels; evaluation under censored return. TRiGS (2604.00538) frames the problem; whether it solves permanence must be read before submission (**UNVERIFIED** — not in `research-wiki/papers/`).
7. *"Hashed sketches are a high-variance estimator."* — Unbiased, ~33 % per-draw relative error at 3 channels (v0.3 §2), used as an SGD gradient with resampled hashes, with the permutation control; the measurement endpoint is the exact contribution logging, not the sketch.

---

## 5. The W/C boundary

| item | W | C | why admissible |
|---|---|---|---|
| **claim: occlusion ≠ absence at the primitive level** | headline finding (Tables 2) | one motivating sentence, cited | C adds no occlusion experiment |
| **claim: training-time vs render-time sign flip** | W4 | cited only | — |
| **claim: presence gate on EL-GS improves ghost core and return** | W3/W5, Table 1 | not reported | different substrate; C never trains EL-GS |
| **claim: T1 + SAM 2 fully estimated gate** | W6, Table 3 | not used (C selects the target once from the exact pre-gap GT mask, v0.3 §4) | — |
| **claim: absence is free, return is hard (STG)** | W7, Table 4 | cited as motivation for a return endpoint | — |
| **claim: persistent roots + notch improve return re-ID** | absent | headline | — |
| **claim: root contribution retention / sketches** | absent | headline mechanism | — |
| **datasets: three authored N3V fixtures** | introduced and released by W | **reused, cited** | dataset reuse across papers is normal; C's tables are all new runs |
| **metric: six event windows, pooled MSE** | introduced by W | secondary endpoint, cited | protocol reuse, no shared numbers |
| **metric: censored-return J&F / latency / swaps / root mass** | absent | introduced by C | — |
| **substrate** | EL-GS (ADAGS) + STG descriptive | Multi4D (+ Ex4DGS optional) | disjoint |
| **figures** | F1–F7 of §3.6 | F1–F6 of §4.5; C's F2 may show one fixture frame with attribution | no figure shared |
| **tables** | Tables 1–5 | Tables C1–C5 | no table shared |
| **text** | related work paragraph on PersistGS/lifespan family; fixture-construction paragraph | one paragraph summarising the fixture with citation; own related work | estimated shared prose < 5 %, far under the 20 % rule |
| **census (DiVa/ImViD/Charge/BEHAVE)** | W's Claim-3 evidence | one sentence, cited | — |
| **timing** | 4-page abstract decided Nov 9 (before C); full W on arXiv Nov 16; full W to a workshop ≈ March 2027 (after C's decisions) | Nov 16 | no > 4-page W is submitted or accepted inside Nov 16 – Feb 22 |

Rule for both drafts: any table cell that appears in W may not appear in C, and vice-versa; C's introduction cites "[W, arXiv]" for the fixtures and the finding and reproduces no number from it beyond one sentence.

---

## 6. Timeline to 16 November (both papers, with gates)

Allocation check today: `saldo -b` shows EUHPC_D36_068 at 20,094 of 144,000 local-h consumed (`LEO`); work area 3.6 of 4.0 TB (415 G free, `LEO df`). **Unit note:** the brief's "~124k GPU-hours" is the local-hour balance; at ≈ 3.5 local-h per GPU-h ([[realdata-gating-lane-2026-09-09]] §1) that is ≈ 35k GPU-h. Still not the constraint.

| week | dates | Paper W | Paper C | gate / verify |
|---|---|---|---|---|
| 1 | Sep 21–27 | freeze W claim list and table schema (this document → `WRITE`); submit STG-crb (≈ 20 A100-h) and product-gate + free-boundary cells (≈ 20) after a flag-off empty-diff test; verify FreeTimeGS++ repo | acquire Mask-Benchmark; fail-closed `eval_seg.py`; `--seed` in `safe_state`; fixture converter; M0 recon on 3 fixtures (8 GPU-h); F6 montage | **G0 (Sep 28):** M0 semantic reproduces published crb number fail-closed; F6 shows the persistent bank ghosts. Fail → C's mechanism is DWP on Multi4D → decide Ex4DGS-only C or ICCV |
| 2 | Sep 28–Oct 4 | draft 1 (intro, protocol, occlusion-vs-absence); F4/F5 scripts from JSON; PhysWorldAI 4-page abstract drafted | presence multiplier + lineage column + tests; oracle test; single-seed screen S0–S5 (16 GPU-h + semantics) | montage of every arm before any table |
| 3 | Oct 5–11 | full W draft; kill-argument on the draft (`kill-argument` skill); claim audit against `absfix-stage2-paired.json` | paired confirmation 18 recon (48.6 GPU-h) | — |
| — | **Oct 12** | — | **G1: v0.3 §8 go/no-go on crb, n = 3.** GO → CVPR package §4.7. NO-GO → C retargets ICCV 2027; all C compute continues at lower priority; W becomes the only Nov deliverable | reducer output hashed before montages viewed |
| 4 | Oct 12–18 | W second draft; abstract to PhysWorldAI (deadline Oct 29) | if GO: 3 scenes × M0/M1/M2(/M3) × 6 seeds submitted (146–195 GPU-h, short walltimes, warden); SAM 2 VOS baseline | every job id in the ledger with a reason |
| 5 | Oct 19–25 | W final; supplementary; hold for arXiv | results in; ablations (sketch off, permutation, occlusion-aware); C draft 1 | **G2 (Oct 26):** C1/C2 tables complete on ≥ 2 scenes at n = 6 or C drops to ICCV |
| 6 | Oct 26–Nov 1 | PhysWorldAI abstract submitted (by Oct 29) | third scene + Ex4DGS probe (optional); C draft 2; kill-argument | — |
| 7 | Nov 2–8 | — | related work (PersistGS, TRiGS, Flow4DGS-SLAM, SharpTimeGS, CubifyGS, MaskGaussian); claim audit; citation audit; fresh-context review | `paper-claim-audit`, `citation-audit` |
| 8 | Nov 9–16 | arXiv W on Nov 16 | register Nov 10; **submit Nov 16 AoE**; supplementary Nov 23 | re-read CVPR 2027 Author Guidelines when live |
| after | Dec–Mar | full W to a CVPR 2027 workshop (≈ March) or ICLR 2027 workshop (Feb 1, then ≤ 4 pages or < 20 %) | if NO-GO: ICCV 2027 (verify deadline when posted) with physical event work | — |

Compute totals: W ≈ 45–60 A100-h; C ≈ 350–400 GPU-h if GO. Wall-time risks: 12-h certificate (chains with Slurm dependencies), start-up stalls (warden), Multi4D host RAM 120 G per job, resume impossible on both substrates.

---

## 7. Open risks, ranked, each with the cheapest check

1. **Multi4D's persistent bank may not ghost in the gap** (TR paints the wall over the roots), making C's mechanism inert on this substrate exactly as the gate was inert on the real occlusion. Check: one M0 reconstruction on the crb fixture + a P1-window montage of the FG-only render (`best/static_persistent/`) — 2.7 GPU-h. Week 1.
2. **Semantic stage does not reproduce / Mask-Benchmark unavailable or huge.** Check: fetch the HF dataset card and size today (CPU); reproduce on crb — week 1. If it fails, C's headline metric has no reference and C goes to ICCV.
3. **Dual-submission text for 2027 differs from 2026.** Check: re-fetch https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines weekly; the plan is robust under either the 4-page carve-out or the bare 20 % rule.
4. **PhysWorldAI is a poor topical fit and the abstract is desk-rejected.** Check: the lead reads the CFP (https://physworld-org.github.io/physworld.github.io/cfp) — 10 minutes; fallback is arXiv + CVPR workshop only (no fast outing).
5. **CVPR 2027 workshop list has no 4D/dynamic-scene workshop.** Check: workshop list ≈ late Dec 2026; fallback ICLR 2027 workshop (≤ 4 pages or < 20 %) or Eurographics short (≤ 4 pages).
6. **The fixture ingestion converter fails open** (extractor skips on directory existence). Check: sha256 of ten edited frames in the Multi4D staging tree against the derived root before any training — CPU, minutes.
7. **Lineage column silently drops at the FG→TR lift or on `restore()`.** Check: unit test asserting N-equality after every mutation and a `capture()/restore()` round trip — zero GPU.
8. **STG-crb run changes the story** (e.g. STG below G on the return only on the confirmatory scenes). Check: it is the first W run; W's abstract sentence is written after it returns.
9. **FreeTimeGS++ repository trains and a reviewer asks why it is absent.** Check: clone and run its README on one scene — ≤ 1 GPU-h if it builds; otherwise "not run" with the reason.
10. **The product gate removes the C1 cost but changes P1/P2** — then W must present it as the formulation. Check: the 4 cells; if it wins, re-run G on all 12 prefixes (≈ 40 A100-h) or present it as an ablation only.
11. **Reviewer P1 (authored ≠ physical) sinks C at CVPR regardless of the gate.** No cheap check; the only retirement is a physical event, which is the ICCV plan. Decide on Oct 12 whether the CVPR attempt is worth the reviewers' first impression.
12. **`peak` vs `best/` and the two SSIM / non-standard LPIPS conventions on Multi4D** leak into C's safety table. Check: label every convention in the table caption; never read `best/per_frame_metrics.json` as the peak (repro §9).
13. **Disk**: 415 G free; Multi4D writes 612–718 MB per run; 72 runs ≈ 50 GB plus semantics — fine, but the approved removals (`runs/realdata_gate` 119 GB) should still run.

---

## 8. Questions for the reviewer (gpt-5.6-sol)

1. Is the 4-page PhysWorldAI abstract + arXiv + March-2027 workshop route the right resolution of the dual-submission problem, or should the full 8-page W be submitted to PhysWorldAI's long track before Nov 16 and rely on the < 20 % overlap of §5? Attack §5's overlap estimate.
2. Is W's contribution list (benchmark + two-sided finding + estimated gate + STG fact) a workshop paper or "an oracle diagnostic around known temporal-opacity machinery" as the kill-argument put it? Which single item would you cut and which would you promote?
3. Is reporting the frozen NOT_MET verdict beside the pooled 12/12 pattern honest enough, or does the paper need a new preregistered rule on a fresh scene?
4. Does the STG comparison (unpaired, different schedule and representation) belong in the abstract, or does its confound structure make it inadmissible even as a descriptive fact?
5. Attack the Oct 12 gate for C: are the v0.3 §8 thresholds screening rules a CVPR reviewer would accept as pre-registration, and is n = 6 paired seeds on three authored fixtures a main-track evidence base for "object permanence"?
6. Given BEHAVE has 4 Kinects, is any physical benchmark reachable before ICCV 2027, and if not, is C's claim publishable at a main venue at all under authored evidence?
7. Is M2 (identical visibility and budget, fresh identities post-gap) actually constructible on Multi4D without a CUDA change, given the FG→TR lift copies rows (code map §4) — or does the lift make "episode split" leak?
8. Is the return-censored J&F protocol self-serving? Propose the one external baseline that would embarrass it.
9. Is the recommendation to write W first (weeks 1–3) and gate C on Oct 12 the right ordering, or does it starve C's only chance at CVPR?
10. Which of the 13 risks in §7 is mis-ranked?

---

## 9. What I could not verify (UNVERIFIED list)

- CVPR 2027 Author Guidelines (HTTP 404); the 2027 page limit; whether the 4-page carve-out and 20 % rule appear verbatim in 2027.
- CVPR 2027 workshop list and deadlines (not announced; 2026 pattern used).
- ICCV 2027 deadline (aggregators only); SIGGRAPH 2027 (official site unreachable).
- ICLR 2027 workshop archival status; Eurographics 2027 short-paper page limit.
- SharpTimeGS's venue (arXiv comment absent; CVPR 2026 claim is third-party).
- FreeTimeGS++ repository (https://github.com/SNU-VGILab/FreeTimeGSPlusPlus) — existence found by the agent; whether it trains on N3V not checked.
- TRASE code availability; Mask-Benchmark size and layout; cost of Multi4D's semantic stage.
- Whether Multi4D's persistent bank ghosts in the authored gap (risk 1) — never rendered.
- Whether the Multi4D semantic stage reproduces its published Neu3D-Mask number here — never run.
- BEHAVE's certified all-camera disappearance count — never censused; its 4-camera count is from the dataset page.
- TRiGS (2604.00538) mechanism — abstract only, no `papers/` page.
- Location of the SAM 2 final masks on Leonardo (`$W/runs/realdata/absfix2/s2` does not exist; the status page says they exist elsewhere).
- The exact GPU-h of one more STG scene (derived from the 30–42 A100-h two-scene estimate).
- Whether `prep_derived.py` (STG) can be reused unchanged for Multi4D's layout — it targets COLMAP layout, Multi4D wants LLFF `poses_bounds.npy` + `fused.ply`; a new small converter is assumed.
