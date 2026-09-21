# Two-paper plan — round 2 (Claude Fable 5.1, 2026-09-21, final revision)

Inputs: `fable_round1.md`, `sol_round1.md`, `orchestrator_notes_r1.md`,
`orchestrator_notes_r2.md`, plus new read-only checks made today (marked `LEO`
for Leonardo, `APOLLO` for the `apollo:` rclone remote, URL for the web). Tags
as before: **REC** on the record · **RUN** must be run (GPU-h) · **WRITE** no
compute · **UNVERIFIED** collected in §9.

Corrections carried without re-litigation (orchestrator, verified): compute
balance **≈ 15,500 GPU-h** (1 GPU-h = 8 local-h, measured from 17 Slurm jobs;
`orchestrator_notes_r1.md` §3); **TRASE is public** (MIT, training + eval +
Docker, https://github.com/yunjinli/TRASE); **Mask-Benchmark is public,
ungated, ~985 MB, CC-BY-NC 4.0** (https://huggingface.co/datasets/yunjinli/Mask-Benchmark);
Sol's ImViD availability statement is **UNVERIFIED** (Codex memory, not this
workspace).

New facts established today, each load-bearing below:

* **Mask-Benchmark `Neu3D-Mask/` contains exactly five scene folders:
  `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_steak`,
  `sear_steak`** (https://huggingface.co/api/datasets/yunjinli/Mask-Benchmark/tree/main/Neu3D-Mask,
  fetched 2026-09-21). So the "reproduce the published crb semantic number" gate
  is possible as written, and all three authored-fixture scenes have GT.
* The card's "536×536" is the Hugging Face parquet-viewer image size; the actual
  ground truth ships as `Mask-Benchmark.zip` (595,236,297 bytes, root of the
  repo) as `Neu3D-Mask/<scene>/gt_masks/*.png` (Multi4D README, `LEO sed
  $W/repo/multi4d/README.md`). The PNG resolution is **UNVERIFIED** until the zip
  is opened; `eval_seg.py:42-43` silently resizes a prediction to the GT shape
  (`LEO`), which is the exact silent-mismatch hazard the orchestrator named and
  is fixed in the fail-closed evaluator (§3.3).
* Multi4D's semantic stage needs TRASE per-(camera, frame) SAM masks at
  `<trase_root>/<scene>/masks/camXX_XXXX.pt`; **"A missing mask does not fail
  the run — it prints `Mask not found` and skips the loss for that view"**
  (README, `LEO`) — a second fail-open, in training, not only in evaluation.
  Distillation ≈ 3 h per scene (README).
* **Cutie** (https://github.com/hkchengrex/Cutie): MIT, pretrained weights via
  `cutie/utils/download_models.py`, first-frame mask then propagation with no
  further prompts (`scripting_demo.py`: `processor.step(image, mask)` at
  `ti == 0`, `processor.step(image)` after) — exactly the censored protocol.
* **ImViD raw takes are on Apollo, not Leonardo**: `rclone lsd
  apollo:/apollo/users/sri/proj_adags/data` lists `imvid` (2026-08-26); no
  `imvid` directory exists under either Leonardo project tree (`LEO find`).
  The record says 15 `scene1_opera` + 6 `scene5_rendition` files verified
  ([[imvid-acquisition-quota-2026-08-24]]:12) and a `scene6_puppy` folder with
  39 MP4s ([[imvid-rig-classification-supersession-2026-08-25]]:204); the two
  puppy events are occlude-and-return, 6.7 s / 6.2 s, on a moving-rig take with
  uncorroborated calibration ([[imvid-event-census-result-and-closure-2026-08-25]] §1;
  rig page :128).
* **DEVA all-camera masks exist only for `cut_roasted_beef` (20 cams),
  `flame_steak` (21), `sear_steak` (21)**; the other three N3V scenes have one
  camera each (`LEO ls $W/repo/sa4d/data/dynerf/*/cam*/pseudo_label/object_mask`).
* **Panoptic Sports is already on Leonardo**: `/leonardo_work/EUHPC_D21_034/proj_adags/data/panopticsports/{basketball,boxes,football,juggle,softball,tennis}`,
  each with `ims/` (31 cameras × 150 frames), `seg/` (27 cameras),
  `init_pt_cld.npz`, `points3d.ply`, `train_meta.json`, `test_meta.json`;
  716 MB total (`LEO`). This is the Dynamic 3D Gaussians release
  ([[papers/luiten2023_dynamic_3d_gaussians]]). It is a 31-camera physical
  capture with per-camera masks that nobody has censused for
  exit-and-return; whether `seg/` is instance-level is **UNVERIFIED**.
* **CVM 2027** (Sol's find, verified): abstracts Oct 23, papers Oct 26 2026,
  "should not exceed 14 pages (including references)", journal-track (TVCG /
  JCST / CVMJ) or Springer proceedings, Singapore Apr 9–11 2027, topics include
  "Datasets and benchmarking of visual media" (https://iccvm.org/2027/callPaper.htm).

---

## 0. Change log — Sol's demands 1–19

| # | demand | disposition | reason (one line) |
|---|---|---|---|
| 1 | "authored counterfactual" everywhere, never "genuine" | **ACCEPT** | the record itself says teacher-rendered, `evidence_bearing: false` ([[absence-fixture-lane-2026-09-10]] :7–16) |
| 2 | re-score with independently adjudicated masks not used in construction or membership | **MODIFY** | evaluation ROIs already come from cam00 DEVA, a camera held out of SA4D training and never used in construction or membership (fixture §0) — but it is the same segmenter family; I add a second-segmenter, human-audited cam00 mask set (SAM 2 from a human click + manual audit on every 5th frame) and **re-profile the stored evaluator renders on CPU** — no retraining, ≈ 1–2 days. The paper reports both mask sets. |
| 3 | rename W6 "estimated-gate feasibility on the admitted subset" | **ACCEPT** | Claim B is DWP on both confirmatory scenes (verdict §0) |
| 4 | cut W8 from contributions | **ACCEPT** | moved to methods + supplement |
| 5 | promote W2 (real-occlusion null) to lead | **MODIFY** | promoted to the first figure and first contribution, but stated as one two-sided contribution (occlusion null + authored-absence positive); a null cannot carry a paper alone |
| 6 | frozen NOT_MET leads, verbatim paragraph | **ACCEPT** | Sol's paragraph adopted verbatim (§2.2) |
| 7 | 12 prefixes nested in 3 scenes; scene-clustered plots; no "12 independent" language | **ACCEPT** | prefix = training seed within a scene; scene is the generalisation unit |
| 8 | STG only with the unpaired/confounded qualifier; no numerical superiority in the abstract | **ACCEPT** | Sol's qualifier sentence adopted; numbers moved to the main text |
| 9 | cut free-boundary and product-gate runs as REQUIRED | **ACCEPT** | both become optional post-abstract items; the method is not W's headline |
| 10 | gates: A Sep 28, B Oct 5, C Oct 12 | **ACCEPT** | adopted verbatim with the concrete pass criteria in §5 |
| 11 | leak-free M2 (8 requirements; lift copies rows) | **ACCEPT** | spec in §3.4; 2–4 days; naive relabelling is banned |
| 12 | Cutie as the embarrassing external baseline under identical censoring | **ACCEPT** | verified runnable today; protocol in §3.5 |
| 13 | physical events are a CVPR prerequisite | **MODIFY** | accepted as the submission gate; what I add is a concrete, zero-acquisition path to certified events from data already on Leonardo (N3V all-camera census on existing DEVA masks; Panoptic Sports 31-camera scout), and a decisive rule: no ≥ 2 certified events by Oct 5 → C goes to ICCV 2027, no authored-only CVPR gamble (§3.6) |
| 14 | drop C3 (W-derived photometric) from C's headline and main table | **ACCEPT** | photometric windows on authored fixtures go to C's supplement only; this is also the largest single reduction of W/C overlap |
| 15 | no sketch M3 before the minimal mechanism passes | **ACCEPT** | M3 is deferred to after Gate B |
| 16 | "< 5 %" overlap unsupported; Sol estimates 15–30 % | **ACCEPT** the criticism, **MODIFY** the estimate | redrawn boundary with a per-section page accounting gives ≈ 10–13 % on C and ≈ 8 % on W (§4); the 25–30 % figure is what the round-1 boundary would have produced with C3 in C's main table |
| 17 | defer the full 8-page W arXiv until after Feb 25 | **REJECT in part** | arXiv is explicitly exempt from the CVF definition of publication (2026 guidelines; identical text at WACV/3DV 2027), so a Nov 17 posting creates no policy risk; the only real cost is reviewer perception of C, which the redrawn boundary addresses. Decision rule: full W to arXiv **on Nov 17 if C's headline is physical + re-ID** (boundary clean, C's reviewers benefit from a citable protocol); **hold** only if C is submitted with authored evidence in its main tables. If C goes to ICCV, W is unconstrained by CVPR and CVM 2027 (Oct 26) becomes the archival option. This is the principal's call (§7). |
| 18 | invert ordering: C's cheap killers first; W bounded to the 4-page abstract | **ACCEPT** | W's remaining work is CPU/human (re-scoring, text); it cannot block C's GPU or engineering lane, so the two run in parallel with C owning the engineer's first week |
| 19 | re-rank risks with authored ≠ physical at #1 | **ACCEPT** | §6 |

---

## 1. Executive decision (revised)

**Paper W** is a *diagnostic benchmark and finding* paper on the EL-GS
substrate: an authored-counterfactual full-multiview absence suite on three N3V
scenes with a frozen six-window event metric and sham/timing/vacuity controls;
the two-sided finding that a training-time presence gate is inert on a naturally
occurring occlusion (within 0.1 dB, 4/4 pairs) and helps on authored absences
(all 12 within-prefix G−U differences positive on ghost core and return, medians
+2.40 / +2.49 dB, under a frozen conjunction that nevertheless returned
NOT_MET); estimated-gate feasibility on the admitted subset; and the unpaired
STG observation, qualified. It goes out as a **4-page non-archival extended
abstract to NeurIPS 2026 PhysWorldAI by Oct 29** (decided Nov 9, outside every
CVPR rule), and as a full 8-page paper to a **CVPR 2027 workshop in ≈ March**,
with the full text on arXiv on Nov 17 if C's boundary is clean. **Paper C**
(persistent object memory on Multi4D: roots with an exact-zero presence notch,
disposable leaves, return-censored segmentation/re-ID headline, leak-free
episode-split control M2, Cutie/TRASE/SAM 2 external baselines) targets **CVPR
2027 only if three dated gates pass — Sep 28 (instruments), Oct 5 (mechanism +
M2 + a physical-event path), Oct 12 (paired confirmation + Cutie + ≥ 1 certified
physical event complete, second committed)** — and otherwise goes to **ICCV
2027 with 3–5 certified physical events**, which is the more probable outcome.
**C's cheap killers run first**; W's 4-page abstract is a bounded CPU/human lane.
The two papers do not collide because W's positive evidence is photometric on
authored fixtures with the EL-GS gate, C's is semantic/identity on Multi4D with
physical events in the main tables and authored fixtures only as supplementary
diagnostics; the redrawn boundary (§4) puts overlap at ≈ 10–13 % of C and
≈ 8 % of W, and no > 4-page W is submitted or accepted inside Nov 16 – Feb 22.

---

## 2. Paper W (revised)

### 2.1 Contributions (a reviewer's tick list)

| # | contribution | status | evidence |
|---|---|---|---|
| W-A | **Occlusion is not absence, measured two ways on one substrate.** (i) Natural occlusion (beef under hand/knife, frames 158–187, `cut_roasted_beef`): render-time deletion of the object's rows −3.39 dB in the box, −0.14 dB over 140–230, one fixed model; trained with the gate (3,017–3,234 rows bound), every paired contrast within 0.1 dB (G−U −0.074 gap / +0.050 return; floor 0.040/0.068 dB; 4 prefixes). (ii) Authored counterfactual absence (wine bottle removed on all cameras, frames 60–89): trained gate vs ungated positive on ghost core and return in every one of 12 within-prefix pairs across 3 scenes; render-time deletion on the same fixture −5.9 dB ghost / +3.2 dB return. | **REC** | [[realdata-gating-lane-2026-09-09]] §4.1; [[absence-fixture-lane-2026-09-10]] §7A, §5, §7B; `assets/absfix-stage2-report.md` |
| W-B | **The benchmark/protocol:** three authored-counterfactual absence scenes (SA4D hull-mode removal inside the segmenter silhouette, 1,720 rows, derived scene byte-identical outside the edit, hull-outlier control), six frozen windows (P1 [63,87] core, P2 [92,99], S1 [100,109], H1 [30,57], H2, C1 [230,259]), pooled-MSE window PSNR on the held-out camera, count-matched (A, B) and local-mass-matched (L) wrong-membership shams with zero truth overlap, timing shams (GMIS, GONES), per-cell mechanism preconditions, fail-closed reducer; released with manifests, masks, frozen windows, reducer and every failed admission | **REC**; **WRITE** the release | fixture §0–2; verdict §3; spec v2 §11, §13.21 |
| W-C | **Estimated-gate feasibility on the admitted subset:** blind visibility-gap estimator (training views only, zero false activations) recovers the authored window within one frame on 10 of 12 prefixes and fails visibly on 2 (flame s3 [60,93]; sear s2 [57,106]); SAM 2 one-click membership precision ≥ 0.9996 / recall 0.978–0.990 on 8 of 8 admitted prefixes, refused on all 4 sear prefixes (cam09 countertop mask); GESTMEM − G within 0.21 dB on the 8 admitted pairs; Claim B DESIGN_WITHOUT_POWER on both confirmatory scenes | **REC** | [[absfix-wave2-stage1-2026-09-13]] §1; verdict §0, §2, §3 |
| (discussion, not a contribution) | STG observation: "An unpaired published-schedule STG comparison suggests that temporal opacity makes the edited gap easy while weakening return; because representation, schedule, capacity and pairing differ, this is not a method ranking." (Sol's wording) | **REC** flame/sear; **RUN** crb ≈ 15–21 GPU-h | verdict §2.1 |
| (methods/supplement) | instrument practice: montage-before-table, vacuity preconditions, gate-off `--val` defect and repair | **WRITE** | fixture §6; gap_map updates |

Cut from the contribution list: W4 (folded into W-A), W5 (credibility infrastructure in results), W7 (discussion), W8 (supplement).

### 2.2 Claims wording (binding for every draft)

* Title: **"Occlusion Is Not Absence: An Authored-Counterfactual Absence Benchmark and a Presence-Gate Diagnostic for 4D Gaussian Splatting."**
* First abstract sentence describing the positive data: "authored counterfactual full-multiview absences on captured multi-view footage". "Genuine", "physical" and "discovery" never modify the positive result.
* Verdict paragraph, verbatim (Sol): "Under the frozen scene-wise conjunction requiring every G–U and G–sham pair to exceed 0.50 dB on both endpoints, Claim A was NOT_MET on both confirmatory scenes: one G–U ghost-core pair measured +0.44 dB, and six of 24 ghost-core membership-sham margins measured +0.23 to +0.43 dB. Post hoc, all 12 within-prefix G–U differences were positive on both the ghost-core and return endpoints, with medians of +2.40 and +2.49 dB. This directional consistency is descriptive and does not supersede the frozen verdict."
* Limitations sentence, verbatim (Sol): "Our evidence establishes behaviour on segmentation-guided counterfactual removals in captured multiview footage; it does not establish automatic discovery of physical scene exit or performance on naturally occurring absence-and-return events."
* Statistics: "12 training runs nested in 3 scenes"; every plot grouped by scene with scene medians; the pooled 97.5 % interval is described as a run-level conditional interval over these fixtures; the 1.59 dB paired sd and n₂ = 269 are reported as "what 12 runs cannot resolve (a 0.30 dB effect)", not as a power statement about the observed medians (Sol §"sd = 1.59").
* "Exact-zero presence" is the mechanism predicate; "zero-error frames" only for LRV3 (21/27) ([[paper-thesis-v1-2026-09-11]] §8.2).

### 2.3 Tables

**Table 1 (REC, re-profiled under demand 2).** Per scene × contrast × window, median (min..max) and sign count over 4 prefixes; contrasts G−U, G−A, G−B, G−L, G−GMIS, GONES−U, G−GEST, G−GESTMEM (admitted prefixes only, admission column); scene medians; **two mask columns**: DEVA-cam00 ROIs (frozen) and the independent human-audited SAM 2 ROIs (new). Source: `stage2_paired.json` for the frozen set; new reducer output for the second set.

**Table 2 (REC).** Occlusion vs absence, same substrate: four rows (natural occlusion render-time / trained; authored absence render-time / trained), columns gap, return, control, n.

**Table 3 (REC).** Estimators and admissions: T1 per prefix (window, onset/offset error, gated groups, false activations); S2 per prefix (precision, recall, IoU, admitted?); the sear refusal and flame s3 failure as rows, not footnotes.

**Table 4 (REC + RUN crb).** STG unpaired: P1/P2/H1/C1 per (scene, seed) with our U and G ranges; caption carries the qualifier verbatim.

**No ablation table is required.** Optional if slack after Oct 29: product gate (4 cells ≈ 10 GPU-h), free boundaries (4 cells ≈ 10 GPU-h), graded membership (12 cells ≈ 30–70 GPU-h) — all labelled post-verdict exploratory.

### 2.4 What must be run / written (W)

RUN (≈ 15–21 GPU-h of the 15,500 available; nothing on C's critical path):
1. STG on the `cut_roasted_beef` fixture, 2 seeds × 6 segment models (`agent-control/realdata/stg/{prep_derived.py,train_stg.sbatch,render_stg.sbatch}` exist, `LEO`) — the only required GPU item; its result is read before the abstract sentence is written.
2. Demand 2 re-scoring (CPU + human): human-click SAM 2 masks on cam00 for the three scenes over frames 30–110 and 230–259, audited every 5th frame; rebuild `absfix/evaluation_rois/{core,ring,object}` from them; re-run `scripts/event_region_frame_profile.py` on the **stored evaluator renders** of all 104 cells (renders are in the work area per spec §13.21) and the reducer with the same spec. ≈ 1–2 days, 0 GPU-h.
3. Optional (§2.3).

WRITE: the 4-page abstract (by Oct 22 for internal review; deadline Oct 29); the 8-page paper (Nov, after C is submitted or deferred); figures: F1 concept + real-occlusion montage (REC `assets/realdata-crb-gate-montage-2026-09-09.jpg`) as the first figure; F2 fixture construction/QA (REC assets); F3 per-arm scored-crop montages (REC, 12 tracked); F4 per-frame box PSNR curves with STG (WRITE from `f_box_profile.csv`); F5 scene-clustered strip plot (WRITE from `absfix-stage2-paired.json`); F6 T1/S2 admission figure (WRITE from `t1_viz` + stage-1 record); the release manifest.

### 2.5 Venue route

| step | date | what | rule check |
|---|---|---|---|
| 1 | by Oct 29 | 4-page extended abstract, PhysWorldAI non-archival track (window Sep 29–Oct 29; notification Nov 9; "3D/4D reconstruction … physically grounded view synthesis" in the CFP, https://physworld-org.github.io/physworld.github.io/cfp) | ≤ 4 pages excl. references → not a publication under the CVF definition; decided before Nov 16 |
| 2 | Nov 17 (conditional, §0 #17) | full 8-page W to arXiv | arXiv exempt ("arXiv preprints … are _not_ considered as publications", https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) |
| 3 | ≈ March 2027 | full W to a CVPR 2027 workshop (2026 had 4D Vision, non-archival, ≤ 8 pages excl. refs, and 4D World Models, archival; 2027 list unannounced) | after C's decision (Feb 25) → outside the review period |
| alt | Oct 26 | **only if C is deferred to ICCV at Gate B (Oct 5):** full W to CVM 2027 (14 pages incl. refs, journal-track archival, https://iccvm.org/2027/callPaper.htm) | then W is a publication before an ICCV C; the same < 20 % boundary (§4) applies and C's ICCV submission (≈ March) is not concurrent if CVM decides first (UNVERIFIED decision date) |

### 2.6 Acceptance odds (W)

PhysWorldAI 4-page: **60–70 %** (agree with Sol; the stretch is topical fit). Full W at a CVPR 2027 4D workshop with the §2.2 wording, the second mask set and the crb STG row: **55–65 %**; without the second mask set: 45–55 %. CVM 2027 full paper (if taken): 50–60 %, journal-track review is slower and graphics-leaning.

---

## 3. Paper C (revised)

### 3.1 Claims

* **C1 (headline).** Under return-censored supervision, persistent roots with a presence notch (M1) improve first-return segmentation and identity association over the shipped representation (M0) **and over a leak-free episode-split control with identical visibility and budget (M2)**, on ≥ 2 certified physical events and the three authored fixtures; paired direction positive in ≥ 5 of 6 (scene, seed) units per comparison.
* **C2 (mechanism).** ≥ +15 pp of return-window contribution mass comes from pre-gap roots or their descendants in M1 vs M0/M2, with a causal cohort ablation.
* **C3 (safety; and the principal's "beat the benchmark" line).** Standard Neu3D-Mask mIoU/mAcc on all five covered scenes, fail-closed: M1 ≥ M0 − 1 point; whole-frame PSNR/SSIM/LPIPS in Multi4D's conventions unchanged. If M1 beats M0 on the standard benchmark that is reported as a result, not assumed (v0.3 §3 predicts 0–2 points).
* Not in C's headline or main tables: photometric event-window PSNR on authored fixtures (supplement only, cited to W).

**On the principal's "novelty scare" remark.** It changes C in one concrete way: C must contain a table a reviewer recognises as *a benchmark* — Neu3D-Mask standard mIoU across the five scenes against Multi4D and TRASE (both public, both with numbers) — in addition to the censored-return protocol C introduces. "Improve the benchmark and beat it" is available on that table; it is *not* available on authored fixtures, where no external number exists. This is why the physical events and the standard table are both prerequisites, and why C3 is kept as a required table rather than a footnote.

### 3.2 Arms

M0 shipped Multi4D (censored semantic supervision) · **M1** persistent-root notch (presence multiplier at the five sites of [[multi4d-code-map-2026-09-19]] §5, including the mask render `:314`) · **M2** leak-free episode split (§3.4) · M4 return-pose residual **only** if the oracle root-landing test fails · M3 sketch loss **only after Gate B** (demand 15). Oracle diagnostic: notch placed at the known event.

### 3.3 Instruments that must exist before any number counts (WRITE, week 1)

1. `eval_seg.py` fail-closed: missing prediction = failure; shape mismatch = failure (no silent resize at `:42-43`); frame count asserted against the GT set.
2. `train_semantic.py` fail-closed on `Mask not found` (README admits the skip).
3. `--seed` in `safe_state` (`utils/general_utils.py:125-127`), never in `setup_seed`.
4. Fixture ingestion converter to Multi4D layout (`poses_bounds.npy` + `fused.ply` symlinked from the six available `_fused_ply/*.ply`, `camNN/images/` populated from the 6,300 edited frames) with **sha256 of ten edited frames asserted against the derived root before training** (the extractor skips on directory existence, repro §3.1).
5. Lineage column `root_id` / `parent_root_id` threaded through creation, all grow/shrink paths (code map §6c), **the FG→TR lift** (`select_fg_to_transient:412-481`), `capture()/restore()`, `save_ply/load_ply` (name must not begin `scale`, `rot`, `f_rest`), with unit tests asserting N-equality after every mutation and a checkpoint round trip.
6. Contribution logging per root/descendant via the existing `max_weight` path (`forward.cu:694-704`, code map §3.7) or a Python-side mask-weighted accumulation.
7. Mask-Benchmark: download the 595 MB zip on the login node; **open it and record the GT PNG resolution, camera and frame set for each Neu3D scene** (UNVERIFIED today); TRASE SAM masks generated for the three fixtures with TRASE's scripts (≈ 19 cams × 300 frames × 3 ≈ 17k images, ≈ 3–6 GPU-h, UNVERIFIED cost).
8. **Fixture GT for return-censored scoring.** Outside frames 60–89 every fixture frame is byte-identical to the original scene (fixture §2), so Mask-Benchmark's GT for the original scene is exact GT on pre-gap and return frames **if** it covers the held-out camera and those frames; inside [60,89] the target's GT mask is empty by construction of the edit. This resolves the orchestrator's issue (c) without new annotation, conditional on the coverage check in item 7 (**UNVERIFIED**). If Mask-Benchmark covers other cameras/frames, the fallback is the DEVA cam00 ids the fixture already carries (fixture §0) plus the human-audited SAM 2 set from W's demand-2 work (shared instrument, disclosed).

### 3.4 Leak-free M2 (Sol's eight requirements, made concrete)

Information boundary: **M1** — pre-gap and post-gap episode slots read the *same* root memory (`root_id`, semantic feature, canonical appearance, deformation state, opacity, presence parameters). **M2** — identical episode slots, identical presence schedule (frozen from the oracle window, so visibility is equal), identical total parameter budget, but the post-gap slot reads a **disjoint, freshly initialised root memory of identical dimension**; no semantic, canonical-appearance, opacity or optimiser state is copied from the pre-gap target.

Implementation on Multi4D (PyTorch only; the presence multiplication and gradient path sit before the rasteriser, code map §5):
1. Target root cohort selected once from the exact pre-gap GT mask (v0.3 §4.3); cohort ids frozen and hashed.
2. In M2, at the return boundary every pre-gap target root **and every descendant** (via `parent_root_id`, including TR rows created by the lift) is terminated or driven to exact-zero presence for all t ≥ return; assertion: their post-gap contribution mass = 0 (logged).
3. A fresh root cohort of identical count and dimension is instantiated at the return with fresh optimiser state; its `root_id`s and ancestors are asserted disjoint from the pre-gap cohort.
4. The FG→TR lift, split and clone propagate `parent_root_id`; a unit test mutates through lift/split/prune/capture/restore and asserts the ancestry sets stay disjoint.
5. Optimizer groups for the two cohorts are separate parameter groups; no `cat_tensors_to_optimizer` path merges their state (code map §6b).
6. A no-leak permutation test: swapping the identity association between two objects' cohorts must change the return identity result while leaving visibility and capacity unchanged; run once per scene before M2 is scored.
7. Shared-information declaration table in the paper: for every state tensor, M1 vs M2 "shared / disjoint".
Cost: 2–4 days of engineering (Sol) inside Sep 29–Oct 4; this is the single largest engineering risk after the lineage column.

### 3.5 External baselines under identical censoring

| baseline | protocol | status |
|---|---|---|
| **Cutie** (CVPR 2024, MIT, pretrained) | per held-out camera: last pre-gap GT mask as the only prompt, propagate through the gap and return with no post-gap prompt (`scripting_demo.py` pattern); score identical windows and swap rate | **RUN**, minutes per camera; build the evaluator in week 1 |
| **TRASE** (MIT, public) | its own feature field on the same reconstruction inputs, semantic supervision censored identically | **RUN** if its Docker builds on Leonardo (login node has network; compute nodes do not); cost UNVERIFIED |
| SAM 2 video propagation from the pre-gap mask | 2D VOS sanity row; `scripts/s2_sam2_propagate.py` exists | **RUN**, minutes |
| Multi4D M0 | primary | **REC** reconstruction; **RUN** semantic |

Reading rule, frozen before any score: if Cutie matches or beats M1 on first-return J&F **and** swap rate, C's headline is not licensed; C's residual value is then C2 (mechanism) + C3, which is not a CVPR paper.

### 3.6 The physical-event path (prerequisite), from data already on hand

Certification protocol (adapted from Sol): calibrated synchronised views; the object visible in ≥ 2 cameras before and after; effectively zero visible support in **every** camera during a sustained gap (≥ 30 frames), with a human-reviewed all-camera montage; independent pre/post identity association by a second annotator; at least one confusing distractor; exit/out-of-view and all-camera-occlusion events labelled separately.

Sources, in order of cost:
1. **N3V `flame_steak` and `sear_steak`: all-camera DEVA masks already exist (21 cams each)** — run the Charge-style mask-empty-interval census per DEVA id over all cameras (CPU, ≤ 1 day) and review candidates by montage. The crb curation ([[crb300-event-mask-curation-2026-08-23]]) asked about occlude-and-return on dynamic content; **no scene has been censused for an object leaving every frustum**. Prior: low-to-moderate (frontal rig; utensils and hands leave the volume).
2. **N3V `coffee_martini`, `cook_spinach`, `flame_salmon_1`**: DEVA on ~19 cameras each (≈ 2 jobs per scene as for crb, jobs 56842373/4; ≈ 6 GPU-h total), then the same census.
3. **Panoptic Sports** (31 cameras, 6 sequences × 150 frames, already on Leonardo): a 15-minute human scout of `cam00` for balls/objects leaving all views; if a candidate exists, SAM 2 masks on 31 cameras (≈ 1 GPU-h) and certification. Note the dome is enclosed and 150 frames is 5 s — low prior, zero cost.
4. **ImViD `scene6_puppy`** on Apollo: two occlude-and-return events (6.7 s, 6.2 s), 39 cameras, moving-rig take with uncorroborated calibration — a transfer of ≥ 39 MP4s (size UNVERIFIED) plus the reprojection gate before use; **not** an exit; only if 1–3 yield nothing and the lead authorises the transfer.
5. **BEHAVE**: 4 Kinects, ≈ 140 GB, MPI non-commercial licence — supporting evidence at most; not before Nov 16.

Rule: **≥ 2 certified events in ≥ 2 captured scenes by Oct 5 (one may be "committed" with certification finishing by Oct 12)** or C retargets ICCV 2027 at Gate B. If 1–3 yield nothing, the honest conclusion is that ICCV with a purpose-built capture or a BEHAVE-scale substrate is the right target, and no authored-only CVPR submission is made.

### 3.7 Tables and figures (C)

C1 return-censored segmentation (rows scene × arm × seed; columns first-return J&F, J&F at +1/+5/+10/+30, recovery latency, gap false-positive area, target–distractor swap rate, IDF1/HOTA where a distractor exists; physical events in the main table, authored fixtures in a second block) · C2 contribution retention (pre-gap roots / descendants / new roots; causal cohort ablation) · C3 standard Neu3D-Mask mIoU/mAcc on five scenes (M0, M1, TRASE published/rerun) + whole-frame metrics with conventions labelled · C4 external baselines (Cutie, SAM 2, TRASE) under identical censoring · C5 (supplement) photometric event windows on authored fixtures · ablations: notch on/off at fixed parameters (non-vacuity, v0.2 §5 P1), occlusion-aware render on/off, KNN smoothing, M3 after Gate B. Figures: F1 formulation; F2 protocol timeline; F3 qualitative return on physical events; F4 contribution-mass flow; F5 the Gate-A diagnostic renders (full / FG / TR / cohort-disabled).

### 3.8 Decisive recommendation: CVPR vs ICCV

**Plan for ICCV 2027 as the base case and let the gates promote C to CVPR.** Reasoning: (i) nothing is implemented and two fail-open instruments must be repaired before the first number exists; (ii) the physical-event supply from data on hand is unknown and has a low-to-moderate prior; (iii) authored-only C at CVPR is a 10–15 % submission that also burns the contribution's first impression; (iv) ICCV with 3–5 certified events, n = 6 seeds, Cutie/TRASE and the standard Neu3D-Mask table is a 40–50 % paper. If Gates A–C all pass, submit to CVPR — the same package is then ≈ 25–35 %, and an ICCV resubmission after a CVPR reject remains available.

### 3.9 Cost (GPU-h, 1 GPU-h = 8 local-h)

Week-1 killers ≈ 15 (M0 on 3 fixtures 8; DEVA on 3 scenes 6; SAM/Cutie < 2) · screen ≈ 20 (S0–S2 + semantics) · paired confirmation 2 scenes × 3 arms × 3 seeds = 18 recon ≈ 49 + semantics ≈ 3 h each ≈ 54 · scale 3–5 scenes × 3 arms × 6 seeds ≈ 54–90 recon ≈ 146–243 + semantics ≈ 160–270 · TRASE mask generation ≈ 5 · total **≈ 400–650 GPU-h**, ≈ 3–4 % of 15,500. Wall time, not GPU-h, is the constraint (2.7 h per reconstruction, 3 h per distillation, 120 G host RAM, no resume, single-GPU jobs disfavoured — walltime is the only lever, repro §8).

### 3.10 Acceptance odds (C)

CVPR 2027, authored only: **≤ 10 %** (I would not submit). CVPR 2027 with Gates A–C passed (leak-free M2, Cutie not erasing the effect, ≥ 2 certified physical events, standard-benchmark table): **25–35 %**. ICCV 2027 with 3–5 physical events and the mature package: **40–50 %**. Probability that Gates A–C all pass by Oct 12 given today's state: my estimate **20–30 %** (the physical-event supply is the binding unknown).

---

## 4. The W/C boundary (redrawn, with a defended overlap estimate)

Boundary rules: W = authored-counterfactual benchmark + EL-GS presence-gate diagnostic + real-occlusion null + photometric event windows + STG observation; **no Multi4D, no roots, no semantics, no re-ID.** C = persistent-root representation + physical events + return-censored segmentation/identity + M2 + Cutie/TRASE + standard Neu3D-Mask table; **authored fixtures only as supplementary diagnostics; no photometric event-window table in the main text; no EL-GS results; W cited for the fixtures and protocol in the third person.**

Overlap accounting (8-page papers; shared *content* counted where it appears in each paper, not literal prose):

| shared element | in W (pages) | in C (pages) | note |
|---|---|---|---|
| object-permanence / occlusion-vs-absence motivation | 0.5 | 0.3 | C states it in one paragraph and cites W and PersistGS |
| description of the three authored fixtures | 1.0 | 0.3 (supplement 0.5) | C describes them as a cited external dataset |
| event-window protocol | 0.5 | 0.1 (supplement 0.4) | C uses it only in the supplement |
| presence gate / notch as a device | 1.0 (EL-GS gate) | 0.4 (root notch) | different substrate, different role (rendering gate vs memory gate); counted as half-shared |
| STG / temporal-opacity observation | 0.3 | 0.1 | one sentence in C |
| related work on PersistGS, lifespan family | 0.4 | 0.4 | unavoidable; counted |
| **sum** | **≈ 0.65 of 8 counted at half weight for the gate → ≈ 0.6 page + shared related work 0.4 ≈ 1.0 of 8 = 8 %** (W) | **≈ 0.3+0.3+0.1+0.2+0.1+0.4 = 1.4 of 8 ≈ 13 %; ≈ 10 % if the fixture description is shortened to a citation** (C) | |

Why Sol's 25–30 % was right about round 1 and is not right about this boundary: round 1 put C3 (photometric windows on the authored fixtures) into C's main table and had no physical events, so C's positive evidence *was* W's positive evidence on a different substrate (≈ 1.5–2 pages of shared results). With C3 in the supplement and physical events in the main table, no result is shared. Residual risk: an area chair who reads "same three fixtures, same gate idea" as substantial similarity regardless of page counts; mitigations are the third-person citation of W, the physical-event main table, and the choice in §0 #17 to hold W's arXiv if C's main tables are authored-only.

Timing: the only > 4-page W submitted before Feb 22 is the optional CVM route, taken only if C is not at CVPR. The 4-page abstract is never a publication. The arXiv posting is exempt.

---

## 5. Dated timeline from 2026-09-21

| dates | C (owns the engineer and the GPU lane) | W (bounded CPU/human lane) | gate |
|---|---|---|---|
| Sep 21–23 | write the M1/M2 information-boundary spec and no-leak tests before any code (§3.4); fail-closed `eval_seg.py` + `train_semantic.py`; `--seed`; converter with frame-hash assertion; download Mask-Benchmark and record GT resolution/camera/frames; TRASE Docker on the login node | rewrite W's claims to §2.2; freeze the second-mask protocol; launch the N3V all-camera exit census on flame/sear DEVA masks (CPU) — shared with C's physical-event path | — |
| Sep 24–27 | M0 on the three fixtures (8 GPU-h) with full / FG-only / TR-only / cohort-disabled renders and per-cohort contribution mass over gap, first-return, settled-return frames; M0 semantic on crb (≈ 3 h) and `eval_seg` fail-closed vs the published number; DEVA on the other three N3V scenes (6 GPU-h) + census; Panoptic Sports scout; Cutie evaluator | STG-crb submitted (15–21 GPU-h); human-click SAM 2 masks on cam00 | **Gate A, Sep 28:** semantics reproduce fail-closed on crb; converter hashes pass; the target is represented in FG roots with measurable pre-gap contribution and incorrect gap contribution (Sol's criterion); logging works; ≥ 1 physical-event candidate has survived visual review. Fail on the mechanism criterion → Multi4D-C is DWP; decide Ex4DGS-only C for ICCV. |
| Sep 29–Oct 4 | implement M1 and leak-free M2 only; single-seed M0/M1/M2 on crb + oracle root-landing test; certify physical events found (second annotator, montages) | re-profile the 104 cells on the second mask set; 4-page abstract draft 1 | **Gate B, Oct 5:** M1 > M0 and M1 > M2 on return J&F; ≥ 15 pp root-contribution difference; no gap-FP increase; ≥ 2 certified physical events (or 1 certified + 1 committed). Fail → **C → ICCV 2027**; W may take the CVM route (lead's call, §7). |
| Oct 6–12 | paired M0/M1/M2, 2 scenes × 3 seeds (≈ 54 GPU-h incl. semantics); Cutie + SAM 2 under identical censoring; one physical event in the paired set; lineage/ancestry audit | abstract draft 2; STG-crb result read; abstract sentence written | **Gate C, Oct 12:** v0.3 §8 thresholds hold vs both M0 and M2; Cutie does not erase the advantage; ≥ 1 physical event complete and a second committed. Pass → CVPR. Fail → ICCV. |
| Oct 13–29 | if CVPR: scale to 3 authored + ≥ 2 physical scenes, n = 6 seeds where variance requires (≈ 160–270 GPU-h, short walltimes, warden, Slurm dependency chains across the 12-h certificate); standard Neu3D-Mask table on five scenes; TRASE row | **submit the 4-page abstract to PhysWorldAI by Oct 29** | — |
| Oct 30–Nov 16 | freeze results; write C + supplementary; leakage and failure audits; kill-argument; claim and citation audits; register Nov 10; **submit Nov 16 AoE**; supplementary Nov 23 | no W expansion | re-read the CVPR 2027 Author Guidelines when live |
| Nov 17–Feb 25 | (if ICCV: build 3–5 certified events; M3; failure atlas) | full 8-page W drafted from the PhysWorldAI feedback; arXiv Nov 17 if §0 #17's condition holds, else after Feb 25 | — |
| ≈ March 2027 | ICCV 2027 submission if deferred (deadline UNVERIFIED) | full W to a CVPR 2027 workshop | — |

---

## 6. Risks, re-ranked

1. **Authored ≠ physical (kill-argument P1).** Retired only by certified physical events; cheapest check: the flame/sear all-camera DEVA census (CPU, this week) and the 15-minute Panoptic Sports scout. If both are empty, C is an ICCV paper — decided at Gate B, not discovered in November.
2. **M2 leaks through the FG→TR copy, descendants, or optimiser state.** Check: the ancestry-disjointness unit test and the identity-swap permutation test before any M2 score (§3.4).
3. **Semantic stage does not reproduce / Mask-Benchmark GT resolution or camera set mismatches the renders.** Check: open the zip and assert resolution and frame set on day 1; the fail-closed evaluator refuses a resize.
4. **Fixture converter fails open** (extractor skips on directory existence). Check: sha256 of ten edited frames asserted before training.
5. **The persistent bank does not exercise the mechanism** (TR covers the gap; roots have nothing to remove or restore — the real-occlusion lesson). Check: Gate A's FG/TR/cohort-disabled renders (Sol: the full composite alone can produce a false negative).
6. **Cutie matches M1 on the censored return.** Check: run it in week 1 on M0's renders' GT — it does not need C's model to exist; if Cutie is already near-perfect on these events, the endpoint is not discriminative and C needs harder events (distractors, longer gaps) before anything is built.
7. **STG-crb changes W's story.** Check: it is the first W run; the abstract is written after it.
8. **Second-mask re-scoring moves the frozen verdict qualitatively** (e.g. a sham crosses G). Check: report both sets; the frozen set remains the verdict of record; a qualitative change becomes a stated result, not a problem.
9. **CVPR 2027 Author Guidelines differ from 2026** (4-page carve-out absent). Plan is robust: the only pre-Nov-16 W is ≤ 4 pages and the 20 % rule is met by §4.
10. **PhysWorldAI desk-rejects on topical fit.** Fallback: arXiv + March workshop; no fast outing.
11. **No 4D/dynamic-scene workshop at CVPR 2027.** Fallback: ICLR 2027 workshop (≤ 4 pages or < 20 %), Eurographics 2027 short (≤ 4 pages), CVM 2027 if C is not at CVPR.
12. **Wall-time**: 12-h certificate, start-up stalls, no resume on either substrate; every chain with Slurm dependencies and a warden.
13. **Disk**: 415 G free; 90 runs ≈ 65 GB; the approved removals should still run.

---

## 7. What the principal must decide personally

1. **Whether an authored-only C ever goes to CVPR.** This plan says no (Gate B); only the principal can overrule it, knowing the ≈ 10 % odds and the first-impression cost.
2. **The physical-event budget**: whether to authorise the ImViD puppy transfer from Apollo (size UNVERIFIED, moving rig, occlusion-class events) or a BEHAVE download under MPI's non-commercial licence, if the N3V/Panoptic scouts come back empty.
3. **W's full-text exposure**: arXiv on Nov 17 (my recommendation if C's boundary is clean) versus Sol's hold until after Feb 25; and whether CVM 2027 (Oct 26, 14 pages, graphics journal track) is an acceptable home for the full W if C is deferred.
4. **PhysWorldAI topical fit**: read the CFP and decide whether a counterfactual-absence benchmark belongs there; the alternative is no pre-November outing.
5. **Authorship and data licences for release**: the fixtures derive from N3V footage; Mask-Benchmark is CC-BY-NC; BEHAVE forbids redistribution — what W and C can release must be decided before the release manifest is written.
6. **The engineer's first week belongs to C** (§0 #18); the principal must accept that W's 8-page text waits until after Oct 5.

---

## 8. Questions for Sol's final round

1. Is the arXiv-on-Nov-17 rule (§0 #17) an acceptable resolution, given the exemption text, or is reviewer perception alone sufficient reason to hold the full W until Feb 25?
2. Does the redrawn boundary's 10–13 % (§4) survive your reading, and which shared element would you cut further?
3. Are the N3V all-camera exit census and the Panoptic Sports scout credible sources of "certified physical events" under your certification list, given frontal rigs and short takes? What is the minimum gap length you would accept?
4. Does the Mask-Benchmark-GT-on-unedited-frames argument (§3.3 item 8) hold, conditional on camera/frame coverage, or does the edit's teacher background inside [60,89] contaminate the return frames' GT?
5. Is the M2 spec in §3.4 complete? In particular: is driving the pre-gap cohort to exact-zero presence after the return a leak-free termination, or must the rows be deleted so that their deformation-field state cannot reach the fresh cohort through the shared HexPlane?
6. If Cutie is near-ceiling on the censored return (risk 6), what event property should be increased first — gap length, distractor count, or camera sparsity?
7. Is the standard Neu3D-Mask table (C3) enough to satisfy the principal's "beat the benchmark" criterion, or does C need a second public benchmark (Technicolor-Mask exists in Mask-Benchmark; Technicolor footage is not on hand)?
8. Given Gates A–C, is my 20–30 % estimate that C reaches CVPR too optimistic?

---

## 9. UNVERIFIED

- CVPR 2027 Author Guidelines (HTTP 404); 2027 page limit; whether the 4-page carve-out and 20 % rule reappear verbatim.
- CVPR 2027 workshop list and deadlines; ICCV 2027 deadline (aggregators ≈ Mar 5–7); CVM 2027 decision date and whether its journal-track review completes before March.
- Mask-Benchmark: actual GT PNG resolution, camera(s) and frame set per Neu3D scene (only the five scene folder names and the 595 MB zip are verified); whether "536×536" is only the parquet preview.
- Whether Panoptic Sports `seg/` masks are instance-level; whether any sequence contains an all-camera exit-and-return.
- Whether any N3V scene contains an object leaving every frustum (never censused).
- ImViD on Apollo: directory exists (`APOLLO rclone lsd`), file inventory not re-listed today; puppy take size; the two events are occlude-and-return, not exits; calibration uncorroborated.
- TRASE Docker build on Leonardo and the cost of its mask generation and training; TRASE's published Neu3D-Mask numbers per scene.
- Cost of Multi4D's semantic distillation on this cluster (README: ≈ 3 h).
- Whether Multi4D's persistent bank ghosts in the authored gap and whether the target is represented in FG roots (Gate A).
- Cutie's behaviour on objects that fully disappear (no explicit statement found in its README).
- PhysWorldAI's reviewers' topical tolerance; ICLR 2027 workshop archival status; Eurographics 2027 short-paper limit.
- SharpTimeGS venue (third-party only); FreeTimeGS++ repository trains on N3V (not checked).
- The GT hypothesis of §3.3 item 8 (unedited frames outside [60,89] → original-scene GT valid).
