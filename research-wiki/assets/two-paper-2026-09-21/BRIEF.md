# Two-paper plan — brief (2026-09-21)

You are planning **two papers from one research programme**, to be reviewed
adversarially and revised. Precision standard: accepted papers at CVPR / ECCV /
NeurIPS and their workshops. The authors want to come out of this with **two
papers**, not one plan.

**Process.** You (Claude Fable 5.1) write round 1. It is then reviewed by
`gpt-5.6-sol` at `xhigh` effort, acting as an adversarial CVPR reviewer. You
revise once. It reviews once more. That is the whole budget — **two rounds each**
— so be complete the first time.

---

## 1. The two papers

**Paper W — a workshop paper or a smaller-conference paper, fast.**
Built by restructuring the existing ADAGS / EL-GS work, which has run for months
and produced a great deal of evidence but no paper. The lead's framing:

> "Like PersistGS, which if you think about it doesn't have too much novelty by
> itself and is just good story, good engineering. Restructuring that /
> incorporating some elements of that as well as a bit of novelty there can be
> good enough for a workshop paper. It needs to be something feasible with the
> datasets I have and can be put out quickly."

**Paper C — the CVPR 2027 main-conference paper.** Object permanence in 4D
Gaussian splatting, per the current plan v0.3 (below).

**The two must not cannibalise each other.** This is a hard requirement, not a
preference — see §6.

---

## 2. Access

**Local files.** Working directory `D:\adags`. Everything in
`research-wiki/` is the durable record. Read it; it is the evidence.

**The Leonardo cluster (read-only).** Two substrate codebases, all run outputs,
and the fixtures live there.

```
C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes siyengar@login.leonardo.cineca.it 'command'
```

From the Bash tool the path is `/c/Windows/System32/OpenSSH/ssh.exe`. **Do not
use Git Bash's own `ssh`** — it cannot reach the Windows ssh-agent and fails
with "Permission denied". Single quotes around the remote command are fine;
avoid inner double quotes and unpaired apostrophes. The credential is a CINECA
12-hour certificate, renewed today.

```
$W = /leonardo_work/EUHPC_D36_068/sri/proj_adags
$W/repo/adags        the ADAGS repository (same as D:\adags)
$W/repo/multi4d      Multi4D @ c483e82 (ECCV 2026) — reproduces 3/3
$W/repo/ex4dgs       Ex4DGS  @ 7fac649 (NeurIPS 2024) — reproduces partially
$W/runs/             every run output
$W/data_staging/     staged scenes
```

**The internet.** Use it — for venues, deadlines, dual-submission policies,
page limits, and for checking prior work. Verify; do not recall.

---

## 3. Hard rules

1. **Read-only on Leonardo.** Do not submit jobs, do not edit files there, do
   not run training or evaluation. Inspection only.
2. **Do not commit, stage or edit anything in `D:\adags`** except your one output
   file (§8).
3. **Every factual claim carries its source**: a file and line, a wiki page, or
   a URL. A number you cannot source does not go in.
4. **Distinguish three categories everywhere, explicitly**: (a) *on the record*
   — measured, with a page to cite; (b) *must be run* — with a cost estimate;
   (c) *must be written* — no compute. A plan that blurs these is not useful.
5. **Never invent a number, a venue, a deadline, a dataset, or a baseline
   result.** Mark anything unverified as unverified.
6. **Report cohort means, not best-of-n.** This project has been caught
   headlining a best-of-six before (see §5).

---

## 4. Where to read — in order

**The research history, compressed.**
* `research-wiki/query_pack.md` — the whole programme, newest first. Long; read
  the 2026-09-19, 09-15, 09-11/13, 09-10, 09-09, 08-24 and 08-23 blocks at
  minimum.
* `research-wiki/gap_map.md`

**The existing paper attempt — the natural core of Paper W.**
* `research-wiki/operations/paper-thesis-v1-2026-09-11.md` — thesis "Occlusion
  is not absence", four claims
* `research-wiki/operations/paper-thesis-v1-kill-argument-2026-09-11.md` — a
  cross-model kill-argument returned **FAIL, 4 unresolved** against it
* `research-wiki/operations/paper-plan-assessment-2026-09-14.md` — claim-by-claim
  evidence table, and the kill-argument's seven points and their state
* `research-wiki/operations/absfix-wave2-stage2-verdict-2026-09-15.md` — the most
  recent result: Claim A NOT_MET under the all-pairs 0.50 dB rule (return met
  8/8, ghost core short on one pair and six sham pairs), Claim B
  DESIGN_WITHOUT_POWER, calibration MET
* `research-wiki/operations/absence-fixture-lane-2026-09-10.md` — the first
  real-footage positive: +4.16 dB ghost, +2.86 dB return, 4/4 pairs
* `research-wiki/operations/literature-check-absence-return-2026-09-10.md`
* `research-wiki/operations/realdata-gating-lane-2026-09-09.md` — occlusion ≠
  absence at the row level
* `research-wiki/operations/elgs-method.md`
* `research-wiki/operations/project-status-2026-09-16.md`

**The substrate work — the natural core of Paper C.**
* `research-wiki/operations/ex4dgs-multi-interval-plan-v0.3-2026-09-21.md` —
  **the current Paper C plan**
* `research-wiki/operations/ex4dgs-multi-interval-review-round1-2026-09-21.md`
  and `...-round2-2026-09-21.md` — the two reviews that produced it
* `research-wiki/operations/multi4d-code-map-2026-09-19.md`,
  `ex4dgs-code-map-2026-09-19.md`
* `research-wiki/operations/multi4d-n3v-leonardo-reproduction-2026-09-18.md`,
  `ex4dgs-n3v-leonardo-reproduction-2026-09-19.md`
* Figures in `research-wiki/assets/` — SVG, readable as text.

---

## 5. Facts you must carry correctly

These have each been gotten wrong at least once in this project.

* **Reproduction gap.** Ex4DGS mean 32.6731 vs published 33.7329 = **1.06 dB**.
  The 0.59 dB figure that circulated was best-of-six and is selection.
* **Noise floors are fixed-seed and are lower bounds.** Multi4D 0.3445 dB
  spread (n=3); Ex4DGS sd 0.3564, range 0.9128 dB (n=6). Neither substrate
  exposes a `--seed`; Multi4D's visible `setup_seed(6666)` is overwritten to 0
  by `safe_state` 28 lines later.
* **The event union is ~0.34% of pixel-times**, so whole-frame PSNR moves by
  ~0.02 dB even for an infinite event-region gain. Whole-frame PSNR cannot be
  the endpoint of either paper.
* **STG** (per-Gaussian temporal opacity, no gate) renders the authored gap
  3.4–7.5 dB *better* than the gated arm but is 1.3–2.5 dB *worse* on the
  return. Absence is easy; **return** is the hard part.
* **The largest standing exposure is authored ≠ physical.** The fixtures are
  SA4D-edited removals. The kill-argument's point P1 was declared out of scope,
  not answered.
* **Multi4D's `eval_seg.py` is fail-open** (verified today): it silently skips
  missing predictions and computes background-dominated full-image accuracy.
* **Round-2 reviewer citations into upstream GitHub were wrong** — two of them
  pointed past the end of a ~520-line file. Its substance held. Re-locate any
  upstream line number in the pinned checkout before relying on it.

---

## 6. Things your plan must resolve, not just mention

1. **Venue and deadline for Paper W, verified.** Today is **2026-09-21**. CVPR
   2027 main papers are due **2026-11-16 AoE**. Find what Paper W can actually
   target: CVPR 2027 workshops, other late-2026 / early-2027 workshops, and
   smaller conferences (e.g. WACV, 3DV, BMVC — check which are still open).
   Give each candidate with its **verified deadline, page limit, and whether its
   proceedings are archival**, with the URL.

2. **Dual submission. This is the one that can sink both papers.** CVPR's
   policy treats some prior workshop publications as prior publication. Find the
   **actual current CVPR dual-submission / prior-publication rule** (URL), the
   page threshold at which a workshop paper counts, and whether non-archival
   workshops are exempt. Then draw an **explicit content boundary** between W and
   C: which claims, figures, experiments and datasets belong to which paper, and
   why the overlap is admissible. If the only safe answer is "W must be
   non-archival" or "W must be ≤ 4 pages", say so.

3. **What is Paper W's contribution, precisely?** The lead's model is PersistGS:
   good story, good engineering, modest novelty. State the contribution as a
   short list a reviewer could tick off, and for each item say whether the
   evidence is already on the record. Candidate material you should weigh (not
   an instruction to use it): the "occlusion is not absence" thesis; the
   authored-disappearance fixtures and event-window metric as a benchmark /
   protocol; the finding that training-time and render-time gating have opposite
   signs; the blind visibility-gap estimator T1; the SAM2 one-click membership
   result; the STG comparison showing absence is easy and return is hard; and the
   instrument lessons (vacuity preconditions, fail-closed evaluation).

4. **The kill-argument FAIL.** Paper W likely inherits `paper-thesis-v1`, which
   a cross-model review failed. Say which of its unresolved points W must
   answer, which it can scope out honestly, and which make W unpublishable
   unless fixed.

5. **Paper C's feasibility by 16 November.** Plan v0.3 proposes a two-week
   screen. Is the full Paper C achievable in the ~8 weeks remaining, given that
   Paper W also needs writing time? If not, say what the minimum publishable
   Paper C is, or whether C should target a later venue.

6. **Datasets actually available.** N3V, six scenes, on Leonardo. Authored
   absence fixtures on three N3V scenes. DiVa-360 partially. ImViD partially and
   rate-limited (~62 GiB, 5.7%). **Technicolor raw footage is NOT available** —
   only pretrained models. BEHAVE and Charge are **not** downloaded; if you rely
   on either, the acquisition is part of the plan and its cost and risk are
   yours to state.

7. **Precision.** For each paper: the exact table of main results with named
   rows and columns; the ablation table; baselines (which exist, which must be
   run, which have no released code); the figure list; and what a CVPR reviewer
   will attack first and the answer.

---

## 7. Compute and calendar

* ~124k GPU-hours remain; **compute is not the constraint**. A100 64GB.
  Ex4DGS 40k ≈ 1.2 GPU-h/run; Multi4D ≈ 2.7 GPU-h/run.
* Leonardo scheduling structurally disfavours 1-GPU jobs; short walltimes are
  the only lever. Queue waits are real — budget for them.
* **Calendar is the constraint**: ~8 weeks to CVPR, and two papers to write.

---

## 8. Output

Write **one file**: `D:\adags\refine-logs\two-paper-2026-09-21\fable_round1.md`.

Structure it as:

1. **Executive decision** — one paragraph: what Paper W is, what Paper C is,
   where each goes, and why they do not collide.
2. **Venue analysis** — verified table (§6.1) and the dual-submission rule with
   URL (§6.2).
3. **Paper W** — title; abstract draft; contribution list with evidence status;
   main-results table; ablations; baselines; figures; the kill-argument points
   and their disposition; what must be run (with cost), what must be written;
   reviewer attacks and answers.
4. **Paper C** — the same, against plan v0.3, with a hard-nosed feasibility
   verdict for 16 November.
5. **The W/C boundary** — claim by claim, figure by figure, dataset by dataset.
6. **Timeline** — week by week to 16 November, both papers, with go/no-go gates.
7. **Open risks** — ranked, each with the cheapest check that retires it.
8. **Questions for the reviewer** — the specific things you want `gpt-5.6-sol`
   to attack.

Be scientifically thorough. The goal is two papers that survive review — a CVPR
workshop review for W and a CVPR main-track review for C.
