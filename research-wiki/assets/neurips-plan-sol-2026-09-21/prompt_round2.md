# Round 2 — follow-up questions from the principal

Thank you. The principal read your round-1 answer and has follow-up questions,
relayed faithfully below. Same rules as before (read-only everywhere, no jobs, no
file edits; verify papers and datasets by fetching them; mark anything unchecked
UNVERIFIED). Use any ARIS skills you want (`C:\Users\sucar\.claude\skills\`).

## The principal's questions, in their words (lightly cleaned)

1. **"The question about datasets still remains, right? Ideally I need to run it on
   all the different datasets and still show improvement, right?"**
2. **"The way I do the gap and return can't just be one thing where it disappears
   and appears. I need to find various ways of doing it — like the object moving,
   and so on."**
3. **"Perhaps synthetic scenes are the right place to start and compare my method
   against other methods?"**

What I need from you on each:

**Q1 — datasets.** Which datasets should this paper evaluate on, concretely, and
what does "show improvement" mean on each? Consider at least: N3V/DyNeRF (the
current fixtures are three of its scenes; the other three are coffee_martini,
cook_spinach, flame_salmon_1), Technicolor (footage NOT on hand — only Ex4DGS
pretrained models locally), Panoptic Sports (on Leonardo: 6 sequences × 31
cameras × 150 frames, `/leonardo_work/EUHPC_D21_034/proj_adags/data/panopticsports`),
ImViD (on Apollo, not Leonardo; rate-limited acquisition), DiVa-360 (census found
0/597 true absences), and any synthetic multiview dynamic dataset (D-NeRF,
Kubric/MOVi, PersistGS's scenes if released, CMU/Blender renders, etc.). For each:
availability (verify), whether it can host authored or natural absence-and-return,
whether published baselines have numbers on it, and cost. Distinguish (a) where the
mechanism must show a gain (event-window endpoints), from (b) where it only needs to
show no harm on standard whole-scene metrics — and whether running "all datasets"
is actually required for an 8-page workshop paper done by Oct 15, or whether a
smaller, well-chosen set is stronger. Be explicit about what a reviewer would
demand versus what is nice to have.

**Q2 — variety of absence/return events.** Design a taxonomy of gap-and-return
event types that a benchmark in this area should cover, e.g.: same pose return;
return at a moved position/orientation (object relocated during the gap); moving
object that exits and re-enters; changed appearance/lighting on return; multiple
gaps; different gap lengths; partial-camera vs full-multiview absence; occlusion vs
absence (our real-occlusion null); distractor/look-alike objects on return;
non-rigid objects; objects removed while something else moves in. For each type:
(i) can it be AUTHORED on real N3V footage with our existing SA4D-based fixture
pipeline (read `research-wiki/operations/absence-fixture-lane-2026-09-10.md` §0–2,
and `scripts/sa4d_absence_edit_render.py`, `scripts/build_absence_fixture_scene.py`
to judge what the editor can do — e.g. can it re-insert the object at a moved pose,
or only delete?), (ii) can it be generated SYNTHETICALLY, (iii) does it change which
mechanism is needed (the Complementary Episode Residuals idea assumes the object
returns unchanged; a moved return needs the return-pose / loop-closure component you
ranked low — does that change your ranking?), and (iv) cost. Then recommend a
minimal set of event types for Oct 15 that a reviewer would accept as "not a single
toy case".

**Q3 — synthetic first?** Should the method be developed and compared on synthetic
scenes first (exact ground truth for the gap, controllable event types, free choice
of object/pose/timing, many seeds, PersistGS-style comparability), with real N3V
authored fixtures second — or the reverse? Consider: this project already has a
synthetic fixture history (LRV1–LRV4 in `research-wiki/query_pack.md`, and
`scripts/eval_lrv1_event.py`), what rendering stack could produce multiview
synthetic dynamic scenes fast (Blender/Kubric/BlenderProc; check what is installed
or installable on Leonardo/workstation without heavy downloads), whether STG,
Ex4DGS and Multi4D can ingest such data with little converter work, and whether
reviewers at a NeurIPS "Physical World AI" workshop would value synthetic control
or see it as a weakness. PersistGS used only synthetic scenes and was accepted at a
CVPR workshop — check exactly what it released (code/data) and whether its scenes
could be reused as a direct comparison (that would let us report against PersistGS
itself).

## Two points from Claude (the orchestrator) to fold in

* Your STG compute estimate (60–84 GPU-h for crb, "four prefixes × six segments")
  looks wrong: STG on flame/sear ran as 2 seeds × 6 fifty-frame segments (63 jobs
  total incl. renders, `research-wiki/operations/absfix-wave2-stage2-verdict-2026-09-15.md`
  §2.1, §4), so crb should be ~15–21 GPU-h. Confirm or correct.
* Your Gate-A kill rule ("close ≥50% of the base-to-STG P1 deficit") may fail for
  reasons unrelated to the mechanism: STG trains separate 50-frame models at 25k
  iterations (more capacity per frame), and its control window C1 is already
  0.4–1.2 dB above our EL-GS arms. Claude proposes gating against unmodified Ex4DGS
  on the same fixture and treating STG-gap parity as a hoped-for outcome, not a kill
  condition. Agree or disagree, with reasons.

## Output

1. Answers to Q1, Q2, Q3, each ending in a clear recommendation.
2. Your response to Claude's two points.
3. A **revised dated plan Sep 22 – Oct 15** incorporating the answers (datasets,
   event types, synthetic/real order, the method, baselines, gates, kill rules,
   compute, wall-clock risks), replacing round 1's §6 where they differ.
4. Updated mock-review score for the revised plan, and what could still sink it.
5. UNVERIFIED list.
