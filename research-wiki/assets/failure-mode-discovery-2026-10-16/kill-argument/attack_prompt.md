You are simulating a hostile CVPR / NeurIPS reviewer. This is a kill-argument adversarial check. Your task is NOT to
give a balanced review. It is to construct the **single strongest argument for rejecting** a paper built on the claim
and evidence below.

There is no LaTeX yet. The "paper" is the claim statement and evidence dossier in the workspace file
research-wiki/assets/failure-mode-discovery-2026-10-16/kill_dossier.md (read it in full). You may also read
research-wiki/operations/failure-mode-discovery-plan-2026-10-16.md (sections 0-16 hold the setting, hypotheses and
census) and the scorer scripts in research-wiki/assets/failure-mode-discovery-2026-10-16/. You have web search: use it
to check whether the failure is already known or solved (temporal opacity, Spacetime Gaussians, 4DGS/4D-Rotor,
FreeTimeGS, RetimeGS, SharpTimeGS, T-3DGS semi-transient distractors, Shape of Motion, MoSca, etc.).
Do not consult any prior reviews.

Focus on the axes most damaging for an empirical failure-analysis + benchmark + fix paper. Pick the most damaging
combination; do not list them all:
1. claim-vs-evidence gap (breadth of the headline vs single seeds, three scenes, authored real absences);
2. scope overclaim ("SOTA methods" vs the configurations actually run; self-inflicted configurations);
3. "known limitation / already solved" (per-primitive temporal opacity exists);
4. a metric or protocol artefact that could manufacture the effect (ring metric, held-out-camera extrapolation,
   monocular sweep construction, authored gap ground truth, masks);
5. a missing obligation (the fix is not built; no temporal-opacity method under the same protocol).

Constraints:
- About 200 words, at most 250. One argument, not a list.
- Cite specific dossier sections / table rows / file paths when accusing.
- Dispassionate but uncompromising. Do not hedge and do not acknowledge mitigations; the defense gets the next pass.

Output: just the rejection memo, nothing else.
