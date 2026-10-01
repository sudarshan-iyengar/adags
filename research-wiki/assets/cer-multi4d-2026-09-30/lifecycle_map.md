# Multi4D at c483e82: render paths, row lifecycle and persistence, for the CER port (2026-09-30)

Read-only study by a subagent of a clean clone of Multi4D at c483e82; not yet independently re-verified (the CER
Multi4D spec re-reads every load-bearing line). `R` = `gaussian_renderer/__init__.py`, `GM` = `scene/gaussian_model.py`,
`T` = `train.py`. Config: `arguments/dynerf.py` over `arguments/__init__.py`: coarse 2,000 then fine 20,000; densify
from 500, interval 100, until 16,000; opacity prune from 500 every 600 at 0.005; opacity reset every 2,000; transient
merging 3,000; phase 3 from 10,000; row caps 90,000 (foreground), 960,000 (background), 1e6 (transient); lift every 50
(≤ 2,000 rows); transient weight prune every 200 after 6,000.

## Headline

From fine iteration 16,000 no density-control function runs (`T:944`, `:955`, `:965` all require `iteration < 16000`),
so no bank's row count, reset or per-row statistic changes; the last possible change is 15,900 (background and
transient densify); the foreground is frozen after 9,900. There is no optimizer step at 20,000 (`T:950`, `:960`, `:973`
require `iteration < 20000`), so the evaluated model is the one after step 19,999.

## 1. Where foreground and background opacity reach a rasterizer

| render | opacity sites | used when | used for |
|---|---|---|---|
| `render_background` `R:18` | `R:59`, `R:86`; dropout `R:87-94`; deformation in a bare `try/except` `R:66-82` | coarse and fine phase 2 (`T:542`, stage forced coarse, `prob_mask = 1 − fg`) ; `final_test_dump` (`utils/eval_utils.py:242`) | composite colour loss, background region losses, separation loss; background densify statistics and weight prune. **Not called in phase 3.** |
| `render_foreground` `R:121` | `R:162`, `R:198`; dropout `R:202-209` | phase 2 (`T:514`); `final_test_dump` (`eval_utils.py:235`) | composite, foreground region losses, separation; foreground densify statistics and weight prune; soft displacement-variance statistics |
| `render_mask` `R:248` | `R:290`, `R:314`; dropout `R:315-322` | phase 2 (`T:505`); `final_test_dump` (`eval_utils.py:246`) | hard foreground mask (`prob_mask` of both standalone renders), region weights, alpha loss (3,001–9,999) |
| `render_rawall3pc` `R:351` | foreground `R:445/468`, background `R:475`, concatenated `opacity_static` `R:486`; dropout `R:492-502`; to the rasterizer `R:520`; transient `R:419` | coarse + phase 2 (`T:547`); **phase 3's only render** (`T:450`); `run_evaluation` every 500 fine iterations incl. 20,000 (`eval_utils.py:68`, best at `:177`); `final_test_dump` (`:251`); the project's final-render wrapper | phase 3: colour L1+SSIM, anchor, diversity, depth TV, depth order, background densify statistics until 15,900 |

Other opacity reads: soft displacement-variance loss reads foreground `get_opacity` (`T:723`); `compute_transfer_state`
(`GM:361-410`, `no_grad`); the semantic scripts (`cluster_semantic_features.py:29,39`, `render_cluster_by_text.py:44,52`,
`train_semantic.py:395,405`) build opacity themselves and would bypass any gate.

For a CER start at ≥ 10,000, one multiply on `opacity_static` after `R:486` and before dropout at `R:492` gates training
and every evaluation render identically; the standalone sites `R:86`, `R:198`, `R:314` then feed only the debug panel.
The gate state must live on the model objects (the wrapper renders after `train()` returns).

## 2. Row lifecycle in the fine stage

- **Foreground** (`maintain_foreground`, only `iteration < 16000 and pre_phase3`, `T:944`): densify 600..9,900 (clone
  `GM:764-781`, split `GM:731-762`); weight prune 3,100..9,900 (`GM:483-494`); opacity prune 600..9,600 (`GM:126-129`);
  opacity reset of a random 50% at 2,000/4,000/6,000/8,000 (`GM:570-581`). Frozen after 9,900.
- **Background** (`maintain_background`, `iteration < 16000`, `T:955`): densify 600..15,900 (clone `GM:1198-1214`, split
  `GM:1170-1196`); opacity prune 600..15,600; opacity reset of a random 70% at 2,000..14,000 (`GM:925-936`); weight prune
  3,500..9,500 (`GM:1217-1228`). Frozen after 15,900.
- **Transient** (`T:965`): densify-and-prune 600..15,900; lift 3,050..9,950; weight prune 6,200..15,800.
- Nothing reorders rows. Upstream oddity: both reset functions start from `get_opacity` (a probability) and overwrite
  only the selected rows with a logit (`GM:572-579`, `930-934`), so unselected rows get σ(σ(ω)).
- Label threading would be needed only for a start before 16,000 (sites listed in the subagent's report: background
  `GM:1088`, `1106-1120`, `1122`, `1148-1168`, `1170-1214`, `1217`; foreground `GM:483`, `631`, `651`, `673`, `700`,
  `731`, `764`).

## 3. Transient bank and the lift

The lift reads foreground rows only (`select_fg_to_transient`, `GM:412-481`, raw `opacity_final` `:442`, +0.05 in
probability `:469-470`, donors kept), inactive after 9,950. Transient rows share the blend with the static set, so gating
changes them only through transmittance. Gating transient rows is possible with a python multiply at `R:419` (the CUDA
marginal then multiplies a 0).

## 4. Persistence and the final evaluation

Foreground `capture` 14-tuple (`GM:192-218`), background 12-tuple (`GM:862-885`), transient none; `torch.save` only at
`--checkpoint_iterations` (default empty). Point clouds `point_cloud.ply`, `point_cloud_second.ply`,
`point_cloud_transient.ply` via `Scene2gs_mixed.save` (`scene/__init__.py:104-128`) at 19,999 and 20,000 and to `best/`.
The score is rendered from the in-memory models after step 19,999.

## 5. Time

`scene/neural_3D_dataset_NDC.py:93-105`: `time = idx / 300`, `idx` the position in the sorted `camXX/images` listing, so
t = f/300. Cameras are rebuilt on each access (`scene/dataset.py:16, 34-41`), `image_path = .../camXX/images/YYYY.png`;
`image_name` is the frame for cam00 only. Exact gap test: parse the frame from `image_path`, or `round(cam.time * 300)`
on the python float.

## 6. Dropout

On at every training render (`T:452, 506, 517, 544, 549`), off at evaluation. For a factor in {0, 1} the gate and dropout
commute exactly; put the gate before dropout. Extra residual rows change the dropout draw count, so q = 0 identity is a
statement about the evaluation render only.

## 7. CUDA

No change needed. Hybrid rasterizer (JIT-built from `diff_gaussian_rasterization_hybrid/`): no static opacity cull in
preprocessing (`forward.cu:432-516`); per-pixel skip `forward.cu:651-654` and in the launched backward
`PerGaussianRenderCUDA` `backward.cu:1373-1375`; static opacity gradient returned as input slot 16
(`diff_gaussian_rasterization_hybrid.py:299`). The factor must be float32 cuda.

## Corrections to `multi4d-code-map-2026-09-19.md` (reported by the study)

§4's 90,000 / 960,000 / 1e6 are row caps, not iteration bounds; the iteration windows are those above. Not in the map:
the reset oddity, the absent step at 20,000, and the frozen window from 16,000.
