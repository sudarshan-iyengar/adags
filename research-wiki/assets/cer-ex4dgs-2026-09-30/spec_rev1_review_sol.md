## Verdict

MATERIAL DEFECT

Independent review; no ARIS skill used. I checked the pinned tracked sources at `7fac64997164fde64732c80261e56adf66ca14df`. The spec needs revision before implementation.

| Criterion | Score | Reason |
|---|---:|---|
| Mathematics and exactness | 6/10 | Opacity factors are coherent; gradient and base-equivalence claims exceed what they prove. |
| Attachment points and substrate facts | 7/10 | Main junction and tuple claims match; clipping, resume and PLY behavior are mishandled. |
| Tensor shapes | 7/10 | Birth tensors fit degree 3; the factor’s required column shape is unspecified. |
| Cohort lifecycle | 3/10 | Resume can bypass initialization or re-enable promotion. PLY loading remains fail-open. |
| Minimality | 6/10 | PLY persistence promises unnecessary training-state equivalence while missing required render metadata. |
| Surgical scope | 6/10 | Fixes need explicit resume/fallback guards beyond the stated changes. |
| Tests | 5/10 | Resume and missing-state failures are uncovered; PLY’s full-state criterion cannot pass as specified. |
| CUDA conclusion | 10/10 | The existing opacity input and backward return support Python gating; no CUDA change is required. |

## Confirmed defects

1. **Major — Resuming the prescribed 30,000 checkpoint silently skips CER.**  
   The spec saves that checkpoint before `begin`, then initializes only when `iteration == densify_until_iter`. Upstream restores `first_iter` and executes `first_iter += 1`: resuming `chkpnt30000` starts at 30,001. ([train.py:63–92](D:/adags-cer/ex4dgs/train.py:63))  
   Resume also resets sampling to `dataset.start_duration` and `need_extract = True`. Promotion at lines 257–282 is outside the densification cutoff.  
   **Fix:** either reject CER training resumes explicitly, or restore the sampling/lifecycle state and initialize CER before the first resumed render when crossing its start boundary. Add boundary-resume and active-CER-resume tests.

2. **Major — The contribution instrument measures different visibility from training and evaluation.**  
   Its `render(...)` call omits clipping arguments. The actual signature defaults to **`near=0.2, far=100.0`**, whereas the supplied config uses **4.0, 300.0**. Training explicitly supplies those configured values. ([renderer:19](D:/adags-cer/ex4dgs/gaussian_renderer/__init__.py:19), [train.py:138](D:/adags-cer/ex4dgs/train.py:138))  
   This affects membership, donors and every contribution gate. T4’s conservation checks could still pass because both sides use the same incorrectly clipped render.  
   **Fix:** pass the configured near/far through every instrument call and test that those settings match the real training/evaluation calls.

3. **Major — “Zero gradient to its parameters” and “equals unmodified Ex4DGS” need narrower qualifications.**  
   The opacity gate suppresses rendered contribution. It does not suppress the regularizers: upstream adds  
   `static_reg * ... _xyz_disp ... .mean()` and a motion penalty over `_xyz_motion`. Both coefficients are nonzero in the supplied config. ([train.py:155–161](D:/adags-cer/ex4dgs/train.py:155))  
   Moreover, upstream uses **RAdam**, not Adam. Zero current gradient does not eliminate updates from existing moments. ([model:449](D:/adags-cer/ex4dgs/scene/c_gaussian_model.py:449), [RAdam equations](https://docs.pytorch.org/docs/2.14/generated/torch.optim.RAdam.html))  
   Appending static residuals also changes the static regularizer’s mean denominator, including at `q=0`. Continued CER training therefore does not reproduce base training exactly.  
   **Fix:** state “zero rendering-loss gradient” and restrict byte identity to a fixed model whose original rows are unchanged. If actual parameter freezing or training identity is required, specify and test it separately.

4. **Major — The donor rule does not select the “largest gap-region contributors.”**  
   Section 5.2 selects by **fraction** `inside / total`, with no ranking by contribution magnitude. A row contributing \(10^{-9}\) entirely inside qualifies; a row contributing 100 inside and 101 outside is rejected.  
   **Fix:** either define birth as majority-region membership and acknowledge the method change, or specify a ranking/coverage rule using absolute inside mass.

5. **Major — T5’s PLY round trip cannot preserve “every tensor” using the proposed sidecar.**  
   `save_ply` writes primitive attributes; it omits optimizer state, accumulators, duration and other checkpoint state. ([model:513–547](D:/adags-cer/ex4dgs/scene/c_gaussian_model.py:513))  
   There is also a concrete shape mismatch: `load_ply` reconstructs keyframe count with **`+1+4`**, while `expand_duration` uses **`+1+2`**. ([model:603](D:/adags-cer/ex4dgs/scene/c_gaussian_model.py:603), [model:1254](D:/adags-cer/ex4dgs/scene/c_gaussian_model.py:1254)) Static positions depend on `t / self.duration`, so missing duration can affect images too.  
   **Fix:** keep full-state equivalence as a checkpoint test. For minimal scope, require checkpoints for CER and reject PLY fallback; otherwise persist necessary render metadata and test PLY only for render-state equivalence.

6. **Major — “Load `cer.pt` if present” still permits silent gate bypass.**  
   Section 5.4 explicitly makes sidecar loading conditional on presence. A CER PLY copied without `cer.pt` therefore loads its residuals as ordinary static rows—the exact failure the section claims to prevent. The fallback is reachable when the checkpoint is absent. ([scene:167–176](D:/adags-cer/ex4dgs/scene/__init__.py:167))  
   **Fix:** missing CER state must raise for a known CER model. Test missing sidecars, mismatched row counts, and stale sidecars when saving an inactive model.

7. **Major — Failed-gate scoring contradicts the stated proof rule.**  
   Section 6 says failed runs are “never dropped”; §9 says mark them failed **“instead of scoring”**. Withholding scores excludes planned repetitions from the comparison, including potentially every moved-return CER run. Training-camera-only gates do not resolve that contradiction.  
   **Fix:** score every planned run, retain its gate status, and separately mark whether it supports the CER mechanism claim. Do not condition the numerical comparison on gate passage.

## Suspected issues

- **Contribution tolerances may be too tight.** The backward kernel reconstructs transmittance with `T = T / (1.f - alpha)` and accumulates color gradients using float `atomicAdd`. ([backward.cu:592–636](D:/adags-cer/ex4dgs/submodules/diff_gaussian_rasterization_df/cuda_rasterizer/backward.cu:592)) Float64 accumulation afterward does not recover rounding already incurred. Repeated instrument runs and an independently calculable synthetic scene would settle T4’s tolerances.
- **The fixed pre-gap region may measure the wrong replacement content under occlusion or pass-through.** This requires inspection of the actual training-camera identity masks and donor contributions; those fixtures are unavailable here.

## Unverified claims

- The referenced handoff, Sol review, fixture lane, `ARS_SPEC_FORMAT.md`, launcher and scorer are absent at the referenced workspace paths. Their alleged thresholds, requirements and prior findings cannot be checked.
- Canonical scenes and checkpoints are absent. I could not verify mask alignment, cohort coverage, nine-scene eligibility, the 1.2 GPU-hour estimate, or the claimed 36/255 cross-GPU differences.
- T0–T7 were not executed; the available Python reports `ModuleNotFoundError: No module named 'torch'`.

## Missing

- Specify `get_cer_factor()` as **float32 CUDA `[N_s + N_d, 1]`**. A `[N]` factor multiplied by `[N,1]` broadcasts to `[N,N]`. ([PyTorch broadcasting rules](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html))
- Require per-bank label-length checks, positive gate denominators, and explicit empty-cohort handling. Total-length checks cannot detect a permutation or a count-preserving bank transfer.
- Resolve the deviation from the supplied core’s instruction to **disable pruning**. Sections 5.3 and 10 retain pruning while claiming a frozen lifecycle.
- Test gradients and optimizer steps, not only rendered mass. Check NaNs **before** upstream `prune_nan_points` removes them; it examines only `_xyz` and `_xyz_motion`, not every parameter. ([model:1229–1241](D:/adags-cer/ex4dgs/scene/c_gaussian_model.py:1229))
- Make T6 verify optimizer-state values under the keep masks, rather than lengths alone, and test loading a base checkpoint into an already CER-active instance.