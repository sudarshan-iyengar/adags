# DEFECT — the temporal marginal is applied TWICE, so every 4D support width renders at 1/sqrt(2) of its stored value (2026-08-26)

Status: **CONFIRMED BY DIRECT SOURCE INSPECTION. NOT REPAIRED.** Found
incidentally during the ImViD paper-parity lane
([[imvid-paper-parity-freeze-2026-08-26]]) by a fresh-context adversarial
review of an unrelated change, and verified independently before being
recorded.

This is **pre-existing** and has nothing to do with that lane's changes. It
is written here rather than inside that lane's records because its scope is
every `gaussian_dim: 4` result this project has produced since the renderer
change that introduced it.

## 1. The two applications

**Python**, `gaussian_renderer/__init__.py`, in the `compute_cov3D_python:
False` branch — the branch every ADAGS config uses:

```python
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            ...
            marginal_t = _temporal_multiplier()
            opacity = opacity * marginal_t
```

`opacity`, `ts` and `scales_t` are then ALL handed to the rasterizer with
`cov3D_precomp = None`.

**CUDA**, `diff-gaussian-rasterization/cuda_rasterizer/forward.cu`, in the
matching non-precomputed-covariance branch:

```c
		if (gaussian_dim == 4){  // no rot_4d
            float dt = ts[idx]-timestamp;
            float sigma = scales_t[idx] * scale_modifier;
		    float marginal_t = __expf(-0.5*dt*dt/sigma);
		    if (marginal_t <= opa_threshold) return;
		    opacity *= marginal_t;
		}
```

Both compute the same quantity from the same inputs and both multiply.

## 2. The consequence, exactly

The intended marginal is `exp(-0.5 dt^2 / sigma)`. What renders is its
square, `exp(-dt^2 / sigma)` — a Gaussian with effective variance
`sigma / 2`. Since a temporal standard deviation is `sqrt(sigma)`:

> **rendered temporal std = stored temporal std / sqrt(2) ~= 0.7071 x**

For the ImViD FG arm's declared bands:

| band | stored std (s) | actually rendered std (s) |
|---|---:|---:|
| compact | 0.133467 | **0.094375** |
| default / abstain | 0.999415 | **0.706693** |
| broad | 2.494158 | **1.763636** |

A second, independent consequence: `forward.cu` culls on
`marginal_t <= opa_threshold` using the SQUARED marginal, so the temporal
support is culled earlier than the stored `sigma` implies.

## 3. What is and is not affected

**Not affected — ratios and comparisons.** The factor is a single global
constant applied identically to every primitive in every arm. Any comparison
between two runs of this trainer — including the ImViD NF/FG pair — is
unaffected, and the uniform-default initialization is unaffected in the sense
that it renders however it has always rendered.

**Affected — every absolute statement about temporal support width.** Any
sentence of the form "this primitive is supported over ~0.13 s" is wrong by
41%. Any parameter chosen to produce a particular support width produced a
narrower one. Any comparison against an EXTERNAL method's stated temporal
extent (for example the ImViD paper's `3e-2` temporal-extent learning rate,
or a published support duration) is comparing incommensurable quantities.

**Unknown and not investigated here:** whether the double application was
deliberate, and whether the trained models have simply absorbed the factor
into the learned `_scaling_t` (they can: `_scaling_t` is a free parameter, so
optimization would move it to compensate, and only the INITIAL widths and any
frozen/authored widths would be off). That question matters for interpreting
historical results and is left open rather than guessed.

## 4. Why it is not repaired here

Repairing it would change the rendered output of every `gaussian_dim: 4`
configuration in the repository, silently invalidating comparisons against
every previously recorded number. That is a decision about the whole project,
not a side-effect of an ImViD baseline lane, and this project's rules are
explicit that adjacent code is not "improved" in passing.

**What was done instead:** the ImViD lane's own documentation now states
rendered widths beside declared ones, and
`tests/test_imvid_point_cloud_time_extent.py` carries a test that asserts
BOTH multiplications are present — so if either is ever removed, that test
goes red and this page is the reason why.

## 5. Provenance of the finding

Proposed by a fresh-context adversarial reviewer with no stake in the code,
which is precisely the review posture that catches a defect nobody was
looking for. **Verified independently before acceptance** by reading both
call sites directly rather than taking the report at face value — the report
also correctly identified that the upstream code it was derived from applied
the marginal only in the `compute_cov3D_python` branch.
