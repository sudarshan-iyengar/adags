# BLOCKING: rendered-flow supervision has no gradient (2026-08-17)

Status: **F and X cells CANNOT be run. The flow lane is stopped at a
scope boundary.** EXPLORATORY diagnostic; no training result.

The bounded N3V plumbing smoke (`scripts/flow_plumbing_smoke.py`,
experiments 128-131) was built to establish that the rendered-flow path
works BEFORE any supervision cell could be interpreted. It found that it
does not.

## What passes

| check | result |
|---|---|
| target units | PIXELS of their own raster |
| resize rescales magnitudes | YES — ratio 0.4998/0.4999 at half size, expected 0.5 |
| rendered vs target shapes at the loss | MATCH, so `compute_flow_loss`'s value-preserving interpolate is a no-op |
| mask coverage | 97.7-99.4% of pixels supervised |
| `training_setup` propagates `enable_rendered_flow` | YES (asserted; the smoke refuses otherwise) |
| rendered flow output is in the autograd graph | `requires_grad=True`, `grad_fn` present |
| projection chain is differentiable | YES — coefficients 1.675e-07, shared basis 5.845e-03 |

## What fails

```
rendered flow (untrained model)      EXACTLY zero
gradient on LoRA coefficients        0.000e+00
gradient on shared basis             0.000e+00

after perturbing coefficients to definitely-nonzero motion:
rendered flow absmax                 8.033e-04   (field DOES respond)
gradient on LoRA coefficients        0.000e+00
gradient on shared basis             0.000e+00
```

**VERDICT: severed at the rasterizer boundary.** The perturbation probe
exists precisely to separate a degenerate zero-init (harmless) from a
dead path (fatal), and it returns the fatal one: the field responds to
motion, so the renderer branch IS taken, and yet no gradient survives.

## Where the defect is — the Python is correct

`gaussian_renderer/diff_gaussian_rasterization.py` was read end to end.
The autograd Function saves `flow_2d`, `backward` accepts `grad_flow`,
passes it into `_C.rasterize_gaussians_backward`, receives `grad_flows`,
and returns it in the 5th slot, which matches `flow_2d`'s position in
the forward argument list. **The wrapper is wired correctly.**

`grad_flows` therefore comes back from the COMPILED CUDA EXTENSION as
zero. The defect is in the rasterizer's CUDA backward, which does not
populate the flow gradient. That code is not in this repository.

This is consistent with the historical record: `enable_rendered_flow`
and `lambda_track_flow` have NEVER been set together in any tracked
config or any recorded run ([[gap_map]] lists the hook as an open gap),
so a non-functional backward would never have been noticed.

## Why this stops the lane rather than being repaired here

Repairing it means modifying and recompiling the differentiable
rasterizer's CUDA sources. That is:

* NOT "target/loss semantics" — the targets are correct (pixels, correct
  resize, 99% mask coverage) and the loss is correct (shapes match,
  `compute_flow_loss` behaves);
* a change to the core rendering machinery, which is
  representation-critical and would need its own review;
* subject to the directive's own precondition that the disabled-flow
  path be proven NUMERICALLY UNCHANGED before experiments 104 and 123
  may still serve as controls — a proof that is much harder after
  touching the rasterizer than before.

**Running F and X in this state would compare two identical no-ops.**
Both cells would train exactly the model the control trains, and any
difference between them would be scheduler noise. The attribution gate
("if X matches F, treat as unattributable") would fire for a reason that
has nothing to do with flow semantics, and would waste the gate.

## A second finding: the direction test is not reliable on this scene

The RGB-warp direction check returned `minus` on two runs and `plus` on a
third, over different probe cameras. The cause is visible in the data:
N3V `cut_roasted_beef` flow has a median magnitude of **0.015-0.035
pixels**. At sub-0.1-pixel motion, warping by +F and by -F produce
almost the same image and the test is decided by noise, not by
convention.

So the sign convention of the SEA-RAFT sidecars relative to the
renderer's `screen(t+dt) - screen(t)` is **NOT established**, and this
page does not claim it. Any future direction test must restrict itself
to pixels whose flow magnitude is well above the noise floor. Recorded
because an unestablished sign is exactly the error that presents as
"flow supervision hurts".

## What is NOT concluded

That flow supervision would or would not help. Nothing about the idea
has been tested — only that the machinery cannot currently express it.
The N3V targets themselves are sound on every axis measured.
