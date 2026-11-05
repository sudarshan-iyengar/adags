# Code review request: "CER on Multi4D, early gate" (round 5 — one test commit, focused)

GPU results on 96e92ee (bottle same-pose base run 1 as stand-in): existing T0-T6, E1-E4 PASS. E5 crashed on a test dtype bug.
E8 FAILED on the background bank only, deterministically (two fresh processes agree exactly): for a subset of 881 bg
witnesses the opacity gradient is nonzero (1e-14 .. 2e-7) but the DC-colour gradient is EXACTLY 0, both at q=0 in the gap
view and in the pre-gap view (e.g. bg rows 139, 147, 9848, 14541; static index = 10552 + row; transient row count ~85k).
All fg witnesses pass. Two explanations: (a) legitimate colour clamping (all three channels < 0 after +0.5); (b) the
inherited `clamped` buffer defect you flagged (rasterizer_impl.cu:160 sized for the transient count, written by the static
preprocess, forward.cu ~502 / ~62-64), i.e. aliasing with transient rows' clamp flags.

Commit 486b78a (D:\adags-cer\multi4d-early, test_cer_early.py only) fixes the E5 dtype and makes E8 separate (a) from (b):
a new preclamp() computes each witness's pre-clamp RGB as the hybrid rasterizer's static path would, excuses fully clamped
witnesses for the DC gradient only, and fails "spurious zero colour gradient" otherwise.

Review `git -C D:\adags-cer\multi4d-early diff 96e92ee..486b78a`. Most important: is preclamp() EXACTLY the colour the
hybrid rasterizer's static path computes and clamps (read diff_gaussian_rasterization_hybrid\cuda_rasterizer\forward.cu
and auxiliary.h: which function the static preprocess calls, its SH degree argument, the direction convention (pos -
campos vs campos - pos), the constants and signs, whether the static path uses the transient degree or its own, how
shs_static [:, :16] is laid out, and for foreground rows whether the deformed means/SH match render_rawall3pc), and does the
backward actually zero the DC gradient through `clamped` the way the test assumes? Also: is the per-row expectation right
when only SOME channels are clamped (the DC gradient of an unclamped channel should then be nonzero, so "any nonzero" is
the correct row-level test)? Any other defect in the commit.

Verdict line READY / READY WITH FIXES / MATERIAL DEFECT, then findings (severity, file:line, fix). Terse.
