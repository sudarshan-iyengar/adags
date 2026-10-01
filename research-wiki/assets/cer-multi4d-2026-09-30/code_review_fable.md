# Fable 5.1 review of the CER Multi4D diff (c483e82..c1e6045), 2026-10-01

Read-only subagent review; condensed transcription.

| # | dimension | score | short reason |
|---|---|---|---|
| 1 | gate correctness | 8 | one `apply` per site after activation, before dropout, per bank before the hybrid concatenation; fail-closed arming; strict frame parse; sentinels on every optimizer-surgery choke point. Deductions: three lifecycle names missing from the list (unreachable), bare asserts, wrapper unverified, `train_views` decodes every camera image (wall clock) |
| 2 | CUDA change | 9 | right quantity (the early exit becomes lossless), sync-safe, index-consistent, training build textually upstream; correctness witnessed indirectly by the per-render conservation assertion |
| 3 | instrument swap | 8 | `_C` resolved at call time in forward (`:161`) and backward (`:281`); `finally` restores; separate `PyInit` symbols; Python loads extensions `RTLD_LOCAL`; a JIT build inside a training job is a new failure mode |
| 4 | shapes / dtypes | 9 | factor `[N,1]` float32 cuda, labels uint8, background append of six groups plus `max_radii2D [D]`, `xyz_gradient_accum [D,1]`, `denom [D,1]` |
| 5 | minimality / scope | 8 | 13 CUDA lines under a macro; spec §8 and §3 stale |
| 6 | tests | 6 | **T4 cannot pass** (`len()` of a generator); T2 hard-codes frame 74; T3's foreground witness chosen by opacity can be vacuous; T4b has no negative control; the 30-minute test job is at risk |

**CUDA verified in detail:** `last_contributor` is the 1-based position of the pixel's last blended splat; the backward
keeps splat `s` iff `s < n_contrib[pix]`; bucket `b` is useful iff `32b < max_pix n_contrib`, exactly the negation of the
early exit at `backward.cu:1229`; each bucket restarts from its own sampled transmittance, so skipping buckets past the
maximum changes no gradient. `tile_id` is `y·horizontal_blocks + x` in both kernels; `bucket_to_tile` from the same scan;
`max_contrib` has `W·H ≥` tile-count entries, written for every tile (empty ones 0); outside-image threads excluded; no
`return` in `renderCUDA` and the batch loop's `break` is collective, so every thread reaches both syncs. Without the
macro the training build is the upstream kernel; T1's 300-frame byte identity confirms the forward binary.

**Minor findings:** T2 frame constant; T3 witness by rendered contribution; `LIFECYCLE` add
`reset_opacity_partially_small`, `add_points`, `densification_postfix_nodenom` and match the T6 regex; `train_views`
decode cost; pre-build the instrument at t = 0 in the launcher; a descriptive training-build comparison in T4b; a
zero-mass guard in `begin`'s report; bare asserts vanish under `python -O` (not used); provenance should hash
`forward.cu`; spec §8 needs a rev-3 note.

**Verdict:** substrate and instrument code ready for the principal's review; the test suite not, until T4 is fixed.

All findings were applied in 531c27b (tests) and 7d8c5e2 (code), and in the launcher, except: the bare asserts are kept
(both launchers run plain `python`), and `train_views`' decode cost is accepted (the tests job now runs on the normal
queue with a 2-hour limit).
