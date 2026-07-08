# Raw Reviewer Response

claim_supported: no

The claim is not supported as stated. More precise: current evidence does not show that a non-oracle Gaussian method can recover the frozen R009 event-crop occlusion/reveal failures. This is not a proof that no future method can do it.

Evidence supports the oracle upper bound, method-form constraints for R026/R027, and weak directional R027 signal over route0. Evidence rules out R017 opacity gating, R025 event-candidate local refinement, and R027 boundary micro-densification as recovery methods under their tested recipes.

Plausible but untested: stronger identity/track-aware reveal matching, better support discovery, learned visibility state, true LPIPS, more windows, larger controlled local capacity, and oracle-support-but-Gaussian-only diagnostics.

Most likely failure modes: support missing/mislocalized reveal regions, local refinement optimizing wrong pixels or too weakly coupled to occlusion identity, opacity/densification removing or blurring content without synthesizing revealed surface, tiny R027 gain being ordinary continuation noise, and limited five-window scope.

Recommended compact experiments: posthoc support-overlap audit, oracle-support Gaussian-only diagnostic, matched route0 continuation control to `6400`, and a tiny one-scene budget sweep only after those diagnostics.

Confidence: high that current artifacts do not support the claim; medium about the broader scientific hypothesis.
