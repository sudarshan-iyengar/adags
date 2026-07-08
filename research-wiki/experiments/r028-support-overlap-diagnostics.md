---
id: experiment:r028-support-overlap-diagnostics
date: 2026-07-08
status: diagnostic_pass
related_idea: idea:event-causal-visibility-gaussians
---

# R028 Support Overlap Diagnostics

R028 is a posthoc diagnostic that reads the frozen R009 windows after support generation. It must not be used as test-time support or threshold tuning.

Artifacts:
- `refine-logs/hide_reveal_poc/r028_support_overlap_diagnostics/r020_candidates/support_overlap_report.md`
- `refine-logs/hide_reveal_poc/r028_support_overlap_diagnostics/r026_boundary_support/support_overlap_report.md`
- `scripts/audit_event_support_overlap.py`

Result: the R020 high-recall candidate boxes covered most of four frozen windows but missed `cut_roasted_beef_hand_tongs_meat_095_110` entirely. Mean support-frame fraction was `0.6375` and mean crop coverage was `0.491371`.

The R026 M2 boundary masks almost completely missed the frozen crop regions. Mean support-frame fraction was `0.0250` and mean crop coverage was `0.000000`.

Interpretation: R027 M2 failure should not be read as a clean rejection of "good support plus micro-densification." It rejects the concrete R026 support generator plus R027 400-iteration micro-densification recipe. R025 remains stronger evidence against the current local-refinement machinery because R020 support had much better overlap yet the method still degraded all windows.
