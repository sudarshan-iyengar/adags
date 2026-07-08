# Raw Reviewer Response

Overall integrity verdict: WARN.

No evidence was found of fake ground truth, self-normalized scores, or phantom result files. The scorer loads separate render and GT folders and computes raw render-vs-GT PSNR/L1/flicker/static-ghost metrics.

Caveats: learned LPIPS was not exercised and proxy L1 is reported instead; scope is only five frozen windows; the bounded audit did not directly verify every unlisted manifest/frame file; R017 used frozen crop support at test time and is not non-oracle; R025/R027 include the oracle `hide_reveal` row as an upper-bound comparison, not a method claim.
