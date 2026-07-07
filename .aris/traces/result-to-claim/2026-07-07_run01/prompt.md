# Result-To-Claim Prompt

Intended claim under test:

A non-oracle Gaussian-model method can produce the event-crop fix suggested by R001-R017 on the five frozen real windows, without using oracle event-crop information at test time.

Experiment:

R025 event_candidate_refine, method M1 non-oracle residual-component local refinement. It uses non-oracle candidate supports from route0/dynamic/mask/flicker diagnostics, resumes route0 checkpoints, locally refines, and renders actual Gaussian output folders. It was evaluated on the frozen R009 windows using manifest `refine-logs/hide_reveal_poc/r025_event_candidate_refine_manifest.json`. Outputs are in `refine-logs/hide_reveal_poc/r025_event_candidate_refine_real_eval/`.

Predeclared PASS gate:

- checkpoint-backed/newly trained Gaussian output, not GT crop compositing
- no R009 frozen crops as test-time support
- at least 3/5 frozen windows improve versus route0, matched_lifespan, and residual_uncertainty on both PSNR and L1/proxy-LPIPS
- at least 3/5 windows do not worsen static ghost versus route0
- mean PSNR improves over route0 by at least +0.5 dB and mean L1 improves over route0 by at least -0.001
- recover at least 25% of oracle upper bound on either mean PSNR or mean L1 reduction

Key results:

- event_candidate_refine: mean PSNR 28.93926057568487, mean L1 0.018874982371926308, mean flicker 0.00847709346562624, mean static ghost 0.12565182149410248, n=5
- route0: mean PSNR 30.50211919273412, mean L1 0.014831560850143432, mean flicker 0.007990826107561588, mean static ghost 0.12733273804187775
- matched_lifespan: mean PSNR 29.818134359321654, mean L1 0.016354632563889027
- residual_uncertainty: mean PSNR 30.073395341209725, mean L1 0.01657234113663435
- oracle/derived hide_reveal upper bound: mean PSNR 41.714903733552326, mean L1 0.0026653554756194352

Computed deltas/gate counts:

- mean delta PSNR vs route0: -1.56285861704925 dB
- mean delta L1 vs route0: +0.004043421521782876
- mean delta flicker vs route0: +0.000486267358064652
- mean delta static ghost vs route0: -0.00168091654777527
- PSNR oracle fraction: -0.1393818468
- L1 oracle fraction: -0.3323486163
- windows improving vs all three baselines on both PSNR and L1: 0/5
- windows improving vs route0 on both PSNR and L1: 0/5
- windows with static ghost no worse than route0: 2/5

Integrity/context:

- R024 eval jobs completed ExitCode 0:0 and each eval folder has 300 renders/gt/static/dynamic frames under test/ours_6200.
- R025 scoring job 48805053 completed ExitCode 0:0 and wrote metrics/report/summary.
- Learned LPIPS and confident-track ID switches are unavailable; the protocol used L1/proxy-LPIPS as predeclared.
- Method appears to be checkpoint-backed Gaussian-rendered output; the quantitative gate is decisive.

Requested review fields:

1. claim_supported: yes | partial | no
2. what_results_support
3. what_results_dont_support
4. missing_evidence
5. suggested_claim_revision
6. next_experiments_needed
7. confidence: high | medium | low
