# DyCheck iPhone runs: Shape of Motion and MoSca (fmd mono2, 2026-10-19)

On SCRATCH, because the work area (/leonardo_work/EUHPC_D36_068) was at 100.7 % of quota on 2026-10-19 15:15:
  X = /leonardo_scratch/large/userexternal/siyengar/fmd
Untouched public data, each method's own released processed copy and published iPhone protocol; code unmodified.
Held-out cameras: DyCheck's two static cameras, id 1 and id 2. Frame names are `<camid>_<time:05d>`.

## Shape of Motion (5 scenes with held-out cameras: apple, block, paper-windmill, spin, teddy; full resolution)
- renders: X/runs/som_iphone/<scene>/results/rgb/{1,2}_NNNNN.png   (its own validator, final epoch)
- GT:      X/data/dycheck/som/scenes/<scene>/rgb/1x/{1,2}_NNNNN.png  (RGBA; use RGB)
- co-visibility mask: X/data/dycheck/som/scenes/<scene>/flow3d_preprocessed/covisible/1x/val/{1,2}_NNNNN.png
- official metrics: X/runs/som_iphone/<scene>/eval_iphone.txt (scripts/evaluate_iphone.py)
- errmap.py: --gt X/data/dycheck/som/scenes/<scene>/rgb/1x/1_{f}.png --method som=X/runs/som_iphone/<scene>/results/rgb/1_{f}.png
  (f = 5-digit time; only times listed in splits/val.json exist)

## MoSca (7 scenes: apple, block, paper-windmill, space-out, spin, teddy, wheel; DyCheck 2x resolution 360x480)
- renders (published protocol, pose-only test-time alignment): X/runs/mosca_iphone/<scene>/logs/iphone_fit_*/tto_test/{1,2}_NNNNN.png
- GT:      X/runs/mosca_iphone/<scene>/test_images/{1,2}_NNNNN.png
- co-visibility mask: X/runs/mosca_iphone/<scene>/test_covisible/{1,2}_NNNNN.png
- official metrics: MoSca's eval_dycheck output files in the same logdir (*dycheck*), and the job log X/logs/fmd_mono2_moscaiph_<scene>_*.out
- errmap.py: --gt X/runs/mosca_iphone/<scene>/test_images/1_{f}.png --method mosca=<logdir>/tto_test/1_{f}.png

Provenance: X/runs/{som_iphone,mosca_iphone}/<scene>/provenance.txt; launchers X/agent/fmd_mono2_{som,mosca}_iphone.sbatch.
