# Real Event-Window PoC Report

Manifest: `refine-logs/hide_reveal_real_windows.json`

## System Summary

### route0
- `n_windows`: 5
- `mean_psnr`: 30.5021
- `mean_l1`: 0.0148316
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0148316
- `mean_flicker`: 0.00799083
- `mean_static_ghost_score`: 0.127333

## Notes

- `lpips_proxy_l1` is not a learned LPIPS metric; use `--compute-lpips` when the LPIPS stack is available.
- Confident-track identity switches are not inferred here; attach them separately if available.
