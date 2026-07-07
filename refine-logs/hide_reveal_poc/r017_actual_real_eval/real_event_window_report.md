# Real Event-Window PoC Report

Manifest: `refine-logs/hide_reveal_poc/r017_actual_real_renders/actual_real_windows_manifest.json`

## System Summary

### actual_hide_reveal
- `n_windows`: 5
- `mean_psnr`: 19.3667
- `mean_l1`: 0.0761056
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0761056
- `mean_flicker`: 0.0162899
- `mean_static_ghost_score`: 0.152789

### matched_lifespan
- `n_windows`: 5
- `mean_psnr`: 29.8181
- `mean_l1`: 0.0163546
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0163546
- `mean_flicker`: 0.00795601
- `mean_static_ghost_score`: 0.127333

### residual_uncertainty
- `n_windows`: 5
- `mean_psnr`: 30.0734
- `mean_l1`: 0.0165723
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0165723
- `mean_flicker`: 0.00803902
- `mean_static_ghost_score`: 0.145702

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
