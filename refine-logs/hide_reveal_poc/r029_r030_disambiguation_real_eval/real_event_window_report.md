# Real Event-Window PoC Report

Manifest: `refine-logs/hide_reveal_poc/r029_r030_disambiguation_manifest.json`

## System Summary

### event_boundary_micro_densify
- `n_windows`: 5
- `mean_psnr`: 30.5591
- `mean_l1`: 0.0147412
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0147412
- `mean_flicker`: 0.00801033
- `mean_static_ghost_score`: 0.125634

### hide_reveal
- `n_windows`: 5
- `mean_psnr`: 41.7149
- `mean_l1`: 0.00266536
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.00266536
- `mean_flicker`: 0.00168586
- `mean_static_ghost_score`: 0.127333

### matched_lifespan
- `n_windows`: 5
- `mean_psnr`: 29.8181
- `mean_l1`: 0.0163546
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0163546
- `mean_flicker`: 0.00795601
- `mean_static_ghost_score`: 0.127333

### oracle_crop_support_micro_densify
- `n_windows`: 5
- `mean_psnr`: 29.9021
- `mean_l1`: 0.015877
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.015877
- `mean_flicker`: 0.00839323
- `mean_static_ghost_score`: 0.123819

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

### route0_continue_6400
- `n_windows`: 5
- `mean_psnr`: 30.3532
- `mean_l1`: 0.0150603
- `mean_lpips`: n/a
- `mean_lpips_proxy_l1`: 0.0150603
- `mean_flicker`: 0.00810518
- `mean_static_ghost_score`: 0.128242

## Notes

- `lpips_proxy_l1` is not a learned LPIPS metric; use `--compute-lpips` when the LPIPS stack is available.
- Confident-track identity switches are not inferred here; attach them separately if available.
