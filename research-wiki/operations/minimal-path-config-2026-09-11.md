---
title: Minimal path for a wave-2 counterfactual-absence cell — configs, flags that must stay off, and the two traps
date: 2026-09-11
evidence_bearing: false
---

# Minimal path — wave-2 counterfactual-absence lane

Configuration note only; no result and no scientific claim. It records
what a wave-2 cell of the counterfactual-absence lane
([[absence-fixture-lane-2026-09-10]]) actually runs, which flag-gated
mechanisms must stay off, and the two traps that have already cost
cells. Wave 1 was six arms × four prefixes: each arm is a continuation
to 12,000 iterations of that prefix's shared ungated 6k checkpoint on
the SA4D-edited (derived) scene.

## 1. The prefix, and the six arms

Prefix (four seeds, un-gated, on the derived scene):
`configs/n3v/b0c_crb300_6k_rp_prefix.yaml` — `b0c_crb300_12k_rp.yaml`
with `iterations: 6_000` (§4 of the lane page). Every continuation is
`--start_checkpoint <prefix>/chkpnt6000.pth` with `iterations: 12_000`
and the same `--source_path` (the derived scene).

| arm | YAML | difference from the plain B0-C baseline (`configs/n3v/b0c_crb300_12k.yaml`) |
|---|---|---|
| U | `configs/n3v/b0c_crb300_12k_rp.yaml` | `elgs_reserved_parity: true` — one line, nothing else |
| G-oracle | `configs/n3v/elgs_local_crb300_12k.yaml`, per-prefix program | the `elgs_*` block (below) + `elgs_reserved_parity: true` |
| G-est | same YAML, `elgs_oracle_episodes` = the T1-estimated program | as G-oracle; only the program path differs |
| G-mis | same YAML, program = same membership, gap moved | as G-oracle; only the program path differs (Lane A's committed twin is `configs/n3v/elgs_local_crb300_12k_mis.yaml`, which differs from G in the program path alone) |
| G-wrongmem | same YAML, program = count-matched random row set, authored gap | as G-oracle; only the program path differs |
| G-ones | same YAML, program = truth rows with a late gap (286–297) | as G-oracle; only the program path differs |

So the lane has **three base configs** (prefix, U, G) and one derived
per-prefix config per gated arm — wave 1 froze 20 of them
(`runs/realdata/absfix/freeze_v1.txt`). The gated arms differ from each
other ONLY in `elgs_oracle_episodes`; U differs from G ONLY in the
`elgs_*` block.

**ON in every arm** (identical values in both base YAMLs): the 4D core
(`gaussian_dim: 4`, `rot_4d: False`, `force_sh_3d: True`,
`eval_shfs_4d: True`, `time_duration: [0.0, 10.0]`); LoRA motion
(`motion_model: "lora"`, rank 8, anchors 32); soft routing
(`enable_soft_routing: true`, `route_logit_init: 4.0`); the
residual-derived dynamic-mask losses (`dynamic_mask_from_residual: true`,
quantile 0.85, `lambda_dynamic_roi: 0.5`,
`lambda_static_exclusion: 0.02`); densification and pruning
(`densify_from_iter: 500`, `densify_until_iter: 30_000` — i.e. on for
the whole 12k run — `densify_grad_threshold: 0.0002`,
`densify_until_num_points: 600000`, `thresh_opa_prune: 0.005`,
`opacity_reset_interval: 40000`, never reached).

**ON in the G arms only** — the `elgs_*` block of
`elgs_local_crb300_12k.yaml`: `elgs_enable: true`,
`elgs_local_presence: true` (localized total gate; without it the
renderer applies a GLOBAL presence to every row),
`elgs_oracle_episodes: <program>.json` (family seeding binds membership
ONCE, at seeding, and `_elgs_family_ids` are inherited by clone/split),
`elgs_rounds_enabled: false` (no structural search),
`elgs_routing_pins_enabled: false`, `elgs_a_lr: 0.0` (boundaries
frozen), `elgs_tracks_dir: ""` (photometric-only: `attach_evidence` is a
no-op, `elgs/trainer_hooks.py:364-366`).

## 2. Flag-gated mechanisms that MUST stay off

All of these are off by default and are explicitly false in both base
YAMLs; none may be turned on in a wave-2 cell without a new frozen spec.

| mechanism | controlling flag | default |
|---|---|---|
| motion scaffold | `motion_scaffold_enable` | `False` (`arguments/__init__.py:173`) |
| hard static conversion | `enable_hard_static_conversion` | `False` (`arguments/__init__.py:112`) |
| CCR packet birth (B1) | `packet_birth_enable` | `False` (`arguments/__init__.py:234`) |
| flow-initialized birth (B1-F/B1-X) | `packet_birth_flow_init` | `False` (`arguments/__init__.py:249`) |
| CSVL-VPL lifecycle | `lifecycle_enable` | `False` (`arguments/__init__.py:195`); `setup_elgs` raises on `elgs_enable` + `lifecycle_enable` (`elgs/trainer_hooks.py:129-130`) |
| polynomial motion | `motion_model` | `"poly"` (`arguments/__init__.py:127`) — the YAMLs set `"lora"`; leaving it unset silently changes the motion model |
| motion-aware densification | `enable_motion_aware_densify` | `False` (`arguments/__init__.py:185`) |
| rendered flow | `enable_rendered_flow` | `False` (`arguments/__init__.py:184`) |
| appearance/opacity pointer edit | `--appearance_edit` (CLI, val-only) | `""` (`main.py:2210`) |

## 3. Trap 1 — reserved parity on the ungated comparator

Every `elgs_enable` run reserves the `(frame_order + camera_order) % 4
== 0` diagonal, ~25% of training units, UNCONDITIONALLY:
`build_reserved_pool` (`elgs/trainer_hooks.py:1437`, the diagonal at
`:1471-1472`) is called from `setup_elgs` (`:211`) for every EL-GS run,
and the trainer drops those units with `filter_elgs_reserved`
(`elgs/trainer_hooks.py:1536`, called at `main.py:1267`). A bare ungated
comparator would therefore train on a third more data than the arm it is
compared with. **An ungated wave-2 comparator MUST set
`elgs_reserved_parity: true`** (`arguments/__init__.py:311`, default
`False`), which routes the same reservation rule through
`reserved_indices_for_parity` (`elgs/trainer_hooks.py:1493`, applied at
`main.py:1272`). It returns `None` — a no-op, never a second filter —
when `elgs_enable` is true, which is why the G configs keep it set for
symmetry. Eight cells of the 2026-09-09 lane were cancelled and rerun
over exactly this.

## 4. Trap 2 — `--val` used to render EL-GS cells with the gate OFF

`validation()` restored a checkpoint and rendered it without calling
`setup_elgs`, so `gaussians.elgs_runtime` stayed None, `elgs_active` was
False (`gaussian_renderer/__init__.py:224`) and every row went through
the ordinary temporal marginal: every `--val` metric of every EL-GS cell
on N3V was gate-off (lane page §6). **Fixed** at `main.py:1044-1085`,
which now runs `training_setup` → `restore(model_params, opt)` →
`setup_elgs` on the checkpoint's own `elgs_state` and refuses a config
that declares `elgs_enable` on a checkpoint carrying no state
(`resolve_seeding_mode`, `scripts/eval_n3v_gated.py:101`).

**Until that fix is deployed on Leonardo, score EL-GS cells only through
`scripts/eval_n3v_gated.py --restore_state`** (`:516`, `:621-642`),
which takes the same restore branch and proves the restored intervals
equal the supplied `--program` by lineage key. U cells are unaffected by
either path.

## 5. What to check before submitting a wave-2 cell

1. `git diff` the cell's YAML against its base: the gated arms may differ
   from `elgs_local_crb300_12k.yaml` in `elgs_oracle_episodes` only, and
   U from `b0c_crb300_12k_rp.yaml` not at all.
2. `elgs_reserved_parity: true` is present in the ungated arm.
3. Every flag in §2 is absent or false; `motion_model: "lora"` is
   present (its default is `"poly"`).
4. The program JSON named by `elgs_oracle_episodes` exists and its
   sha256 is the frozen one — seeding raises if it does not, but a wrong
   program raises nothing.
5. `eval: True` and `resolution: 1` in `ModelParams` (with `eval: False`
   the reader merges the test split into training).
6. `--start_checkpoint` is that prefix's own `chkpnt6000.pth`, and
   `iterations: 12_000`.
7. The precondition extractor is handed the program the cell actually
   trained with — it fails closed on a mismatch, which is what marked
   eight Lane A post-steps FAILED.
8. Score through `eval_n3v_gated.py --restore_state` (§4), and look at a
   montage of the scored crop on every arm before reading any table.
