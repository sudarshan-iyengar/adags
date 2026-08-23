#!/usr/bin/env python3
"""Lane-4 synthetic falsification of the B2 directional donor DC edit.

EXPLORATORY. Takes the trained LRV3 B1-packets state and asks the one
question the N3V ladder cannot answer: when the edit under test
(`scene.appearance_edit`, "dc" arm — recipient rows render with the
DONOR row's base radiance and keep their own higher-order SH) is applied
to a link that is CORRECT BY AUTHORED CONSTRUCTION, does the CCR
certificate (ccr-method-2026-08-20 §3.2 item 3) admit it, and does the
event region actually improve?

Three links are built from authored ground truth rather than from the
descriptor proposer, so proposal ambiguity is removed from the question:

  L1 oracle-correct   episode-1 rows of the event object -> its return rows
  L2 wrong-identity   descriptor-close rows OFF the object -> the same
                      return rows (a known negative; the fixture is the
                      only place known identity negatives exist, §3.3)
  L3 same-identity    episode-1 rows of the object split by row-index
     no-op            parity, donor half -> recipient half (trained
                      appearance expected near-identical: the vacuity
                      reference)

and, OPT-IN under `--with_l4`, a fourth:

  L4 within-recipient RECIPIENT rows permuted among themselves, so
     permutation      identity is destroyed while the temporal window,
                      the row population and the row count are held fixed

L3 was originally read as isolating L1's damage as a property of the
cross-episode link. That reading is WITHDRAWN (2026-08-23, fresh-context
review): L3 edits DONOR rows, whose support ends at
`donor_support_upper_max`, while `event_return` is scored on the RETURN
frames — so L3's edited rows are largely invisible to the metric and
L1-vs-L3 varies identity AND metric visibility at once. Two hypotheses
survive that comparison equally: the payload carries nothing (so an
oracle-correct edit cannot help), or ANY reshuffle of this tensor over
the recipient rows costs about the same (so the damage is a property of
the edit machinery, not of identity). A monotone damage-versus-magnitude
curve fits both measured points, so the wrong-identity rejection does
not discriminate them either. L4 is the control that does, and it is the
only one here that does.

Row sets come from the authored oracle region and the effective temporal
support mu +/- 2 sigma_t. That support is an OPERATIONAL MATCHING
INTERVAL at a declared cutoff — never the exact support of a Gaussian
temporal lobe, which is unbounded.

The comparative anti-vacuity gate runs, and is PRINTED, before any
reconstruction render: L1's pre-edit DC distance must strictly exceed
L3's, i.e. the oracle pair must be more different than the same-surface
no-op pair. A run where L1 rows already carry L3-level appearance
distances cannot falsify anything, so no delta is computed at all —
`falsification_flow` never calls the measurement function when the gate
returns a verdict.

The certificate RULE is the unchanged CCR one — reserved units from
`elgs.trainer_hooks.build_reserved_pool` (the b1_packets config sets
`elgs_reserved_parity`, so training dropped exactly these units), paired
per-unit photometric deltas, admit iff mean + 3*SE < 0 AND both
per-side means <= 0 — with ONE amended allocation semantics, decided
before any render on this fixture: links SHARE reserved units instead of
consuming disjoint slots (every link is scored against the same identity
base, never an accumulating admitted state, and LRV3's return window
holds only ~12 reserved units), and L3's slot windows are (W1, W1)
where its rows actually render. See the inline SLOT AMENDMENT comment.
Held-out cameras are REPORT-ONLY: the event_return PSNR delta is a
diagnostic, never a certificate input.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

SCHEMA = "ccr-b2-falsification-v1"

# ---------------------------------------------------------------------------
# Frozen constants (any change is a new spec, not a flag)
# ---------------------------------------------------------------------------

FRAME_DT = 1.0 / 6.0                       # LRV3 fixture: 6 fps, model seconds
WINDOW_EPISODE1 = (0.0, 29.0 / 6.0)        # W1  = [0, 4.8333]  frames 0-29
WINDOW_RETURN = (57.0 / 6.0, 59.0 / 6.0)   # WR  = [9.5, 9.8333] frames 57-59
RETURN_FRAMES = (57, 58, 59)
DONOR_PROBE_T = 2.5                        # mid-episode-1 position probe
RECIPIENT_PROBE_T = 9.6                    # mid-return position probe
DONOR_SUPPORT_UPPER_MAX = 5.0
RECIPIENT_SUPPORT_LOWER_MIN = 9.3
WRONG_IDENTITY_RADIUS_SCALE = 1.75
GAP_WINDOW = (30.0 / 6.0, 56.0 / 6.0)      # LRV3 gap = [5.0, 9.3333] frames 30-56
MIN_DONOR_ROWS = 8
MIN_RECIPIENT_ROWS = 4
# mirrors scripts/consolidate_packets.T_SUPPORT_SIGMA / ADMIT_K_SE /
# ROW_MAP_CHUNK; re-declared rather than imported because that module
# imports the CUDA renderer at import time and these constants must stay
# reachable from CPU tests.
T_SUPPORT_SIGMA = 2.0
ADMIT_K_SE = 3.0
ROW_MAP_CHUNK = 4096
WINDOW_EPS = 1e-9

EDIT_MODE = "dc"                           # the B2 PRIMARY arm, unchanged

# ---- payloads -------------------------------------------------------------
# `--payload dc` is the DEFAULT and reproduces the recorded 2026-08-20 run
# exactly. `--payload opacity` swaps ONLY the redirected tensor and the
# anti-vacuity quantity; the row SETS, the row MAP, the certificate rule,
# the reserved-slot allocation and the report-only event_return metric are
# untouched, so the two arms are directly comparable.
PAYLOAD_DC = "dc"
PAYLOAD_OPACITY = "opacity"
PAYLOADS = (PAYLOAD_DC, PAYLOAD_OPACITY)
#: payload -> (model attribute redirected, scene.appearance_edit mode)
PAYLOAD_TENSOR = {
    PAYLOAD_DC: "_features_dc",
    PAYLOAD_OPACITY: "_opacity",
}
PAYLOAD_MODE = {
    PAYLOAD_DC: "dc",
    PAYLOAD_OPACITY: "opacity",
}

LINK_L1 = "L1_oracle_correct"
LINK_L2 = "L2_wrong_identity"
LINK_L3 = "L3_same_identity_noop"
LINK_ORDER = (LINK_L1, LINK_L2, LINK_L3)   # prespecified; slots are consumed
                                           # in exactly this order

# ---- L4: within-recipient permutation (OPT-IN, --with_l4) -----------------
# Added 2026-08-23 after a fresh-context review found the L1-vs-L3 comparison
# cannot attribute. L3 edits DONOR rows, whose support ends at
# `donor_support_upper_max`, while `event_return` is scored on the RETURN
# frames; L3's edited rows are therefore largely invisible to the metric, so
# L1 and L3 differ in TWO variables at once (identity AND whether the edited
# rows render where the metric looks).
#
# L4 holds the second variable fixed: it edits RECIPIENT rows — the same rows
# L1 edits, in the same temporal window, at the same row count per half — and
# destroys only identity, by permuting them among themselves.
#
#   L4 ~ 0 dB   reshuffling identity among same-window rows is harmless, so
#               L1's damage is attributable to the CROSS-EPISODE link;
#   L4 ~ L1     any reshuffle costs the same, so L1's number is a property of
#               the edit machinery on this tensor and says nothing about
#               identity.
#
# OPT-IN so that a default `--payload dc` run reproduces the recorded
# three-link report byte-for-byte.
LINK_L4 = "L4_recipient_permutation"
#: Every link this tool knows, in the prespecified consumption order. The
#: order of a given RUN is this list restricted to the links actually built
#: (`link_order_for`), so a three-link run is unchanged.
ALL_LINKS = LINK_ORDER + (LINK_L4,)
#: Offset added to `--seed` for the L4 permutation generator, so the
#: permutation is reproducible from the recorded seed alone and cannot
#: collide with any other seeded draw in the run.
L4_SEED_OFFSET = 7717

VERDICT_INVALID_SETS = "INVALID_SETS"
VERDICT_INVALID_VACUOUS = "INVALID_VACUOUS"
VERDICT_INVALID_SLOTS = "INVALID_SLOTS"
VERDICT_COMPLETED = "COMPLETED"


# ---------------------------------------------------------------------------
# Fixture protocol — the constants above are LRV3's values; every one of
# them is a function of the fixture's own event_spec.json
# ---------------------------------------------------------------------------
#
# Hardcoding them was safe while LRV3 was the only fixture. It stops being
# safe the moment a fixture moves its return: LRV4 returns on frame 59
# ALONE, so its return window is the single instant t = 59/6 = 9.8333 and
# LRV3's recipient probe of 9.6 would land in LRV4's GAP — the probe would
# silently sample positions while the object is ABSENT and the recipient
# row set would be selected on nonsense. `validate_protocol` refuses that
# case rather than letting it through.

FalsificationProtocol = namedtuple("FalsificationProtocol", (
    "scene_id",
    "frame_dt",
    "return_frames",
    "window_episode1",
    "window_return",
    "gap_window",
    "donor_probe_t",
    "recipient_probe_t",
    "donor_support_upper_max",
    "recipient_support_lower_min",
    "scalars_source",
))

#: The recorded LRV3 protocol: exactly the module constants above, so
#: `protocol=None` anywhere is the pre-2026-08-23 behaviour verbatim.
DEFAULT_PROTOCOL = FalsificationProtocol(
    scene_id="LRV3",
    frame_dt=FRAME_DT,
    return_frames=RETURN_FRAMES,
    window_episode1=WINDOW_EPISODE1,
    window_return=WINDOW_RETURN,
    gap_window=GAP_WINDOW,
    donor_probe_t=DONOR_PROBE_T,
    recipient_probe_t=RECIPIENT_PROBE_T,
    donor_support_upper_max=DONOR_SUPPORT_UPPER_MAX,
    recipient_support_lower_min=RECIPIENT_SUPPORT_LOWER_MIN,
    scalars_source="frozen",
)

#: Scenes whose four hand-chosen scalars are PINNED to the values their
#: recorded experiments actually ran with.
#:
#: Two of LRV3's four are not what `derive_protocol_scalars` yields:
#: the recipient probe is 9.6 where the mid-return-frame rule gives
#: 9.66667, and the recipient support floor is 9.3 where the gap-end rule
#: gives 9.33333. Both were round numbers chosen by hand when the LRV3
#: protocol was frozen. They are pinned here, not "corrected", because
#: experiments 213/233/236 selected their row sets with them and every
#: recorded LRV3 number is only comparable against these exact values.
#: The other two (donor probe 2.5, donor support cap 5.0) the rule
#: reproduces exactly, and `test_lrv3_derivation_divergence_is_recorded`
#: keeps the divergence visible rather than buried.
FROZEN_PROTOCOL_SCALARS = {
    "LRV3": {
        "donor_probe_t": DONOR_PROBE_T,
        "recipient_probe_t": RECIPIENT_PROBE_T,
        "donor_support_upper_max": DONOR_SUPPORT_UPPER_MAX,
        "recipient_support_lower_min": RECIPIENT_SUPPORT_LOWER_MIN,
    },
}


def frame_time(frame, fps):
    """Model-time seconds of a frame index."""
    return float(frame) / float(fps)


def derive_protocol_scalars(episode_1, gap, episode_2, fps):
    """The four hand-chosen scalars, as RULES over the fixture's frames.

    * donor probe        — the mid frame of episode 1
                           (LRV3: frame 15 -> 2.5, exact match)
    * recipient probe    — the mid frame of the return
                           (LRV4: the only return frame, 59 -> 9.8333)
    * donor support cap  — the first gap frame
                           (LRV3: frame 30 -> 5.0, exact match)
    * recipient support  — the last gap frame
      floor

    Episode 1 is identical in LRV3 and LRV4, so the donor side is
    identical by construction — which is what makes the two fixtures'
    donor rows comparable.
    """
    return {
        "donor_probe_t": frame_time((episode_1[0] + episode_1[1] + 1) // 2, fps),
        "recipient_probe_t": frame_time(
            (episode_2[0] + episode_2[1] + 1) // 2, fps),
        "donor_support_upper_max": frame_time(gap[0], fps),
        "recipient_support_lower_min": frame_time(gap[1], fps),
    }


def protocol_from_event_spec(spec):
    """Build the falsification protocol from a fixture's event_spec.json.

    Everything except the four hand-chosen scalars is an exact function
    of the spec, so LRV3 reproduces its frozen windows and return frames
    with no table at all.
    """
    if spec.get("kind") != "synthetic_leave_and_return":
        raise ContractError(
            "fixture kind {!r} is not a leave-and-return scene; the authored "
            "row sets are defined for that kind only".format(spec.get("kind")))
    fps = float(spec["fps"])
    if fps <= 0:
        raise ContractError("fixture declares a non-positive fps")
    presence = spec["presence_frames"]
    episode_1 = tuple(int(f) for f in presence["episode_1"])
    gap = tuple(int(f) for f in presence["gap"])
    episode_2 = tuple(int(f) for f in presence["episode_2"])
    return_frames = tuple(sorted(int(f) for f in spec["return_frames"]))
    if return_frames != tuple(range(episode_2[0], episode_2[1] + 1)):
        raise ContractError(
            "fixture return_frames {} disagree with its episode_2 {}".format(
                return_frames, episode_2))

    scene_id = str(spec.get("scene_id", ""))
    scalars = derive_protocol_scalars(episode_1, gap, episode_2, fps)
    source = "derived"
    if scene_id in FROZEN_PROTOCOL_SCALARS:
        scalars = dict(FROZEN_PROTOCOL_SCALARS[scene_id])
        source = "frozen"

    protocol = FalsificationProtocol(
        scene_id=scene_id,
        frame_dt=1.0 / fps,
        return_frames=return_frames,
        window_episode1=(frame_time(episode_1[0], fps),
                         frame_time(episode_1[1], fps)),
        window_return=(frame_time(episode_2[0], fps),
                       frame_time(episode_2[1], fps)),
        gap_window=(frame_time(gap[0], fps), frame_time(gap[1], fps)),
        scalars_source=source,
        **scalars
    )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    """Fail closed on a protocol that would select the wrong rows.

    The load-bearing check is that each POSITION PROBE lies inside the
    window it is supposed to sample. LRV3's 9.6 recipient probe against
    LRV4's single-instant return window is exactly the failure this
    catches: 9.6 < 9.8333, i.e. the probe is in the gap and the object is
    absent there.
    """
    w1, wr, gap = (protocol.window_episode1, protocol.window_return,
                   protocol.gap_window)
    if not protocol.return_frames:
        raise ContractError("protocol has an empty return window")
    if not (w1[0] <= w1[1] and wr[0] <= wr[1] and gap[0] <= gap[1]):
        raise ContractError("protocol has a reversed window")
    if not w1[1] < wr[0]:
        raise ContractError(
            "episode 1 and the return overlap; they must be disjoint in time")
    if not (w1[0] - WINDOW_EPS <= protocol.donor_probe_t <= w1[1] + WINDOW_EPS):
        raise ContractError(
            "donor probe {} lies outside episode-1 window {}; it would sample "
            "positions where the object is not present".format(
                protocol.donor_probe_t, w1))
    if not (wr[0] - WINDOW_EPS <= protocol.recipient_probe_t
            <= wr[1] + WINDOW_EPS):
        raise ContractError(
            "recipient probe {} lies outside the return window {}; it would "
            "sample positions where the object is ABSENT and the recipient "
            "rows would be selected on nonsense".format(
                protocol.recipient_probe_t, wr))
    if protocol.donor_support_upper_max > wr[0] + WINDOW_EPS:
        raise ContractError(
            "donor support cap {} reaches into the return window {}".format(
                protocol.donor_support_upper_max, wr))
    if protocol.recipient_support_lower_min < w1[1] - WINDOW_EPS:
        raise ContractError(
            "recipient support floor {} reaches back into episode 1 {}".format(
                protocol.recipient_support_lower_min, w1))
    if protocol.recipient_support_lower_min > wr[1] + WINDOW_EPS:
        raise ContractError(
            "recipient support floor {} is past the end of the return window "
            "{}; no row could qualify".format(
                protocol.recipient_support_lower_min, wr))
    return True


def protocol_block(protocol):
    """The protocol, as reported in a JSON artifact."""
    return {
        "scene_id": protocol.scene_id,
        "frame_dt": protocol.frame_dt,
        "return_frames": list(protocol.return_frames),
        "window_episode1": list(protocol.window_episode1),
        "window_return": list(protocol.window_return),
        "gap_window": list(protocol.gap_window),
        "probe_times": [protocol.donor_probe_t, protocol.recipient_probe_t],
        "donor_support_upper_max": protocol.donor_support_upper_max,
        "recipient_support_lower_min": protocol.recipient_support_lower_min,
        "scalars_source": protocol.scalars_source,
        "support_sigma_multiple": T_SUPPORT_SIGMA,
    }


# ---------------------------------------------------------------------------
# Pure functions: windows, row sets, row maps, gate, report assembly
# ---------------------------------------------------------------------------


def effective_support(t_mean, scaling_t_log):
    """Operational matching interval per row: mu +/- 2 sigma_t, with
    sigma_t = exp(_scaling_t) (the parameter is stored in log scale).

    NOT the exact support of the row's temporal lobe; a declared cutoff.
    """
    t_mean = torch.as_tensor(t_mean).reshape(-1).to(torch.float64)
    sigma_t = torch.exp(torch.as_tensor(scaling_t_log).reshape(-1).to(torch.float64))
    return t_mean - T_SUPPORT_SIGMA * sigma_t, t_mean + T_SUPPORT_SIGMA * sigma_t


def window_intersects(support_lo, support_hi, window):
    lo, hi = float(window[0]), float(window[1])
    return (support_lo <= hi + WINDOW_EPS) & (support_hi >= lo - WINDOW_EPS)


def build_row_sets(position_episode1, position_return, support_lo, support_hi,
                   features_dc, centre, radius, protocol=None):
    """The authored row sets (frozen design item 1).

    All inputs are CPU tensors over the SAME row order:
      position_episode1  (N,3)  get_dynamic_xyz(protocol.donor_probe_t)
      position_return    (N,3)  get_dynamic_xyz(protocol.recipient_probe_t)
      support_lo/hi      (N,)   effective_support(...)
      features_dc        (N,C)  flattened DC features

    `protocol` carries the fixture's windows and support edges;
    `protocol=None` is the frozen LRV3 protocol, i.e. the module
    constants, i.e. the behaviour every recorded LRV3 run had.

    Returns a dict of LongTensors plus the counts the report carries.
    Spanning rows are excluded from BOTH sets and counted. (With the
    frozen edge conditions the exclusion is structurally implied — a row
    with support_hi <= 5.0 cannot reach WR — so it is defence in depth,
    and it stays explicit because the edge conditions are the part a
    round-2 spec is most likely to relax.)
    """
    protocol = protocol or DEFAULT_PROTOCOL
    centre = torch.as_tensor(centre, dtype=torch.float64).reshape(1, 3)
    radius = float(radius)
    p1 = torch.as_tensor(position_episode1, dtype=torch.float64).reshape(-1, 3)
    pr = torch.as_tensor(position_return, dtype=torch.float64).reshape(-1, 3)
    lo = torch.as_tensor(support_lo, dtype=torch.float64).reshape(-1)
    hi = torch.as_tensor(support_hi, dtype=torch.float64).reshape(-1)
    dc = torch.as_tensor(features_dc, dtype=torch.float64)
    dc = dc.reshape(dc.shape[0], -1)
    n = p1.shape[0]
    if not (pr.shape[0] == lo.shape[0] == hi.shape[0] == dc.shape[0] == n):
        raise ContractError("row-set inputs disagree on the row count")

    d_ep1 = (p1 - centre).norm(dim=1)
    d_ret = (pr - centre).norm(dim=1)
    hits_w1 = window_intersects(lo, hi, protocol.window_episode1)
    hits_wr = window_intersects(lo, hi, protocol.window_return)
    inside_ep1 = d_ep1 <= radius
    inside_ret = d_ret <= radius

    donor_cap = protocol.donor_support_upper_max
    recipient_floor = protocol.recipient_support_lower_min
    spanning_mask = hits_w1 & hits_wr & (inside_ep1 | inside_ret)
    donor_mask = (inside_ep1 & hits_w1
                  & (hi <= donor_cap) & ~spanning_mask)
    recipient_mask = (inside_ret & hits_wr
                      & (lo >= recipient_floor) & ~spanning_mask)
    wrong_pool_mask = ((d_ep1 > radius * WRONG_IDENTITY_RADIUS_SCALE) & hits_w1
                       & (hi <= donor_cap) & ~spanning_mask)

    donor = torch.nonzero(donor_mask).reshape(-1)
    recipient = torch.nonzero(recipient_mask).reshape(-1)
    spanning = torch.nonzero(spanning_mask).reshape(-1)
    wrong_pool = torch.nonzero(wrong_pool_mask).reshape(-1)

    # Dw: descriptor-close but a DIFFERENT surface — ranked by DC
    # distance to the recipient set's DC medoid (median, the same medoid
    # convention consolidate_packets.packet_table uses), truncated to |D|.
    if recipient.numel() > 0 and wrong_pool.numel() > 0 and donor.numel() > 0:
        medoid = dc[recipient].median(dim=0).values
        dist = (dc[wrong_pool] - medoid).norm(dim=1)
        order = torch.argsort(dist)
        wrong = wrong_pool[order][:int(donor.numel())]
    else:
        wrong = torch.zeros(0, dtype=torch.long)

    donor_a, donor_b = split_by_row_parity(donor)
    return {
        "donor": donor,
        "recipient": recipient,
        "wrong": wrong,
        "spanning": spanning,
        "wrong_pool": wrong_pool,
        "donor_a": donor_a,
        "donor_b": donor_b,
    }


def split_by_row_parity(rows):
    """No-op link split: D_a = even row indices, D_b = odd (frozen
    design item 1). Disjoint by construction, and independent of any
    appearance quantity, so the no-op reference cannot be tuned."""
    rows = torch.as_tensor(rows).reshape(-1).to(torch.long)
    return rows[(rows % 2) == 0], rows[(rows % 2) == 1]


def recipient_permutation_link(recipient_rows, seed):
    """L4: the recipient set permuted AMONG ITSELF.

    Returns ``(r_rows, d_rows, meta)``: `r_rows` are the recipient rows
    that get edited, `d_rows` the recipient rows their payload is taken
    from, and `meta` the provenance the report carries.

    THE ONE-HOP INVARIANT IS PRESERVED, NOT BYPASSED. A permutation of a
    set onto itself makes every row both a donor and a recipient, which
    `scene.appearance_edit` forbids and `link_pointer` refuses. So the
    recipient set is first partitioned by ROW-INDEX PARITY — the same
    split convention L3 uses, and independent of any payload quantity so
    the control cannot be tuned — into halves A (sources only) and B
    (edited only). Both halves are truncated to the shorter so the map is
    a bijection, and B is mapped onto A by a random permutation drawn
    from `torch.Generator().manual_seed(seed)`.

    WHAT THE PARTITION COSTS, stated plainly: only |B| ~ half the
    recipient set is edited, so L4's edit VOLUME is about half L1's. The
    comparison against L1 is therefore per-row (`pre_edit_distance_mean`)
    rather than per-set, and the report records both counts. What a full
    self-permutation would have bought — the multiset of payload values
    over the edited set preserved EXACTLY — is weakened to: every value
    written comes from another row of the same recipient population, in
    the same temporal window. The identity destruction, which is the
    point of the control, is complete either way.

    There are no fixed points: A and B are disjoint by construction, so
    no row is ever mapped to itself.
    """
    rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    if int(rows.numel()) < 2:
        raise ContractError(
            "the within-recipient permutation needs at least 2 recipient "
            "rows; got {}".format(int(rows.numel())))
    source_half, edited_half = split_by_row_parity(rows)
    n = min(int(source_half.numel()), int(edited_half.numel()))
    if n < 1:
        raise ContractError(
            "the recipient parity split left an empty half ({} / {}); the "
            "permutation control cannot be built".format(
                int(source_half.numel()), int(edited_half.numel())))
    source_half = source_half[:n]
    edited_half = edited_half[:n]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(n, generator=generator)
    d_rows = source_half[order]
    r_rows = edited_half
    if bool((r_rows == d_rows).any()):
        raise ContractError(
            "the within-recipient permutation produced a fixed point, which "
            "the parity partition makes structurally impossible")
    meta = {
        "seed": int(seed),
        "permutation_sha256": permutation_hash(r_rows, d_rows),
        "recipient_rows_total": int(rows.numel()),
        "rows_edited": int(r_rows.numel()),
        "source_half_rows": int(source_half.numel()),
        "edited_half_rows": int(edited_half.numel()),
        "edit_volume_vs_l1": (int(r_rows.numel()) / float(rows.numel())
                              if rows.numel() else 0.0),
        "one_hop": "preserved by a row-index-parity partition; A rows are "
                   "sources only, B rows are edited only, and the halves are "
                   "disjoint",
        "cost": "only the edited half is redirected, so L4's edit VOLUME is "
                "about half L1's; compare per-row pre_edit_distance_mean, "
                "not per-set totals",
    }
    return r_rows, d_rows, meta


def permutation_hash(recipient_rows, donor_rows):
    """SHA-256 over the mapped pairs — the permutation's identity."""
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    hasher = hashlib.sha256()
    hasher.update(r_rows.cpu().numpy().tobytes())
    hasher.update(b"->")
    hasher.update(d_rows.cpu().numpy().tobytes())
    return hasher.hexdigest()


def link_order_for(link_stats):
    """The prespecified link order, restricted to the links actually
    built. A three-link run yields exactly `LINK_ORDER`."""
    unknown = set(link_stats) - set(ALL_LINKS)
    if unknown:
        raise ContractError(
            "unknown link(s) in the report: {}".format(sorted(unknown)))
    return tuple(name for name in ALL_LINKS if name in link_stats)


def sets_summary(sets, protocol=None):
    protocol = protocol or DEFAULT_PROTOCOL
    return {
        "donor_rows": int(sets["donor"].numel()),
        "recipient_rows": int(sets["recipient"].numel()),
        "wrong_identity_rows": int(sets["wrong"].numel()),
        "wrong_identity_pool_rows": int(sets["wrong_pool"].numel()),
        "spanning_rows_excluded": int(sets["spanning"].numel()),
        "noop_donor_rows": int(sets["donor_a"].numel()),
        "noop_recipient_rows": int(sets["donor_b"].numel()),
        "min_donor_rows": MIN_DONOR_ROWS,
        "min_recipient_rows": MIN_RECIPIENT_ROWS,
        "window_episode1": list(protocol.window_episode1),
        "window_return": list(protocol.window_return),
        "probe_times": [protocol.donor_probe_t, protocol.recipient_probe_t],
        "support_sigma_multiple": T_SUPPORT_SIGMA,
        "support_semantics": "operational matching interval at a declared "
                             "cutoff, NOT exact temporal support",
    }


def sets_are_sufficient(sets):
    return (int(sets["donor"].numel()) >= MIN_DONOR_ROWS
            and int(sets["recipient"].numel()) >= MIN_RECIPIENT_ROWS)


def assert_sets_disjoint(sets):
    """No link may make a row both a donor and a recipient (the one-hop
    invariant of scene.appearance_edit), and the no-op split must be a
    genuine partition."""
    pairs = (
        ("donor", "recipient"), ("wrong", "recipient"), ("wrong", "donor"),
        ("donor_a", "donor_b"),
    )
    for left, right in pairs:
        overlap = (set(sets[left].tolist()) & set(sets[right].tolist()))
        if overlap:
            raise ContractError(
                f"row sets {left} and {right} share {len(overlap)} row(s); "
                "the one-hop invariant would be violated")
    return True


def nearest_row_map(values, recipient_rows, donor_rows):
    """recipient row -> donor row by nearest `values`, chunked.

    The generalized nearest-map. `values` is any (N, ...) per-row tensor;
    rows are flattened to vectors and matched by Euclidean distance in
    that space. `nearest_dc_row_map` is this function on `_features_dc`
    and is the FROZEN primary correspondence; a payload-native map is
    this function on the payload's own tensor.
    """
    matrix = torch.as_tensor(values, dtype=torch.float64)
    matrix = matrix.reshape(matrix.shape[0], -1)
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    if r_rows.numel() == 0 or d_rows.numel() == 0:
        raise ContractError("row map needs a non-empty donor and recipient set")
    d_values = matrix[d_rows]
    r_values = matrix[r_rows]
    idx_chunks = []
    for start in range(0, r_values.shape[0], ROW_MAP_CHUNK):
        chunk = r_values[start:start + ROW_MAP_CHUNK]
        idx_chunks.append(torch.cdist(chunk, d_values).argmin(dim=1))
    return r_rows, d_rows[torch.cat(idx_chunks)]


def nearest_dc_row_map(features_dc, recipient_rows, donor_rows):
    """recipient row -> donor row by nearest DC, chunked — mirrors
    consolidate_packets.row_map semantics on explicit row sets.

    The FROZEN primary correspondence: every payload arm uses it, so
    every arm is scored on the same authored row pairing.
    """
    return nearest_row_map(features_dc, recipient_rows, donor_rows)


def link_pointer(n_rows, recipient_rows, donor_rows):
    """The pointer column for one link, with the one-hop invariant
    checked on the SETS (no row may be both donor and recipient)."""
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    if r_rows.shape != d_rows.shape:
        raise ContractError("row map is not one-to-one with the recipient set")
    overlap = set(r_rows.tolist()) & set(d_rows.tolist())
    if overlap:
        raise ContractError(
            f"one-hop invariant violated: {len(overlap)} row(s) are both "
            "donor and recipient in the same link")
    pointer = torch.arange(int(n_rows), dtype=torch.long)
    pointer[r_rows] = d_rows
    return pointer


def pre_edit_stats(values, recipient_rows, donor_rows, payload=None):
    """Anti-vacuity measurement: how different are the mapped pairs
    BEFORE the edit? Computed with no render.

    The measured quantity is whichever per-row tensor the payload
    redirects. The legacy `pre_edit_dc_distance_*` keys are ALWAYS
    emitted so the recorded `ccr-b2-falsification-v1` schema and every
    downstream reader keep working unchanged; when `payload` is given
    (i.e. anything other than the default DC run) payload-neutral keys
    are added alongside, and the DC-named keys carry the payload's
    distance rather than a DC distance.
    """
    matrix = torch.as_tensor(values, dtype=torch.float64)
    matrix = matrix.reshape(matrix.shape[0], -1)
    r_rows = torch.as_tensor(recipient_rows).reshape(-1).to(torch.long)
    d_rows = torch.as_tensor(donor_rows).reshape(-1).to(torch.long)
    dist = (matrix[r_rows] - matrix[d_rows]).norm(dim=1)
    changed = int((r_rows != d_rows).sum())
    mean = float(dist.mean()) if dist.numel() else 0.0
    maximum = float(dist.max()) if dist.numel() else 0.0
    stat = {
        "pre_edit_dc_distance_mean": mean,
        "pre_edit_dc_distance_max": maximum,
        "rows_changed": changed,
        "rows_changed_fraction": (changed / int(r_rows.numel())
                                  if r_rows.numel() else 0.0),
    }
    if payload is not None:
        stat["pre_edit_distance_mean"] = mean
        stat["pre_edit_distance_max"] = maximum
        stat["payload"] = str(payload)
        stat["payload_tensor"] = PAYLOAD_TENSOR.get(payload)
    return stat


def pre_edit_dc_stats(features_dc, recipient_rows, donor_rows):
    """The DC-payload anti-vacuity measurement (unchanged surface)."""
    return pre_edit_stats(features_dc, recipient_rows, donor_rows)


def stat_distance_mean(stat):
    """Payload-neutral read of a link stat's mean pre-edit distance.

    Falls back to the legacy DC-named key so hand-built stats and every
    recorded report stay readable.
    """
    if "pre_edit_distance_mean" in stat:
        return float(stat["pre_edit_distance_mean"])
    return float(stat["pre_edit_dc_distance_mean"])


def stat_distance_max(stat):
    """Payload-neutral read of a link stat's max pre-edit distance."""
    if "pre_edit_distance_max" in stat:
        return float(stat["pre_edit_distance_max"])
    return float(stat["pre_edit_dc_distance_max"])


def gate_decision(link_stats):
    """The comparative anti-vacuity gate (frozen design item 3).

    `link_stats` maps link name -> dict with at least
    `pre_edit_dc_distance_mean`, `rows_changed`, `slot_ok`.

    The measured quantity is whatever the payload redirects; the rule
    itself is payload-agnostic and unchanged. Payload-neutral keys are
    mirrored in only when the stats carry them (i.e. never on the
    default DC run, whose report stays byte-identical).

    Returns (verdict_or_None, anti_vacuity_block). A non-None verdict
    means NO reconstruction delta may be computed. Precedence:
    INVALID_VACUOUS (the comparison is degenerate — a scientific
    refusal) before INVALID_SLOTS (an infrastructural refusal), so a
    vacuous fixture is never reported as a scheduling problem.
    """
    l1 = link_stats[LINK_L1]
    l3 = link_stats[LINK_L3]
    comparative_ok = bool(stat_distance_mean(l1) > stat_distance_mean(l3))
    rows_ok = int(l1["rows_changed"]) >= 1
    order = link_order_for(link_stats)
    slots_ok = all(bool(link_stats[name].get("slot_ok")) for name in order)

    verdict = None
    if not (comparative_ok and rows_ok):
        verdict = VERDICT_INVALID_VACUOUS
    elif not slots_ok:
        verdict = VERDICT_INVALID_SLOTS

    anti = {
        "rule": ("L1 pre-edit mean DC distance STRICTLY > L3's, L1 rows "
                 "changed >= 1, and every link's reserved slot satisfiable"),
        "l1_pre_edit_dc_distance_mean": stat_distance_mean(l1),
        "l3_pre_edit_dc_distance_mean": stat_distance_mean(l3),
        "comparative_ok": comparative_ok,
        "l1_rows_changed": int(l1["rows_changed"]),
        "l1_rows_changed_ok": rows_ok,
        "slots_ok": slots_ok,
        "per_link": {
            name: _per_link_gate_entry(link_stats[name])
            for name in order
        },
        "verdict": verdict,
    }
    _mirror_payload_keys(anti, l1, prefix="l1_")
    _mirror_payload_keys(anti, l3, prefix="l3_", identity_only=True)
    if order != LINK_ORDER:
        # Only a run that added a link says so; the frozen three-link
        # report is untouched.
        anti["links"] = list(order)
    return verdict, anti


def _per_link_gate_entry(stat):
    entry = {
        "pre_edit_dc_distance_mean": stat_distance_mean(stat),
        "pre_edit_dc_distance_max": stat_distance_max(stat),
        "rows_changed": int(stat["rows_changed"]),
        "rows_changed_fraction": float(stat["rows_changed_fraction"]),
        "slot_ok": bool(stat.get("slot_ok")),
        "slot_units_per_side": stat.get("slot_units_per_side"),
        "slot_available_per_side": stat.get("slot_available_per_side"),
    }
    if "pre_edit_distance_mean" in stat:
        entry["pre_edit_distance_mean"] = stat_distance_mean(stat)
        entry["pre_edit_distance_max"] = stat_distance_max(stat)
        entry["payload"] = stat.get("payload")
        entry["payload_tensor"] = stat.get("payload_tensor")
    return entry


def _mirror_payload_keys(block, stat, prefix="", identity_only=False):
    """Add the payload-neutral mirror keys, and only then."""
    if "pre_edit_distance_mean" not in stat:
        return
    block[prefix + "pre_edit_distance_mean"] = stat_distance_mean(stat)
    if identity_only:
        return
    block["payload"] = stat.get("payload")
    block["payload_tensor"] = stat.get("payload_tensor")


def _empty_link_entry(name, stat):
    entry = {
        "name": name,
        "pre_edit_dc_distance_mean": stat_distance_mean(stat),
        "pre_edit_dc_distance_max": stat_distance_max(stat),
        "rows_changed": int(stat["rows_changed"]),
        "rows_changed_fraction": float(stat["rows_changed_fraction"]),
        "slot_units_per_side": stat.get("slot_units_per_side"),
        "slot_available_per_side": stat.get("slot_available_per_side"),
        "raw_slot_delta_mean": None,
        "raw_slot_delta_se": None,
        "return_side_delta_mean": None,
        "event_return_psnr_base": None,
        "event_return_psnr_edited": None,
        "event_return_psnr_delta": None,
        "certificate": {
            "stage_reached": "slot" if not stat.get("slot_ok") else None,
            "admitted": False,
            "pooled_mean": None,
            "pooled_se": None,
            "side_means": None,
        },
    }
    if "pre_edit_distance_mean" in stat:
        entry["pre_edit_distance_mean"] = stat_distance_mean(stat)
        entry["pre_edit_distance_max"] = stat_distance_max(stat)
        entry["payload"] = stat.get("payload")
        entry["payload_tensor"] = stat.get("payload_tensor")
    return entry


def falsification_flow(sets_block, link_stats, measure_fn, packet_block=None):
    """Assemble the report. `measure_fn(name, stat) -> dict` is the ONLY
    place a reconstruction render happens, and it is called exclusively
    after the gate returns no verdict — that is the structural guarantee
    that a failed gate cannot be followed by a delta."""
    verdict, anti = gate_decision(link_stats)
    links = []
    for name in link_order_for(link_stats):
        entry = _empty_link_entry(name, link_stats[name])
        if verdict is None:
            entry.update(measure_fn(name, link_stats[name]))
        links.append(entry)
    report = {
        "schema": SCHEMA,
        "sets": sets_block,
        "anti_vacuity": anti,
        "links": links,
        "verdict": verdict or VERDICT_COMPLETED,
    }
    if packet_block is not None:
        report["packets"] = packet_block
    return report


def invalid_sets_report(sets_block, packet_block=None):
    """The refusal report when the authored sets are too small to ask
    the question at all (|D| < 8 or |R| < 4)."""
    report = {
        "schema": SCHEMA,
        "sets": sets_block,
        "anti_vacuity": None,
        "links": [],
        "verdict": VERDICT_INVALID_SETS,
    }
    if packet_block is not None:
        report["packets"] = packet_block
    return report


def packet_summary(packet_ids, sets):
    """Descriptive only: B1 packet membership of the authored sets. The
    falsification never uses packet ids to build a link."""
    ids = torch.as_tensor(packet_ids).reshape(-1).to(torch.long)
    valid = ids >= 0
    donor = sets["donor"]
    recipient = sets["recipient"]
    return {
        "rows": int(ids.numel()),
        "rows_with_packet": int(valid.sum()),
        "packets": int(len(set(int(p) for p in ids[valid].tolist()))),
        "donor_rows_with_packet": int(valid[donor].sum()) if donor.numel() else 0,
        "recipient_rows_with_packet": (int(valid[recipient].sum())
                                       if recipient.numel() else 0),
        "recipient_packet_ids": sorted(
            set(int(p) for p in ids[recipient][valid[recipient]].tolist())
        )[:32] if recipient.numel() else [],
    }


def mean_se(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    return mean, (var / n) ** 0.5


def certificate_stage(pooled_mean, pooled_se, side_means):
    """The UNCHANGED CCR admission rule (§3.2 item 3): admit iff
    mean + 3*SE < 0 AND every per-side mean <= 0. Returns
    (stage_reached, admitted) with the FAILING stage named."""
    pooled_ok = (pooled_mean + ADMIT_K_SE * pooled_se) < 0
    side_ok = all(m <= 0 for m in side_means)
    if not pooled_ok:
        return "pooled-rule", False
    if not side_ok:
        return "side-rule", False
    return "admitted", True


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(1.0) - 10.0 * np.log10(mse))


def _merge_config(args, config_path):
    """YAML-over-argparse merge, identical to scripts/eval_lrv1_event.py."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def recursive_merge(key, host):
        if isinstance(host[key], dict):
            for kk in host[key]:
                recursive_merge(kk, host[key])
        else:
            assert hasattr(args, key), f"unknown config key {key}"
            setattr(args, key, host[key])

    for key in config:
        recursive_merge(key, config)


# ---------------------------------------------------------------------------
# Render-dependent helpers (thin; every CUDA import is lazy)
# ---------------------------------------------------------------------------


def set_pointer(gaussians, pointer, mode=EDIT_MODE):
    """Install one pointer column in `mode`.

    The mode selects WHICH per-row tensor the column redirects
    (scene.appearance_edit); the column itself is payload-independent.
    """
    gaussians._appearance_source_idx = pointer.to(gaussians._features_dc.device)
    gaussians._appearance_share_mode = mode


def payload_values(gaussians, payload):
    """The CPU float64 (N, C) matrix a payload's anti-vacuity distance
    and its native row map are measured in — the RAW stored parameter,
    matching what the pointer actually redirects."""
    if payload not in PAYLOAD_TENSOR:
        raise ContractError(f"unknown payload: {payload!r}")
    name = PAYLOAD_TENSOR[payload]
    tensor = getattr(gaussians, name, None)
    if tensor is None or int(tensor.numel()) == 0:
        raise ContractError(
            f"payload {payload!r} needs {name}, which is empty on this model")
    n_rows = int(gaussians._xyz.shape[0])
    if int(tensor.shape[0]) != n_rows:
        raise ContractError(
            f"{name} has {int(tensor.shape[0])} rows, model has {n_rows}")
    return tensor.detach().reshape(n_rows, -1).cpu().to(torch.float64)


def event_return_psnr(gaussians, scene, pipe, background, gt_dir, obj_id,
                      protocol=None):
    """Pooled PSNR over the event object's front-most mask at the RETURN
    frames, on the held-out cameras, under the model's CURRENT pointer
    state. Mask loading follows scripts/eval_lrv1_event.py exactly
    (source_path/gt_identity/cam%02d_f%03d.npy, front-most == the event
    object id). REPORT-ONLY: never an input to the certificate.

    The return frames come from the fixture's protocol; `protocol=None`
    is LRV3's frozen (57, 58, 59)."""
    from gaussian_renderer import render  # lazy: CUDA JIT at import time

    protocol = protocol or DEFAULT_PROTOCOL
    return_frames = set(protocol.return_frames)

    sq = 0.0
    count = 0
    frames = 0
    with torch.no_grad():
        for item in scene.getTestCameras():
            cam = item[1] if isinstance(item, (tuple, list)) else item
            name = cam.image_name                       # cam<NN>_f<FFF>
            frame = int(name.split("_f")[1])
            if frame not in return_frames:
                continue
            cam_idx = int(name.split("_")[0][3:])
            mask_path = Path(gt_dir) / ("cam%02d_f%03d.npy" % (cam_idx, frame))
            if not mask_path.is_file():
                raise ContractError(
                    f"held-out view {name} has no ground-truth identity buffer; "
                    "the event metric would be computed over an incomplete region")
            obj = torch.from_numpy(np.load(mask_path)).cuda() == obj_id
            if not bool(obj.any()):
                continue
            gt = (item[0] if isinstance(item, (tuple, list))
                  else cam.original_image)[0:3].cuda().float().clamp(0.0, 1.0)
            pred = render(cam.cuda(), gaussians, pipe,
                          background)["render"].clamp(0.0, 1.0)
            sel = obj.unsqueeze(0).expand_as(pred)
            diff = (pred - gt)[sel]
            sq += float((diff ** 2).sum())
            count += int(diff.numel())
            frames += 1
    if count == 0:
        raise ContractError(
            "no held-out return-frame pixel carried the event object id")
    return psnr_from_mse(sq / count), count // 3, frames


# ---------------------------------------------------------------------------
# state loading — shared with scripts/payload_headroom.py so both tools see
# the SAME restored state and therefore the same authored row sets
# ---------------------------------------------------------------------------


def resolve_model_path(args):
    """Fill `model_path` from $ADAGS_RUN_DIR and make sure it exists."""
    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if not str(getattr(args, "model_path", "") or "").strip():
        if not run_dir:
            raise ContractError("--model_path is required when ADAGS_RUN_DIR is unset")
        args.model_path = run_dir
    os.makedirs(args.model_path, exist_ok=True)
    return args.model_path


def load_oracle_region(path):
    """(centre, radius) of the authored sphere the row sets live on."""
    region = json.loads(Path(path).read_text())["region"]
    if region.get("kind") != "sphere":
        raise ContractError(
            f"oracle region kind {region.get('kind')!r} is not a sphere; the "
            "authored row sets are defined on a sphere only")
    return region["centre"], float(region["radius"])


def load_event_spec(source_path):
    """Read the fixture and derive its protocol: (spec, obj_id, gt_dir,
    protocol).

    This REPLACES the old `return_frames == (57, 58, 59)` refusal, which
    admitted exactly one fixture and would have silently mis-probed any
    other. The guard is now structural rather than literal: the scene
    must be a leave-and-return fixture, its return frames must agree with
    its own episode 2, and `validate_protocol` must accept the windows
    and probes the spec implies. LRV3 passes it with byte-identical
    windows, probes and support edges."""
    spec = json.loads((Path(source_path) / "event_spec.json").read_text())
    obj_id = int(spec["event_object"]["id"])
    gt_dir = Path(source_path) / "gt_identity"
    protocol = protocol_from_event_spec(spec)
    return spec, obj_id, gt_dir, protocol


def restore_model_and_scene(args, lp, op, pp):
    """Restore the trained state exactly as the falsification does.

    Returns (gaussians, scene, dataset, opt, pipe). No render happens
    here and no optimizer step is ever taken: the checkpoint carries
    every per-row parameter, so an edit can be installed and scored
    without retraining.
    """
    from scene import Scene, GaussianModel
    from scene.appearance_edit import clear_appearance_edit

    torch.manual_seed(args.seed)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(dataset, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio,
                  time_duration=args.time_duration, shuffle=False)
    scene.opt = opt
    gaussians.training_setup(opt)
    model_params, _ = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt)
    clear_appearance_edit(gaussians)
    if bool(getattr(opt, "elgs_enable", False)):
        raise ContractError(
            "this falsification is specified on the b1_packets substrate "
            "(elgs_enable false); an EL-GS cell would need its presence "
            "program restored before any render")
    return gaussians, scene, dataset, opt, pipe


def probe_row_state(gaussians, protocol=None):
    """The probe of the restored state the row sets are built from: both
    probe positions, the DC matrix and the effective support.

    Payload-independent on purpose — every payload arm must select the
    SAME authored donor/recipient rows. `protocol=None` is LRV3's frozen
    probe pair (2.5, 9.6).
    """
    protocol = protocol or DEFAULT_PROTOCOL
    n_rows = int(gaussians._xyz.shape[0])
    with torch.no_grad():
        pos_ep1 = gaussians.get_dynamic_xyz(
            protocol.donor_probe_t).detach().cpu()
        pos_ret = gaussians.get_dynamic_xyz(
            protocol.recipient_probe_t).detach().cpu()
        dc = gaussians._features_dc.detach().reshape(n_rows, -1).cpu()
        sup_lo, sup_hi = effective_support(
            gaussians._t.detach().cpu(), gaussians._scaling_t.detach().cpu())
    return {
        "n_rows": n_rows, "pos_ep1": pos_ep1, "pos_ret": pos_ret,
        "dc": dc, "sup_lo": sup_lo, "sup_hi": sup_hi,
        "protocol": protocol,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = ArgumentParser(description="Lane-4 B2 edit falsification (LRV3)")
    from arguments import ModelParams, OptimizationParams, PipelineParams

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--packet_state", default="")
    parser.add_argument("--oracle_region",
                        default=str(REPO_ROOT / "configs" / "lrv3"
                                    / "oracle_correct.json"))
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--payload", choices=PAYLOADS, default=PAYLOAD_DC,
                        help="which per-row tensor the pointer redirects; "
                             "'dc' (default) reproduces the recorded run")
    parser.add_argument("--with_l4", action="store_true",
                        help="add the L4 within-recipient permutation "
                             "control, which separates 'the payload carries "
                             "nothing' from 'any reshuffle of this tensor "
                             "costs the same'. Default OFF so a three-link "
                             "report stays byte-identical to the recorded "
                             "one. Deterministic under --seed.")
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float,
                        default=[0.0, 10.0])
    parser.add_argument("--num_pts", type=int, default=50_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _merge_config(args, args.config)

    from elgs.trainer_hooks import build_reserved_pool
    from scene.appearance_edit import clear_appearance_edit
    from scripts.consolidate_packets import paired_delta, pick_confirmation_slot

    payload = args.payload
    edit_mode = PAYLOAD_MODE[payload]
    # The DC arm keeps the recorded report byte-for-byte: no payload
    # annotation is threaded, so `pre_edit_stats` emits the legacy keys
    # only. Any other payload annotates.
    stats_payload = None if payload == PAYLOAD_DC else payload

    resolve_model_path(args)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    centre, radius = load_oracle_region(args.oracle_region)
    # The protocol is the FIXTURE's, not this tool's: windows, return
    # frames, probes and support edges all come from event_spec.json, so
    # pointing --source_path at LRV4 moves the recipient probe onto
    # LRV4's single return instant instead of silently sampling its gap.
    spec, obj_id, gt_dir, protocol = load_event_spec(args.source_path)

    gaussians, scene, dataset, opt, pipe = restore_model_and_scene(
        args, lp, op, pp)

    probe = probe_row_state(gaussians, protocol)
    n_rows = probe["n_rows"]
    pos_ep1, pos_ret = probe["pos_ep1"], probe["pos_ret"]
    dc = probe["dc"]
    sup_lo, sup_hi = probe["sup_lo"], probe["sup_hi"]
    # The anti-vacuity quantity follows the payload; the row SETS and the
    # row MAP never do (both stay DC-derived), so the arms are comparable.
    measured = dc if payload == PAYLOAD_DC else payload_values(gaussians, payload)

    sets = build_row_sets(pos_ep1, pos_ret, sup_lo, sup_hi, dc, centre, radius,
                          protocol)
    sets_block = sets_summary(sets, protocol)
    sets_block["rows"] = n_rows
    sets_block["oracle_region"] = {"centre": list(centre), "radius": radius,
                                   "source": str(args.oracle_region)}
    sets_block["protocol"] = protocol_block(protocol)

    packet_block = None
    if str(args.packet_state or "").strip():
        state = torch.load(args.packet_state, map_location="cpu")
        packet_ids = state["packet_ids"]
        if int(packet_ids.shape[0]) != n_rows:
            raise ContractError(
                f"packet state has {int(packet_ids.shape[0])} rows, model has "
                f"{n_rows}")
        packet_block = packet_summary(packet_ids, sets)
        packet_block["schema_version"] = str(state.get("schema_version", ""))

    def _emit(report):
        Path(args.out_report).write_text(
            json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps(report, indent=1, sort_keys=True), flush=True)
        return 0

    if not sets_are_sufficient(sets):
        return _emit(invalid_sets_report(sets_block, packet_block))
    assert_sets_disjoint(sets)

    # ---- links, row maps and pointers (no render)
    link_rows = {
        LINK_L1: (sets["recipient"], sets["donor"]),
        LINK_L2: (sets["recipient"], sets["wrong"]),
        LINK_L3: (sets["donor_b"], sets["donor_a"]),
    }
    link_stats = {}
    for name, (r_set, d_set) in link_rows.items():
        if r_set.numel() == 0 or d_set.numel() == 0:
            raise ContractError(
                f"link {name} has an empty side ({int(r_set.numel())} recipient "
                f"rows, {int(d_set.numel())} donor rows)")
        r_rows, d_rows = nearest_dc_row_map(dc, r_set, d_set)
        stat = dict(pre_edit_stats(measured, r_rows, d_rows, stats_payload))
        stat["pointer"] = link_pointer(n_rows, r_rows, d_rows)
        link_stats[name] = stat

    # L4 is built from a seeded PERMUTATION rather than the nearest-payload
    # map — that is the whole point of the control — but everything after
    # this line is the identical machinery: same pointer install, same
    # reserved slots, same certificate, same event_return measurement.
    l4_block = None
    if args.with_l4:
        r_rows, d_rows, l4_block = recipient_permutation_link(
            sets["recipient"], args.seed + L4_SEED_OFFSET)
        stat = dict(pre_edit_stats(measured, r_rows, d_rows, stats_payload))
        stat["pointer"] = link_pointer(n_rows, r_rows, d_rows)
        link_stats[LINK_L4] = stat
        l4_block["payload"] = payload
        l4_block["l1_rows_edited"] = int(sets["recipient"].numel())

    # ---- reserved pool and per-link disjoint slots (no render)
    train_units = scene.getTrainCameras()
    cameras_meta = scene.train_cameras[1.0]
    if len(cameras_meta) != len(train_units):
        raise ContractError("camera metadata / dataset length mismatch")
    reserved = build_reserved_pool(cameras_meta)
    slot_pool = list(reserved)
    window_w1 = protocol.window_episode1
    window_wr = protocol.window_return
    avail_d_all = [u for u, t in slot_pool
                   if window_w1[0] <= t <= window_w1[1]]
    avail_r_all = [u for u, t in slot_pool
                   if window_wr[0] <= t <= window_wr[1]]
    return_units = list(avail_r_all)

    # SLOT AMENDMENT (2026-08-20, decided by the primary BEFORE any
    # reconstruction render on this fixture; the worker's static
    # feasibility measurement showed the CCR-style disjoint-slot rule is
    # infeasible on LRV3 — only ~12 reserved units exist in WR, so L2/L3
    # would starve after L1's pick and the run would be INVALID_SLOTS by
    # construction). Here every link is measured against the SAME
    # identity base state, never an accumulating admitted state, so the
    # sequential-disjointness rationale of the CCR pass does not apply:
    # links SHARE reserved units, which additionally makes their deltas
    # directly comparable. Per-link windows: L1 and L2 measure donor-side
    # W1 / recipient-side WR; L3's rows all live in episode 1, so its
    # photometric null is measured on (W1, W1) — the frozen (W1, WR)
    # windows would have made it structurally near-null instead of a
    # genuine control. pick_confirmation_slot keeps the two sides of one
    # link internally disjoint.
    link_windows = {
        LINK_L1: (window_w1, window_wr),
        LINK_L2: (window_w1, window_wr),
        LINK_L3: (window_w1, window_w1),
        # L4 takes L1's windows, NOT the (WR, WR) form the L3 amendment
        # would suggest, for two stated reasons. (1) The L1-vs-L4
        # comparison is the entire purpose of the control, and it is only
        # valid on identical slot geometry — same units, same budget, same
        # certificate. (2) L1's edited rows are the RECIPIENTS, which
        # render in WR exactly as L4's do, so the two links already have
        # the same relationship to each side; (WR, WR) would additionally
        # be infeasible here, since LRV3's return window holds ~12
        # reserved units against the 16-unit floor.
        LINK_L4: (window_w1, window_wr),
    }
    for name in link_order_for(link_stats):    # prespecified order
        stat = link_stats[name]
        (d_lo, d_hi), (r_lo, r_hi) = link_windows[name]
        if (d_lo, d_hi) == (r_lo, r_hi):
            # Identical windows defeat pick_confirmation_slot: its
            # recipient side excludes the ENTIRE donor-side availability
            # list (correct for the CCR pass's disjoint windows, fatal
            # here — experiment 212 returned INVALID_SLOTS on 120/120
            # available units). Allocate directly: first 8 units in the
            # window to the donor side, the NEXT 8 to the recipient
            # side — internally disjoint by construction.
            in_window = [u for u, t in slot_pool if d_lo <= t <= d_hi]
            slot = ((in_window[:8], in_window[8:16])
                    if len(in_window) >= 16 else None)
        else:
            slot = pick_confirmation_slot(
                slot_pool, set(),              # fresh per link: shared units
                d_lo, d_hi, r_lo, r_hi)
        stat["slot_available_per_side"] = [
            len([u for u, t in slot_pool if d_lo <= t <= d_hi]),
            len([u for u, t in slot_pool if r_lo <= t <= r_hi]),
        ]
        stat["slot_windows"] = [[d_lo, d_hi], [r_lo, r_hi]]
        if slot is None:
            stat["slot"] = None
            stat["slot_ok"] = False
            stat["slot_units_per_side"] = None
            continue
        side_d, side_r = slot
        stat["slot"] = (side_d, side_r)
        stat["slot_ok"] = True
        stat["slot_units_per_side"] = [len(side_d), len(side_r)]

    # ---- the anti-vacuity gate, PRINTED before any reconstruction render
    gate_verdict, anti_preview = gate_decision(link_stats)
    print(json.dumps({"ccr_b2_falsification_gate": {
        "sets": sets_block, "anti_vacuity": anti_preview,
        "reserved_units": len(reserved),
        "reserved_units_in_W1": len(avail_d_all),
        "reserved_units_in_WR": len(avail_r_all),
    }}, sort_keys=True), flush=True)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")
    identity = torch.arange(n_rows, dtype=torch.long)
    base_psnr = {"value": None}

    def measure(name, stat):
        """Every render in this tool happens here, and this runs only
        when the gate returned no verdict (falsification_flow)."""
        pointer = stat["pointer"]
        side_d, side_r = stat["slot"]
        with torch.no_grad():
            # `paired_delta` is reused UNCHANGED: the payload rides its
            # existing `mode` parameter, which is exactly the flag
            # scene.appearance_edit dispatches on.
            dd = paired_delta([train_units[u] for u in side_d], gaussians, pipe,
                              background, identity, pointer, edit_mode)
            dr = paired_delta([train_units[u] for u in side_r], gaussians, pipe,
                              background, identity, pointer, edit_mode)
            pooled_mean, pooled_se = mean_se(dd + dr)
            side_means = [sum(dd) / len(dd), sum(dr) / len(dr)]
            stage, admitted = certificate_stage(pooled_mean, pooled_se,
                                                side_means)
            ret = paired_delta([train_units[u] for u in return_units], gaussians,
                               pipe, background, identity, pointer, edit_mode)
            return_mean, _ = mean_se(ret)

            if base_psnr["value"] is None:
                set_pointer(gaussians, identity, edit_mode)
                base_psnr["value"] = event_return_psnr(
                    gaussians, scene, pipe, background, gt_dir, obj_id,
                    protocol)
            set_pointer(gaussians, pointer, edit_mode)
            edited = event_return_psnr(gaussians, scene, pipe, background,
                                       gt_dir, obj_id, protocol)
            clear_appearance_edit(gaussians)
        base = base_psnr["value"]
        return {
            "raw_slot_delta_mean": pooled_mean,
            "raw_slot_delta_se": pooled_se,
            "return_side_delta_mean": return_mean,
            "return_side_units": len(return_units),
            "event_return_psnr_base": base[0],
            "event_return_psnr_edited": edited[0],
            "event_return_psnr_delta": edited[0] - base[0],
            "event_return_pixels": edited[1],
            "event_return_frames": edited[2],
            "certificate": {
                "stage_reached": stage,
                "admitted": bool(admitted),
                "pooled_mean": pooled_mean,
                "pooled_se": pooled_se,
                "side_means": side_means,
            },
        }

    report = falsification_flow(sets_block, link_stats, measure, packet_block)
    report["checkpoint"] = str(args.start_checkpoint)
    report["config"] = str(args.config)
    report["arm"] = edit_mode          # == EDIT_MODE on the default DC arm
    if stats_payload is not None:
        report["payload"] = payload
        report["payload_tensor"] = PAYLOAD_TENSOR[payload]
    report["evidence_bearing"] = False
    if l4_block is not None:
        report["l4_permutation"] = l4_block
    report["protocol"] = protocol_block(protocol)
    report["reserved_units"] = len(reserved)
    report["reserved_units_in_W1"] = len(avail_d_all)
    report["reserved_units_in_WR"] = len(avail_r_all)
    if gate_verdict is not None and report["verdict"] != gate_verdict:
        raise ContractError("gate verdict changed between preview and flow")
    return _emit(report)


if __name__ == "__main__":
    raise SystemExit(main())
