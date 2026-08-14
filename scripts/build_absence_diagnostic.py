"""M1 true-absence diagnostic PRIMARY evaluator (prereg revision 4).

Authority: ``configs/elgs/prereg_m1_absence_diagnostic_v1.json`` REVISION 4
(r1 initial freeze; r2 owner pre-data multi-view tightening; r3
hostile-review repairs B1-B9/N1-N7; r4 scoped re-review repairs R1-R13).
EVERY predicate, constant, decision rule and output below is read from that
file at runtime -- no scientific constant is hardcoded here, so a pre-data
amendment is a config edit rather than a code edit. The file's sha256 is
recorded in the output and the loader fails closed on ``revision != 4`` or on
any missing expected key.

What this evaluator does: classify every ``true_absence_candidate`` window
emitted by the sealed M1-A0 census into exactly one of the frozen terminal
classes using ONLY calibration, masks, the sealed consensus anchor ``ltp`` and
the sealed tracker reports. It performs no tracking, no rendering, no
training, and it re-scores no census statistic. No output is gate-bearing, and
no output may be read as a statement about PHYSICAL absence before the
mandatory M1-A0b audit (whose sample this evaluator emits) returns.

Commensurability with the instrument under test is a hard requirement, so the
mask/pixel/geometry primitives are IMPORTED from ``scripts.build_m1_census``
rather than reimplemented: ``load_component_labels``, ``frustum_contains``,
``round_half_up``, ``index_tracks``, ``ReportIndex``, ``rig_radius``,
``angular_separation_floor``, ``MASK_THRESHOLD``, ``MIN_COMPONENT_PX``,
``VIS_THRESHOLD``. The prereg's frozen constants are asserted EQUAL to the
census module's constants at load time; a divergence fails the run closed
because the reused primitives cannot honour a different value.

Pass structure (prereg ``mandated_pass_structure_N3_R9``): per SEQUENCE,
camera-major then frame-major, decoding EVERY frame of the realized census
window. Connected-component labels for ``(c, t)`` are computed AT MOST ONCE
and reused by every window that needs ``t``; the three S5 area floors are
applied to the SAME labelling. The 144 decision-relevant grid cells are
RE-AGGREGATIONS of one cached boolean tensor, never re-passes.

Fail-closed conditions: any substrate mismatch; any non-empty C4 or C6 (both
asserted empty, tautology checks per N2/R9); the R6 operative census-
reproduction guard (``|W| + bridged == n_frames`` and maximal bridged runs ==
``bridged_interruptions``); the retained r3 tautology checks on W; any missing
artifact.

Import-time constraint: torch-free (numpy + cv2 + PIL only; cv2/PIL are
imported lazily by the reused census primitives).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.canonical import canonical_json_bytes, sha256_file  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from scripts.build_elgs_tracks import (  # noqa: E402
    CameraModel,
    SceneBundle,
    load_temporal_scene,
)
from scripts.build_m1_census import (  # noqa: E402
    MASK_THRESHOLD,
    MIN_COMPONENT_PX,
    VIS_THRESHOLD,
    ReportIndex,
    angular_separation_floor,
    frustum_contains,
    index_tracks,
    load_component_labels,
    rig_radius,
    round_half_up,
)

DIAGNOSTIC_SCHEMA = "elgs-m1-absence-diagnostic-v1"
PREREG_REVISION_REQUIRED = 5
PROVENANCE_FILENAME = "diva360_conversion_provenance.json"
AUDIT_SAMPLE_SEED = 20260814
AUDIT_SAMPLE_MIN = 60

# ---------------------------------------------------------------------------
# Frozen vocabulary. These are NAMES, not thresholds: every one is verified to
# occur verbatim in the loaded prereg text, so a renamed class or status in an
# amended file fails the run closed instead of silently vanishing.
# ---------------------------------------------------------------------------

CLASS_C1A = "C1a_GENUINE_ABSENCE_CORROBORATED"
CLASS_C1B = "C1b_SUBTHRESHOLD_FOREGROUND_ONLY"
CLASS_C1C = "C1c_ANCHOR_UNSUPPORTED"
CLASS_C2 = "C2_TRACK_OR_REPORT_LOSS"
CLASS_C3 = "C3_VISIBILITY_OR_ASSOCIATION_FAILURE"
CLASS_C4 = "C4_SUBSTRATE_OR_PROJECTION"
CLASS_C5_NI = "C5_NON_IDENTIFIABLE"
CLASS_C5_SS = "C5_STRUCTURALLY_SILENT"
CLASS_C5_NS = "C5_NO_SUSTAINED_ANCHOR_OCCUPANCY"
CLASS_C6 = "C6_OTHER"

# The step-0..4 partition. C5_STRUCTURALLY_SILENT is not a step outcome: step 2
# assigns C5_NON_IDENTIFIABLE and tallies the flag (see A11). It IS a terminal
# level for the audit sample's stratification (frozen R10 class order).
CLASS_NAMES = (
    CLASS_C1A,
    CLASS_C1B,
    CLASS_C1C,
    CLASS_C2,
    CLASS_C3,
    CLASS_C4,
    CLASS_C5_NI,
    CLASS_C5_NS,
    CLASS_C6,
)
C5_MEMBERS = (CLASS_C5_NI, CLASS_C5_NS)
# frozen R10 "class" order, used for the audit sample strata
AUDIT_CLASS_ORDER = (
    CLASS_C1A,
    CLASS_C1B,
    CLASS_C1C,
    CLASS_C2,
    CLASS_C3,
    CLASS_C5_NI,
    CLASS_C5_SS,
    CLASS_C5_NS,
)
# the SHORT tokens the frozen R10 "class" string uses, in the same order
AUDIT_CLASS_TOKENS = (
    "C1a",
    "C1b",
    "C1c",
    "C2",
    "C3",
    "C5_NON_IDENTIFIABLE",
    "C5_STRUCTURALLY_SILENT",
    "C5_NO_SUSTAINED_ANCHOR_OCCUPANCY",
)

ST_NEVER_QUERIED = 0
ST_MISS_TOKEN = 1
ST_OUT_OF_DOMAIN = 2
ST_LOW_VISIBILITY = 3
ST_OFF_COMPONENT = 4
ST_ASSOCIATED = 5
STATUS_NAMES = (
    "NEVER_QUERIED",
    "MISS_TOKEN",
    "OUT_OF_DOMAIN",
    "LOW_VISIBILITY",
    "OFF_COMPONENT",
    "ASSOCIATED",
)
# prereg R_grouping_for_the_decision_list: the ONLY statuses that may push a
# window to C2. NEVER_QUERIED belongs to neither group and never contributes.
STATUS_C2_GROUP = (ST_MISS_TOKEN, ST_OUT_OF_DOMAIN)

D2_SURVIVES = "LINEAGE_SURVIVES"
D2_ENDS = "LINEAGE_ENDS"
D2_ENTRY_UNAVAILABLE = "LINEAGE_ENTRY_UNAVAILABLE"
D2_VERDICTS = (D2_SURVIVES, D2_ENDS, D2_ENTRY_UNAVAILABLE)

STATUS_1 = "status_1_ANCHOR_OCCUPANCY_CONSISTENT"
STATUS_2 = "status_2_ANCHOR_OCCUPANCY_INCONSISTENT"
STATUS_3 = "status_3_PARTIALLY_CONFOUNDED"
STATUS_4 = "status_4_NOT_IDENTIFIABLE"
STATUS_5 = "status_5_UNRESOLVED"
STATUS_NAMES_ORDERED = (STATUS_1, STATUS_2, STATUS_3, STATUS_4, STATUS_5)
READING_LABEL_UNRESOLVED = "UNRESOLVED"

# R15 robustness grade (replaces the r4 strict-unanimity rule)
GRADE_UNANIMOUS = "UNANIMOUS"
GRADE_ROBUST = "ROBUST"
GRADE_FRAGILE = "FRAGILE"

# S2 radius ladder. The occupancy RADIUS varies in both decision steps jointly
# (R5: T2b's radius scales identically); the eligibility filter is what
# distinguishes T2 from T2b.
RADII: tuple[tuple[str, float], ...] = (("0.25x", 0.25), ("1x", 1.0), ("2x", 2.0))
RADIUS_INDEX = {name: i for i, (name, _) in enumerate(RADII)}
OCC_T1 = "T1"
OCC_T2_025 = "T2_0.25x_tol"
OCC_T2 = "T2"
OCC_T2_2X = "T2_2x_tol"
OCC_RADIUS = {OCC_T2_025: "0.25x", OCC_T2: "1x", OCC_T2_2X: "2x"}
# frozen (step1, step2) pairs, in the order the prereg enumerates them
S2_PAIRS = (
    (OCC_T2_025, OCC_T1),
    (OCC_T2, OCC_T1),
    (OCC_T2, OCC_T2),
    (OCC_T2_2X, OCC_T2),
)
S2_PRIMARY_INDEX = 1  # "(T2, T1) [primary]"

S1_FROZEN = "frozen_containing_cameras"
S1_PER_FRAME = "per_frame_frustum"

PANELS = ("primary", "secondary")

W_STRATA = ("lt_0.125s", "0.125_to_0.5s", "gt_0.5s")

POOLING_A = "a_pooled"
POOLING_B = "b_unweighted_mean"

# ---------------------------------------------------------------------------
# Disclosed readings: every place the frozen text does not determine a unique
# computation, or contradicts itself. Recorded verbatim in the output so the
# independent reducer can compare its own resolutions against these (prereg
# what_agreement_means_N6 requires the reducer to report ITS underdetermined
# readings in its own words).
# ---------------------------------------------------------------------------

DISCLOSED_READINGS = {
    "A1_S7_singleton_pair_clause": (
        "step 2 requires '|A_t| >= min_occupancy_cameras AND A_t contains a "
        "camera pair whose optical-axis angular separation is >= the frozen "
        "angular floor'. Read literally a SINGLETON A_t contains no pair, so "
        "reading S7 = 1 would be identical to S7 = 2 and could not 'reproduce "
        "revision 1's single-camera rule' as S7_note asserts. RESOLUTION: the "
        "angular-pair clause is applied only when the reading's "
        "min_occupancy_cameras >= 2; at 1 it is waived. This affects ONLY the "
        "S7 = 1 grid rows -- the primary reading (2) is identical either way. "
        "The literal alternative is ALSO computed and published as a "
        "descriptive row outside the 144-cell grid. NOTE: neither reading "
        "exactly reproduces revision 1, whose rule was per-CAMERA sustained "
        "occupancy rather than a per-frame camera set; S7 = 1 here is a "
        "superset of revision 1's C2/C3."
    ),
    "A2_C4_is_unreachable_and_fails_the_run_if_reached": (
        "r4/R9 widens step 0 to a TAUTOLOGY CHECK on BOTH limbs and states 'a "
        "non-empty C4 count fails the run closed'. RESOLUTION: a "
        "sequence-level substrate mismatch raises ContractError and stops the "
        "run before any window is classified; the per-window secondary limb is "
        "still evaluated (z <= 0, or the continuous or round-half-up "
        "projection outside the DECODED raster, in a camera S asserts contains) "
        "and a non-empty C4 raises ContractError as an implementation "
        "inconsistency. Its passing is NOT evidence."
    ),
    "A3_C4_projection_test_uses_both_coordinate_forms": (
        "the secondary limb says 'the anchor projects outside the decoded "
        "raster' without saying whether the test is continuous or rounded. "
        "RESOLUTION: a camera fails projection when z <= 0, OR the continuous "
        "(x, y) falls outside [0, dw-1] x [0, dh-1] of the DECODED raster, OR "
        "the round-half-up pixel falls outside it. The rounded pixel is the "
        "coordinate that indexes the mask; both are recorded per camera."
    ),
    "A4_report_status_precedence": (
        "R_report_status defines six mutually exclusive statuses, which forces "
        "the evaluation order: no track row for (seed, camera) -> "
        "NEVER_QUERIED; else miss token -> MISS_TOKEN; else round-half-up "
        "pixel out of bounds -> OUT_OF_DOMAIN; else v < visibility_threshold "
        "-> LOW_VISIBILITY; else eligible label -> ASSOCIATED, otherwise "
        "OFF_COMPONENT. This differs from build_m1_census.index_tracks, which "
        "discards low-visibility reports BEFORE the bounds test; the two agree "
        "exactly on the ASSOCIATED case, the only case index_tracks retains, "
        "and that agreement is asserted at runtime."
    ),
    "A5_missing_report_row_maps_to_MISS_TOKEN": (
        "MISS_TOKEN is 'a track row exists and this frame's report is a miss "
        "token'. A track row with no report row for a frame is unspecified. "
        "RESOLUTION: it maps to MISS_TOKEN -- the tracker was asked (a row "
        "exists) and did not place the identity here, which is the R_grouping "
        "semantics. The frozen text records that this does not arise; the "
        "realized count is published as "
        "missing_report_rows_mapped_to_MISS_TOKEN."
    ),
    "A6_reading_induced_associated_denominator": (
        "R4 excludes reading-induced ASSOCIATED pairs from BOTH the numerator "
        "and the denominator of step 2's report-status fraction, but does not "
        "say what happens when that empties the denominator. RESOLUTION: an "
        "empty denominator yields fraction 0.0, hence C3 -- the tracker "
        "demonstrably DID place the identity on an eligible component at every "
        "surviving frame, which is the opposite of C2's 'asked and did not "
        "place'. The count is published per window as "
        "reading_induced_associated."
    ),
    "A7_D2_verdict_precedence": (
        "the three D2 verdicts do not partition the case 'entry available in "
        "some cameras of S_q, unavailable in others, none surviving'. "
        "RESOLUTION, in order: (1) no camera of S_q has an associated frame "
        "strictly before the window -> LINEAGE_ENTRY_UNAVAILABLE; (2) some "
        "camera's lineage survives every frame of W(record) -> "
        "LINEAGE_SURVIVES; (3) otherwise -> LINEAGE_ENDS. Total, and agrees "
        "with each literal definition wherever it applies."
    ),
    "A8_split_event_is_undefined_in_the_frozen_text": (
        "the D2 construction requires merge and split events. Merge is defined "
        "by its r2 parenthetical and implemented as written (two or more live "
        "lineages in the same camera and frame accepting the same current "
        "component). SPLIT is not defined. RESOLUTION: a split is recorded "
        "when the previous component X_t overlaps two or more ELIGIBLE "
        "components of frame t+1 each at IoU >= lineage_iou_min -- the only "
        "frozen threshold available. Both counts are descriptive: they never "
        "enter a verdict or a class."
    ),
    "A9_S4_and_S5_scope_for_D2": (
        "S4 (lineage IoU) affects only D2; S5 affects both the class and D2. "
        "RESOLUTION: the class distribution is computed under every S5 level as "
        "the grid requires. D2 is computed one-at-a-time -- the three S4 IoU "
        "levels at the primary eligibility, and the three S5 levels at the "
        "primary IoU. Merge/split counts are recorded at the primary "
        "(eligibility, IoU) reading only. Both axes are descriptive."
    ),
    "A10_T1_does_not_imply_T2_at_every_S2_radius": (
        "T1 places the ROUND-HALF-UP pixel inside an eligible component, so the "
        "nearest eligible pixel is at most sqrt(0.5) px from the CONTINUOUS "
        "projection. That is within tol_c only when tol_c >= sqrt(0.5). At the "
        "r4 0.25x level a small tol_c could break the implication. RESOLUTION: "
        "no T1 => T2 short circuit is taken at any radius; every radius is "
        "evaluated by its own disc test."
    ),
    "A11_C5_STRUCTURALLY_SILENT_is_a_flag_in_the_partition_and_a_class_in_the_sample": (
        "step 2 assigns C5_NON_IDENTIFIABLE and 'additionally tallies' "
        "C5_STRUCTURALLY_SILENT, while classification.binding says C5 "
        "'aggregates' all three names and the frozen R10 audit spec lists "
        "C5_STRUCTURALLY_SILENT as its own TERMINAL class. Summing all three "
        "would double-count and break classification.totality. RESOLUTION: the "
        "step-2 partition assigns C5_NON_IDENTIFIABLE with a "
        "structurally_silent flag; the C5 total used by every decision rule is "
        "C5_NON_IDENTIFIABLE + C5_NO_SUSTAINED_ANCHOR_OCCUPANCY; and the audit "
        "sample's TERMINAL class maps a flagged window to "
        "C5_STRUCTURALLY_SILENT per R10. Under this reading the flag holds for "
        "every C5_NON_IDENTIFIABLE window, so the terminal C5_NON_IDENTIFIABLE "
        "stratum is empty by construction. Both counts are published."
    ),
    "A12_status_1_is_pooling_symmetric_by_construction": (
        "r5/R14_stability_definition makes status_1 require its three "
        "conjuncts under pooling (a) AS WELL AS (b), and states the disclosed "
        "consequence that status_1 therefore has the same truth value under "
        "both and can never contribute to POOLING DISAGREEMENT. RESOLUTION: "
        "implemented exactly so -- status_1 = core(b) AND core(a), and POOLING "
        "DISAGREEMENT is evaluated on status_2 and status_3 only. status_1's "
        "per-pooling cores are published separately so the symmetry is "
        "checkable rather than assumed."
    ),
    "A13_status_3_variation_is_superseded_but_reported": (
        "r5 SUPERSEDES status_3's '<= 20% relative variation across every "
        "reading' clause with the R15 robustness grade, and retains the "
        "(max - min)/max definition 'ONLY as a reported diagnostic'. "
        "RESOLUTION: the clause is NOT applied to status_3; the figure is "
        "computed over the pooled primary-scope C1a COUNT across the 144 grid "
        "cells (taken as 0.0 when max == 0) and published as "
        "C1a_relative_variation_across_grid."
    ),
    "A22_FRAGILE_placement_versus_the_precedence_ladder": (
        "CONTRADICTION IN THE FROZEN TEXT: R15 says 'a FRAGILE grade FORCES "
        "status_4 regardless of the primary reading's label', while "
        "what_the_statuses_mean.precedence puts status_2 and status_1 BEFORE "
        "status_4, and status_4's own definition lists FRAGILE as one of its "
        "disjuncts. Under the first reading FRAGILE dominates everything; "
        "under the second, a holding status_2 or status_1 outranks it. "
        "RESOLUTION: the PRECEDENCE ladder governs -- FRAGILE is evaluated as "
        "a status_4 disjunct in its precedence position -- because status_4's "
        "definition places it there explicitly and because R15's stated "
        "purpose and DISCLOSED_DIRECTION are to make status_4 LESS likely, "
        "which the dominant reading would reverse. The FRAGILE-dominant "
        "alternative is ALSO computed and published as "
        "primary_status_if_FRAGILE_dominates, so the owner can see whether the "
        "two readings differ on this run."
    ),
    "A23_strict_unanimity_counterfactual_is_reported": (
        "required_outputs.sensitivity_table demands 'an explicit statement of "
        "whether the STRICT UNANIMITY rule (the superseded r4 behaviour) would "
        "have produced a different primary status -- and if so, which'. "
        "RESOLUTION: the r4 rule is reproduced as 'reading_label not constant "
        "over the grid forces status_4', applied to the same per-reading and "
        "per-pooling inputs, and published as "
        "primary_status_under_strict_unanimity together with a boolean "
        "differs_from_primary_status."
    ),
    "A14_end_truncation_scope": (
        "status_1 says 'END_TRUNCATED windows excluded from the numerator' and "
        "stratification_N4_R4 says they 'never count toward the "
        "adequacy-status numerator (they remain in the denominator)'. "
        "RESOLUTION: the C1a numerator of the status_1 predicate counts only "
        "windows that are C1a AND NOT END_TRUNCATED; the denominator is the "
        "full window count. Only status_1 uses this adjusted numerator; "
        "status_3's 'C1a >= 0.10' uses the unadjusted C1a. Both counts are "
        "published."
    ),
    "A15_binding_pooling_b_may_be_undefined": (
        "R2 makes pooling (b) -- the unweighted mean over sequences carrying "
        ">= 10 windows -- BINDING for every status predicate, and gives no "
        "fallback when no such sequence exists (which happens for any partial "
        "invocation over small sequences). RESOLUTION: when (b) is undefined "
        "the primary status is emitted as null with an explicit note; it is "
        "NEVER silently replaced by pooling (a). Every input to the rules is "
        "still published so the owner can complete the pooled run and decide."
    ),
    "A16_status_scope_is_the_invocation": (
        "the decision rules bind to the 597 primary-scope windows. "
        "RESOLUTION: the status is computed only for --panel primary and only "
        "over the sequences present in THIS invocation; the scope is published "
        "as measurement_closure.scope so a partial run is never mistaken for "
        "the pooled verdict. For --panel secondary the status is null, per "
        "evidence_boundary N5 (the secondary panel is a SUPERSET of primary "
        "rows, not an independent sample, and reports no coverage-derived "
        "statistic)."
    ),
    "A17_T2_disc_radius_is_ceil_tol_plus_one": (
        "the frozen text mandates (N3) a disc test of radius ceil(tol_c) "
        "around the projected pixel rather than a full-image distance "
        "transform. The disc is centred on the ROUNDED pixel while the "
        "predicate measures distance from the CONTINUOUS projection, so a "
        "radius of exactly ceil(tol_c) can miss an eligible pixel that is "
        "within tol_c of the continuous point but up to sqrt(0.5) px further "
        "from the rounded one. RESOLUTION: the search box is built at radius "
        "ceil(tol_c) + 1 -- a strict superset of the mandated disc, still "
        "local, no distance transform -- and the inclusive '<= tol_c' "
        "comparison is then evaluated EXACTLY against the continuous point. "
        "This widens the search box by one pixel, never the predicate."
    ),
    "A18_tolerance_and_area_medians_cover_the_decoded_pass": (
        "tolerance_B9 asks for the median eligible-component area and the "
        "median foreground area fraction per (camera, frame). RESOLUTION: both "
        "are taken over exactly the (camera, frame) pairs this pass decoded -- "
        "every frame of the realized census window for every camera in the "
        "union of the windows' applicable sets -- and the decoded pair count "
        "is published so the base is explicit. Cameras in no window's "
        "applicable set are never decoded and contribute to nothing."
    ),
    "A19_audit_sample_rng_stream": (
        "RESOLVED BY r5 token_repairs: 'the RNG is constructed FRESH PER "
        "STRATUM (a new default_rng(20260814) for each stratum), not shared "
        "across strata'. Implemented exactly so; the earlier shared-stream "
        "reading is abandoned. Tercile boundaries are NEAREST-RANK (the "
        "ceil(n/3)-th and ceil(2n/3)-th of the sorted pooled anchor_staleness "
        "values, no interpolation) with ties to the LOWER tercile, as frozen."
    ),
    "A21_W_tautology_checks_are_recorded_not_operative": (
        "r5 stopping_conditions drops 'EMPTY W' and 'min/max disagree' as "
        "operative stop conditions and retains them as labelled tautology "
        "checks (R6), while adding 'ANY VIOLATION OF "
        "operative_fail_closed_guard_R6'. RESOLUTION: the two demoted checks "
        "are EVALUATED and RECORDED per window but do not raise; the R6 guard "
        "raises. An empty W would still stop the run indirectly, because the "
        "decision list then returns C6 and a non-empty C6 fails closed."
    ),
}


# ---------------------------------------------------------------------------
# Prereg loading (fail-closed; every scientific constant comes from here)
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = (
    "schema_version",
    "revision",
    "scientific_question",
    "disclosed_prior_knowledge",
    "evidence_boundary",
    "units",
    "applicable_camera_set",
    "anchor",
    "per_pair_tests",
    "substrate_check",
    "identity_evidence",
    "classification",
    "frozen_constants",
    "sensitivity_readings",
    "required_outputs",
    "what_the_statuses_mean",
    "what_this_diagnostic_may_not_do",
    "execution_and_verification",
)

_REQUIRED_CONSTANTS = (
    "occupancy_frame_fraction_min",
    "min_occupancy_cameras",
    "report_majority_fraction",
    "lineage_iou_min",
    "component_min_px",
    "mask_binarize_threshold",
    "visibility_threshold",
    "ltp_on_eligible_component_min",
    "pixel_rounding",
    "distance_metric",
)

_REQUIRED_SENSITIVITY = (
    "decision_relevant_grid",
    "S2_occupancy_tolerance",
    "S3_occupancy_fraction",
    "S5_component_eligibility_px",
    "S7_min_occupancy_cameras",
    "declared_decision_irrelevant_by_construction",
)

_REQUIRED_PAIR_TESTS = (
    "T1_strict_anchor_occupancy",
    "T2_ball_anchor_occupancy",
    "T2b_subthreshold_anchor_occupancy",
    "T3_any_foreground",
    "R_report_status",
    "ASSOCIATED_handling_r4",
    "operative_fail_closed_guard_R6",
    "R_grouping_for_the_decision_list",
)

_REQUIRED_OUTPUT_KEYS = (
    "per_sequence_and_pooled",
    "anchor_quality_B3",
    "query_gap_B4",
    "tolerance_B9",
    "stratification_N4_R4",
    "pooling_B7_R2",
    "relationships",
    "sensitivity_table",
    "audit_sample_B8",
    "evidence_grading",
)

_REQUIRED_AUDIT_KEYS = (
    "population",
    "class",
    "terciles",
    "strata_order",
    "within_stratum_order",
    "allocation",
    "emitted_fields",
    "supersession_DISCLOSED",
    "bridged_frame_bias_DISCLOSED",
    "scope",
)

_REQUIRED_R4_AMENDMENTS = (
    "R2_binding_pooling_and_precedence",
    "R3_reading_label_device",
    "R4_W_is_reading_invariant",
    "R5_T2b_tolerance_under_S2",
    "R6_real_census_reproduction_guard",
    "R7_tolerance_computed_first_and_a_reachable_adequacy_regime",
    "R8_status_2_symmetry",
    "R9_decode_scope_and_dead_clauses",
    "R10_audit_sample_frozen",
)

_REQUIRED_R5_AMENDMENTS = (
    "R14_binding_reading",
    "R14_stability_definition",
    "R15_OWNER_robustness_grade_replaces_unanimity",
    "token_repairs",
)


@dataclasses.dataclass(frozen=True)
class Prereg:
    """Every scientific constant, read from the frozen file at runtime."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    occupancy_frame_fraction_min: float
    min_occupancy_cameras: int
    report_majority_fraction: float
    lineage_iou_min: float
    component_min_px: int
    mask_binarize_threshold: int
    visibility_threshold: float
    ltp_on_eligible_component_min: float
    s2_pairs: tuple[tuple[str, str], ...]
    s3_occupancy_fraction: tuple[float, ...]
    s5_component_px: tuple[int, ...]
    s7_min_cameras: tuple[int, ...]
    s4_lineage_iou: tuple[float, ...]
    s6_report_majority: tuple[float, ...]
    robustness_supermajority: float
    status_1_pooled_c1a_min: float = 0.80
    status_1_sequence_c1a_min: float = 0.60
    status_1_c23_max: float = 0.10
    status_2_c23_min: float = 0.50
    status_3_c1a_min: float = 0.10
    status_3_variation_max: float = 0.20
    status_4_c5_min: float = 0.50
    sequence_window_floor: int = 10

    @property
    def primary_level_index(self) -> int:
        return self.s5_component_px.index(self.component_min_px)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _numeric_members(value: Any, where: str, caster) -> tuple:
    """Numeric members of a prereg list, ignoring trailing prose members.

    The ``declared_decision_irrelevant_by_construction`` lists append an
    explanatory string to the numeric ladder; the numeric prefix is the ladder.
    """

    _require(isinstance(value, list) and value, f"prereg {where} must be a non-empty list")
    out = []
    for item in value:
        if isinstance(item, str):
            continue
        _require(
            isinstance(item, (int, float)) and not isinstance(item, bool),
            f"prereg {where} member {item!r} is neither numeric nor prose",
        )
        out.append(caster(item))
    _require(bool(out), f"prereg {where} has no numeric members")
    return tuple(out)


def load_prereg(prereg_path: Path) -> Prereg:
    """Load, verify and freeze the diagnostic prereg. Fails closed."""

    prereg_path = Path(prereg_path)
    if not prereg_path.is_file():
        raise ContractError(f"prereg missing: {prereg_path}")
    payload = json.loads(prereg_path.read_text(encoding="utf-8"))

    if payload.get("revision") != PREREG_REVISION_REQUIRED:
        raise ContractError(
            f"prereg revision {payload.get('revision')!r} != required "
            f"{PREREG_REVISION_REQUIRED} (the r4 re-review-repaired revision)"
        )
    for key in _REQUIRED_TOP_KEYS:
        _require(key in payload, f"prereg missing required key {key!r}")
    _require(
        "amendment_2026_08_14_r4_rereview" in payload,
        "prereg missing amendment_2026_08_14_r4_rereview",
    )
    r4 = payload["amendment_2026_08_14_r4_rereview"]
    for key in _REQUIRED_R4_AMENDMENTS:
        _require(key in r4, f"prereg r4 amendment missing {key!r}")
    _require(
        "amendment_2026_08_14_r5_signoff" in payload,
        "prereg missing amendment_2026_08_14_r5_signoff",
    )
    r5 = payload["amendment_2026_08_14_r5_signoff"]
    for key in _REQUIRED_R5_AMENDMENTS:
        _require(key in r5, f"prereg r5 amendment missing {key!r}")
    r15 = r5["R15_OWNER_robustness_grade_replaces_unanimity"]
    for key in ("defect", "repair", "robustness_supermajority", "DISCLOSED_DIRECTION"):
        _require(key in r15, f"prereg R15 block missing {key!r}")
    supermajority = r15["robustness_supermajority"]
    _require(
        isinstance(supermajority, (int, float))
        and not isinstance(supermajority, bool)
        and 0.0 < float(supermajority) <= 1.0,
        f"prereg R15 robustness_supermajority {supermajority!r} is not a fraction in (0, 1]",
    )
    supermajority = float(supermajority)
    _require(
        str(len(S2_PAIRS) * 3 * 3 * 4) in str(payload["required_outputs"]["sensitivity_table"]),
        "prereg required_outputs.sensitivity_table does not state the realized "
        f"{len(S2_PAIRS) * 3 * 3 * 4}-cell grid",
    )

    constants = payload["frozen_constants"]
    _require(isinstance(constants, dict), "prereg frozen_constants must be an object")
    for key in _REQUIRED_CONSTANTS:
        _require(key in constants, f"prereg frozen_constants missing {key!r}")

    pair_tests = payload["per_pair_tests"]
    for key in _REQUIRED_PAIR_TESTS:
        _require(key in pair_tests, f"prereg per_pair_tests missing {key!r}")

    sensitivity = payload["sensitivity_readings"]
    for key in _REQUIRED_SENSITIVITY:
        _require(key in sensitivity, f"prereg sensitivity_readings missing {key!r}")

    statuses = payload["what_the_statuses_mean"]
    for key in (*STATUS_NAMES_ORDERED, "precedence", "scope_B8", "binding_pooling_R2"):
        _require(key in statuses, f"prereg what_the_statuses_mean missing {key!r}")

    outputs = payload["required_outputs"]
    for key in _REQUIRED_OUTPUT_KEYS:
        _require(key in outputs, f"prereg required_outputs missing {key!r}")

    units = payload["units"]
    _require("temporal_unit_W" in units, "prereg units missing temporal_unit_W")
    _require(
        "W(record)" in str(units["temporal_unit_W"]),
        "prereg units.temporal_unit_W does not define W(record)",
    )
    _require(
        "queried_subset" in payload["applicable_camera_set"],
        "prereg applicable_camera_set missing queried_subset (S_q)",
    )

    classification = payload["classification"]
    for key in ("ordered_decision_list", "totality", "binding"):
        _require(key in classification, f"prereg classification missing {key!r}")
    decision_text = "\n".join(str(step) for step in classification["ordered_decision_list"])
    _require(
        len(classification["ordered_decision_list"]) == 5,
        "prereg ordered_decision_list must have exactly the five steps 0..4, got "
        f"{len(classification['ordered_decision_list'])}",
    )
    for index in range(5):
        _require(
            f"step {index}" in decision_text,
            f"prereg ordered_decision_list does not mention 'step {index}'",
        )
    for name in (*CLASS_NAMES, CLASS_C5_SS):
        _require(
            name in decision_text,
            f"prereg ordered_decision_list does not name the class {name!r}",
        )

    status_text = str(pair_tests["R_report_status"])
    for name in STATUS_NAMES:
        _require(
            name in status_text,
            f"prereg per_pair_tests.R_report_status does not name the status {name!r}",
        )
    grouping_text = str(pair_tests["R_grouping_for_the_decision_list"])
    _require(
        "NEVER_QUERIED" in grouping_text and "NEVER contributes to C2" in grouping_text,
        "prereg R_grouping_for_the_decision_list does not bar NEVER_QUERIED from C2",
    )
    guard_text = str(pair_tests["operative_fail_closed_guard_R6"])
    _require(
        "bridged_associated_frames" in guard_text and "bridged_interruptions" in guard_text,
        "prereg operative_fail_closed_guard_R6 does not state the census-reproduction guard",
    )

    identity = payload["identity_evidence"]
    _require(
        "secondary_instrument_D2_deterministic_component_lineage" in identity,
        "prereg identity_evidence missing the D2 block",
    )
    _require(
        "mandatory_successor_B8_FROZEN_R10" in identity,
        "prereg identity_evidence missing mandatory_successor_B8_FROZEN_R10",
    )
    audit = identity["mandatory_successor_B8_FROZEN_R10"]
    for key in _REQUIRED_AUDIT_KEYS:
        _require(key in audit, f"prereg frozen audit-sample spec missing {key!r}")
    # The frozen audit spec names the classes in SHORT form; verify both
    # membership and the ORDER, since the strata order is frozen by it.
    audit_class_text = str(audit["class"])
    positions = []
    for token in AUDIT_CLASS_TOKENS:
        _require(
            token in audit_class_text,
            f"prereg audit-sample class order does not name {token!r}",
        )
        positions.append(audit_class_text.index(token))
    _require(
        positions == sorted(positions),
        "prereg audit-sample class order is not the order this evaluator strata by: "
        f"{list(AUDIT_CLASS_TOKENS)}",
    )
    _require(
        str(AUDIT_SAMPLE_SEED) in str(audit["within_stratum_order"]),
        f"prereg audit-sample within_stratum_order does not fix the seed {AUDIT_SAMPLE_SEED}",
    )
    _require(
        str(AUDIT_SAMPLE_MIN) in str(audit["allocation"]),
        f"prereg audit-sample allocation does not fix the minimum {AUDIT_SAMPLE_MIN}",
    )

    d2 = identity["secondary_instrument_D2_deterministic_component_lineage"]
    for key in ("construction", "verdicts", "role", "independence_disclosure"):
        _require(key in d2, f"prereg D2 block missing {key!r}")
    for name in D2_VERDICTS:
        _require(name in str(d2["verdicts"]), f"prereg D2 verdicts does not name {name!r}")

    substrate = payload["substrate_check"]
    for key in ("rule", "also_recorded"):
        _require(key in substrate, f"prereg substrate_check missing {key!r}")

    def _scalar(key: str, caster):
        value = constants[key]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"prereg frozen_constants.{key} must be numeric, got {value!r}",
        )
        return caster(value)

    occupancy_frame_fraction_min = _scalar("occupancy_frame_fraction_min", float)
    min_occupancy_cameras = _scalar("min_occupancy_cameras", int)
    report_majority_fraction = _scalar("report_majority_fraction", float)
    lineage_iou_min = _scalar("lineage_iou_min", float)
    component_min_px = _scalar("component_min_px", int)
    mask_binarize_threshold = _scalar("mask_binarize_threshold", int)
    visibility_threshold = _scalar("visibility_threshold", float)
    ltp_on_eligible_component_min = _scalar("ltp_on_eligible_component_min", float)

    # ---- commensurability with the instrument under test ------------------
    _require(
        component_min_px == MIN_COMPONENT_PX,
        f"prereg component_min_px={component_min_px} != build_m1_census."
        f"MIN_COMPONENT_PX={MIN_COMPONENT_PX}; the reused component primitive "
        "cannot honour a different eligibility floor",
    )
    _require(
        mask_binarize_threshold == MASK_THRESHOLD,
        f"prereg mask_binarize_threshold={mask_binarize_threshold} != "
        f"build_m1_census.MASK_THRESHOLD={MASK_THRESHOLD}",
    )
    _require(
        visibility_threshold == VIS_THRESHOLD,
        f"prereg visibility_threshold={visibility_threshold} != "
        f"build_m1_census.VIS_THRESHOLD={VIS_THRESHOLD}",
    )
    _require(
        "round-half-up" in str(constants["pixel_rounding"]),
        f"prereg pixel_rounding {constants['pixel_rounding']!r} is not round-half-up",
    )
    _require(
        round_half_up(0.5) == 1 and round_half_up(-0.5) == 0 and round_half_up(12.49) == 12,
        "build_m1_census.round_half_up does not implement floor(x + 0.5)",
    )
    _require(
        "Euclidean" in str(constants["distance_metric"])
        and "<=" in str(constants["distance_metric"]),
        f"prereg distance_metric {constants['distance_metric']!r} is not the "
        "inclusive Euclidean rule",
    )

    # ---- sensitivity readings --------------------------------------------
    s2_text = str(sensitivity["S2_occupancy_tolerance"])
    for token in ("(0.25x-tol T2, T1)", "(T2, T1)", "(T2, T2)", "(2x-tol T2, T2)"):
        _require(
            token in s2_text,
            f"prereg S2_occupancy_tolerance {s2_text!r} does not enumerate {token}",
        )
    _require(
        "[primary]" in s2_text and s2_text.index("(T2, T1)") < s2_text.index("[primary]"),
        "prereg S2_occupancy_tolerance does not mark (T2, T1) as the primary pair",
    )
    grid_text = str(sensitivity["decision_relevant_grid"])
    _require(
        "S2 x S3 x S5 x S7" in grid_text,
        f"prereg decision_relevant_grid {grid_text!r} is not the S2 x S3 x S5 x S7 grid",
    )

    s3 = _numeric_members(sensitivity["S3_occupancy_fraction"], "S3_occupancy_fraction", float)
    s5 = _numeric_members(
        sensitivity["S5_component_eligibility_px"], "S5_component_eligibility_px", int
    )
    s7 = _numeric_members(
        sensitivity["S7_min_occupancy_cameras"], "S7_min_occupancy_cameras", int
    )
    irrelevant = sensitivity["declared_decision_irrelevant_by_construction"]
    for key in ("S1_camera_set", "S4_lineage_iou", "S6_report_majority_fraction"):
        _require(
            key in irrelevant,
            f"prereg declared_decision_irrelevant_by_construction missing {key!r}",
        )
    s4 = _numeric_members(irrelevant["S4_lineage_iou"], "S4_lineage_iou", float)
    s6 = _numeric_members(
        irrelevant["S6_report_majority_fraction"], "S6_report_majority_fraction", float
    )

    expected_grid = len(S2_PAIRS) * len(s3) * len(s5) * len(s7)
    _require(
        str(expected_grid) in grid_text,
        f"prereg decision_relevant_grid does not state the realized combination "
        f"count {expected_grid}",
    )

    _require(
        occupancy_frame_fraction_min in s3,
        f"prereg S3 {s3} does not contain the primary occupancy_frame_fraction_min "
        f"{occupancy_frame_fraction_min}",
    )
    _require(
        lineage_iou_min in s4,
        f"prereg S4 {s4} does not contain the primary lineage_iou_min {lineage_iou_min}",
    )
    _require(
        component_min_px in s5,
        f"prereg S5 {s5} does not contain the primary component_min_px {component_min_px}",
    )
    _require(
        report_majority_fraction in s6,
        f"prereg S6 {s6} does not contain the primary report_majority_fraction "
        f"{report_majority_fraction}",
    )
    _require(
        min_occupancy_cameras in s7,
        f"prereg S7 {s7} does not contain the primary min_occupancy_cameras "
        f"{min_occupancy_cameras}",
    )
    _require(all(level > 0 for level in s5), f"prereg S5 levels must be positive, got {s5}")
    _require(all(count >= 1 for count in s7), f"prereg S7 counts must be >= 1, got {s7}")

    return Prereg(
        path=prereg_path,
        sha256=sha256_file(prereg_path),
        payload=payload,
        occupancy_frame_fraction_min=occupancy_frame_fraction_min,
        min_occupancy_cameras=min_occupancy_cameras,
        report_majority_fraction=report_majority_fraction,
        lineage_iou_min=lineage_iou_min,
        component_min_px=component_min_px,
        mask_binarize_threshold=mask_binarize_threshold,
        visibility_threshold=visibility_threshold,
        ltp_on_eligible_component_min=ltp_on_eligible_component_min,
        s2_pairs=S2_PAIRS,
        s3_occupancy_fraction=s3,
        s5_component_px=s5,
        s7_min_cameras=s7,
        s4_lineage_iou=s4,
        s6_report_majority=s6,
        robustness_supermajority=supermajority,
    )


# ---------------------------------------------------------------------------
# Reading (one point of the decision-relevant grid)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Reading:
    step1: str
    step2: str
    occupancy_fraction: float
    component_px: int
    min_cameras: int
    report_majority: float
    camera_set: str = S1_FROZEN
    literal_pair_clause: bool = False

    def label(self) -> str:
        base = (
            f"S2=({self.step1},{self.step2});S3={self.occupancy_fraction};"
            f"S5={self.component_px};S7={self.min_cameras}"
        )
        extras = []
        if self.camera_set != S1_FROZEN:
            extras.append(f"S1={self.camera_set}")
        if self.report_majority is not None:
            extras.append(f"S6={self.report_majority}")
        if self.literal_pair_clause:
            extras.append("pair_clause=literal")
        return base + "".join(f";{item}" for item in extras)

    def as_dict(self) -> dict[str, Any]:
        return {
            "S2_step1": self.step1,
            "S2_step2": self.step2,
            "S3_occupancy_fraction": self.occupancy_fraction,
            "S5_component_eligibility_px": self.component_px,
            "S7_min_occupancy_cameras": self.min_cameras,
            "S6_report_majority_fraction": self.report_majority,
            "S1_camera_set": self.camera_set,
            "literal_pair_clause": self.literal_pair_clause,
        }


def primary_reading(prereg: Prereg) -> Reading:
    step1, step2 = prereg.s2_pairs[S2_PRIMARY_INDEX]
    return Reading(
        step1=step1,
        step2=step2,
        occupancy_fraction=prereg.occupancy_frame_fraction_min,
        component_px=prereg.component_min_px,
        min_cameras=prereg.min_occupancy_cameras,
        report_majority=prereg.report_majority_fraction,
    )


def decision_grid(prereg: Prereg) -> list[Reading]:
    """The frozen S2 x S3 x S5 x S7 decision-relevant grid, in a fixed order."""

    grid: list[Reading] = []
    for step1, step2 in prereg.s2_pairs:
        for fraction in prereg.s3_occupancy_fraction:
            for level in prereg.s5_component_px:
                for minimum in prereg.s7_min_cameras:
                    grid.append(
                        Reading(
                            step1=step1,
                            step2=step2,
                            occupancy_fraction=fraction,
                            component_px=level,
                            min_cameras=minimum,
                            report_majority=prereg.report_majority_fraction,
                        )
                    )
    return grid


def descriptive_readings(prereg: Prereg) -> list[Reading]:
    """S1 / S6 / literal-pair-clause readings, declared decision-irrelevant."""

    base = primary_reading(prereg)
    rows = [dataclasses.replace(base, camera_set=S1_PER_FRAME)]
    rows += [dataclasses.replace(base, report_majority=v) for v in prereg.s6_report_majority]
    rows += [
        dataclasses.replace(base, min_cameras=v, literal_pair_clause=True)
        for v in prereg.s7_min_cameras
        if v == 1
    ]
    unique: list[Reading] = []
    seen = {base.label()}
    for reading in rows:
        if reading.label() in seen:
            continue
        seen.add(reading.label())
        unique.append(reading)
    return unique


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def anchor_projection(camera: CameraModel, point: np.ndarray) -> tuple[float, float, float]:
    """Project ``point`` EXACTLY as build_m1_census.frustum_contains does."""

    cam_point = camera.w2c[:3, :3] @ point + camera.w2c[:3, 3]
    z = float(cam_point[2])
    if z <= 0.0:
        return math.nan, math.nan, z
    uv = camera.K @ cam_point
    return float(uv[0] / uv[2]), float(uv[1] / uv[2]), z


def anchor_tolerance(camera: CameraModel, z: float, r_site_census: float) -> float:
    """tol_c = r_site_census * fl_x_c / z_c (prereg anchor.tolerance)."""

    return float(r_site_census) * float(camera.K[0, 0]) / float(z)


class Disc:
    """The frozen disc test (N3): a LOCAL box around the projected pixel with
    an EXACT inclusive Euclidean membership mask against the CONTINUOUS point.

    The box half-width is ``ceil(tol) + 1`` rather than the mandated
    ``ceil(tol)`` (disclosed reading A17): the disc is centred on the rounded
    pixel while the predicate measures from the continuous projection, and the
    extra pixel makes the box a provable superset of every pixel within
    ``tol``. No full-image distance transform is used anywhere.
    """

    __slots__ = ("r0", "r1", "c0", "c1", "within", "empty")

    def __init__(self, x: float, y: float, col: int, row: int, tol: float, height: int, width: int):
        if not (tol >= 0.0):
            raise ContractError(f"non-finite or negative anchor tolerance {tol}")
        radius = int(math.ceil(tol)) + 1
        self.r0, self.r1 = max(0, row - radius), min(height, row + radius + 1)
        self.c0, self.c1 = max(0, col - radius), min(width, col + radius + 1)
        if self.r0 >= self.r1 or self.c0 >= self.c1:
            self.within = np.zeros((0, 0), dtype=bool)
            self.empty = True
            return
        rows = np.arange(self.r0, self.r1, dtype=np.float64) - y
        cols = np.arange(self.c0, self.c1, dtype=np.float64) - x
        self.within = (rows[:, None] ** 2 + cols[None, :] ** 2) <= tol * tol
        self.empty = not bool(self.within.any())

    def hits(self, labels: np.ndarray, lookup: np.ndarray | None) -> bool:
        """True iff some pixel of the (looked-up) foreground set lies inside.

        ``lookup is None`` selects ANY foreground label (the T2b form, which
        does not apply the component_min_px filter).
        """

        if self.empty:
            return False
        block = labels[self.r0 : self.r1, self.c0 : self.c1]
        selected = (block > 0) if lookup is None else lookup[block]
        return bool((selected & self.within).any())


# ---------------------------------------------------------------------------
# Mask frame
# ---------------------------------------------------------------------------


class MaskFrame:
    """One decoded mask, labelled ONCE by the reused census primitive.

    ``load_component_labels`` returns the eligible set at the frozen
    ``MIN_COMPONENT_PX``. The other S5 levels are derived from component AREAS
    recovered from the SAME labelling via ``np.bincount`` -- one decode, one
    connected-component pass, three area floors -- and the derived primary
    level is asserted identical to the primitive's own answer.
    """

    __slots__ = ("labels", "areas", "shape", "lookup", "eligible", "any_eligible", "n_pixels")

    def __init__(self, mask_path: Path, levels: Sequence[int], primary_index: int):
        labels, eligible_primary = load_component_labels(mask_path)
        self.labels = labels
        self.shape = labels.shape
        self.n_pixels = int(labels.size)
        self.areas = np.bincount(labels.ravel())
        self.lookup: list[np.ndarray] = []
        self.eligible: list[frozenset[int]] = []
        self.any_eligible: list[bool] = []
        for level in levels:
            lut = self.areas >= int(level)
            lut[0] = False
            self.lookup.append(lut)
            members = frozenset(int(v) for v in np.nonzero(lut)[0])
            self.eligible.append(members)
            self.any_eligible.append(bool(members))
        if self.eligible[primary_index] != frozenset(int(v) for v in eligible_primary):
            raise ContractError(
                f"{mask_path}: area-derived eligibility at the primary level "
                "disagrees with build_m1_census.load_component_labels; the S5 "
                "derivation is not commensurate with the instrument"
            )

    @property
    def foreground_fraction(self) -> float:
        background = int(self.areas[0]) if len(self.areas) else self.n_pixels
        return (self.n_pixels - background) / self.n_pixels if self.n_pixels else 0.0


# ---------------------------------------------------------------------------
# Raw report index (preserves every R_report_status distinction)
# ---------------------------------------------------------------------------

PRE_MISS = 0
PRE_OUT_OF_DOMAIN = 1
PRE_LOW_VISIBILITY = 2
PRE_NEEDS_MASK = 3


@dataclasses.dataclass(frozen=True)
class RawReports:
    """(seed, camera, frame) -> (precursor, col, row), plus the track-row set.

    ``ReportIndex`` keeps only the associated-candidate pixels, which cannot
    distinguish NEVER_QUERIED from MISS_TOKEN from OUT_OF_DOMAIN from
    LOW_VISIBILITY. This index preserves every precursor while using the SAME
    thresholds and the SAME round-half-up rule, and its NEEDS_MASK cells are
    asserted identical to ``ReportIndex.pixels`` for every in-scope seed.
    """

    table: dict[tuple[int, int, int], tuple[int, int, int]]
    track_rows: set[tuple[int, int]]
    candidate_frames: dict[tuple[int, int], tuple[int, ...]]
    out_of_domain_count: int


def build_raw_reports(
    artifact: dict[str, Any],
    scene: SceneBundle,
    index: ReportIndex,
    seeds_in_scope: Iterable[int],
    prereg: Prereg,
) -> RawReports:
    scope = {int(seed) for seed in seeds_in_scope}
    table: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    track_rows: set[tuple[int, int]] = set()
    candidates: dict[tuple[int, int], list[int]] = {}
    out_of_domain = 0
    for track in artifact["tracks"]:
        seed_id = int(track["seed_id"])
        camera_id = int(track["camera_id"])
        if camera_id not in scene.cameras or seed_id not in scope:
            continue
        track_rows.add((seed_id, camera_id))
        camera = scene.cameras[camera_id]
        for report in track["reports"]:
            # Same frame key expression as build_m1_census.index_tracks.
            frame = int(round(float(report["frame"])))
            key = (seed_id, camera_id, frame)
            if report.get("is_miss", False):
                table[key] = (PRE_MISS, -1, -1)
                continue
            col = round_half_up(float(report["x"]))
            row = round_half_up(float(report["y"]))
            # A4: bounds precede visibility.
            if not (0 <= col <= camera.width - 1 and 0 <= row <= camera.height - 1):
                table[key] = (PRE_OUT_OF_DOMAIN, -1, -1)
                out_of_domain += 1
                continue
            if float(report["v"]) < prereg.visibility_threshold:
                table[key] = (PRE_LOW_VISIBILITY, -1, -1)
                continue
            table[key] = (PRE_NEEDS_MASK, col, row)
            candidates.setdefault((seed_id, camera_id), []).append(frame)

    # Commensurability assertion against the instrument's own index.
    for key, pixel in index.pixels.items():
        if key[0] not in scope:
            continue
        entry = table.get(key)
        if entry is None or entry[0] != PRE_NEEDS_MASK or (entry[1], entry[2]) != pixel:
            raise ContractError(
                "raw report index disagrees with build_m1_census.index_tracks at "
                f"{key}: {entry!r} vs {pixel!r}"
            )
    for key, entry in table.items():
        if entry[0] == PRE_NEEDS_MASK and index.pixels.get(key) != (entry[1], entry[2]):
            raise ContractError(
                f"raw report index has an association candidate at {key} that "
                "build_m1_census.index_tracks does not"
            )

    return RawReports(
        table=table,
        track_rows=track_rows,
        candidate_frames={key: tuple(sorted(set(v))) for key, v in candidates.items()},
        out_of_domain_count=out_of_domain,
    )


# ---------------------------------------------------------------------------
# Substrate check (a mismatch STOPS the run)
# ---------------------------------------------------------------------------


def _decode_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image

    if not path.is_file():
        raise ContractError(f"substrate_check: file missing: {path}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.ndim != 2:
        raise ContractError(f"substrate_check: {path} did not decode to a 2-D raster")
    return int(array.shape[1]), int(array.shape[0])


def substrate_check(
    scene: SceneBundle, scene_dir: Path
) -> tuple[dict[str, Any], dict[int, tuple[int, int]]]:
    """Decode one mask and one undistorted frame per TRAINING camera and
    compare against that camera's declared (w, h). Any mismatch fails closed."""

    first_frame = int(scene.frame_indices[0])
    cameras: dict[str, Any] = {}
    mismatches: list[str] = []
    decoded: dict[int, tuple[int, int]] = {}
    for camera_id in scene.tracking_ids:
        camera = scene.cameras[camera_id]
        mask_path = scene.mask_path(camera_id, first_frame)
        mask_w, mask_h = _decode_dimensions(mask_path)
        entry: dict[str, Any] = {
            "declared": [int(camera.width), int(camera.height)],
            "mask_path": str(mask_path),
            "mask_decoded": [mask_w, mask_h],
            "mask_matches": bool(mask_w == camera.width and mask_h == camera.height),
        }
        if not entry["mask_matches"]:
            mismatches.append(
                f"camera {camera_id} mask decoded {mask_w}x{mask_h} != declared "
                f"{camera.width}x{camera.height}"
            )
        image_path = scene.image_path(camera_id, first_frame)
        entry["image_path"] = str(image_path)
        if image_path.is_file():
            image_w, image_h = _decode_dimensions(image_path)
            entry["image_decoded"] = [image_w, image_h]
            entry["image_matches"] = bool(image_w == camera.width and image_h == camera.height)
            if not entry["image_matches"]:
                mismatches.append(
                    f"camera {camera_id} undistorted frame decoded {image_w}x{image_h} "
                    f"!= declared {camera.width}x{camera.height}"
                )
        else:
            entry["image_decoded"] = None
            entry["image_matches"] = None
            mismatches.append(f"camera {camera_id} undistorted frame missing: {image_path}")
        cameras[str(camera_id)] = entry
        decoded[camera_id] = (mask_w, mask_h)

    provenance: dict[str, Any] = {"path": None, "sha256": None, "train_archive_path": None}
    for candidate in (scene_dir / PROVENANCE_FILENAME, scene_dir.parent / PROVENANCE_FILENAME):
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            provenance = {
                "path": str(candidate),
                "sha256": sha256_file(candidate),
                "train_archive_path": (
                    payload.get("archive_selection", {}).get("train", {}).get("archive_path")
                ),
            }
            break

    report = {
        "frame_checked": first_frame,
        "cameras": cameras,
        "mismatches": mismatches,
        "passed": not mismatches,
        "conversion_provenance": provenance,
    }
    if mismatches:
        raise ContractError(
            f"substrate_check FAILED CLOSED for {scene_dir.name}: " + "; ".join(mismatches)
        )
    return report, decoded


# ---------------------------------------------------------------------------
# Window state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WindowEval:
    sequence: str
    seed_id: int
    first_frame: int
    last_frame: int
    record_n_frames: int
    bridged_interruptions: int
    ltp_frame: int
    anchor: np.ndarray
    frozen_S: tuple[int, ...]
    queried_S: tuple[int, ...]
    per_frame_S: tuple[int, ...]
    cams: tuple[int, ...]
    interval: tuple[int, ...]
    geom: dict[int, dict[str, Any]]
    projection_failures: list[dict[str, Any]]
    pair_confirmable: np.ndarray
    end_truncated: bool
    fps: float
    anchor_quality: dict[str, Any]
    # evidence over the CLOSED interval; W is derived after the sweep
    t1: np.ndarray  # (levels, cams, interval)
    t2: np.ndarray  # (radii, levels, cams, interval)
    t2b: np.ndarray  # (radii, cams, interval)
    t3: np.ndarray  # (levels, cams, interval)
    status: np.ndarray  # (levels, cams, interval) int8
    evaluated: np.ndarray
    ltp_on_eligible: np.ndarray  # (levels, cams) bool at ltp_frame
    ltp_frame_evaluated: bool = False
    w_mask: np.ndarray | None = None
    d2: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.sequence, self.seed_id, self.first_frame)

    def camera_index(self, camera_id: int) -> int:
        return self.cams.index(camera_id)

    def member_mask(self, camera_set: str) -> np.ndarray:
        source = set(self.frozen_S if camera_set == S1_FROZEN else self.per_frame_S)
        return np.array([c in source for c in self.cams], dtype=bool)

    def occupancy(self, which: str, level_index: int) -> np.ndarray:
        if which == OCC_T1:
            return self.t1[level_index]
        return self.t2[RADIUS_INDEX[OCC_RADIUS[which]], level_index]

    def subthreshold(self, step1: str) -> np.ndarray:
        # R5: T2b's radius tracks S2 exactly; eligibility is never applied.
        return self.t2b[RADIUS_INDEX[OCC_RADIUS[step1]]]

    @property
    def w_frames(self) -> np.ndarray:
        if self.w_mask is None:
            raise ContractError("W(record) has not been derived yet")
        return self.w_mask


# ---------------------------------------------------------------------------
# Classification (prereg classification.ordered_decision_list, exactly)
# ---------------------------------------------------------------------------


def classify(window: WindowEval, reading: Reading, prereg: Prereg) -> dict[str, Any]:
    """The ordered decision list as a total function of the frozen inputs.

    D2 is NOT an argument: by construction the lineage evidence cannot change
    the class (prereg D2 role: "CORROBORATIVE ONLY").
    """

    w = window.w_frames
    n_w = int(w.sum())
    if n_w == 0:
        # Provably unreachable (R6 tautology check); retained so the decision
        # list stays a total function.
        return {"class": CLASS_C6, "reason": "W(record) is empty", "w_frames": n_w}

    # step 0 -- TAUTOLOGY CHECK (A2); a non-empty C4 fails the run closed.
    if window.projection_failures:
        return {
            "class": CLASS_C4,
            "reason": "anchor projects outside the decoded raster of a camera S asserts contains",
            "cameras": [item["camera_id"] for item in window.projection_failures],
            "w_frames": n_w,
        }

    level_index = prereg.s5_component_px.index(reading.component_px)
    member = window.member_mask(reading.camera_set)[:, None]
    selector = member & w[None, :]

    # step 1
    step1 = window.occupancy(reading.step1, level_index)
    if not bool((step1 & selector).any()):
        t2b_any = bool((window.subthreshold(reading.step1) & selector).any())
        applicable = window.member_mask(reading.camera_set)
        n_applicable = int(applicable.sum())
        on_eligible = (
            float((window.ltp_on_eligible[level_index] & applicable).sum()) / n_applicable
            if n_applicable
            else 0.0
        )
        detail = {
            # |W| is a per-window REQUIRED OUTPUT on every decision path, not
            # only the step-2 path (prereg required_outputs; owner fix).
            "w_frames": n_w,
            "t2b_any": t2b_any,
            "ltp_on_eligible_component": on_eligible,
            "ltp_on_eligible_component_min": prereg.ltp_on_eligible_component_min,
        }
        if not t2b_any and on_eligible >= prereg.ltp_on_eligible_component_min:
            return {
                "class": CLASS_C1A,
                "reason": "no foreground of any size within the anchor ball",
                **detail,
            }
        if t2b_any:
            return {
                "class": CLASS_C1B,
                "reason": "only sub-threshold foreground within the anchor ball",
                **detail,
            }
        return {
            "class": CLASS_C1C,
            "reason": "the anchor itself was not on an eligible component at ltp_frame",
            **detail,
        }

    # step 2
    occupancy = window.occupancy(reading.step2, level_index) & selector
    counts = occupancy.sum(axis=0)
    enough = counts >= reading.min_cameras
    if reading.min_cameras >= 2 or reading.literal_pair_clause:
        separated = np.zeros(len(window.interval), dtype=bool)
        for ti in np.nonzero(enough)[0]:
            active = np.nonzero(occupancy[:, ti])[0]
            if len(active) < 2:
                continue
            if bool(window.pair_confirmable[np.ix_(active, active)].any()):
                separated[ti] = True
        confirmed = enough & separated
    else:
        confirmed = enough  # A1: pair clause waived at min_occupancy_cameras = 1
    confirmed &= w

    m_count = int(confirmed.sum())
    base = {"m_count": m_count, "w_frames": n_w, "m_fraction": m_count / n_w}
    if (m_count / n_w) >= reading.occupancy_fraction:
        queried = np.array([c in set(window.queried_S) for c in window.cams], dtype=bool)
        in_m = occupancy & confirmed[None, :]
        a_q_star = queried & in_m.any(axis=1)
        if not bool(a_q_star.any()):
            return {
                "class": CLASS_C5_NI,
                "structurally_silent": True,
                "reason": "sustained multi-view occupancy but no occupying camera was ever queried",
                **base,
            }
        per_camera = np.where(a_q_star, in_m.sum(axis=1), -1)
        best = int(per_camera.max())
        star_index = int(np.nonzero(per_camera == best)[0][0])  # cams sorted ascending
        star_frames = np.nonzero(in_m[star_index])[0]
        if len(star_frames) == 0:
            return {"class": CLASS_C6, "reason": "c* has no occupancy frame inside M", **base}
        statuses = window.status[level_index, star_index, star_frames]
        # R4/A6: reading-induced ASSOCIATED pairs leave BOTH numerator and
        # denominator; an emptied denominator yields fraction 0.0 -> C3.
        induced = int((statuses == ST_ASSOCIATED).sum())
        kept = statuses[statuses != ST_ASSOCIATED]
        loss = int(np.isin(kept, np.array(STATUS_C2_GROUP, dtype=statuses.dtype)).sum())
        fraction = (loss / len(kept)) if len(kept) else 0.0
        detail = {
            **base,
            "c_star": int(window.cams[star_index]),
            "c_star_frames": int(len(star_frames)),
            "c_star_denominator": int(len(kept)),
            "c_star_loss_frames": loss,
            "c_star_loss_fraction": fraction,
            "reading_induced_associated": induced,
            "c_star_status_counts": {
                name: int((statuses == code).sum()) for code, name in enumerate(STATUS_NAMES)
            },
        }
        if fraction >= reading.report_majority:
            return {
                "class": CLASS_C2,
                "reason": "sustained multi-view occupancy; tracker asked and did not place the identity",
                **detail,
            }
        return {
            "class": CLASS_C3,
            "reason": "sustained multi-view occupancy; report present but unqualified",
            **detail,
        }

    # step 3
    return {
        "class": CLASS_C5_NS,
        "reason": "foreground near the anchor but no sustained multi-view occupancy at it",
        **base,
    }


def terminal_class(class_name: str, structurally_silent: bool) -> str:
    """The TERMINAL level used by the frozen R10 audit strata (A11)."""

    if class_name == CLASS_C5_NI and structurally_silent:
        return CLASS_C5_SS
    return class_name


# ---------------------------------------------------------------------------
# Sequence evaluation
# ---------------------------------------------------------------------------


def _census_block(payload: dict[str, Any], sequence_name: str, census_path: Path) -> dict[str, Any]:
    per_sequence = payload.get("per_sequence")
    if isinstance(per_sequence, dict):
        if sequence_name in per_sequence:
            return per_sequence[sequence_name]
        raise ContractError(
            f"{census_path}: sequence {sequence_name!r} not in per_sequence {sorted(per_sequence)}"
        )
    if "records" in payload and "constants" in payload:
        return payload
    raise ContractError(f"{census_path}: not a recognisable M1-A0 census artifact")


def _pair_confirmable(scene: SceneBundle, cams: Sequence[int], angular_floor: float) -> np.ndarray:
    """Optical axis of camera c = w2c[:3,:3].T @ [0,0,1] -- identical to the
    census's angular_separation_floor / r2prime_holds construction."""

    axes = [scene.cameras[c].w2c[:3, :3].T @ np.array([0.0, 0.0, 1.0]) for c in cams]
    n = len(cams)
    out = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            cosine = float(np.clip(np.dot(axes[i], axes[j]), -1.0, 1.0))
            if float(np.arccos(cosine)) >= angular_floor:
                out[i, j] = out[j, i] = True
    return out


def _d2_keys(prereg: Prereg) -> tuple[tuple[int, float], ...]:
    """The S4 IoU ladder at primary eligibility plus the S5 ladder at the
    primary IoU, de-duplicated, primary first. Both axes are descriptive (A9)."""

    keys: list[tuple[int, float]] = [(prereg.component_min_px, prereg.lineage_iou_min)]
    for iou in prereg.s4_lineage_iou:
        candidate = (prereg.component_min_px, float(iou))
        if candidate not in keys:
            keys.append(candidate)
    for level in prereg.s5_component_px:
        candidate = (int(level), prereg.lineage_iou_min)
        if candidate not in keys:
            keys.append(candidate)
    return tuple(keys)


def _d2_key_name(key: tuple[int, float]) -> str:
    return f"px{key[0]}_iou{key[1]}"


def _maximal_runs(flags: Sequence[bool]) -> int:
    """Number of maximal runs of True in a boolean sequence (R6 guard)."""

    runs = 0
    previous = False
    for value in flags:
        if value and not previous:
            runs += 1
        previous = bool(value)
    return runs


def evaluate_sequence(
    *,
    sequence_name: str,
    scene_dir: Path,
    tracks_path: Path,
    census_path: Path,
    prereg: Prereg,
) -> dict[str, Any]:
    """Classify every true_absence_candidate window of ONE sequence."""

    started = time.monotonic()
    scene_dir = Path(scene_dir)
    scene = load_temporal_scene(scene_dir)
    substrate, decoded_dims = substrate_check(scene, scene_dir)

    census_payload = json.loads(Path(census_path).read_text(encoding="utf-8"))
    block = _census_block(census_payload, sequence_name, Path(census_path))
    records = block["records"]["true_absence_candidates"]
    if "r_site_census" not in block.get("constants", {}):
        raise ContractError(f"{census_path}: sequence block has no constants.r_site_census")
    r_site_census = float(block["constants"]["r_site_census"])

    angular_floor = angular_separation_floor(scene)
    sealed_floor = block.get("angular_floor_rad")
    if sealed_floor is not None and not math.isclose(
        float(sealed_floor), angular_floor, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ContractError(
            f"{sequence_name}: recomputed angular floor {angular_floor!r} != sealed {sealed_floor!r}"
        )

    levels = prereg.s5_component_px
    level_primary = prereg.primary_level_index
    n_levels = len(levels)
    n_radii = len(RADII)
    scene_frames = [int(f) for f in scene.frame_indices]
    frame_set = set(scene_frames)
    fps = float(scene.fps)

    artifact = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    index = index_tracks(artifact, scene)  # fail-closed schema validation
    seeds_in_scope = {int(record["seed_id"]) for record in records}
    raw = build_raw_reports(artifact, scene, index, seeds_in_scope, prereg)
    seed_meta = {
        int(seed["seed_id"]): {"n_cam": int(seed.get("n_cam", 0)), "reproj_rms_px": None}
        for seed in artifact["seeds"]
        if int(seed["seed_id"]) in seeds_in_scope
    }
    diagnostics = artifact.get("diagnostics", {}) or {}
    reproj_map = diagnostics.get("reproj_rms_px", {}) or {}
    fb_map = diagnostics.get("fb_rms_px", {}) or {}
    for seed_id, meta in seed_meta.items():
        value = reproj_map.get(str(seed_id))
        meta["reproj_rms_px"] = None if value is None else float(value)
    consensus_at: dict[tuple[int, int], dict[str, Any]] = {}
    for key, entries in (artifact.get("consensus", {}) or {}).items():
        seed_id = int(key)
        if seed_id not in seeds_in_scope:
            continue
        for entry in entries:
            consensus_at[(seed_id, int(round(float(entry["frame"]))))] = entry
    del artifact, index
    gc.collect()

    # ---- window construction ---------------------------------------------
    d2_keys = _d2_keys(prereg)
    windows: list[WindowEval] = []
    frame_mismatches: list[dict[str, Any]] = []
    for record in records:
        seed_id = int(record["seed_id"])
        first_frame = int(record["first_frame"])
        last_frame = int(record["last_frame"])
        ltp_frame = int(record["ltp_frame"])
        anchor = np.asarray(record["ltp"], dtype=np.float64)
        frozen_S = tuple(sorted(int(c) for c in record["containing_cameras"]))
        for camera_id in frozen_S:
            if camera_id not in scene.cameras:
                raise ContractError(
                    f"{sequence_name} seed {seed_id} window {first_frame}: containing "
                    f"camera {camera_id} is not a training camera of the scene"
                )
        queried_S = tuple(c for c in frozen_S if (seed_id, c) in raw.track_rows)
        # R12: the anchor is frozen and the rig is static, so per-frame frustum
        # containment is a provable no-op; computed with the census primitive
        # and published descriptively.
        per_frame_S = tuple(
            c for c in scene.tracking_ids if frustum_contains(scene.cameras[c], anchor)
        )
        cams = tuple(sorted(set(frozen_S) | set(per_frame_S)))
        interval = tuple(f for f in range(first_frame, last_frame + 1) if f in frame_set)
        if not interval:
            raise ContractError(
                f"{sequence_name} seed {seed_id}: window [{first_frame}, {last_frame}] "
                "does not intersect the realized frame indices"
            )
        if len(interval) != int(record["n_frames"]):
            frame_mismatches.append(
                {
                    "seed_id": seed_id,
                    "first_frame": first_frame,
                    "record_n_frames": int(record["n_frames"]),
                    "realized_n_frames": len(interval),
                }
            )

        geom: dict[int, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        for camera_id in cams:
            camera = scene.cameras[camera_id]
            x, y, z = anchor_projection(camera, anchor)
            in_frustum = bool(frustum_contains(camera, anchor))
            derived = z > 0.0 and 0.0 <= x <= camera.width - 1.0 and 0.0 <= y <= camera.height - 1.0
            if derived != in_frustum:
                raise ContractError(
                    f"{sequence_name} seed {seed_id}: anchor projection disagrees with "
                    f"build_m1_census.frustum_contains in camera {camera_id}"
                )
            dw, dh = decoded_dims[camera_id]
            if z > 0.0:
                col, row = round_half_up(x), round_half_up(y)
                continuous_ok = 0.0 <= x <= dw - 1.0 and 0.0 <= y <= dh - 1.0
                rounded_ok = 0 <= col <= dw - 1 and 0 <= row <= dh - 1
                tol = anchor_tolerance(camera, z, r_site_census)
            else:
                col = row = -1
                continuous_ok = rounded_ok = False
                tol = None
            entry = {
                "camera_id": camera_id,
                "x": None if math.isnan(x) else x,
                "y": None if math.isnan(y) else y,
                "z": z,
                "col": col,
                "row": row,
                "tol_px": tol,
                "in_frustum_declared": in_frustum,
                "continuous_in_decoded_raster": bool(continuous_ok),
                "rounded_in_decoded_raster": bool(rounded_ok),
                "decoded_raster": [dw, dh],
            }
            entry["projection_ok"] = bool(z > 0.0 and continuous_ok and rounded_ok)
            geom[camera_id] = entry
            if camera_id in frozen_S and not entry["projection_ok"]:
                failures.append(
                    {
                        "camera_id": camera_id,
                        "z": z,
                        "continuous_in_decoded_raster": bool(continuous_ok),
                        "rounded_in_decoded_raster": bool(rounded_ok),
                    }
                )

        consensus_entry = consensus_at.get((seed_id, ltp_frame), {})
        anchor_quality = {
            "anchor_staleness": first_frame - ltp_frame,
            "ltp_frame": ltp_frame,
            "ltp_n_cam": (
                None if consensus_entry.get("n_cam") is None else int(consensus_entry["n_cam"])
            ),
            "ltp_reproj_rms": (
                None
                if consensus_entry.get("reproj_rms") is None
                else float(consensus_entry["reproj_rms"])
            ),
            "seed_n_cam": seed_meta.get(seed_id, {}).get("n_cam"),
            "seed_reproj_rms_px": seed_meta.get(seed_id, {}).get("reproj_rms_px"),
            "fb_rms_px": {
                str(c): (float(fb_map[f"{seed_id}:{c}"]) if f"{seed_id}:{c}" in fb_map else None)
                for c in frozen_S
            },
        }

        n_cams, n_interval = len(cams), len(interval)
        windows.append(
            WindowEval(
                sequence=sequence_name,
                seed_id=seed_id,
                first_frame=first_frame,
                last_frame=last_frame,
                record_n_frames=int(record["n_frames"]),
                bridged_interruptions=int(record.get("bridged_interruptions", 0)),
                ltp_frame=ltp_frame,
                anchor=anchor,
                frozen_S=frozen_S,
                queried_S=queried_S,
                per_frame_S=per_frame_S,
                cams=cams,
                interval=interval,
                geom=geom,
                projection_failures=failures,
                pair_confirmable=_pair_confirmable(scene, cams, angular_floor),
                end_truncated=bool(last_frame == scene_frames[-1]),
                fps=fps,
                anchor_quality=anchor_quality,
                t1=np.zeros((n_levels, n_cams, n_interval), dtype=bool),
                t2=np.zeros((n_radii, n_levels, n_cams, n_interval), dtype=bool),
                t2b=np.zeros((n_radii, n_cams, n_interval), dtype=bool),
                t3=np.zeros((n_levels, n_cams, n_interval), dtype=bool),
                status=np.full((n_levels, n_cams, n_interval), -1, dtype=np.int8),
                evaluated=np.zeros((n_cams, n_interval), dtype=bool),
                ltp_on_eligible=np.zeros((n_levels, n_cams), dtype=bool),
                d2={"cameras": {_d2_key_name(k): {} for k in d2_keys}},
            )
        )

    # ---- streaming pass: per SEQUENCE, camera-major then frame-major ------
    # R9: decode EVERY frame of the realized census window, for every camera in
    # the union of the windows' applicable sets.
    stats = {"mask_decodes": 0, "missing_report_rows_mapped_to_MISS_TOKEN": 0}
    component_areas: list[int] = []
    foreground_fractions: list[float] = []
    camera_universe = sorted({c for w in windows for c in w.cams})
    interval_index = [{frame: i for i, frame in enumerate(w.interval)} for w in windows]
    by_camera: dict[int, list[int]] = {}
    for wi, window in enumerate(windows):
        for camera_id in window.cams:
            by_camera.setdefault(camera_id, []).append(wi)

    for camera_id in camera_universe:
        window_indices = by_camera.get(camera_id, [])
        eval_by_frame: dict[int, list[int]] = {}
        ltp_by_frame: dict[int, list[int]] = {}
        for wi in window_indices:
            for frame in windows[wi].interval:
                eval_by_frame.setdefault(frame, []).append(wi)
            ltp_by_frame.setdefault(windows[wi].ltp_frame, []).append(wi)

        height, width = decoded_dims[camera_id][1], decoded_dims[camera_id][0]
        discs: dict[int, list[Disc] | None] = {}
        for wi in window_indices:
            entry = windows[wi].geom[camera_id]
            if not entry["projection_ok"]:
                discs[wi] = None
                continue
            tol = float(entry["tol_px"])
            discs[wi] = [
                Disc(entry["x"], entry["y"], entry["col"], entry["row"], tol * scale, height, width)
                for _, scale in RADII
            ]

        # D2 lineage bookkeeping for this camera
        lineage: dict[tuple[int, str], dict[str, Any]] = {}
        lineage_start: dict[int, int] = {}
        for wi in window_indices:
            window = windows[wi]
            if camera_id not in window.queried_S:
                continue
            candidates = raw.candidate_frames.get((window.seed_id, camera_id), ())
            earlier = [f for f in candidates if f < window.first_frame and f in frame_set]
            if not earlier:
                continue
            lineage_start[wi] = earlier[0]
            for key in d2_keys:
                lineage[(wi, _d2_key_name(key))] = {
                    "level": levels.index(key[0]),
                    "iou_min": key[1],
                    "label": None,
                    "alive": False,
                    "t0": None,
                    "entry_label": None,
                    "entry_corroborated": False,
                    "alive_flags": np.zeros(len(window.interval), dtype=bool),
                    "merges": 0,
                    "splits": 0,
                    "end_frame": None,
                }
        primary_name = _d2_key_name(d2_keys[0])

        previous_mask: MaskFrame | None = None
        for frame in scene_frames:
            mask_frame = MaskFrame(scene.mask_path(camera_id, frame), levels, level_primary)
            stats["mask_decodes"] += 1
            foreground_fractions.append(mask_frame.foreground_fraction)
            for label in mask_frame.eligible[level_primary]:
                component_areas.append(int(mask_frame.areas[label]))

            # ---- ltp_on_eligible_component (B3), at ltp_frame --------------
            for wi in ltp_by_frame.get(frame, ()):
                window = windows[wi]
                ci = window.camera_index(camera_id)
                entry = window.geom[camera_id]
                window.ltp_frame_evaluated = True
                if entry["projection_ok"]:
                    label = int(mask_frame.labels[entry["row"], entry["col"]])
                    for li in range(n_levels):
                        window.ltp_on_eligible[li, ci] = bool(mask_frame.lookup[li][label])

            # ---- per-pair tests over the CLOSED interval -------------------
            for wi in eval_by_frame.get(frame, ()):
                window = windows[wi]
                ci = window.camera_index(camera_id)
                entry = window.geom[camera_id]
                ti = interval_index[wi][frame]
                if window.evaluated[ci, ti]:
                    raise ContractError(
                        f"{sequence_name}: pair (camera {camera_id}, frame {frame}) "
                        f"evaluated twice for window {window.key}"
                    )
                window.evaluated[ci, ti] = True
                report = raw.table.get((window.seed_id, camera_id, frame))
                queried = (window.seed_id, camera_id) in raw.track_rows
                if not queried:
                    precursor = None
                elif report is None:
                    precursor = PRE_MISS  # A5
                    stats["missing_report_rows_mapped_to_MISS_TOKEN"] += 1
                else:
                    precursor = report[0]
                window_discs = discs[wi]
                if window_discs is not None:
                    for ri in range(n_radii):
                        window.t2b[ri, ci, ti] = window_discs[ri].hits(mask_frame.labels, None)
                anchor_label = (
                    int(mask_frame.labels[entry["row"], entry["col"]])
                    if entry["projection_ok"]
                    else None
                )
                if precursor == PRE_NEEDS_MASK:
                    if not (
                        0 <= report[2] < mask_frame.shape[0]
                        and 0 <= report[1] < mask_frame.shape[1]
                    ):
                        raise ContractError(
                            f"{sequence_name}: in-domain report pixel ({report[1]}, {report[2]}) "
                            f"outside the decoded raster {mask_frame.shape} at camera "
                            f"{camera_id} frame {frame}"
                        )
                    report_label = int(mask_frame.labels[report[2], report[1]])
                else:
                    report_label = None
                for li in range(n_levels):
                    lut = mask_frame.lookup[li]
                    window.t3[li, ci, ti] = mask_frame.any_eligible[li]
                    if anchor_label is not None:
                        window.t1[li, ci, ti] = bool(lut[anchor_label])
                    # A10: no T1 => T2 short circuit at any radius.
                    if window_discs is not None:
                        for ri in range(n_radii):
                            window.t2[ri, li, ci, ti] = window_discs[ri].hits(
                                mask_frame.labels, lut
                            )
                    # ---- report status (A4) ------------------------------
                    if precursor is None:
                        status = ST_NEVER_QUERIED
                    elif precursor == PRE_MISS:
                        status = ST_MISS_TOKEN
                    elif precursor == PRE_OUT_OF_DOMAIN:
                        status = ST_OUT_OF_DOMAIN
                    elif precursor == PRE_LOW_VISIBILITY:
                        status = ST_LOW_VISIBILITY
                    else:
                        status = ST_ASSOCIATED if lut[report_label] else ST_OFF_COMPONENT
                    window.status[li, ci, ti] = status

            # ---- D2 lineage --------------------------------------------------
            if lineage:
                contingency: dict[int, tuple[int, np.ndarray]] = {}
                accepted: dict[int, list[tuple[int, str]]] = {}
                for (wi, name), state in lineage.items():
                    window = windows[wi]
                    if frame < lineage_start.get(wi, math.inf) or frame > window.last_frame:
                        continue
                    li = state["level"]
                    lut = mask_frame.lookup[li]
                    report = raw.table.get((window.seed_id, camera_id, frame))
                    # (Re)seed at every ASSOCIATED frame STRICTLY before the
                    # window; the last such reseed leaves the lineage seeded at
                    # t0 exactly as the frozen construction requires.
                    reseeded = False
                    if (
                        frame < window.first_frame
                        and report is not None
                        and report[0] == PRE_NEEDS_MASK
                    ):
                        label = int(mask_frame.labels[report[2], report[1]])
                        if lut[label]:
                            entry_geom = window.geom[camera_id]
                            state.update(
                                label=label,
                                alive=True,
                                t0=frame,
                                entry_label=label,
                                end_frame=None,
                                entry_corroborated=bool(
                                    entry_geom["projection_ok"]
                                    and int(
                                        mask_frame.labels[entry_geom["row"], entry_geom["col"]]
                                    )
                                    == label
                                ),
                            )
                            reseeded = True
                    if not reseeded and state["alive"]:
                        if previous_mask is None:
                            state["alive"] = False
                            state["end_frame"] = frame
                        else:
                            prev_label = int(state["label"])
                            cached = contingency.get(prev_label)
                            if cached is None:
                                picked = previous_mask.labels == prev_label
                                cached = (
                                    int(picked.sum()),
                                    np.bincount(
                                        mask_frame.labels[picked].ravel(),
                                        minlength=len(mask_frame.areas),
                                    ),
                                )
                                contingency[prev_label] = cached
                            area_prev, inter = cached
                            best_label, best_iou, overlaps = -1, -1.0, 0
                            for candidate in sorted(mask_frame.eligible[li]):
                                intersection = (
                                    int(inter[candidate]) if candidate < len(inter) else 0
                                )
                                if intersection == 0:
                                    continue
                                union = area_prev + int(mask_frame.areas[candidate]) - intersection
                                iou = intersection / union if union > 0 else 0.0
                                if iou >= state["iou_min"]:
                                    overlaps += 1
                                if iou > best_iou:
                                    best_iou, best_label = iou, candidate
                            if overlaps >= 2 and name == primary_name:
                                state["splits"] += 1
                            if best_iou >= state["iou_min"]:
                                state["label"] = best_label
                            else:
                                state["alive"] = False
                                state["end_frame"] = frame
                    if state["alive"]:
                        ti = interval_index[wi].get(frame)
                        if ti is not None:
                            state["alive_flags"][ti] = True
                        if name == primary_name:
                            accepted.setdefault(int(state["label"]), []).append((wi, name))
                for holders in accepted.values():
                    if len(holders) >= 2:
                        for wi, name in holders:
                            lineage[(wi, name)]["merges"] += 1

            previous_mask = mask_frame

        # ---- fold this camera's lineage into the windows -------------------
        for wi in window_indices:
            window = windows[wi]
            never_queried = camera_id in window.frozen_S and camera_id not in window.queried_S
            for key in d2_keys:
                name = _d2_key_name(key)
                state = lineage.get((wi, name))
                if state is None or state["t0"] is None:
                    window.d2["cameras"][name][str(camera_id)] = {
                        "entry": None,
                        "never_queried": never_queried,
                    }
                    continue
                window.d2["cameras"][name][str(camera_id)] = {
                    "entry": state["t0"],
                    "entry_label": state["entry_label"],
                    "entry_corroborated": bool(state["entry_corroborated"]),
                    "alive_flags": state["alive_flags"],
                    "end_frame": state["end_frame"],
                    "merge_events": int(state["merges"]),
                    "split_events": int(state["splits"]),
                }
        del lineage, discs
        previous_mask = None
        gc.collect()

    # ---- derive W(record) and enforce the fail-closed guards --------------
    tautology_checks: list[dict[str, Any]] = []
    for window in windows:
        if not window.evaluated.all():
            raise ContractError(
                f"{sequence_name}: window {window.key} left "
                f"{int((~window.evaluated).sum())} (camera, frame) pairs unevaluated"
            )
        if (window.status < 0).any():
            raise ContractError(f"{sequence_name}: window {window.key} has an unset report status")
        if window.ltp_frame in frame_set and not window.ltp_frame_evaluated:
            raise ContractError(
                f"{sequence_name}: window {window.key} never evaluated its ltp_frame "
                f"{window.ltp_frame}"
            )
        member = window.member_mask(S1_FROZEN)
        associated = (
            (window.status[prereg.primary_level_index] == ST_ASSOCIATED) & member[:, None]
        ).any(axis=0)
        window.w_mask = ~associated
        w_frames = [f for f, keep in zip(window.interval, window.w_mask) if keep]
        # A21: the r3 W conditions are RECORDED tautology checks, not operative
        # stops (r5 stopping_conditions). An empty W still stops the run
        # indirectly: the decision list then returns C6, which fails closed.
        tautology = {
            "W_non_empty": bool(w_frames),
            "min_W_equals_first_frame": bool(w_frames and w_frames[0] == window.first_frame),
            "max_W_equals_last_frame": bool(w_frames and w_frames[-1] == window.last_frame),
        }
        tautology_checks.append({"window": list(window.key), **tautology})
        # ---- R6 OPERATIVE census-reproduction guard ----------------------
        # THE ONLY CHECK that this diagnostic's recomputed association
        # reproduces the sealed census's own. Either mismatch fails the
        # SEQUENCE closed.
        bridged = int(associated.sum())
        if len(w_frames) + bridged != window.record_n_frames:
            raise ContractError(
                f"{sequence_name}: window {window.key} violates "
                f"operative_fail_closed_guard_R6: |W| {len(w_frames)} + bridged {bridged} "
                f"!= record n_frames {window.record_n_frames}"
            )
        runs = _maximal_runs(associated.tolist())
        if runs != window.bridged_interruptions:
            raise ContractError(
                f"{sequence_name}: window {window.key} violates "
                f"operative_fail_closed_guard_R6: maximal bridged runs {runs} != record "
                f"bridged_interruptions {window.bridged_interruptions}"
            )

    # ---- D2 verdicts (A7) --------------------------------------------------
    for window in windows:
        n_w = int(window.w_mask.sum())
        verdicts: dict[str, str] = {}
        for key in d2_keys:
            name = _d2_key_name(key)
            bucket = window.d2["cameras"][name]
            live = [v for v in bucket.values() if v.get("entry") is not None]
            if not live:
                verdicts[name] = D2_ENTRY_UNAVAILABLE
                continue
            survives = False
            for value in live:
                survived = int((value["alive_flags"] & window.w_mask).sum())
                value["survived_frames"] = survived
                value["survived_fraction"] = survived / n_w if n_w else 0.0
                if survived >= n_w:
                    survives = True
            verdicts[name] = D2_SURVIVES if survives else D2_ENDS
        for name in list(window.d2["cameras"]):
            for value in window.d2["cameras"][name].values():
                value.pop("alive_flags", None)
        window.d2["verdicts"] = verdicts
        window.d2["entry_corroborated_any"] = bool(
            any(
                v.get("entry_corroborated")
                for v in window.d2["cameras"][_d2_key_name(d2_keys[0])].values()
            )
        )

    return {
        "sequence": sequence_name,
        "scene_dir": str(scene_dir),
        "windows": windows,
        "substrate": substrate,
        "census_block": block,
        "r_site_census": r_site_census,
        "angular_floor_rad": angular_floor,
        "fps": fps,
        "realized_window": {
            "first_frame": scene_frames[0],
            "last_frame": scene_frames[-1],
            "n_frames": len(scene_frames),
        },
        "training_cameras": [int(c) for c in scene.tracking_ids],
        "held_out_cameras": [int(c) for c in scene.held_out_ids],
        "window_frame_count_mismatches": frame_mismatches,
        "decode_stats": stats,
        "out_of_domain_reports": raw.out_of_domain_count,
        "tautology_checks_W": {
            "all_passed": all(
                row["W_non_empty"]
                and row["min_W_equals_first_frame"]
                and row["max_W_equals_last_frame"]
                for row in tautology_checks
            ),
            "note": DISCLOSED_READINGS["A21_W_tautology_checks_are_recorded_not_operative"],
            "failures": [
                row
                for row in tautology_checks
                if not (
                    row["W_non_empty"]
                    and row["min_W_equals_first_frame"]
                    and row["max_W_equals_last_frame"]
                )
            ],
        },
        "median_eligible_component_area": (
            float(np.median(component_areas)) if component_areas else None
        ),
        "median_foreground_area_fraction": (
            float(np.median(foreground_fractions)) if foreground_fractions else None
        ),
        "decoded_pairs": len(foreground_fractions),
        "wall_seconds": time.monotonic() - started,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in CLASS_NAMES}


def _fraction(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator else None


def _c5_total(counts: dict[str, int]) -> int:
    return sum(counts[name] for name in C5_MEMBERS)


def classify_all(
    windows: Sequence[WindowEval], reading: Reading, prereg: Prereg
) -> tuple[list[str], list[bool]]:
    assigned: list[str] = []
    silent: list[bool] = []
    for window in windows:
        verdict = classify(window, reading, prereg)
        assigned.append(verdict["class"])
        silent.append(bool(verdict.get("structurally_silent", False)))
    return assigned, silent


@dataclasses.dataclass(frozen=True)
class Tally:
    """Everything the decision rules need from ONE reading's assignment."""

    counts: dict[str, int]
    per_sequence: dict[str, dict[str, int]]
    end_truncated_c1a: dict[str, int]
    structurally_silent: int
    total: int
    supply: dict[str, int]
    prereg: Prereg

    # ---- pooling (a): pooled over all windows -----------------------------
    def a_c23(self) -> float:
        return _fraction(self.counts[CLASS_C2] + self.counts[CLASS_C3], self.total) or 0.0

    def a_c5(self) -> float:
        return _fraction(_c5_total(self.counts), self.total) or 0.0

    def a_c1a(self) -> float:
        return _fraction(self.counts[CLASS_C1A], self.total) or 0.0

    def a_c1a_adequacy(self) -> float:
        # A14: END_TRUNCATED leaves the numerator, not the denominator.
        return (
            _fraction(self.counts[CLASS_C1A] - self.end_truncated_c1a["pooled"], self.total) or 0.0
        )

    # ---- pooling (b): unweighted mean over sequences with >= floor windows -
    def big_sequences(self) -> list[str]:
        floor = self.prereg.sequence_window_floor
        return [name for name in sorted(self.supply) if self.supply[name] >= floor]

    def b_fractions(self) -> dict[str, float] | None:
        big = self.big_sequences()
        if not big:
            return None  # A15
        out: dict[str, float] = {}
        for key in (*CLASS_NAMES, "C5_total", "C1a_adequacy", "C2_plus_C3"):
            values = []
            for name in big:
                block = self.per_sequence[name]
                if key == "C5_total":
                    numerator = _c5_total(block)
                elif key == "C1a_adequacy":
                    numerator = block[CLASS_C1A] - self.end_truncated_c1a.get(name, 0)
                elif key == "C2_plus_C3":
                    numerator = block[CLASS_C2] + block[CLASS_C3]
                else:
                    numerator = block[key]
                values.append(numerator / self.supply[name])
            out[key] = float(np.mean(values))
        return out

    # ---- pooling (c): leave-scissor-out ------------------------------------
    def c_fractions(self) -> dict[str, float | None]:
        members = [name for name in self.supply if name != "scissor"]
        total = sum(self.supply[name] for name in members)
        counts = _empty_counts()
        for name in members:
            for key in CLASS_NAMES:
                counts[key] += self.per_sequence[name][key]
        out: dict[str, float | None] = {key: _fraction(counts[key], total) for key in CLASS_NAMES}
        out["C5_total"] = _fraction(_c5_total(counts), total)
        out["sequences"] = sorted(members)  # type: ignore[assignment]
        out["windows"] = total  # type: ignore[assignment]
        return out

    # ---- pooling-independent per-sequence conditions -----------------------
    def per_sequence_c1a_adequacy_ok(self) -> bool:
        floor = self.prereg.sequence_window_floor
        for name, block in self.per_sequence.items():
            if self.supply[name] >= floor:
                numerator = block[CLASS_C1A] - self.end_truncated_c1a.get(name, 0)
                if (numerator / self.supply[name]) < self.prereg.status_1_sequence_c1a_min:
                    return False
        return True

    def prefix_c23_ok(self, prefix: Sequence[str]) -> bool:
        members = [name for name in prefix if name in self.per_sequence]
        if not members:
            return False
        return all(
            (
                _fraction(
                    self.per_sequence[name][CLASS_C2] + self.per_sequence[name][CLASS_C3],
                    self.supply[name],
                )
                or 0.0
            )
            >= self.prereg.status_2_c23_min
            for name in members
        )

    def largest_supply_sequence(self) -> str | None:
        if not self.supply:
            return None
        return max(self.supply, key=lambda n: (self.supply[n], n))

    def largest_supply_c1a_fraction(self) -> float | None:
        largest = self.largest_supply_sequence()
        if largest is None or not self.supply[largest]:
            return None
        return self.per_sequence[largest][CLASS_C1A] / self.supply[largest]

    # ---- pooling-parameterised accessors -----------------------------------
    def fractions(self, pooling: str) -> dict[str, float] | None:
        if pooling == POOLING_A:
            return {
                CLASS_C1A: self.a_c1a(),
                "C1a_adequacy": self.a_c1a_adequacy(),
                "C2_plus_C3": self.a_c23(),
                "C5_total": self.a_c5(),
            }
        values = self.b_fractions()
        if values is None:
            return None
        return {
            CLASS_C1A: values[CLASS_C1A],
            "C1a_adequacy": values["C1a_adequacy"],
            "C2_plus_C3": values["C2_plus_C3"],
            "C5_total": values["C5_total"],
        }


def summarize_assignment(
    windows: Sequence[WindowEval],
    assigned: Sequence[str],
    silent: Sequence[bool],
    supply: dict[str, int],
    prereg: Prereg,
) -> Tally:
    counts = _empty_counts()
    per_sequence = {name: _empty_counts() for name in supply}
    end_truncated = {"pooled": 0, **{name: 0 for name in supply}}
    for window, value in zip(windows, assigned):
        counts[value] += 1
        per_sequence[window.sequence][value] += 1
        if value == CLASS_C1A and window.end_truncated:
            end_truncated["pooled"] += 1
            end_truncated[window.sequence] += 1
    return Tally(
        counts=counts,
        per_sequence=per_sequence,
        end_truncated_c1a=end_truncated,
        structurally_silent=sum(1 for value in silent if value),
        total=len(assigned),
        supply=supply,
        prereg=prereg,
    )


# ---- the frozen status predicates -----------------------------------------


def status_1_core(tally: Tally, pooling: str) -> bool | None:
    """The three status_1 conjuncts at ONE reading under ONE pooling.

    r5/R14: status_1 is these three conjuncts at the PRIMARY READING under
    pooling (b) AND the same three under pooling (a). Cross-reading behaviour
    is carried by the R15 robustness grade, not by an all-readings conjunct.
    """

    values = tally.fractions(pooling)
    if values is None:
        return None
    p = tally.prereg
    return bool(
        values["C1a_adequacy"] >= p.status_1_pooled_c1a_min
        and values["C2_plus_C3"] <= p.status_1_c23_max
        and tally.per_sequence_c1a_adequacy_ok()
    )


def status_2_local(tally: Tally, prefix: Sequence[str], pooling: str) -> bool | None:
    """status_2's two disjuncts at ONE reading under ONE pooling.

    r5 supersedes the r4 cross-reading quantifier on BOTH disjuncts with the
    R15 grade, which applies symmetrically, so the r4 asymmetry that made the
    single-sequence route cheap stays closed.
    """

    values = tally.fractions(pooling)
    if values is None:
        return None
    return bool(
        values["C2_plus_C3"] >= tally.prereg.status_2_c23_min or tally.prefix_c23_ok(prefix)
    )


def status_3_local(tally: Tally, prefix: Sequence[str], pooling: str) -> bool | None:
    """status_3's two C1a conjuncts at ONE reading under ONE pooling.

    r5 supersedes the '<= 20% relative variation' clause with the R15 grade;
    the figure survives only as a reported diagnostic (A13).
    """

    values = tally.fractions(pooling)
    if values is None:
        return None
    p = tally.prereg
    if status_2_local(tally, prefix, pooling) or status_1_core(tally, pooling):
        return False
    return bool(
        values[CLASS_C1A] >= p.status_3_c1a_min
        and (tally.largest_supply_c1a_fraction() or 0.0) >= p.status_3_c1a_min
    )


def reading_label(tally: Tally, prefix: Sequence[str]) -> str:
    """The FROZEN R3 device: reading-local predicates under BINDING pooling
    (b), in precedence order 2, 1, 3, else UNRESOLVED. Never a status."""

    if tally.fractions(POOLING_B) is None:
        return READING_LABEL_UNRESOLVED
    if status_2_local(tally, prefix, POOLING_B):
        return STATUS_2
    if status_1_core(tally, POOLING_B):
        return STATUS_1
    if status_3_local(tally, prefix, POOLING_B):
        return STATUS_3
    return READING_LABEL_UNRESOLVED


def robustness_grade(
    labels: Sequence[str], primary_label: str, supermajority: float
) -> tuple[str, float]:
    """The FROZEN R15 grade over the decision-relevant grid.

    UNANIMOUS when reading_label is constant; ROBUST when it agrees with the
    PRIMARY READING's label in at least ``supermajority`` of cells; else
    FRAGILE. Returns ``(grade, agreement_fraction)``.
    """

    if not labels:
        return GRADE_FRAGILE, 0.0
    agreement = sum(1 for value in labels if value == primary_label) / len(labels)
    if len(set(labels)) == 1:
        return GRADE_UNANIMOUS, agreement
    if agreement >= supermajority:
        return GRADE_ROBUST, agreement
    return GRADE_FRAGILE, agreement


def _minimal_prefix(supply: dict[str, int]) -> list[str]:
    """The MINIMAL descending-absence-count prefix summing to >= 50% of total
    (prereg status_2; for the frozen 597-window scope this is {scissor})."""

    total = sum(supply.values())
    ordered = sorted(supply.items(), key=lambda item: (-item[1], item[0]))
    chosen: list[str] = []
    running = 0
    for name, count in ordered:
        if total and running * 2 >= total:
            break
        chosen.append(name)
        running += count
    return chosen


def _w_stratum(n_w: int, fps: float) -> str:
    seconds = n_w / fps if fps else 0.0
    if seconds < 0.125:
        return W_STRATA[0]
    if seconds <= 0.5:
        return W_STRATA[1]
    return W_STRATA[2]


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3:
        return None
    xa, xb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if float(xa.std()) == 0.0 or float(xb.std()) == 0.0:
        return None
    return float(np.corrcoef(xa, xb)[0, 1])


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3:
        return None

    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    return _pearson(rank(a), rank(b))


def _terciles(values: Sequence[int]) -> tuple[float, float]:
    """Nearest-rank tercile cut points; ties fall to the LOWER tercile because
    membership is tested with ``<=`` (frozen R10)."""

    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    n = len(ordered)
    lower = ordered[min(n - 1, max(0, math.ceil(n / 3.0) - 1))]
    upper = ordered[min(n - 1, max(0, math.ceil(2.0 * n / 3.0) - 1))]
    return (float(lower), float(upper))


def _tercile_label(value: int, cuts: tuple[float, float]) -> str:
    if value <= cuts[0]:
        return "staleness_T1"
    if value <= cuts[1]:
        return "staleness_T2"
    return "staleness_T3"


def build_audit_sample(
    windows: Sequence[WindowEval],
    assigned: Sequence[str],
    silent: Sequence[bool],
    prereg: Prereg,
) -> dict[str, Any]:
    """The FROZEN R10 stratified M1-A0b sample. Emitting it is in scope;
    RUNNING the audit is not."""

    staleness = [int(w.anchor_quality["anchor_staleness"]) for w in windows]
    cuts = _terciles(staleness)
    strata: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for window, value, flag in zip(windows, assigned, silent):
        terminal = terminal_class(value, flag)
        if terminal not in AUDIT_CLASS_ORDER:
            # C4/C6 are asserted empty and fail the run before this point.
            raise ContractError(
                f"audit sample: terminal class {terminal!r} is not in the frozen R10 order"
            )
        stale = int(window.anchor_quality["anchor_staleness"])
        key = (window.sequence, AUDIT_CLASS_ORDER.index(terminal), _tercile_label(stale, cuts))
        strata.setdefault(key, []).append(
            {
                "sequence": window.sequence,
                "seed_id": window.seed_id,
                "first_frame": window.first_frame,
                "last_frame": window.last_frame,
                "W_frames": int(window.w_mask.sum()),
                "bridged_associated_frames": len(window.interval) - int(window.w_mask.sum()),
                # bridged_frame_bias_DISCLOSED: audit frames are drawn uniformly
                # over the CENSUS window, so a frame landing on a bridged frame
                # can never yield an ABSENT-ABSENT confirmation.
                "has_bridged_frames": bool(len(window.interval) > int(window.w_mask.sum())),
                "class": terminal,
                "anchor_staleness": stale,
                "containing_cameras": list(window.frozen_S),
                "ltp": [float(v) for v in window.anchor],
                "ltp_frame": window.ltp_frame,
            }
        )
    # A19 / r5 token_repairs: the RNG is constructed FRESH PER STRATUM.
    ordered_keys = sorted(strata)
    pools: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for key in ordered_keys:
        members = sorted(
            strata[key], key=lambda item: (item["sequence"], item["seed_id"], item["first_frame"])
        )
        order = np.random.default_rng(AUDIT_SAMPLE_SEED).permutation(len(members))
        pools[key] = [members[i] for i in order]
    sample: list[dict[str, Any]] = []
    cursor = {key: 0 for key in ordered_keys}
    rounds = 0
    while True:
        drew = False
        for key in ordered_keys:
            if cursor[key] < len(pools[key]):
                sample.append(pools[key][cursor[key]])
                cursor[key] += 1
                drew = True
        rounds += 1
        # "until at least 60 have been drawn AND every non-empty stratum has
        # been visited at least once" -- rounds are never truncated mid-way.
        if not drew or (len(sample) >= AUDIT_SAMPLE_MIN and rounds >= 1):
            break
    audit_spec = prereg.payload["identity_evidence"]["mandatory_successor_B8_FROZEN_R10"]
    return {
        "seed": AUDIT_SAMPLE_SEED,
        "target_minimum": AUDIT_SAMPLE_MIN,
        "n_selected": len(sample),
        "n_available": len(windows),
        "rounds": rounds,
        "strata_definition": "sequence x terminal class (frozen R10 order) x anchor_staleness tercile",
        "class_order": list(AUDIT_CLASS_ORDER),
        "anchor_staleness_tercile_cuts": list(cuts),
        "strata_sizes": {
            f"{key[0]}|{AUDIT_CLASS_ORDER[key[1]]}|{key[2]}": len(strata[key])
            for key in ordered_keys
        },
        "rng_stream_note": DISCLOSED_READINGS["A19_audit_sample_rng_stream"],
        "frozen_spec": audit_spec,
        "windows": sample,
        "note": (
            "EMITTING this sample is in scope; RUNNING the M1-A0b audit is NOT. No "
            "statement about PHYSICAL absence may be recorded before it returns."
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _peak_rss_bytes() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        if ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            handle, ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
        return None
    except Exception:  # pragma: no cover - platform dependent
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return None


def run_diagnostic(
    inputs: Sequence[tuple[str, Path, Path, Path]],
    prereg_path: Path,
    *,
    panel: str = "primary",
) -> dict[str, Any]:
    if not inputs:
        raise ContractError("at least one (sequence, scene, tracks, census) input is required")
    if panel not in PANELS:
        raise ContractError(f"unknown panel {panel!r}; expected one of {PANELS}")
    started = time.monotonic()
    prereg = load_prereg(Path(prereg_path))
    base = primary_reading(prereg)

    evaluations: dict[str, dict[str, Any]] = {}
    artifact_ids: dict[str, Any] = {}
    for sequence_name, scene_dir, tracks_path, census_path in inputs:
        if sequence_name in evaluations:
            raise ContractError(f"duplicate sequence name in diagnostic input: {sequence_name}")
        result = evaluate_sequence(
            sequence_name=sequence_name,
            scene_dir=Path(scene_dir),
            tracks_path=Path(tracks_path),
            census_path=Path(census_path),
            prereg=prereg,
        )
        evaluations[sequence_name] = result
        artifact_ids[sequence_name] = {
            "scene_dir": str(Path(scene_dir)),
            "transforms_train_sha256": sha256_file(Path(scene_dir) / "transforms_train.json"),
            "tracks_path": str(Path(tracks_path)),
            "tracks_sha256": sha256_file(Path(tracks_path)),
            "census_path": str(Path(census_path)),
            "census_sha256": sha256_file(Path(census_path)),
            "conversion_provenance": result["substrate"]["conversion_provenance"],
        }

    all_windows: list[WindowEval] = [w for r in evaluations.values() for w in r["windows"]]
    names = sorted(evaluations)
    supply = {name: len(evaluations[name]["windows"]) for name in names}
    prefix = _minimal_prefix(supply)

    # ---- primary classification ------------------------------------------
    primary_assignment, primary_silent = classify_all(all_windows, base, prereg)
    primary_tally = summarize_assignment(
        all_windows, primary_assignment, primary_silent, supply, prereg
    )
    primary_counts = primary_tally.counts
    for empty_class in (CLASS_C4, CLASS_C6):
        if primary_counts[empty_class]:
            offenders = [
                all_windows[i].key
                for i, n in enumerate(primary_assignment)
                if n == empty_class
            ]
            raise ContractError(
                f"implementation inconsistency: {empty_class} is a TAUTOLOGY CHECK asserted "
                f"EMPTY but {len(offenders)} window(s) reached it: {offenders[:5]}"
            )

    # ---- decision-relevant grid (144 re-aggregations, no re-passes) -------
    grid = decision_grid(prereg)
    grid_rows: list[dict[str, Any]] = []
    grid_tallies: list[Tally] = []
    grid_class_sets: dict[tuple[str, int, int], set[str]] = {}
    for reading in grid:
        assigned, silent = classify_all(all_windows, reading, prereg)
        tally = summarize_assignment(all_windows, assigned, silent, supply, prereg)
        grid_tallies.append(tally)
        grid_rows.append(
            {
                "reading": reading.as_dict(),
                "label": reading.label(),
                "counts": tally.counts,
                "C5_total": _c5_total(tally.counts),
                "C5_STRUCTURALLY_SILENT": tally.structurally_silent,
                "per_sequence": tally.per_sequence,
                "reading_label": reading_label(tally, prefix),
                "status_1_core_pooling_a": status_1_core(tally, POOLING_A),
                "status_1_core_pooling_b": status_1_core(tally, POOLING_B),
                "status_2_local_pooling_a": status_2_local(tally, prefix, POOLING_A),
                "status_2_local_pooling_b": status_2_local(tally, prefix, POOLING_B),
            }
        )
        for window, value in zip(all_windows, assigned):
            grid_class_sets.setdefault(window.key, set()).add(value)
    stable_windows = sum(1 for values in grid_class_sets.values() if len(values) == 1)

    descriptive_rows = []
    for reading in descriptive_readings(prereg):
        assigned, silent = classify_all(all_windows, reading, prereg)
        tally = summarize_assignment(all_windows, assigned, silent, supply, prereg)
        descriptive_rows.append(
            {
                "reading": reading.as_dict(),
                "label": reading.label(),
                "counts": tally.counts,
                "C5_total": _c5_total(tally.counts),
                "C5_STRUCTURALLY_SILENT": tally.structurally_silent,
                "declared_decision_irrelevant_by_construction": True,
            }
        )

    # ---- per-sequence roll-ups -------------------------------------------
    per_sequence_out: dict[str, Any] = {}
    window_records: list[dict[str, Any]] = []
    end_truncated_c1a = primary_tally.end_truncated_c1a
    level = prereg.primary_level_index
    primary_radius = RADIUS_INDEX[OCC_RADIUS[base.step1]]
    d2_primary = _d2_key_name(_d2_keys(prereg)[0])
    cursor = 0
    for name in names:
        result = evaluations[name]
        windows = result["windows"]
        counts = _empty_counts()
        silent_count = 0
        cross_tab = {status: 0 for status in STATUS_NAMES}
        cross_tab_by_t1 = {status: {"t1_true": 0, "t1_false": 0} for status in STATUS_NAMES}
        pairs = t3_true = never_queried_pairs = bridged_total = 0
        d2_dist = {verdict: 0 for verdict in D2_VERDICTS}
        agreement: dict[str, dict[str, int]] = {}
        w_table: dict[str, dict[str, int]] = {s: _empty_counts() for s in W_STRATA}
        trunc_table = {"END_TRUNCATED": _empty_counts(), "NOT_END_TRUNCATED": _empty_counts()}
        tolerances: list[float] = []
        query_gaps: list[float] = []
        fully_queried = 0
        for window in windows:
            assigned = primary_assignment[cursor]
            flag = primary_silent[cursor]
            cursor += 1
            counts[assigned] += 1
            silent_count += int(flag)
            detail = classify(window, base, prereg)
            member = window.member_mask(S1_FROZEN)
            selected = member[:, None] & window.w_mask[None, :]
            n_w = int(window.w_mask.sum())
            bridged = len(window.interval) - n_w
            bridged_total += bridged
            statuses = window.status[level][selected]
            t1_values = window.t1[level][selected]
            for code, status_name in enumerate(STATUS_NAMES):
                hits = statuses == code
                cross_tab[status_name] += int(hits.sum())
                cross_tab_by_t1[status_name]["t1_true"] += int((hits & t1_values).sum())
                cross_tab_by_t1[status_name]["t1_false"] += int((hits & ~t1_values).sum())
            pairs += int(selected.sum())
            t3_true += int(window.t3[level][selected].sum())
            never_queried = int((statuses == ST_NEVER_QUERIED).sum())
            never_queried_pairs += never_queried
            verdict = window.d2["verdicts"][d2_primary]
            d2_dist[verdict] += 1
            agreement.setdefault(assigned, {v: 0 for v in D2_VERDICTS})[verdict] += 1
            stratum = _w_stratum(n_w, window.fps)
            w_table[stratum][assigned] += 1
            trunc_table["END_TRUNCATED" if window.end_truncated else "NOT_END_TRUNCATED"][
                assigned
            ] += 1
            window_tols = [
                window.geom[c]["tol_px"]
                for c in window.frozen_S
                if window.geom[c]["tol_px"] is not None
            ]
            tolerances.extend(float(v) for v in window_tols)
            gap = 1.0 - (len(window.queried_S) / len(window.frozen_S)) if window.frozen_S else 0.0
            query_gaps.append(gap)
            if len(window.queried_S) == len(window.frozen_S):
                fully_queried += 1
            window_records.append(
                {
                    "sequence": name,
                    "seed_id": window.seed_id,
                    "first_frame": window.first_frame,
                    "last_frame": window.last_frame,
                    "record_n_frames": window.record_n_frames,
                    "interval_frames": len(window.interval),
                    "W_frames": n_w,
                    "bridged_associated_frames": bridged,
                    "bridged_interruptions": window.bridged_interruptions,
                    "W_seconds": n_w / window.fps if window.fps else None,
                    "n_frames_seconds": (
                        window.record_n_frames / window.fps if window.fps else None
                    ),
                    "W_stratum": stratum,
                    "end_truncated": window.end_truncated,
                    "n_applicable_cameras": len(window.frozen_S),
                    "n_queried_cameras": len(window.queried_S),
                    "applicable_cameras": list(window.frozen_S),
                    "queried_cameras": list(window.queried_S),
                    "never_queried_pair_share": _fraction(never_queried, int(selected.sum())),
                    "s1_matches_frozen_S": sorted(window.per_frame_S) == sorted(window.frozen_S),
                    "class": assigned,
                    "terminal_class": terminal_class(assigned, flag),
                    "structurally_silent": bool(flag),
                    "class_detail": {k: v for k, v in detail.items() if k != "class"},
                    "class_stable_across_grid": len(grid_class_sets[window.key]) == 1,
                    "classes_across_grid": sorted(grid_class_sets[window.key]),
                    "anchor_quality": window.anchor_quality,
                    "ltp_on_eligible_component": _fraction(
                        int((window.ltp_on_eligible[level] & member).sum()), int(member.sum())
                    ),
                    "tol_px": {str(c): window.geom[c]["tol_px"] for c in window.frozen_S},
                    "tallies": {
                        "pairs": int(selected.sum()),
                        "t1_true": int(window.t1[level][selected].sum()),
                        "t2_true": int(window.t2[primary_radius, level][selected].sum()),
                        "t2b_true": int(window.t2b[primary_radius][selected].sum()),
                        "t2_true_by_radius": {
                            radius: int(window.t2[ri, level][selected].sum())
                            for ri, (radius, _) in enumerate(RADII)
                        },
                        "t2b_true_by_radius": {
                            radius: int(window.t2b[ri][selected].sum())
                            for ri, (radius, _) in enumerate(RADII)
                        },
                        "t3_true": int(window.t3[level][selected].sum()),
                        "status": {
                            status_name: int((statuses == code).sum())
                            for code, status_name in enumerate(STATUS_NAMES)
                        },
                        "associated_in_W_by_S5_level": {
                            str(level_px): int(
                                ((window.status[li] == ST_ASSOCIATED) & selected).sum()
                            )
                            for li, level_px in enumerate(prereg.s5_component_px)
                        },
                    },
                    "projection_failures": window.projection_failures,
                    "d2": {
                        "verdicts": window.d2["verdicts"],
                        "entry_corroborated_any": window.d2["entry_corroborated_any"],
                        "per_camera": window.d2["cameras"][d2_primary],
                    },
                }
            )
        block = result["census_block"]
        total = len(windows)
        per_sequence_out[name] = {
            # The r3/r4 W conditions are RECORDED tautology checks (prereg R6);
            # evaluate_sequence computes them and they must SURVIVE into the
            # per-sequence block, or "recorded, not operative" is unverifiable
            # from the artifact (owner fix).
            "tautology_checks_W": result["tautology_checks_W"],
            "true_absence_candidate_count": total,
            "classes": counts,
            "C5_total": _c5_total(counts),
            "C5_STRUCTURALLY_SILENT": silent_count,
            "confirmed_absence_fraction_C1a": _fraction(counts[CLASS_C1A], total),
            "C1a_not_end_truncated": counts[CLASS_C1A] - end_truncated_c1a.get(name, 0),
            "track_coverage_upper_bound": block["statistics"]["track_coverage_upper_bound"],
            "components_total": block["coverage_tallies"]["components_total"],
            "components_covered": block["coverage_tallies"]["components_covered"],
            "report_status_cross_tabulation": cross_tab,
            "report_status_by_T1": cross_tab_by_t1,
            "pairs_evaluated": pairs,
            "t3_any_foreground_rate": _fraction(t3_true, pairs),
            "never_queried_pair_share": _fraction(never_queried_pairs, pairs),
            "bridged_associated_frames": bridged_total,
            "d2_verdict_distribution": d2_dist,
            "d2_agreement_with_assigned_class": agreement,
            "stratification_by_W_seconds": w_table,
            "stratification_by_end_truncation": trunc_table,
            "query_gap": {
                "mean_fraction_of_S_never_queried": (
                    float(np.mean(query_gaps)) if query_gaps else None
                ),
                "max_fraction_of_S_never_queried": (
                    float(np.max(query_gaps)) if query_gaps else None
                ),
                "windows_fully_queried": fully_queried,
                "windows_fully_queried_fraction": _fraction(fully_queried, total),
            },
            "tolerance": {
                "tol_px_median": float(np.median(tolerances)) if tolerances else None,
                "tol_px_min": float(np.min(tolerances)) if tolerances else None,
                "tol_px_max": float(np.max(tolerances)) if tolerances else None,
                "tol_px_p5": float(np.percentile(tolerances, 5)) if tolerances else None,
                "tol_px_p95": float(np.percentile(tolerances, 95)) if tolerances else None,
                "median_eligible_component_area": result["median_eligible_component_area"],
                "median_foreground_area_fraction": result["median_foreground_area_fraction"],
                "decoded_camera_frame_pairs": result["decoded_pairs"],
                "note": DISCLOSED_READINGS["A18_tolerance_and_area_medians_cover_the_decoded_pass"],
            },
            "r_site_census": result["r_site_census"],
            "angular_floor_rad": result["angular_floor_rad"],
            "fps": result["fps"],
            "realized_window": result["realized_window"],
            "training_cameras": result["training_cameras"],
            "held_out_cameras": result["held_out_cameras"],
            "window_frame_count_mismatches": result["window_frame_count_mismatches"],
            "out_of_domain_reports": result["out_of_domain_reports"],
            "decode_stats": result["decode_stats"],
            "wall_seconds": result["wall_seconds"],
        }

    pooled_total = primary_tally.total
    pooled_cross = {status: 0 for status in STATUS_NAMES}
    pooled_pairs = pooled_t3 = pooled_never = pooled_bridged = 0
    pooled_d2 = {verdict: 0 for verdict in D2_VERDICTS}
    for block in per_sequence_out.values():
        for status in STATUS_NAMES:
            pooled_cross[status] += block["report_status_cross_tabulation"][status]
        pooled_pairs += block["pairs_evaluated"]
        pooled_t3 += int(round((block["t3_any_foreground_rate"] or 0.0) * block["pairs_evaluated"]))
        pooled_never += block["report_status_cross_tabulation"]["NEVER_QUERIED"]
        pooled_bridged += block["bridged_associated_frames"]
        for verdict in D2_VERDICTS:
            pooled_d2[verdict] += block["d2_verdict_distribution"][verdict]

    pooled_w_table = {s: _empty_counts() for s in W_STRATA}
    pooled_trunc = {"END_TRUNCATED": _empty_counts(), "NOT_END_TRUNCATED": _empty_counts()}
    for block in per_sequence_out.values():
        for stratum in W_STRATA:
            for class_name in CLASS_NAMES:
                pooled_w_table[stratum][class_name] += block["stratification_by_W_seconds"][
                    stratum
                ][class_name]
        for key in pooled_trunc:
            for class_name in CLASS_NAMES:
                pooled_trunc[key][class_name] += block["stratification_by_end_truncation"][key][
                    class_name
                ]

    # ---- three-way pooling (B7 / R2) --------------------------------------
    pooling = {
        "binding": POOLING_B,
        "binding_rule": prereg.payload["what_the_statuses_mean"]["binding_pooling_R2"],
        "a_pooled": {
            **{name: _fraction(primary_counts[name], pooled_total) for name in CLASS_NAMES},
            "C5_total": _fraction(_c5_total(primary_counts), pooled_total),
            "C1a_adequacy": primary_tally.a_c1a_adequacy(),
            "C2_plus_C3": primary_tally.a_c23(),
            "windows": pooled_total,
        },
        "b_unweighted_mean_over_sequences_with_at_least_10_windows": primary_tally.b_fractions(),
        "b_sequences": primary_tally.big_sequences(),
        "b_note": DISCLOSED_READINGS["A15_binding_pooling_b_may_be_undefined"],
        "c_leave_scissor_out_pooled": primary_tally.c_fractions(),
    }

    # ---- relationships ----------------------------------------------------
    coverage = [per_sequence_out[n]["track_coverage_upper_bound"] for n in names]
    relationships: dict[str, Any] = {
        "n_sequences": len(names),
        "note": (
            "pearson/spearman need >= 3 sequences in one invocation; with fewer they "
            "are null and a reducer must pool the published per-sequence rates. The "
            "post-hoc coverage GROUP is never a decision input."
        ),
        "per_sequence_rates": {},
        "correlations": {},
    }
    for class_name in (*CLASS_NAMES, "C5_total"):
        rates, fractions = [], []
        for name in names:
            counts = primary_tally.per_sequence[name]
            numerator = _c5_total(counts) if class_name == "C5_total" else counts[class_name]
            components = per_sequence_out[name]["components_total"]
            rates.append(numerator / components if components else 0.0)
            fractions.append(_fraction(numerator, supply[name]) or 0.0)
        relationships["per_sequence_rates"][class_name] = {
            "per_eligible_component": dict(zip(names, rates)),
            "fraction_of_windows": dict(zip(names, fractions)),
        }
        relationships["correlations"][class_name] = {
            "coverage_vs_per_eligible_component": {
                "pearson": _pearson(coverage, rates),
                "spearman": _spearman(coverage, rates),
            },
            "coverage_vs_fraction_of_windows": {
                "pearson": _pearson(coverage, fractions),
                "spearman": _spearman(coverage, fractions),
            },
        }

    # ---- audit sample (frozen R10) ----------------------------------------
    audit_sample = build_audit_sample(all_windows, primary_assignment, primary_silent, prereg)

    # ---- measurement closure (r4 precedence) ------------------------------
    if panel == "primary":
        labels = [row["reading_label"] for row in grid_rows]
        # R14: every unquantified clause is evaluated at the PRIMARY READING.
        primary_label = reading_label(primary_tally, prefix)
        grade, agreement = robustness_grade(
            labels, primary_label, prereg.robustness_supermajority
        )
        binding_defined = primary_tally.fractions(POOLING_B) is not None

        # R14_stability_definition: status_1 requires its three conjuncts under
        # (b) AND (a); it is therefore pooling-symmetric by construction and
        # never contributes to POOLING DISAGREEMENT (A12).
        s1_core = {p: status_1_core(primary_tally, p) for p in (POOLING_A, POOLING_B)}
        s2_by_pooling = {
            p: status_2_local(primary_tally, prefix, p) for p in (POOLING_A, POOLING_B)
        }
        s3_by_pooling = {
            p: status_3_local(primary_tally, prefix, p) for p in (POOLING_A, POOLING_B)
        }
        status_1 = (
            None
            if s1_core[POOLING_A] is None or s1_core[POOLING_B] is None
            else bool(s1_core[POOLING_A] and s1_core[POOLING_B])
        )
        pooling_disagreement = None
        if binding_defined:
            pooling_disagreement = bool(
                s2_by_pooling[POOLING_A] != s2_by_pooling[POOLING_B]
                or s3_by_pooling[POOLING_A] != s3_by_pooling[POOLING_B]
            )
        b_values = primary_tally.fractions(POOLING_B)
        c5_binding = b_values["C5_total"] if b_values else None
        # reported diagnostic only (A13); no longer a status_3 conjunct
        c1a_values = [t.counts[CLASS_C1A] for t in grid_tallies]
        c1a_max, c1a_min = (max(c1a_values), min(c1a_values)) if c1a_values else (0, 0)
        c1a_variation = ((c1a_max - c1a_min) / c1a_max) if c1a_max else 0.0

        status_4 = None
        if binding_defined:
            status_4 = bool(
                (c5_binding or 0.0) >= prereg.status_4_c5_min
                or grade == GRADE_FRAGILE
                or pooling_disagreement
            )

        def _apply_precedence(fragile_dominates: bool, unanimity_rule: bool) -> str | None:
            if not binding_defined:
                return None
            if pooling_disagreement:
                return STATUS_4
            if fragile_dominates and grade == GRADE_FRAGILE:
                return STATUS_4
            four = bool(
                (c5_binding or 0.0) >= prereg.status_4_c5_min
                or pooling_disagreement
                or (grade == GRADE_FRAGILE)
                or (unanimity_rule and len(set(labels)) > 1)
            )
            if s2_by_pooling[POOLING_B]:
                return STATUS_2
            if status_1:
                return STATUS_1
            if four:
                return STATUS_4
            if s3_by_pooling[POOLING_B]:
                return STATUS_3
            return STATUS_5

        # A22: the precedence ladder governs; the FRAGILE-dominant alternative
        # is computed and published beside it.
        primary_status = _apply_precedence(fragile_dominates=False, unanimity_rule=False)
        status_if_fragile_dominates = _apply_precedence(
            fragile_dominates=True, unanimity_rule=False
        )
        # A23: the superseded r4 STRICT UNANIMITY behaviour, as required.
        status_under_strict_unanimity = _apply_precedence(
            fragile_dominates=False, unanimity_rule=True
        )
        closure = {
            "primary_status": primary_status,
            "robustness_grade": grade,
            "robustness_agreement_fraction": agreement,
            "robustness_supermajority": prereg.robustness_supermajority,
            "robustness_margin_disclosure": prereg.payload["amendment_2026_08_14_r5_signoff"][
                "R15_OWNER_robustness_grade_replaces_unanimity"
            ]["margin_disclosure"],
            "primary_reading_label": primary_label,
            "primary_status_if_FRAGILE_dominates": status_if_fragile_dominates,
            "primary_status_under_strict_unanimity": status_under_strict_unanimity,
            "strict_unanimity_differs_from_primary_status": bool(
                status_under_strict_unanimity != primary_status
            ),
            "binding_pooling": POOLING_B,
            "binding_pooling_defined": binding_defined,
            "binding_reading": primary_reading(prereg).as_dict(),
            "undefined_note": (
                None
                if binding_defined
                else DISCLOSED_READINGS["A15_binding_pooling_b_may_be_undefined"]
            ),
            "mapping_to_the_user_facing_question": prereg.payload["what_the_statuses_mean"][
                "mapping_to_the_user_facing_question"
            ],
            "scope_caveat": prereg.payload["what_the_statuses_mean"]["scope_B8"],
            "precedence": prereg.payload["what_the_statuses_mean"]["precedence"],
            "fragile_placement_note": DISCLOSED_READINGS[
                "A22_FRAGILE_placement_versus_the_precedence_ladder"
            ],
            "conditions": {
                "status_1": status_1,
                "status_1_core_by_pooling": s1_core,
                "status_2_by_pooling": s2_by_pooling,
                "status_3_by_pooling": s3_by_pooling,
                "status_4": status_4,
                "pooling_disagreement": pooling_disagreement,
                "pooling_disagreement_scope": "status_2 and status_3 only (R14)",
            },
            "inputs": {
                "pooled_total_windows": pooled_total,
                "pooling_a": pooling["a_pooled"],
                "pooling_b": b_values,
                "reading_label_distinct": sorted(set(labels)),
                "reading_label_constant": len(set(labels)) <= 1,
                "reading_label_counts": {
                    value: labels.count(value) for value in sorted(set(labels))
                },
                "minimal_descending_supply_prefix": prefix,
                "prefix_c23_ok_at_primary_reading": primary_tally.prefix_c23_ok(prefix),
                "C1a_count_max_across_grid": c1a_max,
                "C1a_count_min_across_grid": c1a_min,
                "C1a_relative_variation_across_grid": c1a_variation,
                "C1a_relative_variation_note": DISCLOSED_READINGS[
                    "A13_status_3_variation_is_superseded_but_reported"
                ],
                "largest_supply_sequence": primary_tally.largest_supply_sequence(),
                "largest_supply_sequence_C1a_fraction": (
                    primary_tally.largest_supply_c1a_fraction()
                ),
                "per_sequence_supply": supply,
                "sequences_with_at_least_10_windows": primary_tally.big_sequences(),
                "windows_class_stable_across_grid": stable_windows,
            },
            "scope": {
                "panel": panel,
                "sequences": names,
                "windows": pooled_total,
                "note": DISCLOSED_READINGS["A16_status_scope_is_the_invocation"],
            },
        }
    else:
        closure = {
            "primary_status": None,
            "scope": {"panel": panel, "sequences": names, "windows": pooled_total},
            "note": (
                "the decision rules bind to the primary scope; the secondary panel is a "
                "SUPERSET of the corresponding primary rows, not an independent sample, "
                "and reports no coverage-derived statistic (evidence_boundary N5)."
            ),
        }

    evidence_grading = {
        "gate_bearing": False,
        "gate_bearing_note": prereg.payload["what_this_diagnostic_may_not_do"],
        "scope_caveat": prereg.payload["what_the_statuses_mean"]["scope_B8"],
        "anchor_provenance_limit": prereg.payload["anchor"]["anchor_provenance_LIMIT_r3"],
        "angular_floor_weakness": prereg.payload["amendment_2026_08_14_r4_rereview"][
            "R13_angular_floor_weakness_DISCLOSED"
        ],
        "withdrawn_integrity_claim": prereg.payload["amendment_2026_08_14_r4_rereview"][
            "R12_WITHDRAWN_INTEGRITY_CLAIM"
        ],
        "per_class": {
            CLASS_C1A: {
                "primary_evidence": "step-1 occupancy false everywhere in W, T2b false everywhere, and the anchor on an eligible component in >= ltp_on_eligible_component_min of S at ltp_frame",
                "independence_from_evaluated_implementation": "PARTIAL -- calibration and masks only, but P is a consensus triangulation of the very reports whose failure is under test (anchor_provenance_LIMIT_r3)",
                "sensitivity_to_residual_semantics": "tol_c dominates (median 49.8 px on poker, 1.22% of the frame); S5 and the S2 step-1 radius are the levers, which is why r4 added the 0.25x level",
                "confidence_grade": "corroborative, never probative of physical absence",
                "role": "diagnostic-only",
            },
            CLASS_C1B: {
                "primary_evidence": "no ELIGIBLE foreground within the ball but sub-threshold foreground is present",
                "independence_from_evaluated_implementation": "PARTIAL -- as C1a",
                "sensitivity_to_residual_semantics": "the component_min_px ladder is decisive by construction",
                "confidence_grade": "explicitly excluded from adequacy",
                "role": "diagnostic-only",
            },
            CLASS_C1C: {
                "primary_evidence": "no foreground of any size in the ball AND the anchor itself was off eligible foreground at ltp_frame",
                "independence_from_evaluated_implementation": "PARTIAL -- as C1a",
                "sensitivity_to_residual_semantics": "ltp_on_eligible_component_min",
                "confidence_grade": "indicates a drifted or phantom anchor; excluded from adequacy",
                "role": "unresolved",
            },
            CLASS_C2: {
                "primary_evidence": "sustained multi-view-confirmed anchor occupancy with a MISS_TOKEN/OUT_OF_DOMAIN majority at an actually-queried c*",
                "independence_from_evaluated_implementation": "PARTIAL -- occupancy is mask-side, the status is read from the evaluated tracker",
                "sensitivity_to_residual_semantics": "S2/S3/S5/S7; S6 only moves mass to C3; the r2 multi-view clause is weak because the angular floor admits ~90% of pairs (R13)",
                "confidence_grade": "occupancy consistency only; asserts nothing about identity, and cannot separate 'still there' from 'the hand now occupies the vacated site'",
                "role": "diagnostic-only",
            },
            CLASS_C3: {
                "primary_evidence": "sustained multi-view-confirmed anchor occupancy with in-domain reports failing visibility or landing off an eligible component",
                "independence_from_evaluated_implementation": "PARTIAL -- as C2",
                "sensitivity_to_residual_semantics": "S2/S3/S5/S7; S6 only moves mass to C2",
                "confidence_grade": "occupancy consistency only; asserts nothing about identity",
                "role": "diagnostic-only",
            },
            CLASS_C4: {
                "primary_evidence": "none -- TAUTOLOGY CHECK (N2/R9), asserted EMPTY; a non-empty C4 fails the run closed",
                "independence_from_evaluated_implementation": "n/a",
                "sensitivity_to_residual_semantics": "n/a",
                "confidence_grade": "its passing is NOT evidence",
                "role": "unresolved if ever non-zero",
            },
            CLASS_C5_NI: {
                "primary_evidence": "sustained multi-view occupancy but no occupying camera was ever queried for this seed",
                "independence_from_evaluated_implementation": "HIGH for the occupancy limb; the query gap is a tracker-side structural fact",
                "sensitivity_to_residual_semantics": "S7 and S2 move the occupancy limb",
                "confidence_grade": "honest non-identifiable",
                "role": "diagnostic-only",
            },
            CLASS_C5_NS: {
                "primary_evidence": "foreground within the ball somewhere in W without multi-view-confirmed sustained occupancy",
                "independence_from_evaluated_implementation": "HIGH -- tracker-free",
                "sensitivity_to_residual_semantics": "S2/S3/S7 directly",
                "confidence_grade": "honest non-identifiable",
                "role": "diagnostic-only",
            },
            CLASS_C6: {
                "primary_evidence": "none -- TAUTOLOGY CHECK (N2), asserted EMPTY; a non-empty C6 fails the run closed",
                "independence_from_evaluated_implementation": "n/a",
                "sensitivity_to_residual_semantics": "n/a",
                "confidence_grade": "its passing is NOT evidence",
                "role": "unresolved",
            },
        },
        "per_window_grade_rule": (
            "STABLE when the window's class is identical under every one of the "
            f"{len(grid)} decision-relevant readings; UNSTABLE otherwise. Published "
            "per window as class_stable_across_grid."
        ),
    }

    result: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "cell": "M1-absence-diagnostic",
        "panel": panel,
        "disclosed_readings": DISCLOSED_READINGS,
        "reused_census_primitives": [
            "load_component_labels",
            "frustum_contains",
            "round_half_up",
            "index_tracks",
            "ReportIndex",
            "rig_radius",
            "angular_separation_floor",
            "MASK_THRESHOLD",
            "MIN_COMPONENT_PX",
            "VIS_THRESHOLD",
        ],
        "frozen_constants": {
            "occupancy_frame_fraction_min": prereg.occupancy_frame_fraction_min,
            "min_occupancy_cameras": prereg.min_occupancy_cameras,
            "report_majority_fraction": prereg.report_majority_fraction,
            "lineage_iou_min": prereg.lineage_iou_min,
            "component_min_px": prereg.component_min_px,
            "mask_binarize_threshold": prereg.mask_binarize_threshold,
            "visibility_threshold": prereg.visibility_threshold,
            "ltp_on_eligible_component_min": prereg.ltp_on_eligible_component_min,
        },
        "substrate_check": {name: evaluations[name]["substrate"] for name in names},
        "per_sequence": per_sequence_out,
        "pooled": {
            "true_absence_candidate_count": pooled_total,
            "classes": primary_counts,
            "C5_total": _c5_total(primary_counts),
            "C5_STRUCTURALLY_SILENT": primary_tally.structurally_silent,
            "confirmed_absence_fraction_C1a": _fraction(primary_counts[CLASS_C1A], pooled_total),
            "C1a_not_end_truncated": primary_counts[CLASS_C1A] - end_truncated_c1a["pooled"],
            "report_status_cross_tabulation": pooled_cross,
            "pairs_evaluated": pooled_pairs,
            "t3_any_foreground_rate": _fraction(pooled_t3, pooled_pairs),
            "never_queried_pair_share": _fraction(pooled_never, pooled_pairs),
            "bridged_associated_frames": pooled_bridged,
            "d2_verdict_distribution": pooled_d2,
            "stratification_by_W_seconds": pooled_w_table,
            "stratification_by_end_truncation": pooled_trunc,
        },
        "pooling_B7_R2": pooling,
        "windows": window_records,
        "sensitivity_table": grid_rows,
        "sensitivity_descriptive": descriptive_rows,
        "sensitivity_d2": {
            _d2_key_name(key): {
                verdict: sum(
                    1 for w in all_windows if w.d2["verdicts"][_d2_key_name(key)] == verdict
                )
                for verdict in D2_VERDICTS
            }
            for key in _d2_keys(prereg)
        },
        "sensitivity_summary": {
            "n_decision_relevant_readings": len(grid_rows),
            "distinct_class_distributions": len(
                {tuple(sorted(row["counts"].items())) for row in grid_rows}
            ),
            "windows_class_stable": stable_windows,
            "windows_total": pooled_total,
            "robustness_grade": closure.get("robustness_grade"),
            "robustness_agreement_fraction": closure.get("robustness_agreement_fraction"),
            "primary_status_under_strict_unanimity": closure.get(
                "primary_status_under_strict_unanimity"
            ),
            "strict_unanimity_differs_from_primary_status": closure.get(
                "strict_unanimity_differs_from_primary_status"
            ),
        },
        "audit_sample_B8": audit_sample,
        "measurement_closure": closure,
        "relationships": relationships,
        "evidence_grading": evidence_grading,
        "inputs": artifact_ids,
        "provenance": {
            "prereg_path": str(prereg.path),
            "prereg_sha256": prereg.sha256,
            "prereg_revision": PREREG_REVISION_REQUIRED,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene-dir", type=Path, action="append", required=True,
        help="repeatable; paired positionally with --tracks/--census/--sequence-name",
    )
    parser.add_argument("--tracks", type=Path, action="append", required=True)
    parser.add_argument("--census", type=Path, action="append", required=True)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=REPO_ROOT / "configs" / "elgs" / "prereg_m1_absence_diagnostic_v1.json",
    )
    parser.add_argument("--sequence-name", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", choices=list(PANELS), default="primary")
    args = parser.parse_args(argv)

    lengths = {
        "--scene-dir": len(args.scene_dir),
        "--tracks": len(args.tracks),
        "--census": len(args.census),
        "--sequence-name": len(args.sequence_name),
    }
    if len(set(lengths.values())) != 1:
        raise ContractError(f"inputs must pair 1:1, got {lengths}")

    result = run_diagnostic(
        list(zip(args.sequence_name, args.scene_dir, args.tracks, args.census)),
        args.prereg,
        panel=args.panel,
    )
    result["config_sha256"] = hashlib.sha256(
        canonical_json_bytes({"argv": list(argv) if argv else sys.argv[1:]})
    ).hexdigest()
    # Plain strict JSON (floats as JSON numbers) -- the canonical writer's
    # binary64_hex tokens are an identity convention with no repo decoder, the
    # same treatment build_m1_census.main applies to the census artifact.
    body = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    atomic_write_bytes(args.out, body.encode("utf-8") + b"\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "panel": args.panel,
                "windows": result["pooled"]["true_absence_candidate_count"],
                "classes": result["pooled"]["classes"],
                "C5_total": result["pooled"]["C5_total"],
                "C5_STRUCTURALLY_SILENT": result["pooled"]["C5_STRUCTURALLY_SILENT"],
                "bridged_associated_frames": result["pooled"]["bridged_associated_frames"],
                "measurement_closure": result["measurement_closure"].get("primary_status"),
                "audit_sample_n": result["audit_sample_B8"]["n_selected"],
                "wall_seconds": result["provenance"]["wall_seconds"],
                "peak_rss_bytes": result["provenance"]["peak_rss_bytes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
