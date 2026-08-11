"""Unit tests for elgs/classification.py (spec §8).

CPU only, unittest. Oracle: the full enumerated truth table over
(all_below_floors, passes_a, render_flag, tracker_flag) with the §8
precedence, each row annotated with the governing spec sentence:

  "EQUIVALENCE-CLASS = all data terms below preregistered floors"
  "PRIOR-PIVOTAL = fails (a)"
  "DATA-SUPPORTED ... = passes (a) AND >= 1 single-term flag"
  "INTERACTION-SUPPORTED = passes (a), no single-term flag"
"""

import unittest

from depth_visibility.errors import ContractError
from elgs.classification import (
    MANDATORY_QUALIFIER,
    DecisionClass,
    DecisionFlags,
    ITTRecord,
    classify,
    printable_label,
    risk_coverage,
)


class TruthTableTests(unittest.TestCase):
    def test_full_enumerated_truth_table(self):
        # (below_floors, passes_a, render_flag, tracker_flag) -> class
        table = {}
        for below in (False, True):
            for passes_a in (False, True):
                for render in (False, True):
                    for tracker in (False, True):
                        if below:
                            expected = DecisionClass.EQUIVALENCE_CLASS
                        elif not passes_a:
                            expected = DecisionClass.PRIOR_PIVOTAL
                        elif render or tracker:
                            expected = DecisionClass.DATA_SUPPORTED
                        else:
                            expected = DecisionClass.INTERACTION_SUPPORTED
                        table[(below, passes_a, render, tracker)] = expected
        self.assertEqual(len(table), 16)
        for (below, passes_a, render, tracker), expected in table.items():
            flags = DecisionFlags(
                passes_prior_removed=passes_a,
                render_only_flag=render,
                tracker_only_flag=tracker,
                all_data_terms_below_floors=below,
            )
            self.assertEqual(classify(flags), expected, (below, passes_a, render, tracker))

    def test_equivalence_class_precedes_prior_pivotal(self):
        # Precedence: floors first, even when (a) also fails.
        flags = DecisionFlags(False, False, False, True)
        self.assertEqual(classify(flags), DecisionClass.EQUIVALENCE_CLASS
                         if flags.all_data_terms_below_floors else DecisionClass.PRIOR_PIVOTAL)
        both = DecisionFlags(
            passes_prior_removed=False,
            render_only_flag=True,
            tracker_only_flag=True,
            all_data_terms_below_floors=True,
        )
        self.assertEqual(classify(both), DecisionClass.EQUIVALENCE_CLASS)

    def test_post_refit_requirement_enforced(self):
        with self.assertRaises(ContractError):
            DecisionFlags(True, True, False, False, at_post_refit=False)


class LabelTests(unittest.TestCase):
    def test_every_label_carries_the_mandatory_qualifier(self):
        for decision_class in DecisionClass:
            label = printable_label(decision_class)
            self.assertIn(decision_class.value, label)
            self.assertIn(MANDATORY_QUALIFIER, label)
        self.assertEqual(
            MANDATORY_QUALIFIER,
            "(fixed-path decision decomposition, not statistical support)",
        )


class ITTTests(unittest.TestCase):
    def test_committed_requires_screened_and_class(self):
        good = ITTRecord("c1", "FISSION", 3, 1, True, True,
                         decision_class="DATA-SUPPORTED")
        self.assertEqual(good.decision_class, "DATA-SUPPORTED")
        with self.assertRaises(ContractError):
            ITTRecord("c2", "BIRTH", 1, 1, False, True, decision_class="DATA-SUPPORTED")
        with self.assertRaises(ContractError):
            ITTRecord("c3", "BIRTH", 1, 1, True, True)
        with self.assertRaises(ContractError):
            ITTRecord("c4", "MERGE", 1, 1, True, False, decision_class="PRIOR-PIVOTAL")
        with self.assertRaises(ContractError):
            ITTRecord("c5", "MERGE", 1, 1, True, False)  # no rejection reason

    def test_risk_coverage_counts_full_inventory(self):
        records = [
            ITTRecord("a", "FISSION", 1, 1, True, True, decision_class="DATA-SUPPORTED"),
            ITTRecord("b", "BIRTH", 2, 1, True, True, decision_class="PRIOR-PIVOTAL"),
            ITTRecord("c", "MERGE", 3, 1, True, False, rejection_reason="acceptance"),
            ITTRecord("d", "TRUNCATE", 4, 1, False, False, rejection_reason="screened_out"),
            ITTRecord("e", "FISSION", 5, 1, True, True, decision_class="DATA-SUPPORTED"),
        ]
        cov = risk_coverage(records)
        self.assertEqual(cov["total_candidates"], 5)
        self.assertEqual(cov["screened_in"], 4)
        self.assertEqual(cov["committed"], 3)
        self.assertEqual(cov["by_class"], {"DATA-SUPPORTED": 2, "PRIOR-PIVOTAL": 1})
        self.assertEqual(cov["qualifier"], MANDATORY_QUALIFIER)


if __name__ == "__main__":
    unittest.main()
