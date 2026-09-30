import unittest

from erp_risk.evaluation import ScoredUnit, evaluate_budgets, positive_ranks


class EvaluationTests(unittest.TestCase):
    def test_top_k_random_reference_and_average_precision_by_hand(self):
        units = [
            ScoredUnit("b", 0.9, False),
            ScoredUnit("a", 0.9, True),
            ScoredUnit("c", 0.1, True),
            ScoredUnit("d", 0.0, False),
        ]
        first, half = evaluate_budgets(units, (0.25, 0.5))
        self.assertEqual((first.k, first.hits), (1, 1))
        self.assertEqual(first.random_expected_hits, 0.5)
        self.assertEqual(first.precision_at_k, 1.0)
        self.assertEqual(first.recall_at_k, 0.5)
        self.assertAlmostEqual(first.average_precision, (1 + 2 / 3) / 2)
        self.assertEqual((first.boundary_tie_count, first.boundary_tie_selected), (2, 1))
        self.assertEqual((first.hits_min_across_ties, first.hits_max_across_ties), (0, 1))
        self.assertEqual((half.k, half.hits), (2, 1))
        self.assertEqual((half.boundary_tie_count, half.boundary_tie_selected), (2, 2))
        self.assertEqual((half.hits_min_across_ties, half.hits_max_across_ties), (1, 1))
        self.assertEqual(positive_ranks(units), [1, 3])

    def test_zero_positive_is_not_applicable(self):
        result, = evaluate_budgets([ScoredUnit("a", 0.0, False)], (0.01,))
        self.assertIsNone(result.recall_at_k)
        self.assertIsNone(result.average_precision)
        self.assertEqual(result.random_expected_hits, 0.0)

    def test_duplicate_or_invalid_score_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_budgets([ScoredUnit("a", 1, True), ScoredUnit("a", 0, False)])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_budgets([ScoredUnit("a", float("nan"), True)])


if __name__ == "__main__":
    unittest.main()
