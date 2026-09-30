import unittest

from bootstrap_uncertainty import bootstrap_top_k_hits


class BootstrapTests(unittest.TestCase):
    def test_fixed_seed_integer_bounds_and_ties(self):
        scores = [2.0, 2.0, 1.0, 0.0] * 25
        labels = [True, False, False, False] * 25
        first = bootstrap_top_k_hits(scores, labels, fraction=0.02, repetitions=100)
        self.assertEqual(first, bootstrap_top_k_hits(scores, labels, fraction=0.02, repetitions=100))
        self.assertEqual(first["k"], 2)
        self.assertLessEqual(first["bootstrap_lower_2_5pct"], first["bootstrap_upper_97_5pct"])

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            bootstrap_top_k_hits([1.0], [], fraction=0.01)
        with self.assertRaisesRegex(ValueError, "finite"):
            bootstrap_top_k_hits([float("nan")], [True], fraction=1.0)


if __name__ == "__main__":
    unittest.main()
