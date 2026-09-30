import tempfile
import unittest
from pathlib import Path

try:
    from erp_risk.xgboost_model import XGBoostRanker
except ImportError:
    XGBoostRanker = None
except Exception as exc:
    if "libomp.dylib" not in str(exc):
        raise
    XGBoostRanker = None

from erp_risk.data import read_documents


HEADER = (
    "Belegnummer,Label,Soll/Haben-Kennz_,Betrag Hauswaehr,"
    "Hauptbuchkonto,Transaktionsart\n"
)


@unittest.skipIf(
    XGBoostRanker is None, "XGBoost or its macOS OpenMP runtime is unavailable"
)
class XGBoostTests(unittest.TestCase):
    def test_model_roundtrip_and_margin_explanation(self):
        rows = [
            f"{i},{'FraudA' if i % 5 == 0 else 'NonFraud'},S,{i * 10},4000,A"
            for i in range(1, 21)
        ]
        development = read_documents((HEADER + "\n".join(rows) + "\n").encode(), "erpsim1")
        test = read_documents(
            (HEADER + "21,NonFraud,S,150,4000,A\n22,FraudA,S,250,5000,B\n").encode(),
            "erpsim2",
        )
        ranker = XGBoostRanker.fit(development, n_estimators=8)
        before = ranker.score(test)
        explained = ranker.explain(test)
        self.assertEqual(len(explained), 2)
        for score, explanation in zip(before, explained, strict=True):
            self.assertAlmostEqual(score, explanation.score_margin, places=5)
            self.assertAlmostEqual(
                score,
                explanation.base_margin
                + sum(item.contribution_to_margin for item in explanation.contributions),
                places=4,
            )
            self.assertEqual(explanation.output_scale, "raw_margin_not_calibrated_probability")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            ranker.save(path)
            loaded = XGBoostRanker.load(path)
            for left, right in zip(before, loaded.score(test), strict=True):
                self.assertAlmostEqual(left, right, places=6)

    def test_one_class_training_fails_explicitly(self):
        normal = read_documents((HEADER + "1,NonFraud,S,10,4000,A\n").encode(), "erpsim1")
        with self.assertRaisesRegex(ValueError, "both target classes"):
            XGBoostRanker.fit(normal)


if __name__ == "__main__":
    unittest.main()
