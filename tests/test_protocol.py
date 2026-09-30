import json
import unittest
from pathlib import Path

from erp_risk.data import read_documents
from erp_risk.protocol import load_config, summarize_split


HEADER = (
    "Belegnummer,Label,Soll/Haben-Kennz_,Betrag Hauswaehr,"
    "Hauptbuchkonto,Transaktionsart\n"
)


class ProtocolTests(unittest.TestCase):
    def test_config_feature_schema_and_split_summary(self):
        config = load_config(Path(__file__).parents[1] / "configs" / "experiment.json")
        development = read_documents(
            (HEADER + "1,NonFraud,S,10,4000,A\n").encode(), "erpsim1"
        )
        test = read_documents((HEADER + "2,FraudA,S,20,5000,B\n").encode(), "erpsim2")
        summary = summarize_split(development, test, config)
        self.assertEqual(summary.development_units, 1)
        self.assertEqual(summary.test_positive_units, 1)
        self.assertFalse(summary.event_independence_verified)
        self.assertEqual(json.loads(summary.to_json())["test_source"], "erpsim2")

    def test_unfrozen_direction_fails(self):
        config = load_config(Path(__file__).parents[1] / "configs" / "experiment.json")
        development = read_documents(
            (HEADER + "1,NonFraud,S,10,4000,A\n").encode(), "other_run"
        )
        test = read_documents((HEADER + "2,FraudA,S,20,5000,B\n").encode(), "erpsim2")
        with self.assertRaisesRegex(ValueError, "not frozen"):
            summarize_split(development, test, config)


if __name__ == "__main__":
    unittest.main()
