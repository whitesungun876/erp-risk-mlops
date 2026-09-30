import unittest

from erp_risk.data import read_documents
from erp_risk.rules import RuleBaseline


HEADER = (
    "Belegnummer,Label,Soll/Haben-Kennz_,Betrag Hauswaehr,"
    "Hauptbuchkonto,Transaktionsart\n"
)


class RuleTests(unittest.TestCase):
    def test_thresholds_use_only_development_units_and_evidence_is_reproducible(self):
        development = read_documents(
            (HEADER + "1,NonFraud,S,10,4000,A\n2,NonFraud,S,20,4000,A\n").encode(),
            "erpsim1",
        )
        test, = read_documents((HEADER + "3,FraudA,S,100,4000,A\n").encode(), "erpsim2")
        rules = RuleBaseline.fit(development, quantile=0.5)
        self.assertEqual(rules.debit_total_threshold, 10.0)
        score, evidence = rules.score(test)
        self.assertEqual(score, 1)
        self.assertEqual(evidence[0].code, "unusually_large_debit_total")
        self.assertEqual(evidence[0].threshold, 10.0)


if __name__ == "__main__":
    unittest.main()
