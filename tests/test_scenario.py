import unittest

from erp_risk.data import read_documents
from erp_risk.scenario import leave_one_raw_label_out, shared_scenario_labels


HEADER = (
    "Belegnummer,Label,Soll/Haben-Kennz_,Betrag Hauswaehr,"
    "Hauptbuchkonto,Transaktionsart\n"
)


class ScenarioTests(unittest.TestCase):
    def test_holds_out_all_target_documents_without_relabeling(self):
        development = read_documents(
            (
                HEADER
                + "1,NonFraud,S,10,4000,A\n"
                + "2,FraudA,S,20,5000,B\n"
                + "2,NonFraud,H,20,6000,B\n"
                + "3,FraudB,S,30,7000,C\n"
            ).encode(),
            "erpsim1",
        )
        test = read_documents(
            (
                HEADER
                + "4,NonFraud,S,10,4000,A\n"
                + "5,FraudA,S,40,5000,B\n"
                + "6,FraudB,S,50,7000,C\n"
            ).encode(),
            "erpsim2",
        )
        self.assertEqual(shared_scenario_labels(development, test), ("FraudA", "FraudB"))
        fold = leave_one_raw_label_out(development, test, "FraudA")
        self.assertEqual(fold.removed_development_units, 1)
        self.assertEqual(fold.test_positive_units, 1)
        self.assertEqual({item.document_number for item in fold.development}, {"1", "3"})
        self.assertEqual({item.document_number for item in fold.test}, {"4", "5"})
        self.assertFalse(fold.event_links_verified)


if __name__ == "__main__":
    unittest.main()
