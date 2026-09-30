import unittest

from erp_risk.data import read_documents
from erp_risk.isolation import IsolationRanker, feature_matrix


HEADER = (
    "Belegnummer,Label,Soll/Haben-Kennz_,Betrag Hauswaehr,"
    "Hauptbuchkonto,Transaktionsart\n"
)


class IsolationTests(unittest.TestCase):
    def test_label_changes_cannot_enter_feature_matrix(self):
        normal, = read_documents(
            (HEADER + "1,NonFraud,S,10,4000,A\n").encode(), "erpsim1"
        )
        fraud, = read_documents(
            (HEADER + "1,FraudA,S,10,4000,A\n").encode(), "erpsim1"
        )
        self.assertEqual(feature_matrix([normal]).tolist(), feature_matrix([fraud]).tolist())

    def test_fit_and_score_on_distinct_batches(self):
        development = read_documents(
            (HEADER + "1,NonFraud,S,10,4000,A\n2,NonFraud,S,20,4000,A\n"
             + "3,FraudA,S,30,5000,B\n").encode(),
            "erpsim1",
        )
        test = read_documents(
            (HEADER + "4,NonFraud,S,15,4000,A\n5,FraudB,S,100,7000,C\n").encode(),
            "erpsim2",
        )
        ranker = IsolationRanker.fit(development, n_estimators=10)
        self.assertEqual(len(ranker.score(test)), 2)
        self.assertEqual(ranker.score(test), ranker.score(test))


if __name__ == "__main__":
    unittest.main()
