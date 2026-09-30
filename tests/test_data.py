import unittest

from erp_risk.data import assert_disjoint_units, read_documents


HEADER = (
    "Belegnummer,Label,Soll/Haben-Kennz_,Betrag Hauswaehr,"
    "Hauptbuchkonto,Transaktionsart\n"
)


class DataAggregationTests(unittest.TestCase):
    def test_mixed_labels_remain_visible_and_amounts_are_separate(self):
        raw = (
            HEADER
            + "100,NonFraud,S,10.25,4000,Materialzugang\n"
            + "100,FraudA,H,10.25,5000,Sachkontenbuchung\n"
        ).encode()
        document, = read_documents(raw, "erpsim1")
        self.assertEqual(document.unit_id, "erpsim1:100")
        self.assertEqual(document.labels, ("FraudA", "NonFraud"))
        self.assertTrue(document.is_positive)
        self.assertEqual(document.features()["debit_total"], 10.25)
        self.assertEqual(document.features()["credit_total"], 10.25)
        self.assertEqual(document.distinct_gl_accounts, 2)
        self.assertEqual(document.source_lines, (2, 3))

    def test_hash_mismatch_fails_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_documents(HEADER.encode(), "erpsim1", expected_sha256="0" * 64)

    def test_split_overlap_fails(self):
        raw = (HEADER + "100,NonFraud,S,1,4000,Sachkontenbuchung\n").encode()
        documents = read_documents(raw, "erpsim1")
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint_units(documents, documents)
        assert_disjoint_units(documents, read_documents(raw, "erpsim2"))

    def test_invalid_debit_credit_code_fails(self):
        raw = (HEADER + "100,NonFraud,X,1,4000,Sachkontenbuchung\n").encode()
        with self.assertRaisesRegex(ValueError, "debit/credit"):
            read_documents(raw, "erpsim1")

    def test_missing_transaction_type_is_retained_as_zero_observed_types(self):
        raw = (HEADER + "100,NonFraud,S,1,4000,\n").encode()
        document, = read_documents(raw, "erpsim1")
        self.assertEqual(document.distinct_transaction_types, 0)


if __name__ == "__main__":
    unittest.main()
