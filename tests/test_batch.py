import json
import sys
import types
import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import Mock, patch

from erp_risk.batch import (
    build_xgboost_batch_rows,
    load_pinned_xgboost_ranker,
    merge_batch_rows,
    resolve_alias_version,
)
from erp_risk.data import Document
from erp_risk.isolation import FEATURE_NAMES
from erp_risk.xgboost_model import FeatureContribution, MarginExplanation


def document(number: str) -> Document:
    return Document(
        source_run="test",
        document_number=number,
        labels=("NonFraud",),
        line_count=1,
        debit_total=Decimal("10"),
        credit_total=Decimal("10"),
        max_line_amount=Decimal("10"),
        distinct_gl_accounts=1,
        distinct_transaction_types=1,
    )


def explanation(score: float) -> MarginExplanation:
    features = document("fixture").features()
    return MarginExplanation(
        base_margin=0.0,
        score_margin=score,
        contributions=tuple(
            FeatureContribution(name, float(features[name]), score if index == 0 else 0.0)
            for index, name in enumerate(FEATURE_NAMES)
        ),
    )


class BatchTests(unittest.TestCase):
    def test_alias_pins_one_version_and_missing_alias_fails(self):
        client = Mock()
        client.get_model_version_by_alias.return_value.version = "7"
        self.assertEqual(resolve_alias_version(client, "catalog.schema.risk"), "7")
        client.get_model_version_by_alias.assert_called_once_with(
            "catalog.schema.risk", "champion"
        )
        client.get_model_version_by_alias.side_effect = LookupError("missing")
        with self.assertRaisesRegex(RuntimeError, "could not be resolved"):
            resolve_alias_version(client, "catalog.schema.risk", "missing")
        with self.assertRaisesRegex(ValueError, "catalog.schema.name"):
            resolve_alias_version(client, "risk")

    def test_registered_model_loads_by_pinned_version_with_schema_check(self):
        client = Mock()
        client.get_model_version_by_alias.return_value.version = "7"
        booster = Mock()
        booster.feature_names = list(FEATURE_NAMES)
        mlflow_module = Mock()
        mlflow_module.xgboost.load_model.return_value = booster
        version, ranker = load_pinned_xgboost_ranker(
            client, mlflow_module, "catalog.schema.risk"
        )
        self.assertEqual(version, "7")
        self.assertIs(ranker.booster, booster)
        mlflow_module.xgboost.load_model.assert_called_once_with(
            "models:/catalog.schema.risk/7"
        )
        booster.feature_names = ["wrong"]
        with self.assertRaisesRegex(ValueError, "unexpected feature schema"):
            load_pinned_xgboost_ranker(client, mlflow_module, "catalog.schema.risk")

    def test_deterministic_ranking_keys_and_label_free_evidence(self):
        docs = [document("B"), document("A")]
        rows = build_xgboost_batch_rows(
            docs,
            [explanation(2.0), explanation(2.0)],
            batch_id="batch-1",
            data_version="sha256:test",
            feature_version="v1",
            model_version="7",
        )
        self.assertEqual([(row.unit_id, row.rank) for row in rows], [("test:A", 1), ("test:B", 2)])
        self.assertEqual(
            {(row.batch_id, row.model_version, row.unit_id) for row in rows},
            {("batch-1", "7", "test:A"), ("batch-1", "7", "test:B")},
        )
        self.assertEqual(rows, build_xgboost_batch_rows(
            docs, [explanation(2.0), explanation(2.0)],
            batch_id="batch-1", data_version="sha256:test",
            feature_version="v1", model_version="7",
        ))
        self.assertEqual(json.loads(rows[0].reason_codes), ["margin_increase:line_count"])
        self.assertNotIn("labels", rows[0].reason_evidence)
        self.assertEqual(json.loads(rows[0].reason_evidence)["features"][0]["value"], 1.0)
        traced = build_xgboost_batch_rows(
            [replace(document("T"), source_lines=(2, 7))], [explanation(1.0)],
            batch_id="batch-1", data_version="sha256:test",
            feature_version="v1", model_version="7",
        )[0]
        self.assertEqual(json.loads(traced.reason_evidence)["source_csv_line_refs"], [2, 7])
        zero_row = build_xgboost_batch_rows(
            [document("Z")], [explanation(0.0)],
            batch_id="batch-1", data_version="sha256:test",
            feature_version="v1", model_version="7",
        )[0]
        self.assertEqual(json.loads(zero_row.reason_codes), [])

    def test_reject_duplicate_units_bad_versions_and_bad_explanations(self):
        kwargs = dict(batch_id="b", data_version="d", feature_version="f", model_version="1")
        with self.assertRaisesRegex(ValueError, "duplicate review unit"):
            build_xgboost_batch_rows([document("A"), document("A")], [explanation(1), explanation(2)], **kwargs)
        with self.assertRaisesRegex(ValueError, "positive registry version"):
            build_xgboost_batch_rows([document("A")], [explanation(1)], **(kwargs | {"model_version": "0"}))
        with self.assertRaisesRegex(ValueError, "do not reconstruct"):
            build_xgboost_batch_rows([document("A")], [replace(explanation(1), score_margin=2)], **kwargs)
        wrong_feature = replace(
            explanation(1),
            contributions=(FeatureContribution("line_count", 9.0, 1.0),)
            + explanation(1).contributions[1:],
        )
        with self.assertRaisesRegex(ValueError, "feature values do not match"):
            build_xgboost_batch_rows([document("A")], [wrong_feature], **kwargs)
        with self.assertRaisesRegex(ValueError, "source line references"):
            build_xgboost_batch_rows(
                [replace(document("A"), source_lines=(2, 2))],
                [explanation(1)], **kwargs,
            )

    def test_delta_merge_uses_full_key_and_refuses_duplicate_source(self):
        rows = build_xgboost_batch_rows(
            [document("A")], [explanation(1)], batch_id="b",
            data_version="d", feature_version="f", model_version="1",
        )
        spark = Mock()
        source = spark.createDataFrame.return_value
        table = Mock()
        table.alias.return_value = table
        table.merge.return_value = table
        table.whenMatchedUpdateAll.return_value = table
        table.whenNotMatchedInsertAll.return_value = table
        fake_delta = types.ModuleType("delta")
        fake_tables = types.ModuleType("delta.tables")
        fake_tables.DeltaTable = Mock()
        fake_tables.DeltaTable.forName.return_value = table
        with patch.dict(sys.modules, {"delta": fake_delta, "delta.tables": fake_tables}):
            merge_batch_rows(spark, "catalog.schema.results", rows)
        fake_tables.DeltaTable.forName.assert_called_once_with(spark, "catalog.schema.results")
        source.alias.assert_called_once_with("source")
        condition = table.merge.call_args.args[1]
        self.assertIn("target.batch_id = source.batch_id", condition)
        self.assertIn("target.model_version = source.model_version", condition)
        self.assertIn("target.unit_id = source.unit_id", condition)
        table.whenMatchedUpdateAll.assert_called_once()
        table.whenNotMatchedInsertAll.assert_called_once()
        table.execute.assert_called_once()
        with self.assertRaisesRegex(ValueError, "duplicate batch upsert key"):
            merge_batch_rows(spark, "catalog.schema.results", rows + rows)
        with self.assertRaisesRegex(ValueError, "catalog.schema.table"):
            merge_batch_rows(spark, "catalog.schema.results;DROP", rows)


if __name__ == "__main__":
    unittest.main()
