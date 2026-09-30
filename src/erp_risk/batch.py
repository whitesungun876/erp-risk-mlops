"""Deterministic batch-scoring contract and Delta upsert boundary.

This module does not silently pick a fallback model or claim that a local test
proves a Databricks job has run. The caller pins an alias once per batch.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .data import Document
from .isolation import FEATURE_NAMES
from .xgboost_model import MarginExplanation, XGBoostRanker


_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_MERGE_KEY = (
    "target.batch_id = source.batch_id AND "
    "target.model_version = source.model_version AND "
    "target.unit_id = source.unit_id"
)


@dataclass(frozen=True)
class BatchRow:
    batch_id: str
    unit_id: str
    data_version: str
    feature_version: str
    model_version: str
    score: float
    rank: int
    reason_codes: str
    reason_evidence: str
    explanation_type: str

    def as_record(self) -> dict[str, str | float | int]:
        return vars(self).copy()


def resolve_alias_version(client: Any, model_name: str, alias: str = "champion") -> str:
    """Fail closed if a requested Unity Catalog model alias is unavailable."""
    if not _TABLE_NAME.fullmatch(model_name) or not _ALIAS.fullmatch(alias):
        raise ValueError("model name must be catalog.schema.name and alias must be simple")
    try:
        version = client.get_model_version_by_alias(model_name, alias).version
    except Exception as exc:
        raise RuntimeError(f"model alias {model_name}@{alias} could not be resolved") from exc
    if not str(version).isdigit() or int(version) < 1:
        raise ValueError("resolved model version must be a positive integer")
    return str(version)


def load_pinned_xgboost_ranker(
    client: Any, mlflow_module: Any, model_name: str, alias: str = "champion"
) -> tuple[str, XGBoostRanker]:
    """Resolve the alias once and load that immutable version, never the alias URI."""
    version = resolve_alias_version(client, model_name, alias)
    booster = mlflow_module.xgboost.load_model(f"models:/{model_name}/{version}")
    if tuple(getattr(booster, "feature_names", ()) or ()) != FEATURE_NAMES:
        raise ValueError("registered XGBoost model has an unexpected feature schema")
    return version, XGBoostRanker(booster=booster)


def build_xgboost_batch_rows(
    documents: list[Document],
    explanations: list[MarginExplanation],
    *,
    batch_id: str,
    data_version: str,
    feature_version: str,
    model_version: str,
) -> list[BatchRow]:
    """Build one ordered, label-free result row per review unit.

    The contribution evidence describes the model margin, not fraud causality.
    Source line references point back to the pinned CSV when available.
    """
    if not documents or len(documents) != len(explanations):
        raise ValueError("documents and explanations need equal, nonzero lengths")
    if not all((batch_id, data_version, feature_version, model_version)):
        raise ValueError("batch and version identifiers are required")
    if not model_version.isdigit() or int(model_version) < 1:
        raise ValueError("model_version must be a positive registry version")
    unit_ids = [document.unit_id for document in documents]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("duplicate review unit in batch")
    paired = []
    for document, explanation in zip(documents, explanations, strict=True):
        if explanation.output_scale != "raw_margin_not_calibrated_probability":
            raise ValueError("unexpected explanation scale")
        if not math.isfinite(explanation.score_margin):
            raise ValueError("nonfinite model margin")
        if not explanation.contributions or any(
            not math.isfinite(item.contribution_to_margin)
            or not math.isfinite(item.value)
            for item in explanation.contributions
        ):
            raise ValueError("missing or nonfinite contribution evidence")
        if tuple(item.feature for item in explanation.contributions) != FEATURE_NAMES:
            raise ValueError("explanation feature order does not match model schema")
        features = document.features()
        if any(
            not math.isclose(item.value, float(features[item.feature]), rel_tol=1e-9, abs_tol=1e-9)
            for item in explanation.contributions
        ):
            raise ValueError("explanation feature values do not match review unit")
        reconstructed = explanation.base_margin + sum(
            item.contribution_to_margin for item in explanation.contributions
        )
        if not math.isclose(reconstructed, explanation.score_margin, abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError("contributions do not reconstruct model margin")
        paired.append((document, explanation))

    paired.sort(key=lambda item: (-item[1].score_margin, item[0].unit_id))
    rows = []
    for rank, (document, explanation) in enumerate(paired, start=1):
        leading = sorted(
            (item for item in explanation.contributions if item.contribution_to_margin != 0),
            key=lambda item: (-abs(item.contribution_to_margin), item.feature),
        )[:3]
        evidence = {
            "base_margin": explanation.base_margin,
            "features": [
                {
                    "feature": item.feature,
                    "value": item.value,
                    "contribution_to_margin": item.contribution_to_margin,
                }
                for item in leading
            ],
            "output_scale": explanation.output_scale,
        }
        if document.source_lines:
            if any(line < 2 for line in document.source_lines) or len(set(document.source_lines)) != len(document.source_lines):
                raise ValueError("source line references must be unique CSV data lines")
            evidence["source_csv_line_refs"] = list(document.source_lines)
        rows.append(
            BatchRow(
                batch_id=batch_id,
                unit_id=document.unit_id,
                data_version=data_version,
                feature_version=feature_version,
                model_version=model_version,
                score=explanation.score_margin,
                rank=rank,
                reason_codes=json.dumps([
                    ("margin_increase:" if item.contribution_to_margin > 0 else "margin_decrease:")
                    + item.feature for item in leading
                ], separators=(",", ":")),
                reason_evidence=json.dumps(evidence, sort_keys=True, separators=(",", ":")),
                explanation_type="xgboost_native_shap_margin",
            )
        )
    return rows


def merge_batch_rows(spark: Any, target_table: str, rows: list[BatchRow]) -> None:
    """Transactionally upsert into an existing Delta table by the fixed key.

    The target table must already have the BatchRow schema. A live rerun is
    required to verify Databricks behavior; local tests cover only the call
    contract and deterministic row construction.
    """
    if not _TABLE_NAME.fullmatch(target_table):
        raise ValueError("target_table must be a catalog.schema.table identifier")
    if not rows:
        raise ValueError("cannot merge an empty batch")
    keys = [(row.batch_id, row.model_version, row.unit_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate batch upsert key")
    from delta.tables import DeltaTable

    source = spark.createDataFrame([row.as_record() for row in rows])
    (
        DeltaTable.forName(spark, target_table)
        .alias("target")
        .merge(source.alias("source"), _MERGE_KEY)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
