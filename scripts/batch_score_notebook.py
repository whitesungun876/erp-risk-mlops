# Databricks notebook source
"""Private ERPsim feature batch: pinned UC model version and Delta upsert.

The input contains only unit IDs and six precomputed structural features.
Scores are raw XGBoost margins, not calibrated fraud probabilities.
"""

import json
import math

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from delta.tables import DeltaTable
from mlflow import MlflowClient
from pyspark.sql import types as T


INPUT_PATH = "/Volumes/workspace/default/erp_risk_inputs/erpsim2_features_with_refs.parquet"
OUTPUT_TABLE = "workspace.default.erp_risk_scores_a3b3cc76"
MODEL_NAME = "workspace.default.erp_risk_xgboost_a3b3cc76"
BATCH_ID = "erpsim2-with-refs-v2"
DATA_VERSION = "472706718323d0a330f1e406db41d1f959304ce27a618468c6df32bffe13ded7"
FEATURE_VERSION = "day3-features-plus-source-refs-v2"

mlflow.set_registry_uri("databricks-uc")
registered = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(
    MODEL_NAME, "champion"
)
version = str(registered.version)
if not version.isdigit() or int(version) < 1:
    raise ValueError("champion alias did not resolve to a valid version")
booster = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{version}")
feature_names = tuple(booster.feature_names or ())
expected_names = (
    "line_count", "debit_total", "credit_total", "max_line_amount",
    "distinct_gl_accounts", "distinct_transaction_types",
)
if feature_names != expected_names:
    raise ValueError("registered model feature schema differs from frozen batch contract")

frame = spark.read.parquet(INPUT_PATH).select("unit_id", *feature_names, "source_lines").toPandas()
if frame.empty or frame.unit_id.isna().any() or frame.unit_id.duplicated().any():
    raise ValueError("empty input, missing unit ID, or duplicate review unit")
matrix = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
if not np.isfinite(matrix).all():
    raise ValueError("input features must be finite")
dmatrix = xgb.DMatrix(matrix, feature_names=list(feature_names))
scores = booster.predict(dmatrix, output_margin=True)
contributions = booster.predict(dmatrix, pred_contribs=True)
if not np.isfinite(scores).all() or not np.isfinite(contributions).all():
    raise ValueError("nonfinite score or contribution")
if not np.allclose(contributions.sum(axis=1), scores, atol=1e-5, rtol=1e-5):
    raise ValueError("SHAP contributions do not reconstruct raw margins")

ranked = sorted(range(len(frame)), key=lambda index: (-float(scores[index]), frame.unit_id.iloc[index]))
records = []
for rank, index in enumerate(ranked, start=1):
    leading = sorted(
        ((name, float(matrix[index, position]), float(contributions[index, position]))
         for position, name in enumerate(feature_names)
         if contributions[index, position] != 0),
        key=lambda item: (-abs(item[2]), item[0]),
    )[:3]
    evidence = {
        "base_margin": float(contributions[index, -1]),
        "features": [
            {"feature": name, "value": value, "contribution_to_margin": contribution}
            for name, value, contribution in leading
        ],
        "output_scale": "raw_margin_not_calibrated_probability",
    }
    source_lines = json.loads(frame.source_lines.iloc[index])
    if (not isinstance(source_lines, list) or not source_lines
            or any(type(line) is not int or line < 2 for line in source_lines)
            or len(set(source_lines)) != len(source_lines)):
        raise ValueError("invalid source CSV line references")
    evidence["source_csv_line_refs"] = source_lines
    records.append((
        BATCH_ID, str(frame.unit_id.iloc[index]), DATA_VERSION, FEATURE_VERSION,
        version, float(scores[index]), rank,
        json.dumps([
            ("margin_increase:" if contribution > 0 else "margin_decrease:") + name
            for name, _, contribution in leading
        ], separators=(",", ":")),
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        "xgboost_native_shap_margin",
    ))

schema = T.StructType([
    T.StructField("batch_id", T.StringType(), False),
    T.StructField("unit_id", T.StringType(), False),
    T.StructField("data_version", T.StringType(), False),
    T.StructField("feature_version", T.StringType(), False),
    T.StructField("model_version", T.StringType(), False),
    T.StructField("score", T.DoubleType(), False),
    T.StructField("rank", T.LongType(), False),
    T.StructField("reason_codes", T.StringType(), False),
    T.StructField("reason_evidence", T.StringType(), False),
    T.StructField("explanation_type", T.StringType(), False),
])
source = spark.createDataFrame(records, schema=schema)
spark.sql(f"""CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
  batch_id STRING, unit_id STRING, data_version STRING, feature_version STRING,
  model_version STRING, score DOUBLE, `rank` BIGINT, reason_codes STRING,
  reason_evidence STRING, explanation_type STRING
) USING DELTA""")
(
    DeltaTable.forName(spark, OUTPUT_TABLE).alias("target")
    .merge(
        source.alias("source"),
        "target.batch_id = source.batch_id AND "
        "target.model_version = source.model_version AND "
        "target.unit_id = source.unit_id",
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)
output = spark.table(OUTPUT_TABLE)
persisted = output.where((output.batch_id == BATCH_ID) & (output.model_version == version))
if persisted.count() != len(frame):
    raise AssertionError("Delta output count differs from batch input count")
unique_output_count = persisted.select("unit_id").distinct().count()
if unique_output_count != len(frame):
    raise AssertionError("Delta output has duplicate or missing review units")
top_one_percent = math.ceil(len(records) * 0.01)
correlated_amount_attributions = sum(
    {"debit_total", "credit_total"}.issubset({
        item["feature"] for item in json.loads(record[8])["features"]
    })
    for record in records[:top_one_percent]
)
dbutils.notebook.exit(json.dumps({
    "batch_id": BATCH_ID,
    "input_count": len(frame),
    "output_count": len(frame),
    "unique_output_count": unique_output_count,
    "top_one_percent_units": top_one_percent,
    "top_one_percent_both_amount_features_in_explanation": correlated_amount_attributions,
    "model_version": version,
    "table": OUTPUT_TABLE,
}))
