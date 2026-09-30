"""Register the tracked primary XGBoost artifact as an operational demo.

This does not select an empirical winner from the outer test. The model is
registered to exercise version pinning and batch delivery, with that caveat
recorded on the model version.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import mlflow.xgboost
from mlflow import MlflowClient

from erp_risk.data import read_documents
from erp_risk.protocol import load_config
from erp_risk.xgboost_model import XGBoostRanker
from probe_workspace import HOST, read_private_token
from track_experiments import metrics_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.source_uri.startswith("models:/m-"):
        raise ValueError("source must be an MLflow 3 logged-model URI")
    config = load_config(args.config)
    os.environ["DATABRICKS_HOST"] = HOST
    os.environ["DATABRICKS_TOKEN"] = read_private_token(args.token_file)
    os.environ.pop("DATABRICKS_AUTH_TYPE", None)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    run = mlflow.get_run(args.run_id)
    if run.info.status != "FINISHED" or run.data.params.get("model") != "xgboost" or run.data.params.get("direction") != "primary":
        raise ValueError("source run is not a finished primary XGBoost experiment")
    if run.data.params.get("erpsim2_sha256") != config["source_sha256"]["erpsim2"]:
        raise ValueError("tracked test hash differs from frozen config")

    name = "workspace.default.erp_risk_xgboost_a3b3cc76"
    registered = mlflow.register_model(args.source_uri, name)
    version = str(registered.version)
    booster = mlflow.xgboost.load_model(f"models:/{name}/{version}")
    ranker = XGBoostRanker(booster=booster)
    test = read_documents(
        args.test_csv.read_bytes(), "erpsim2",
        expected_sha256=config["source_sha256"]["erpsim2"],
    )
    metrics, _ = metrics_for(test, ranker.score(test), tuple(config["review_fractions"]))
    if metrics["hits_1pct"] != run.data.metrics["hits_1pct"] or metrics["k_1pct"] != run.data.metrics["k_1pct"]:
        raise AssertionError("registered model does not reproduce tracked outer-test ranking")

    client = MlflowClient(registry_uri="databricks-uc")
    client.set_model_version_tag(
        name, version, "selection_basis",
        "operational_demo_not_outer_test_winner_claim",
    )
    client.set_registered_model_alias(name, "champion", version)
    resolved = client.get_model_version_by_alias(name, "champion")
    if str(resolved.version) != version:
        raise AssertionError("champion alias did not resolve to the checked version")
    print(json.dumps({
        "model_name": name,
        "version": version,
        "alias": "champion",
        "source_run_id": run.info.run_id,
        "test_units": len(test),
        "k_1pct": int(metrics["k_1pct"]),
        "hits_1pct": int(metrics["hits_1pct"]),
        "selection_basis": "operational_demo_not_outer_test_winner_claim",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
