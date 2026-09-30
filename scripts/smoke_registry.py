"""Register and round-trip only the Day 1 synthetic MLflow smoke model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from probe_workspace import HOST, read_private_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.source_uri.startswith("models:/m-"):
        raise ValueError("source must be an MLflow 3 logged-model URI")
    if len(args.run_id) != 32 or not all(c in "0123456789abcdef" for c in args.run_id):
        raise ValueError("run ID must be a 32-digit lowercase hex string")

    os.environ["DATABRICKS_HOST"] = HOST
    os.environ["DATABRICKS_TOKEN"] = read_private_token(args.token_file)
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    model_name = f"workspace.default.erp_risk_day1_{args.run_id[:8]}"
    registered = mlflow.register_model(args.source_uri, model_name)
    version = str(registered.version)

    reloaded = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    prediction = float(reloaded.predict(np.array([[4.0]], dtype=np.float64))[0])
    if abs(prediction - 4.0) > 1e-6:
        raise AssertionError("registered model round-trip changed its synthetic prediction")

    client = MlflowClient(registry_uri="databricks-uc")
    client.set_registered_model_alias(model_name, "champion", version)
    resolved = client.get_model_version_by_alias(model_name, "champion")
    if str(resolved.version) != version:
        raise AssertionError("champion alias did not resolve to the registered version")
    print(json.dumps({
        "model_name": model_name,
        "version": version,
        "alias": "champion",
        "resolved_version": str(resolved.version),
        "synthetic_prediction": prediction,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
