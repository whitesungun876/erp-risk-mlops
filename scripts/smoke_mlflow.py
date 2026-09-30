"""One small, synthetic Databricks MLflow tracking smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression

from probe_workspace import HOST, read_private_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    os.environ["DATABRICKS_HOST"] = HOST
    os.environ["DATABRICKS_TOKEN"] = read_private_token(args.token_file)
    os.environ.pop("DATABRICKS_AUTH_TYPE", None)

    client = WorkspaceClient(host=HOST, token=os.environ["DATABRICKS_TOKEN"])
    user = client.current_user.me()
    if not user.user_name:
        raise RuntimeError("current user has no workspace user name")
    experiment = f"/Users/{user.user_name}/erp-risk-mlops-day1"
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)

    features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    target = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    model = LinearRegression().fit(features, target)
    signature = infer_signature(features, model.predict(features))
    with mlflow.start_run(run_name="day1-synthetic-smoke") as run:
        mlflow.log_param("dataset", "synthetic_4_rows")
        mlflow.log_metric("max_abs_error", float(np.max(abs(model.predict(features) - target))))
        model_info = mlflow.sklearn.log_model(
            model,
            name="synthetic_model",
            signature=signature,
            input_example=features[:1],
        )
        print(json.dumps({
            "tracking_uri": "databricks",
            "registry_uri": "databricks-uc",
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": model_info.model_uri,
            "synthetic_rows": len(features),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
