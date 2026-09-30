"""Submit a one-time serverless job for the private, label-free feature batch."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import Environment
from databricks.sdk.service.jobs import JobEnvironment, NotebookTask, SubmitTask
from databricks.sdk.service.workspace import ImportFormat, Language

from probe_workspace import HOST, read_private_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--attempt", choices=("first", "retry", "rerun"), required=True)
    args = parser.parse_args()
    client = WorkspaceClient(host=HOST, token=read_private_token(args.token_file))
    user = client.current_user.me()
    if not user.user_name:
        raise RuntimeError("current user has no workspace user name")
    folder = f"/Users/{user.user_name}/erp-risk-mlops"
    path = f"{folder}/batch_score_a3b3cc76"
    client.workspace.mkdirs(folder)
    if args.attempt in {"first", "retry"}:
        source = Path(__file__).with_name("batch_score_notebook.py").read_bytes()
        client.workspace.import_(
            path, content=base64.b64encode(source).decode("ascii"),
            format=ImportFormat.SOURCE, language=Language.PYTHON,
            overwrite=args.attempt == "retry",
        )
    status = client.workspace.get_status(path)
    if not status.object_type or status.object_type.value != "NOTEBOOK":
        raise RuntimeError("batch source is not a Databricks notebook")
    waiter = client.jobs.submit(
        run_name=f"erp-risk-real-batch-{args.attempt}",
        idempotency_token=f"erp-risk-batch-a3b3cc76-{args.attempt}",
        tasks=[SubmitTask(
            task_key="score_batch",
            notebook_task=NotebookTask(notebook_path=path),
            environment_key="batch_scoring",
        )],
        environments=[JobEnvironment(
            environment_key="batch_scoring",
            spec=Environment(
                environment_version="5",
                dependencies=["mlflow==3.16.1", "xgboost-cpu==3.4.1", "numpy==2.5.3"],
            ),
        )],
        timeout_seconds=1200,
    )
    print(json.dumps({"run_id": waiter.run_id, "attempt": args.attempt}, sort_keys=True))


if __name__ == "__main__":
    main()
