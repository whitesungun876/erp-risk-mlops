"""Upload and submit a single synthetic Databricks serverless notebook run."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, SubmitTask
from databricks.sdk.service.workspace import ImportFormat, Language

from probe_workspace import HOST, read_private_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="the already verified Day 1 MLflow run ID")
    args = parser.parse_args()
    if len(args.run_id) != 32 or not all(c in "0123456789abcdef" for c in args.run_id):
        raise ValueError("run ID must be a 32-digit lowercase hex string")

    client = WorkspaceClient(host=HOST, token=read_private_token(args.token_file))
    user = client.current_user.me()
    if not user.user_name:
        raise RuntimeError("current user has no workspace user name")
    folder = f"/Users/{user.user_name}/erp-risk-mlops"
    notebook_path = f"{folder}/day1_smoke_{args.run_id[:8]}"
    source = Path(__file__).with_name("day1_smoke_notebook.py").read_bytes()

    client.workspace.mkdirs(folder)
    client.workspace.import_(
        notebook_path,
        content=base64.b64encode(source).decode("ascii"),
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=False,
    )
    status = client.workspace.get_status(notebook_path)
    if not status.object_type or status.object_type.value != "NOTEBOOK":
        raise RuntimeError("uploaded source is not a Databricks notebook")

    waiter = client.jobs.submit(
        run_name="erp-risk-day1-synthetic-batch",
        idempotency_token=f"erp-risk-day1-{args.run_id}",
        tasks=[SubmitTask(
            task_key="synthetic_batch",
            notebook_task=NotebookTask(notebook_path=notebook_path),
        )],
        timeout_seconds=900,
    )
    print(json.dumps({
        "run_id": waiter.run_id,
        "task_key": "synthetic_batch",
        "notebook_uploaded": True,
        "job_submitted": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
