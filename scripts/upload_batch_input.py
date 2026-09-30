"""Upload only the private, label-free feature Parquet to a managed UC volume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import VolumeType

from probe_workspace import HOST, read_private_token


VOLUME = "workspace.default.erp_risk_inputs"
REMOTE_ROOT = "/Volumes/workspace/default/erp_risk_inputs"
ALLOWED_NAMES = {"erpsim2_features.parquet", "erpsim2_features_with_refs.parquet"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    source = args.input
    if source.name not in ALLOWED_NAMES or not source.is_file():
        raise ValueError("expected the prepared, label-free erpsim2 feature Parquet")
    remote_path = f"{REMOTE_ROOT}/{source.name}"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    client = WorkspaceClient(host=HOST, token=read_private_token(args.token_file))
    try:
        client.volumes.read(VOLUME)
    except NotFound:
        client.volumes.create(
            catalog_name="workspace", schema_name="default", name="erp_risk_inputs",
            volume_type=VolumeType.MANAGED,
            comment="Private derived feature input for ERP risk batch demonstration; no labels or raw CSV",
        )
    with source.open("rb") as handle:
        client.files.upload(remote_path, handle, overwrite=False)
    print(json.dumps({
        "volume": VOLUME,
        "remote_path": remote_path,
        "bytes": source.stat().st_size,
        "parquet_sha256": digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
