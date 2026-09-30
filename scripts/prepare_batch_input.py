"""Build a private, label-free Parquet input from the pinned ERPsim test run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from erp_risk.data import read_documents
from erp_risk.isolation import FEATURE_NAMES
from erp_risk.protocol import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source_run = config["primary_split"]["test"]
    documents = read_documents(
        args.source.read_bytes(), source_run,
        expected_sha256=config["source_sha256"][source_run],
    )
    records = [
        {"unit_id": document.unit_id, **document.features(),
         "source_lines": json.dumps(document.source_lines, separators=(",", ":"))}
        for document in documents
    ]
    frame = pd.DataFrame.from_records(records, columns=["unit_id", *FEATURE_NAMES, "source_lines"])
    if frame.unit_id.duplicated().any() or frame.isna().any().any():
        raise ValueError("batch input contains duplicate units or missing features")
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing batch input")
    frame.to_parquet(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "parquet_sha256": digest,
        "source_run": source_run,
        "source_sha256": config["source_sha256"][source_run],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
