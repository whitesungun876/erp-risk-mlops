"""Independently reconcile private batch features and CSV line references.

Only aggregate counts are printed. No source rows or unit-level results leave
this process; the source file must match the pinned hash in experiment.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from erp_risk.protocol import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source_run = config["primary_split"]["test"]
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != config["source_sha256"][source_run]:
        raise ValueError("source CSV hash differs from frozen experiment")
    rows = {}
    with args.source.open("r", encoding="utf-8-sig", newline="") as handle:
        for line, record in enumerate(csv.DictReader(handle), start=2):
            rows[line] = record
    frame = pd.read_parquet(args.parquet)
    if frame.unit_id.duplicated().any():
        raise ValueError("duplicate review units in Parquet")
    referenced = set()
    for record in frame.itertuples(index=False):
        lines = json.loads(record.source_lines)
        if not lines or lines != sorted(set(lines)) or len(lines) != record.line_count:
            raise ValueError("missing, repeated or out-of-order source lines")
        source_records = []
        for line in lines:
            if type(line) is not int or line not in rows or line in referenced:
                raise ValueError("invalid or reused source CSV line reference")
            source_record = rows[line]
            if f"{source_run}:{source_record['Belegnummer'].strip()}" != record.unit_id:
                raise ValueError("source line belongs to another review unit")
            referenced.add(line)
            source_records.append(source_record)
        debit = sum(
            (Decimal(row["Betrag Hauswaehr"]) for row in source_records
             if row["Soll/Haben-Kennz_"] == "S"), Decimal(0)
        )
        credit = sum(
            (Decimal(row["Betrag Hauswaehr"]) for row in source_records
             if row["Soll/Haben-Kennz_"] == "H"), Decimal(0)
        )
        maximum = max(Decimal(row["Betrag Hauswaehr"]) for row in source_records)
        if (float(debit) != record.debit_total or float(credit) != record.credit_total
                or float(maximum) != record.max_line_amount
                or len({row["Hauptbuchkonto"].strip() for row in source_records}) != record.distinct_gl_accounts
                or len({row["Transaktionsart"].strip() for row in source_records
                        if row["Transaktionsart"].strip()}) != record.distinct_transaction_types):
            raise ValueError("feature does not reconcile with referenced source lines")
    if referenced != rows.keys():
        raise ValueError("not all CSV data lines are covered by review units")
    print(json.dumps({"review_units_verified": len(frame), "source_lines_verified": len(referenced),
                      "source_sha256": config["source_sha256"][source_run]}, sort_keys=True))


if __name__ == "__main__":
    main()
