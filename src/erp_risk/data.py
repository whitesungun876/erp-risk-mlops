"""Strict, deterministic aggregation of public ERPsim CSV records."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


REQUIRED_COLUMNS = {
    "Belegnummer",
    "Label",
    "Soll/Haben-Kennz_",
    "Betrag Hauswaehr",
    "Hauptbuchkonto",
    "Transaktionsart",
}


@dataclass(frozen=True)
class Document:
    """A candidate review unit, not an independently validated fraud event."""

    source_run: str
    document_number: str
    labels: tuple[str, ...]
    line_count: int
    debit_total: Decimal
    credit_total: Decimal
    max_line_amount: Decimal
    distinct_gl_accounts: int
    distinct_transaction_types: int
    source_lines: tuple[int, ...] = ()

    @property
    def unit_id(self) -> str:
        return f"{self.source_run}:{self.document_number}"

    @property
    def is_positive(self) -> bool:
        return any(label != "NonFraud" for label in self.labels)

    def features(self) -> dict[str, int | float]:
        """Features known from the exported record, not a pre-payment snapshot."""
        return {
            "line_count": self.line_count,
            "debit_total": float(self.debit_total),
            "credit_total": float(self.credit_total),
            "max_line_amount": float(self.max_line_amount),
            "distinct_gl_accounts": self.distinct_gl_accounts,
            "distinct_transaction_types": self.distinct_transaction_types,
        }


def read_documents(
    raw_csv: bytes, source_run: str, *, expected_sha256: str | None = None
) -> list[Document]:
    """Parse one source run, preserving all labels within each document number.

    The caller may pin the raw file hash. No fitted statistics or test labels are
    used to construct the structural features.
    """
    if not source_run or ":" in source_run:
        raise ValueError("source_run must be a nonempty identifier without ':'")
    actual_sha256 = hashlib.sha256(raw_csv).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError("source file SHA-256 does not match pinned version")
    try:
        text = raw_csv.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("source CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        raise ValueError(f"missing required columns: {sorted(missing)}")

    groups: dict[str, dict] = {}
    for line_number, row in enumerate(reader, start=2):
        document_number = (row["Belegnummer"] or "").strip()
        label = (row["Label"] or "").strip()
        side = (row["Soll/Haben-Kennz_"] or "").strip()
        account = (row["Hauptbuchkonto"] or "").strip()
        transaction_type = (row["Transaktionsart"] or "").strip()
        if not document_number or not label or not account:
            raise ValueError(f"missing identity or label field at CSV line {line_number}")
        if side not in {"S", "H"}:
            raise ValueError(f"invalid debit/credit code at CSV line {line_number}")
        try:
            amount = Decimal((row["Betrag Hauswaehr"] or "").strip())
        except InvalidOperation as exc:
            raise ValueError(f"invalid amount at CSV line {line_number}") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError(f"nonfinite or negative amount at CSV line {line_number}")
        group = groups.setdefault(
            document_number,
            {"labels": set(), "lines": 0, "debit": Decimal(0),
             "credit": Decimal(0), "max_amount": Decimal(0),
             "accounts": set(), "transaction_types": set(),
             "source_lines": []},
        )
        group["source_lines"].append(line_number)
        group["labels"].add(label)
        group["lines"] += 1
        group["debit" if side == "S" else "credit"] += amount
        group["max_amount"] = max(group["max_amount"], amount)
        group["accounts"].add(account)
        if transaction_type:
            group["transaction_types"].add(transaction_type)

    documents = [
        Document(
            source_run=source_run,
            document_number=number,
            labels=tuple(sorted(group["labels"])),
            line_count=group["lines"],
            debit_total=group["debit"],
            credit_total=group["credit"],
            max_line_amount=group["max_amount"],
            distinct_gl_accounts=len(group["accounts"]),
            distinct_transaction_types=len(group["transaction_types"]),
            source_lines=tuple(group["source_lines"]),
        )
        for number, group in groups.items()
    ]
    return sorted(documents, key=lambda document: document.unit_id)


def assert_disjoint_units(development: list[Document], test: list[Document]) -> None:
    """Refuse overlap in the candidate review-unit key across a split."""
    development_ids = {document.unit_id for document in development}
    test_ids = {document.unit_id for document in test}
    if len(development_ids) != len(development) or len(test_ids) != len(test):
        raise ValueError("duplicate review units within one split")
    overlap = development_ids & test_ids
    if overlap:
        raise ValueError(f"review-unit leakage across split: {len(overlap)} units")
