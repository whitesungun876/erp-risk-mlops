"""Frozen two-run split contract and compact, non-record-level evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Document, assert_disjoint_units
from .isolation import FEATURE_NAMES


@dataclass(frozen=True)
class SplitSummary:
    development_source: str
    test_source: str
    development_units: int
    test_units: int
    development_positive_units: int
    test_positive_units: int
    development_unit_id_sha256: str
    test_unit_id_sha256: str
    event_independence_verified: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _unit_id_sha256(documents: list[Document]) -> str:
    identifiers = sorted(document.unit_id for document in documents)
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


def summarize_split(
    development: list[Document], test: list[Document], config: dict
) -> SplitSummary:
    """Validate a whole-run holdout without claiming event-level isolation."""
    assert_disjoint_units(development, test)
    if not development or not test:
        raise ValueError("both split sides need review units")
    development_sources = {document.source_run for document in development}
    test_sources = {document.source_run for document in test}
    if len(development_sources) != 1 or len(test_sources) != 1:
        raise ValueError("each split side must contain one complete source run")
    development_source = next(iter(development_sources))
    test_source = next(iter(test_sources))
    pairs = (config["primary_split"], config["sensitivity_split"])
    if not any(
        pair["development"] == development_source and pair["test"] == test_source
        for pair in pairs
    ):
        raise ValueError("split direction is not frozen in experiment config")
    if tuple(config["features"]) != FEATURE_NAMES:
        raise ValueError("configured feature order differs from model feature order")
    if config["review_unit"] != ["source_run", "Belegnummer"]:
        raise ValueError("unexpected review unit definition")
    return SplitSummary(
        development_source=development_source,
        test_source=test_source,
        development_units=len(development),
        test_units=len(test),
        development_positive_units=sum(document.is_positive for document in development),
        test_positive_units=sum(document.is_positive for document in test),
        development_unit_id_sha256=_unit_id_sha256(development),
        test_unit_id_sha256=_unit_id_sha256(test),
    )


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(config["features"]) != FEATURE_NAMES:
        raise ValueError("configured feature order differs from model feature order")
    if not config["source_revision"] or set(config["source_sha256"]) != {
        "erpsim1", "erpsim2"
    }:
        raise ValueError("source revision or pinned hashes are missing")
    return config
