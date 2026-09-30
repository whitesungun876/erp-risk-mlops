"""Frozen, domain-informed review rules fitted only on development features."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Document


def nearest_rank_quantile(values: list[float], quantile: float) -> float:
    """Deterministic threshold: sorted[ceil(q * n) - 1]."""
    if not values or not 0 < quantile < 1:
        raise ValueError("need nonempty values and quantile within (0, 1)")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("reference values must be finite")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


@dataclass(frozen=True)
class RuleEvidence:
    code: str
    feature: str
    value: float
    threshold: float


@dataclass(frozen=True)
class RuleBaseline:
    line_count_threshold: float
    debit_total_threshold: float

    @classmethod
    def fit(cls, development: list[Document], quantile: float = 0.99) -> RuleBaseline:
        if not development:
            raise ValueError("cannot fit rules on empty development set")
        return cls(
            line_count_threshold=nearest_rank_quantile(
                [float(document.line_count) for document in development], quantile
            ),
            debit_total_threshold=nearest_rank_quantile(
                [float(document.debit_total) for document in development], quantile
            ),
        )

    def score(self, document: Document) -> tuple[int, tuple[RuleEvidence, ...]]:
        evidence = []
        if document.line_count > self.line_count_threshold:
            evidence.append(
                RuleEvidence(
                    "unusually_many_lines", "line_count", float(document.line_count),
                    self.line_count_threshold,
                )
            )
        debit_total = float(document.debit_total)
        if debit_total > self.debit_total_threshold:
            evidence.append(
                RuleEvidence(
                    "unusually_large_debit_total", "debit_total", debit_total,
                    self.debit_total_threshold,
                )
            )
        return len(evidence), tuple(evidence)
