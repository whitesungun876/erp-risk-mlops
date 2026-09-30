"""Unsupervised anomaly ranking fitted on development documents only."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import Document


FEATURE_NAMES = (
    "line_count",
    "debit_total",
    "credit_total",
    "max_line_amount",
    "distinct_gl_accounts",
    "distinct_transaction_types",
)


def feature_matrix(documents: list[Document]) -> np.ndarray:
    """Keep the feature order fixed and exclude all labels and source IDs."""
    if not documents:
        raise ValueError("need at least one review unit")
    matrix = np.asarray(
        [[document.features()[name] for name in FEATURE_NAMES] for document in documents],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("all structural features must be finite")
    return matrix


@dataclass
class IsolationRanker:
    model: Pipeline

    @classmethod
    def fit(
        cls, development: list[Document], *, n_estimators: int = 100,
        random_state: int = 20260930,
    ) -> IsolationRanker:
        matrix = feature_matrix(development)
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "forest",
                    IsolationForest(
                        n_estimators=n_estimators,
                        contamination="auto",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(matrix)
        return cls(model=model)

    def score(self, documents: list[Document]) -> list[float]:
        """Higher values mean more anomalous, not calibrated fraud probability."""
        matrix = feature_matrix(documents)
        scores = -self.model.decision_function(matrix)
        if not all(math.isfinite(float(score)) for score in scores):
            raise ValueError("model returned a nonfinite score")
        return [float(score) for score in scores]
