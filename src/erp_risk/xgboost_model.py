"""Small, fixed XGBoost risk ranker with native margin-scale SHAP values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from .data import Document
from .isolation import FEATURE_NAMES, feature_matrix


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    contribution_to_margin: float


@dataclass(frozen=True)
class MarginExplanation:
    base_margin: float
    score_margin: float
    contributions: tuple[FeatureContribution, ...]
    output_scale: str = "raw_margin_not_calibrated_probability"


@dataclass
class XGBoostRanker:
    booster: xgb.Booster

    @classmethod
    def fit(
        cls,
        development: list[Document],
        *,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        random_state: int = 20260930,
    ) -> XGBoostRanker:
        matrix = feature_matrix(development)
        target = np.asarray([document.is_positive for document in development], dtype=np.int8)
        n_positive = int(target.sum())
        n_negative = len(target) - n_positive
        if not n_positive or not n_negative:
            raise ValueError("XGBoost development set needs both target classes")
        if n_estimators < 1 or max_depth < 1 or learning_rate <= 0:
            raise ValueError("invalid fixed XGBoost configuration")
        train = xgb.DMatrix(matrix, label=target, feature_names=list(FEATURE_NAMES))
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "max_depth": max_depth,
                "eta": learning_rate,
                "scale_pos_weight": n_negative / n_positive,
                "seed": random_state,
                "nthread": 1,
            },
            train,
            num_boost_round=n_estimators,
        )
        return cls(booster=booster)

    def _matrix(self, documents: list[Document]) -> xgb.DMatrix:
        return xgb.DMatrix(feature_matrix(documents), feature_names=list(FEATURE_NAMES))

    def score(self, documents: list[Document]) -> list[float]:
        """Larger raw margins receive higher review priority."""
        scores = self.booster.predict(self._matrix(documents), output_margin=True)
        if not np.isfinite(scores).all():
            raise ValueError("XGBoost returned a nonfinite margin")
        return [float(score) for score in scores]

    def explain(self, documents: list[Document]) -> list[MarginExplanation]:
        """Check SHAP additivity on the exact raw-margin score used for ranking."""
        matrix = self._matrix(documents)
        margins = self.booster.predict(matrix, output_margin=True)
        contributions = self.booster.predict(matrix, pred_contribs=True)
        if contributions.shape != (len(documents), len(FEATURE_NAMES) + 1):
            raise ValueError("unexpected XGBoost contribution shape")
        if not np.allclose(contributions.sum(axis=1), margins, atol=1e-5, rtol=1e-5):
            raise ValueError("feature contributions do not add to raw margins")
        output = []
        for document, margin, contribution_row in zip(
            documents, margins, contributions, strict=True
        ):
            features = document.features()
            output.append(
                MarginExplanation(
                    base_margin=float(contribution_row[-1]),
                    score_margin=float(margin),
                    contributions=tuple(
                        FeatureContribution(
                            feature=name,
                            value=float(features[name]),
                            contribution_to_margin=float(value),
                        )
                        for name, value in zip(
                            FEATURE_NAMES, contribution_row[:-1], strict=True
                        )
                    ),
                )
            )
        return output

    def save(self, path: str | Path) -> None:
        """Save in XGBoost JSON format so feature names survive round-trips."""
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostRanker:
        booster = xgb.Booster()
        booster.load_model(str(path))
        if tuple(booster.feature_names or ()) != FEATURE_NAMES:
            raise ValueError("loaded XGBoost model has an unexpected feature schema")
        return cls(booster=booster)
