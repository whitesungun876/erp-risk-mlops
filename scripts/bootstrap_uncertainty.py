"""Exploratory fixed-model, candidate-unit bootstrap of Top-K hit counts.

This is a descriptive sensitivity band only. Documents may share an unknown
fraud event, and model fitting is not repeated within each resample.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from erp_risk.data import Document, read_documents
from erp_risk.isolation import IsolationRanker
from erp_risk.protocol import load_config, summarize_split
from erp_risk.rules import RuleBaseline
from erp_risk.xgboost_model import XGBoostRanker


def bootstrap_top_k_hits(
    scores: list[float], labels: list[bool], *, fraction: float,
    repetitions: int = 1000, seed: int = 20260930,
) -> dict[str, int]:
    if len(scores) != len(labels) or not scores or not 0 < fraction <= 1 or repetitions < 100:
        raise ValueError("invalid bootstrap inputs")
    value = np.asarray(scores, dtype=float)
    positive = np.asarray(labels, dtype=bool)
    if not np.isfinite(value).all():
        raise ValueError("scores must be finite")
    n = len(value)
    k = math.ceil(n * fraction)
    rng = np.random.default_rng(seed)
    hits = np.empty(repetitions, dtype=np.int64)
    for iteration in range(repetitions):
        sample = rng.integers(0, n, size=n)
        # Original sorted unit order is the frozen tie-break; repeat copies
        # of the same unit carry the same label, so their mutual order is moot.
        order = np.lexsort((sample, -value[sample]))
        hits[iteration] = positive[sample[order[:k]]].sum()
    full_order = np.lexsort((np.arange(n), -value))
    return {
        "k": k,
        "observed_hits": int(positive[full_order[:k]].sum()),
        "bootstrap_lower_2_5pct": int(np.quantile(hits, 0.025, method="lower")),
        "bootstrap_upper_97_5pct": int(np.quantile(hits, 0.975, method="higher")),
        "repetitions": repetitions,
    }


def score_models(development: list[Document], test: list[Document], config: dict) -> dict[str, list[float]]:
    rules = RuleBaseline.fit(
        development, quantile=config["domain_informed_rule_baseline"]["line_count_quantile"]
    )
    isolation = IsolationRanker.fit(development, **{
        key: config["isolation_forest"][key]
        for key in ("n_estimators", "random_state")
    })
    xgboost = XGBoostRanker.fit(development, **config["xgboost"])
    return {
        "rules": [float(rules.score(document)[0]) for document in test],
        "isolation_forest": isolation.score(test),
        "xgboost": xgboost.score(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--erpsim1", type=Path, required=True)
    parser.add_argument("--erpsim2", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    documents = {
        source_run: read_documents(
            path.read_bytes(), source_run,
            expected_sha256=config["source_sha256"][source_run],
        )
        for source_run, path in (("erpsim1", args.erpsim1), ("erpsim2", args.erpsim2))
    }
    output = {}
    for direction in ("primary", "sensitivity"):
        split = config[f"{direction}_split"]
        development = documents[split["development"]]
        test = documents[split["test"]]
        summarize_split(development, test, config)
        labels = [document.is_positive for document in test]
        output[direction] = {
            model: bootstrap_top_k_hits(scores, labels, fraction=0.01)
            for model, scores in score_models(development, test, config).items()
        }
    print(json.dumps({
        "interpretation": "descriptive_fixed_model_unit_bootstrap_not_event_cluster_confidence_interval",
        "results": output,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
