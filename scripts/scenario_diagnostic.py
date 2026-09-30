"""Raw-label holdout diagnostic, not independent-event generalization proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from erp_risk.data import Document, read_documents
from erp_risk.evaluation import ScoredUnit, evaluate_budgets, positive_ranks
from erp_risk.isolation import IsolationRanker
from erp_risk.protocol import load_config, summarize_split
from erp_risk.rules import RuleBaseline
from erp_risk.scenario import leave_one_raw_label_out, shared_scenario_labels
from erp_risk.xgboost_model import XGBoostRanker


def _summarize_scores(
    test: tuple[Document, ...], scores: list[float], fractions: tuple[float, ...]
) -> dict:
    units = [
        ScoredUnit(document.unit_id, score, document.is_positive)
        for document, score in zip(test, scores, strict=True)
    ]
    if len(units) != len(test):
        raise ValueError("missing scores in scenario fold")
    return {
        "positive_ranks": positive_ranks(units),
        "budget_hits": [
            {
                "fraction": result.fraction,
                "k": result.k,
                "hits": result.hits,
                "hits_min_across_ties": result.hits_min_across_ties,
                "hits_max_across_ties": result.hits_max_across_ties,
            }
            for result in evaluate_budgets(units, fractions)
        ],
    }


def diagnose_direction(
    development: list[Document], test: list[Document], config: dict
) -> dict:
    summary = summarize_split(development, test, config)
    output = {"development_source": summary.development_source,
              "test_source": summary.test_source, "folds": []}
    fractions = tuple(config["review_fractions"])
    rule_quantile = config["domain_informed_rule_baseline"]["line_count_quantile"]
    for label in shared_scenario_labels(development, test):
        fold = leave_one_raw_label_out(development, test, label)
        train = list(fold.development)
        filtered_test = list(fold.test)
        if not any(document.is_positive for document in train):
            output["folds"].append(
                {"label": label, "status": "infeasible_no_training_positives"}
            )
            continue
        rules = RuleBaseline.fit(train, quantile=rule_quantile)
        isolation = IsolationRanker.fit(
            train,
            n_estimators=config["isolation_forest"]["n_estimators"],
            random_state=config["isolation_forest"]["random_state"],
        )
        xgboost = XGBoostRanker.fit(train, **config["xgboost"])
        output["folds"].append(
            {
                "label": label,
                "status": "raw_label_proxy_only",
                "event_links_verified": False,
                "removed_development_units": fold.removed_development_units,
                "test_positive_units": fold.test_positive_units,
                "rules": _summarize_scores(
                    fold.test,
                    [float(rules.score(document)[0]) for document in filtered_test],
                    fractions,
                ),
                "isolation_forest": _summarize_scores(
                    fold.test, isolation.score(filtered_test), fractions
                ),
                "xgboost": _summarize_scores(
                    fold.test, xgboost.score(filtered_test), fractions
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--erpsim1", type=Path, required=True)
    parser.add_argument("--erpsim2", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    first = read_documents(
        args.erpsim1.read_bytes(), "erpsim1",
        expected_sha256=config["source_sha256"]["erpsim1"],
    )
    second = read_documents(
        args.erpsim2.read_bytes(), "erpsim2",
        expected_sha256=config["source_sha256"]["erpsim2"],
    )
    print(
        json.dumps(
            {
                "status": "untracked_raw_label_proxy_not_event_level_proof",
                "primary": diagnose_direction(first, second, config),
                "sensitivity": diagnose_direction(second, first, config),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
