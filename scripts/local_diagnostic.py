"""Untracked local diagnostic; never substitutes for Databricks MLflow evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from erp_risk.data import Document, read_documents
from erp_risk.evaluation import ScoredUnit, evaluate_budgets
from erp_risk.isolation import IsolationRanker
from erp_risk.protocol import load_config, summarize_split
from erp_risk.rules import RuleBaseline


def _metrics(documents: list[Document], scores: list[float], fractions: tuple[float, ...]) -> list[dict]:
    if len(documents) != len(scores):
        raise ValueError("score count does not match review-unit count")
    return [
        asdict(result)
        for result in evaluate_budgets(
            [
                ScoredUnit(document.unit_id, score, document.is_positive)
                for document, score in zip(documents, scores, strict=True)
            ],
            fractions,
        )
    ]


def compare_direction(
    development: list[Document], test: list[Document], config: dict,
    *, include_xgboost: bool = False,
) -> dict:
    manifest = summarize_split(development, test, config)
    fractions = tuple(config["review_fractions"])
    rules = RuleBaseline.fit(
        development,
        quantile=config["domain_informed_rule_baseline"]["line_count_quantile"],
    )
    if config["domain_informed_rule_baseline"]["line_count_quantile"] != config[
        "domain_informed_rule_baseline"
    ]["debit_total_quantile"]:
        raise ValueError("current rule implementation requires the two quantiles to match")
    rule_scores = [float(rules.score(document)[0]) for document in test]
    isolation = IsolationRanker.fit(
        development,
        n_estimators=config["isolation_forest"]["n_estimators"],
        random_state=config["isolation_forest"]["random_state"],
    )
    result = {
        "split": asdict(manifest),
        "rules": {
            "line_count_threshold": rules.line_count_threshold,
            "debit_total_threshold": rules.debit_total_threshold,
            "budgets": _metrics(test, rule_scores, fractions),
        },
        "isolation_forest": {
            "budgets": _metrics(test, isolation.score(test), fractions)
        },
    }
    if include_xgboost:
        from erp_risk.xgboost_model import XGBoostRanker

        settings = config["xgboost"]
        ranker = XGBoostRanker.fit(
            development,
            n_estimators=settings["n_estimators"],
            max_depth=settings["max_depth"],
            learning_rate=settings["learning_rate"],
            random_state=settings["random_state"],
        )
        scores = ranker.score(test)
        top_indices = sorted(
            range(len(test)), key=lambda index: (-scores[index], test[index].unit_id)
        )[:20]
        explanations = ranker.explain([test[index] for index in top_indices])
        max_explanation_residual = max(
            abs(
                explanation.score_margin
                - explanation.base_margin
                - sum(
                    item.contribution_to_margin
                    for item in explanation.contributions
                )
            )
            for explanation in explanations
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "xgboost.json"
            ranker.save(model_path)
            loaded_scores = XGBoostRanker.load(model_path).score(test)
        max_roundtrip_difference = max(
            abs(left - right)
            for left, right in zip(scores, loaded_scores, strict=True)
        )
        result["xgboost"] = {
            "budgets": _metrics(test, scores, fractions),
            "top20_max_shap_margin_residual": max_explanation_residual,
            "max_model_roundtrip_score_difference": max_roundtrip_difference,
            "score_scale": "raw_margin_not_calibrated_probability",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--erpsim1", type=Path, required=True)
    parser.add_argument("--erpsim2", type=Path, required=True)
    parser.add_argument("--include-xgboost", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    first = read_documents(
        args.erpsim1.read_bytes(),
        "erpsim1",
        expected_sha256=config["source_sha256"]["erpsim1"],
    )
    second = read_documents(
        args.erpsim2.read_bytes(),
        "erpsim2",
        expected_sha256=config["source_sha256"]["erpsim2"],
    )
    report = {
        "status": "untracked_local_diagnostic_not_databricks_run",
        "protocol_version": config["protocol_version"],
        "source_revision": config["source_revision"],
        "primary": compare_direction(
            first, second, config, include_xgboost=args.include_xgboost
        ),
        "sensitivity": compare_direction(
            second, first, config, include_xgboost=args.include_xgboost
        ),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
