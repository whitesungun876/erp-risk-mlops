"""Track the frozen two-direction, three-model ERPsim study in Databricks.

The CSVs are read-only local inputs. Only aggregate metrics, split hashes,
configuration, and fitted model artifacts leave this process; no raw CSV or
per-document prediction table is logged as an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from databricks.sdk import WorkspaceClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from erp_risk.data import Document, read_documents
from erp_risk.evaluation import ScoredUnit, evaluate_budgets
from erp_risk.isolation import FEATURE_NAMES, IsolationRanker
from erp_risk.protocol import load_config, summarize_split
from erp_risk.rules import RuleBaseline
from erp_risk.xgboost_model import XGBoostRanker
from probe_workspace import HOST, read_private_token


def code_snapshot_sha256(project: Path) -> str:
    paths = sorted((project / "src" / "erp_risk").glob("*.py"))
    paths += [project / "configs" / "experiment.json", Path(__file__)]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def metrics_for(test: list[Document], scores: list[float], fractions: tuple[float, ...]) -> tuple[dict, list[dict]]:
    if len(test) != len(scores):
        raise ValueError("model score count differs from test units")
    budgets = [asdict(item) for item in evaluate_budgets(
        [ScoredUnit(doc.unit_id, score, doc.is_positive)
         for doc, score in zip(test, scores, strict=True)], fractions,
    )]
    metrics = {"test_units": float(len(test)), "test_positives": float(sum(doc.is_positive for doc in test))}
    for item in budgets:
        suffix = f"{round(item['fraction'] * 100)}pct"
        metrics[f"k_{suffix}"] = float(item["k"])
        metrics[f"hits_{suffix}"] = float(item["hits"])
        metrics[f"precision_{suffix}"] = item["precision_at_k"]
        metrics[f"random_expected_hits_{suffix}"] = item["random_expected_hits"]
        metrics[f"tie_hits_min_{suffix}"] = float(item["hits_min_across_ties"])
        metrics[f"tie_hits_max_{suffix}"] = float(item["hits_max_across_ties"])
        if item["recall_at_k"] is not None:
            metrics[f"recall_{suffix}"] = item["recall_at_k"]
        if item["average_precision"] is not None:
            metrics["average_precision"] = item["average_precision"]
    return metrics, budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--erpsim1", type=Path, required=True)
    parser.add_argument("--erpsim2", type=Path, required=True)
    args = parser.parse_args()
    project = args.config.resolve().parent.parent
    config = load_config(args.config)
    os.environ["DATABRICKS_HOST"] = HOST
    os.environ["DATABRICKS_TOKEN"] = read_private_token(args.token_file)
    os.environ.pop("DATABRICKS_AUTH_TYPE", None)
    client = WorkspaceClient(host=HOST, token=os.environ["DATABRICKS_TOKEN"])
    user = client.current_user.me()
    if not user.user_name:
        raise RuntimeError("current user has no workspace user name")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/{user.user_name}/erp-risk-mlops/formal-experiment")

    documents = {
        "erpsim1": read_documents(args.erpsim1.read_bytes(), "erpsim1", expected_sha256=config["source_sha256"]["erpsim1"]),
        "erpsim2": read_documents(args.erpsim2.read_bytes(), "erpsim2", expected_sha256=config["source_sha256"]["erpsim2"]),
    }
    snapshot = code_snapshot_sha256(project)
    fractions = tuple(config["review_fractions"])
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", name) for name in FEATURE_NAMES]),
        outputs=Schema([ColSpec("double")]),
    )
    records = []
    for direction in ("primary", "sensitivity"):
        split = config[f"{direction}_split"]
        development = documents[split["development"]]
        test = documents[split["test"]]
        manifest = summarize_split(development, test, config)
        for model_name in ("rules", "isolation_forest", "xgboost"):
            model = None
            if model_name == "rules":
                settings = config["domain_informed_rule_baseline"]
                if settings["line_count_quantile"] != settings["debit_total_quantile"]:
                    raise ValueError("current rule implementation requires equal quantiles")
                model = RuleBaseline.fit(development, quantile=settings["line_count_quantile"])
                scores = [float(model.score(doc)[0]) for doc in test]
            elif model_name == "isolation_forest":
                model = IsolationRanker.fit(development, **{
                    key: config["isolation_forest"][key]
                    for key in ("n_estimators", "random_state")
                })
                scores = model.score(test)
            else:
                model = XGBoostRanker.fit(development, **config["xgboost"])
                scores = model.score(test)
            metrics, budgets = metrics_for(test, scores, fractions)
            with mlflow.start_run(run_name=f"{direction}-{model_name}") as run:
                mlflow.log_params({
                    "model": model_name,
                    "direction": direction,
                    "development_source": manifest.development_source,
                    "test_source": manifest.test_source,
                    "source_revision": config["source_revision"],
                    "erpsim1_sha256": config["source_sha256"]["erpsim1"],
                    "erpsim2_sha256": config["source_sha256"]["erpsim2"],
                    "protocol_version": config["protocol_version"],
                    "code_snapshot_sha256": snapshot,
                })
                mlflow.set_tags({
                    "result_interpretation": "exploratory_simulation_previously_inspected",
                    "review_unit": "source_run_plus_Belegnummer_candidate",
                    "score_direction": "higher_means_review_first",
                    "raw_csv_logged": "false",
                })
                mlflow.log_metrics(metrics)
                mlflow.log_dict({"split": asdict(manifest), "budgets": budgets}, "aggregate_evaluation.json")
                model_uri = None
                if model_name == "rules":
                    mlflow.log_dict({
                        "line_count_threshold": model.line_count_threshold,
                        "debit_total_threshold": model.debit_total_threshold,
                        "baseline_type": "domain_informed_not_blind",
                    }, "fitted_rules.json")
                elif model_name == "isolation_forest":
                    mlflow.set_tag("inference_note", "review_score_is_negative_decision_function_not_pyfunc_predict")
                    model_uri = mlflow.sklearn.log_model(
                        model.model, name="isolation_pipeline",
                        skops_trusted_types=["sklearn.tree._tree.Tree"],
                    ).model_uri
                else:
                    mlflow.set_tag("score_scale", "raw_margin_not_calibrated_probability")
                    model_uri = mlflow.xgboost.log_model(
                        model.booster, name="risk_ranker", model_format="json",
                        signature=signature,
                        pip_requirements=["xgboost-cpu==3.4.1", "numpy==2.5.3"],
                    ).model_uri
                records.append({
                    "direction": direction,
                    "model": model_name,
                    "run_id": run.info.run_id,
                    "model_uri": model_uri,
                    "hits_1pct": int(metrics["hits_1pct"]),
                    "k_1pct": int(metrics["k_1pct"]),
                })
                print(json.dumps(records[-1], sort_keys=True), flush=True)
    print(json.dumps({"completed_runs": len(records), "code_snapshot_sha256": snapshot}, sort_keys=True))


if __name__ == "__main__":
    main()
