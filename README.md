# ERP Risk Ranking — Databricks Batch MLOps

[![Tests](https://github.com/whitesungun876/erp-risk-mlops/actions/workflows/erp-risk-mlops-tests.yml/badge.svg)](https://github.com/whitesungun876/erp-risk-mlops/actions/workflows/erp-risk-mlops-tests.yml)

A portfolio project for prioritising exported ERP accounting records for human review. It compares domain-informed rules, Isolation Forest and XGBoost, then delivers a versioned model through a Databricks batch workflow.

**Python · scikit-learn · XGBoost · MLflow · Unity Catalog · Delta Lake · GitHub Actions**

## Business question

A finance or internal-control team can review only a small share of each batch. Which ranking method surfaces more labelled cases at the same review budget, and can the scoring pipeline be reproduced and rerun without duplicate outputs?

This project uses **simulated ERP data**, not client records. It supports review prioritisation, not automatic fraud decisions or audit opinions.

## Pipeline

Pinned CSV inputs → document aggregation and six structural features → batch-held-out evaluation → MLflow tracking → Unity Catalog model version → serverless batch scoring → idempotent Delta MERGE.

- Review units are `(source_run, Belegnummer)`, not verified independent fraud events.
- Features capture line count, debit and credit totals, maximum line amount, account count and transaction-type count.
- Evaluation includes 1%, 2% and 5% budgets, random-review expectations, reverse-batch sensitivity and raw-label holdout diagnostics.
- XGBoost native SHAP contributions explain raw model margins and link back to pinned CSV line references. They are not causal explanations.
- The batch job resolves the model alias once and pins the version. Its write key is `(batch_id, model_version, unit_id)`.

## Results at a 1% review budget

| Development → test | Review budget | Positive units | Rules hits | Isolation Forest hits | XGBoost hits | Random expected hits |
|---|---:|---:|---:|---:|---:|---:|
| erpsim1 → erpsim2 | 103 / 10,207 | 14 | 0 | 5 | 6 | 0.141 |
| erpsim2 → erpsim1 | 103 / 10,267 | 12 | 0 | 3 | 3 | 0.120 |

These are exploratory results from two related simulations with few positive cases. They do **not** establish a statistically reliable winner. Isolation Forest has higher average precision in the primary direction; XGBoost was registered to demonstrate deployment, not because it universally performed best.

See the [full experiment report](reports/FORMAL_EXPERIMENT.md) for all budgets, descriptive bootstrap ranges and interpretation limits, and the [raw-label holdout diagnostics](reports/LOCAL_DIAGNOSTIC.md) for failure cases.

## Verified engineering scope

- Six formal model runs recorded in a single Databricks MLflow experiment.
- A registered Unity Catalog model reloaded and reproduced the primary ranking result.
- Real serverless batch scoring wrote 10,207 candidate units; repeated runs preserved the per-batch row count and unique keys.
- Source references reconciled locally against 37,407 CSV rows.
- 23 unit tests cover data contracts, leakage checks, ranking metrics, model round trips and batch behaviour.

Cloud execution was verified in a private workspace. Its resources and credentials are not public; GitHub CI runs the local tests and does **not** independently verify that workspace. See [acceptance notes](PROGRESS.md).

## Run the tests

From this repository's root, with Docker installed:

```sh
docker build -f Dockerfile.local -t erp-risk-mlops:local .
docker run --rm -v "$PWD:/work:ro" -e PYTHONPATH=/work/src:/work/scripts erp-risk-mlops:local python -m unittest discover -s /work/tests -v
```

Alternatively, use Python 3.12 on Linux:

```sh
python -m pip install xgboost-cpu==3.4.1 scikit-learn==1.9.0
PYTHONPATH=src:scripts python -m unittest discover -s tests -v
```

## Data and cloud configuration

Obtain the data from the [original university provider](https://www.informatik.uni-wuerzburg.de/datascience/projects/deepscan/erp-fraud-data/). The [data card](docs/DATA_CARD.md) records the exact revision, hashes, field checks and usage limits. Raw CSVs, derived per-document records and model artifacts are not distributed in this repository.

The cloud scripts are a reference implementation, not a one-command hosted demo. Configure your own `DATABRICKS_HOST`, authenticate with the official Databricks CLI and supply a private short-lived token file where a script requires `--token-file`. The file must be owner-readable only; never commit it or paste its contents into logs. No personal login helper or credentials are included.

Before deployment, review the explicit model/table/volume names in `scripts/`, prepare the private feature input and register your own model. `databricks.yml` defines a **manual job with no schedule**; `databricks bundle validate` uses your configured host/profile. Deployment and execution consume your own workspace resources. Local tests need no cloud account.

## Limitations

- No verified event IDs: leave-one-raw-label testing is a proxy, not evidence of unseen-event generalisation.
- Rules were written with knowledge of scenario descriptions; they are domain-informed, not blind.
- Unknown event correlations limit document-level bootstrap interpretation.
- Debit and credit totals are correlated; individual SHAP contributions should not be narrated as distinct causes.
- No demonstrated pre-payment availability, real-client effectiveness, real-time serving or production readiness.
- An explicit raw-data redistribution licence was not identified; public availability is not treated as redistribution permission.

The public repository is a sanitised standalone export. Original experiment snapshot hashes refer to the implementation used for those runs, before publication-related configuration and documentation changes. Supporting data notes retain the original Chinese text; the main overview and formal results are in English.
