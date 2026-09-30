# Formal Databricks MLflow experiment — exploratory ERPsim comparison

Six runs were read back from Databricks MLflow as `FINISHED` on 2026-09-30. The frozen code snapshot SHA-256 is `a3b3cc7662e2aec1f9ca5af82e8f73302b6280b29e95077d3d37da492768cb4b`. The development batch is never used to fit features for its opposite-batch test. These simulated data have already been inspected during project design, so these are **exploratory held-out batch checks**, not a claim of untouched confirmatory validation or real-world fraud detection.

| Development → test | Model | Test units / positives | Top 1% hits / K | Top 2% hits / K | Top 5% hits / K | Average precision | MLflow run ID |
|---|---|---:|---:|---:|---:|---:|---|
| erpsim1 → erpsim2 | Domain-informed rules | 10,207 / 14 | 0 / 103 | 6 / 205 | 6 / 511 | 0.0098 | `bf4d2285ffcc4f98834478b95f3858bd` |
| erpsim1 → erpsim2 | Isolation Forest | 10,207 / 14 | 5 / 103 | 5 / 205 | 7 / 511 | 0.1048 | `840d0e38b2ce4ca58c8752ccc5bf58cb` |
| erpsim1 → erpsim2 | XGBoost | 10,207 / 14 | 6 / 103 | 7 / 205 | 7 / 511 | 0.0390 | `091c04061d664755b20a73eb5d3ca359` |
| erpsim2 → erpsim1 | Domain-informed rules | 10,267 / 12 | 0 / 103 | 2 / 206 | 2 / 514 | 0.0028 | `21f31603c43c4e079ca986aa058c2fc4` |
| erpsim2 → erpsim1 | Isolation Forest | 10,267 / 12 | 3 / 103 | 4 / 206 | 6 / 514 | 0.0264 | `9f3f595895884c459a662b6ab54c64be` |
| erpsim2 → erpsim1 | XGBoost | 10,267 / 12 | 3 / 103 | 5 / 206 | 6 / 514 | 0.0443 | `f3b9877fa973420fa1345bbffe5caa44` |

At Top 1%, uniform random review expects `K × positives / test units`: about **0.141** hits in the primary direction and **0.120** in the sensitivity direction. These are expectations, not observed random samples. The rule score has wide ties; its reported hits depend on the frozen unit-ID tie-break. See each run's `aggregate_evaluation.json` for the tie-boundary hit range and exact unrounded metrics.

The primary XGBoost run was registered as `workspace.default.erp_risk_xgboost_a3b3cc76` version 1 with `champion` alias as an **operational deployment demonstration**. This is not a claim that XGBoost won model selection: Isolation Forest has higher average precision in the primary direction, and no unbiased development-side choice or uncertainty interval has established a winner. The registered model was reloaded and reproduced the primary Top-1% result (6/103).

## Ranking stability and explanation caveats

The frozen models were rescored on both batches and a fixed-seed, 1,000-replicate candidate-unit bootstrap was run at the 1% review budget. The numbers below are **descriptive sensitivity bands conditional on these fitted models and batches**, not confidence intervals for new companies or independent fraud events. Unknown event links can make document-level resampling too optimistic. The interval also does not include model-refit uncertainty.

| Direction | Model | Observed hits / 103 | Unit bootstrap 2.5–97.5% |
|---|---|---:|---:|
| erpsim1 → erpsim2 | Rules | 0 | 0–0 |
| erpsim1 → erpsim2 | Isolation Forest | 5 | 1–10 |
| erpsim1 → erpsim2 | XGBoost | 6 | 1–11 |
| erpsim2 → erpsim1 | Rules | 0 | 0–0 |
| erpsim2 → erpsim1 | Isolation Forest | 3 | 0–7 |
| erpsim2 → erpsim1 | XGBoost | 3 | 0–6 |

The 0–0 rules bands should **not** be read as certainty that rules cannot work: they reflect this model's highly tied scores and the frozen tie-break, so the run's tie-hit bounds matter more. No statistically defensible winner claim follows from these overlapping bands and two related simulations.

The private batch job stores XGBoost native contribution values on the **raw margin** scale and fixed CSV source-line references for each of the 10,207 units. A local independent read reconciled every reference to the pinned CSV and all six structural features (37,407 source rows); the serverless job checked finite contributions and margin reconstruction before writing. In the highest 1% of scores, zero units had both `debit_total` and `credit_total` among their three displayed contributions. This does not eliminate the interpretability risk: those two features encode the same balanced amount in this data, so credit assigned to one versus the other should not be narrated as a distinct causal reason for fraud. No per-unit prediction or source row is published here.

No original CSV or per-document prediction table was logged to MLflow. The input files remain pinned by SHA-256 in the data card; formal runs contain source hashes, split manifest, aggregate metrics and fitted model artifacts.
