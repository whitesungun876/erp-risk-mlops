# Acceptance notes

The original Day 1–15 scope was accepted on 2026-09-30, with the raw-label holdout proxy explicitly replacing an unverified event-level scenario split.

| Stage | Evidence and scope |
|---|---|
| Days 1–3: data and environment | Pinned source hashes, accounting-field checks, frozen features and batch split; synthetic MLflow/registry/serverless smoke checks. |
| Days 4–8: experiments | Six formal three-model/two-direction MLflow runs; fixed review budgets and random expectations; source-linked explanations; exploratory seven-label holdout diagnostics. |
| Days 9–12: delivery | Model registration and alias reload; real batch runs with 10,207 outputs and unique units; repeated Delta writes; missing-alias error check; 23 Linux unit tests. |
| Days 13–15: handoff | Results, data card, limitations and demonstration sequence consolidated. |

## Evidence available here

- [Formal results](reports/FORMAL_EXPERIMENT.md): aggregate metrics, historical MLflow run identifiers, code snapshot hash and interpretation limits.
- [Local diagnostics](reports/LOCAL_DIAGNOSTIC.md): exploratory label-holdout results, separate from formal tracked runs.
- [Data card](docs/DATA_CARD.md): source revision, hashes and data semantics.
- [Public CI](https://github.com/whitesungun876/erp-risk-mlops/actions): standalone local test runs; no cloud credentials.

Private workspace execution was read back during implementation. Cloud resources, original private CI links, personal workspace addresses and operational login files are intentionally omitted from this public export. Reported cloud evidence is historical, not publicly executable proof. New users must provision their own resources.

## Remaining qualifications

- No verified event-level isolation or real-world fraud generalisation.
- No raw-data redistribution permission inferred from availability.
- The MLflow-enabled tracking image was assembled in a temporary Linux container; a clean build of `Dockerfile.tracking` was not verified. The core test image was exercised separately.
- Independent Jev code review was unavailable during the original implementation; local checks are not represented as an independent review.
- This is a portfolio implementation, not a client deployment or an automatically submitted job application.
