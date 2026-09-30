"""Raw-label scenario holdout proxy, with explicit event-link limitation."""

from __future__ import annotations

from dataclasses import dataclass

from .data import Document, assert_disjoint_units


@dataclass(frozen=True)
class ScenarioFold:
    label: str
    development: tuple[Document, ...]
    test: tuple[Document, ...]
    removed_development_units: int
    test_positive_units: int
    event_links_verified: bool = False


def shared_scenario_labels(
    development: list[Document], test: list[Document]
) -> tuple[str, ...]:
    def labels(documents: list[Document]) -> set[str]:
        return {
            label
            for document in documents
            for label in document.labels
            if label != "NonFraud"
        }

    return tuple(sorted(labels(development) & labels(test)))


def leave_one_raw_label_out(
    development: list[Document], test: list[Document], label: str
) -> ScenarioFold:
    """Remove label s from development, test against normal background plus s.

    This is not leave-one-independent-event-out: no event linkage is present in
    the flat CSV, and other labels from the same fraud family can remain.
    """
    assert_disjoint_units(development, test)
    if label == "NonFraud" or label not in shared_scenario_labels(development, test):
        raise ValueError("scenario label is not a shared positive label")
    filtered_development = tuple(
        document for document in development if label not in document.labels
    )
    filtered_test = tuple(
        document
        for document in test
        if not document.is_positive or label in document.labels
    )
    removed = len(development) - len(filtered_development)
    positives = sum(document.is_positive for document in filtered_test)
    if not filtered_development or not positives:
        raise ValueError("scenario fold has no development or target-positive units")
    return ScenarioFold(
        label=label,
        development=filtered_development,
        test=filtered_test,
        removed_development_units=removed,
        test_positive_units=positives,
    )
