import csv
import json
import re

import numpy as np
import pytest
import logging
from pytest import approx

from govuk_chat_evaluation.output_guardrails.evaluate import (
    AggregateResults,
    EvaluationResult,
    evaluate_and_output_results,
)


@pytest.fixture
def result_true_positive() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="TP",
        expected_triggered=True,
        actual_triggered=True,
        expected_guardrails={"g1": True, "g3": True},
        actual_guardrails={"g1": True, "g3": True},
    )


@pytest.fixture
def result_false_positive() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="FP",
        expected_triggered=False,
        actual_triggered=True,
        expected_guardrails={"g1": False, "g3": False},
        actual_guardrails={"g1": True, "g3": True},
    )


@pytest.fixture
def result_false_negative() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="FN",
        expected_triggered=True,
        actual_triggered=False,
        expected_guardrails={"g1": True, "g3": True},
        actual_guardrails={"g1": False, "g3": False},
    )


@pytest.fixture
def result_true_negative() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="TN",
        expected_triggered=False,
        actual_triggered=False,
        expected_guardrails={"g1": False, "g3": False},
        actual_guardrails={"g1": False, "g3": False},
    )


@pytest.fixture
def result_mixed_guardrails() -> EvaluationResult:
    return EvaluationResult(
        question="Mixed",
        expected_triggered=True,
        actual_triggered=True,
        expected_guardrails={"g1": True, "g2": False, "g3": True},
        actual_guardrails={
            "g1": True,
            "g2": True,
            "g3": False,
            "g4": True,
        },  # g1 TP, g2 FP, g3 TN, g4 FP
    )


class TestEvaluationResult:
    @pytest.mark.parametrize(
        "fixture_name, expected_classification",
        [
            ("result_true_positive", "true_positive"),
            ("result_true_negative", "true_negative"),
            ("result_false_positive", "false_positive"),
            ("result_false_negative", "false_negative"),
        ],
    )
    def test_classification_triggered(
        self, request, fixture_name, expected_classification
    ):
        result: EvaluationResult = request.getfixturevalue(fixture_name)
        assert result.classification_triggered == expected_classification

    def test_for_csv(self, result_mixed_guardrails):
        result = result_mixed_guardrails
        expected_csv_dict = {
            "question": "Mixed",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_guardrails": {"g1": True, "g2": False, "g3": True},
            "actual_guardrails": {"g1": True, "g2": True, "g3": False, "g4": True},
            "classification": "true_positive",
        }
        assert result.for_csv() == expected_csv_dict


class TestAggregateResults:
    @pytest.fixture
    def sample_results(
        self,
        result_true_positive,
        result_true_negative,
        result_false_positive,
        result_false_negative,
    ) -> list[EvaluationResult]:
        return [
            result_true_positive,
            result_true_positive,
            result_true_negative,
            result_true_negative,
            result_true_negative,
            result_false_positive,
            result_false_negative,
        ]

    def test_confusion_matrix_counts(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.true_positive == 2
        assert aggregate.true_negative == 3
        assert aggregate.false_positive == 1
        assert aggregate.false_negative == 1

    def test_precision_any_triggered(self, sample_results):
        aggregate = AggregateResults(sample_results)
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
        assert aggregate.precision() == approx(2 / 3)

    def test_precision_any_triggered_nan(
        self, result_false_negative, result_true_negative
    ):
        # No positive predictions (TP=0, FP=0)
        aggregate = AggregateResults([result_false_negative, result_true_negative])
        assert np.isnan(aggregate.precision())

    def test_recall_any_triggered(self, sample_results):
        aggregate = AggregateResults(sample_results)
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
        assert aggregate.recall() == approx(2 / 3)

    def test_recall_any_triggered_nan(
        self, result_true_negative, result_false_positive
    ):
        # No actual positive cases (TP=0, FN=0)
        aggregate = AggregateResults([result_true_negative, result_false_positive])
        assert np.isnan(aggregate.recall())

    @pytest.fixture
    def per_guardrail_eval_results(self) -> list[EvaluationResult]:
        """Fixture providing sample results for per-guardrail testing."""
        return [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_guardrails={"g1": True, "g2": False, "g3": True},
                actual_guardrails={"g1": True, "g2": False, "g3": True},
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=True,
                actual_triggered=True,
                expected_guardrails={"g1": True, "g2": False, "g3": False},
                actual_guardrails={"g1": True, "g2": True, "g3": False},
            ),
            EvaluationResult(
                question="Q3",
                expected_triggered=True,
                actual_triggered=True,
                expected_guardrails={"g1": True, "g2": True, "g3": True},
                actual_guardrails={"g1": True, "g2": True, "g3": False},
            ),
            EvaluationResult(
                question="Q4",
                expected_triggered=False,
                actual_triggered=False,
                expected_guardrails={"g1": False, "g2": False, "g3": False},
                actual_guardrails={"g1": False, "g2": False, "g3": False},
            ),
            EvaluationResult(
                question="Q5",
                expected_triggered=False,
                actual_triggered=True,
                expected_guardrails={"g1": False, "g2": False, "g3": False},
                actual_guardrails={"g1": True, "g2": False, "g3": False},
            ),
            EvaluationResult(
                question="Q6",
                expected_triggered=True,
                actual_triggered=False,
                expected_guardrails={"g1": False, "g2": True, "g3": False},
                actual_guardrails={"g1": False, "g2": False, "g3": False},
            ),
        ]

    def test_guardrail_names_discovery(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        assert aggregate.guardrail_names == ["g1", "g2", "g3"]

    def test_expected_actual_vectors(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        expected_vectors, actual_vectors = aggregate._expected_actual_guardrails_vectors

        expected_ground_truth = [
            [1, 0, 1],
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
        actual_predictions = [
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]

        assert expected_vectors == expected_ground_truth
        assert actual_vectors == actual_predictions

    def test_expected_actual_vectors_empty(self):
        aggregate = AggregateResults([])
        assert aggregate.guardrail_names == []
        expected_vectors, actual_vectors = aggregate._expected_actual_guardrails_vectors
        assert expected_vectors == []
        assert actual_vectors == []

    def test_precision_per_guardrail(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        precision = aggregate.precision_per_guardrail()

        # Calculate based on vectors: TP / (TP + FP)
        # g1: TP=3 (Q1,Q2,Q3), FP=1 (Q5) -> P=3/4 = 0.75
        # g2: TP=1 (Q3), FP=1 (Q2) -> P=1/2 = 0.5
        # g3: TP=1 (Q1), FP=0 -> P=1/1 = 1.0
        expected_precision = [0.75, 0.5, 1.0]
        assert precision == approx(expected_precision)

    def test_precision_per_guardrail_empty(self):
        aggregate = AggregateResults([])
        precision = aggregate.precision_per_guardrail()
        assert precision == []

    def test_recall_per_guardrail(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        recall = aggregate.recall_per_guardrail()

        # Calculate based on vectors: TP / (TP + FN)
        # g1: TP=3 (Q1,Q2,Q3), FN=0 -> R = 3/3 = 1.0
        # g2: TP=1 (Q3), FN=1 (Q6) -> R = 1/2 = 0.5
        # g3: TP=1 (Q1), FN=1 (Q3) -> R = 1/2 = 0.5
        expected_recall = [1.0, 0.5, 0.5]
        assert recall == approx(expected_recall)

    def test_recall_per_guardrail_empty(self):
        aggregate = AggregateResults([])
        recall = aggregate.recall_per_guardrail()
        assert recall == []

    def test_f1_per_guardrail(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        f1 = aggregate.f1_per_guardrail()

        # Calculate F1 = 2 * (P * R) / (P + R)
        # g1: P=0.75, R=1.0 -> F1 = 2 * (0.75 * 1.0) / (0.75 + 1.0) = 1.5 / 1.75 = 6/7
        # g2: P=0.5, R=0.5 -> F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5 / 1.0 = 0.5
        # g3: P=1.0, R=0.5 -> F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 2/3
        expected_f1 = [6 / 7, 0.5, 2 / 3]
        assert f1 == approx(expected_f1)

    def test_f1_per_guardrail_empty(self):
        aggregate = AggregateResults([])
        f1 = aggregate.f1_per_guardrail()
        assert f1 == []

    def test_to_dict(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        assert aggregate.to_dict() == {
            "Evaluated": 6,
            "Any-triggered False negatives": 1,
            "Any-triggered False positives": 1,
            "Any-triggered Precision": 0.75,
            "Any-triggered Recall": 0.75,
            "Any-triggered True negatives": 1,
            "Any-triggered True positives": 3,
            "F1 [g1]": 6 / 7,
            "F1 [g2]": 0.5,
            "F1 [g3]": 2 / 3,
            "Precision [g1]": 0.75,
            "Precision [g2]": 0.5,
            "Precision [g3]": 1.0,
            "Recall [g1]": 1.0,
            "Recall [g2]": 0.5,
            "Recall [g3]": 0.5,
        }

    def test_for_csv(self, per_guardrail_eval_results):
        aggregate = AggregateResults(per_guardrail_eval_results)
        expected_csv = [
            {"property": k, "value": v} for k, v in aggregate.to_dict().items()
        ]

        assert aggregate.for_csv() == expected_csv

        assert aggregate.for_csv() == [
            {"property": "Evaluated", "value": len(per_guardrail_eval_results)},
            {"property": "Any-triggered Precision", "value": 0.75},
            {"property": "Any-triggered Recall", "value": 0.75},
            {"property": "Any-triggered True positives", "value": 3},
            {"property": "Any-triggered True negatives", "value": 1},
            {"property": "Any-triggered False positives", "value": 1},
            {"property": "Any-triggered False negatives", "value": 1},
            {"property": "Precision [g1]", "value": 0.75},
            {"property": "Recall [g1]", "value": 1.0},
            {"property": "F1 [g1]", "value": 6 / 7},
            {"property": "Precision [g2]", "value": 0.5},
            {"property": "Recall [g2]", "value": 0.5},
            {"property": "F1 [g2]", "value": 0.5},
            {"property": "Precision [g3]", "value": 1.0},
            {"property": "Recall [g3]", "value": 0.5},
            {"property": "F1 [g3]", "value": 2 / 3},
        ]


@pytest.fixture
def mock_evaluation_data_file(tmp_path):
    file_path = tmp_path / "evaluation_data.jsonl"
    data = [
        {
            "question": "Question 1",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_guardrails": {"g1": True, "g2": False},
            "actual_guardrails": {"g1": True, "g2": False},
        },
        {
            "question": "Question 2",
            "expected_triggered": True,
            "actual_triggered": False,
            "expected_guardrails": {"g1": False, "g2": True},
            "actual_guardrails": {"g1": False, "g2": False},
        },
        {
            "question": "Question 3",
            "expected_triggered": False,
            "actual_triggered": False,
            "expected_guardrails": {"g1": False, "g2": False},
            "actual_guardrails": {"g1": False, "g2": False},
        },
        {
            "question": "Question 4",
            "expected_triggered": False,
            "actual_triggered": True,
            "expected_guardrails": {"g1": False, "g2": False},
            "actual_guardrails": {"g1": True, "g2": False},
        },
    ]

    with open(file_path, "w", encoding="utf8") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

    return file_path


def test_evaluate_and_output_results_writes_results(
    mock_project_root, mock_evaluation_data_file
):
    """Check if results.csv is created with expected columns."""
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)
    results_file = mock_project_root / "results.csv"

    assert results_file.exists()

    with open(results_file, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)

        assert headers is not None
        assert "question" in headers


def test_evaluate_and_output_results_writes_aggregates(
    mock_project_root, mock_evaluation_data_file
):
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)
    aggregates_file = mock_project_root / "aggregate.csv"

    assert aggregates_file.exists()
    with open(aggregates_file, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)

        assert headers is not None
        assert "property" in headers


def test_evaluate_and_output_results_prints_aggregates(
    mock_project_root, mock_evaluation_data_file, caplog
):
    caplog.set_level(logging.INFO)
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)

    captured = caplog.text
    assert "Aggregate Results" in captured
    assert re.search(r"Evaluated\s+\d+", captured)
