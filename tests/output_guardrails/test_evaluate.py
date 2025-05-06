import csv
import json
import re

import numpy as np
import pytest
import logging

from govuk_chat_evaluation.output_guardrails.evaluate import (
    AggregateResults,
    EvaluationResult,
    evaluate_and_output_results,
)


class TestEvaluationResult:
    @pytest.mark.parametrize(
        "expected, actual, expected_classification",
        [
            (True, True, "true_positive"),
            (False, False, "true_negative"),
            (False, True, "false_positive"),
            (True, False, "false_negative"),
        ],
    )
    def test_classification(self, expected, actual, expected_classification):
        result = EvaluationResult(
            question="Test question",
            expected_outcome=expected,
            actual_outcome=actual,
        )
        assert result.classification == expected_classification

    def test_for_csv(self):
        result = EvaluationResult(
            question="Test question",
            expected_outcome=True,
            actual_outcome=True,
        )

        assert result.for_csv() == {
            "question": "Test question",
            "expected_outcome": True,
            "actual_outcome": True,
            "classification": "true_positive",
        }


class TestAggregateResults:
    @pytest.fixture
    def sample_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                question="Q1",
                expected_outcome=True,
                actual_outcome=True,
            ),  # TP
            EvaluationResult(
                question="Q2",
                expected_outcome=True,
                actual_outcome=True,
            ),  # TP
            EvaluationResult(
                question="Q3",
                expected_outcome=True,
                actual_outcome=True,
            ),  # TP
            EvaluationResult(
                question="Q4",
                expected_outcome=True,
                actual_outcome=True,
            ),  # TP
            EvaluationResult(
                question="Q5",
                expected_outcome=False,
                actual_outcome=False,
            ),  # TP
            EvaluationResult(
                question="Q6",
                expected_outcome=False,
                actual_outcome=False,
            ),  # TP
            EvaluationResult(
                question="Q7",
                expected_outcome=False,
                actual_outcome=False,
            ),  # TP
            EvaluationResult(
                question="Q8",
                expected_outcome=False,
                actual_outcome=True,
            ),  # TP
            EvaluationResult(
                question="Q9",
                expected_outcome=True,
                actual_outcome=False,
            ),  # TP
        ]

    def test_precision_value(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_outcome=True,
                actual_outcome=True,
            ),
            EvaluationResult(
                question="Q2",
                expected_outcome=False,
                actual_outcome=True,
            ),
        ]
        aggregate = AggregateResults(results)
        assert aggregate.precision() == 0.5

    def test_precision_nan(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_outcome=True,
                actual_outcome=False,
            ),
            EvaluationResult(
                question="Q2",
                expected_outcome=False,
                actual_outcome=False,
            ),
        ]
        aggregate = AggregateResults(results)
        assert np.isnan(aggregate.precision())

    def test_recall_value(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_outcome=True,
                actual_outcome=True,
            ),
            EvaluationResult(
                question="Q2",
                expected_outcome=True,
                actual_outcome=False,
            ),
        ]
        aggregate = AggregateResults(results)
        assert aggregate.recall() == 0.5

    def test_recall_nan(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_outcome=False,
                actual_outcome=False,
            ),
            EvaluationResult(
                question="Q2",
                expected_outcome=False,
                actual_outcome=True,
            ),
        ]
        aggregate = AggregateResults(results)
        assert np.isnan(aggregate.recall())

    def test_to_dict(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.to_dict() == {
            "Evaluated": 9,
            "Precision": aggregate.precision(),
            "Recall": aggregate.recall(),
            "True positives": 4,
            "True negatives": 3,
            "False positives": 1,
            "False negatives": 1,
        }

    def test_for_csv(self, sample_results):
        aggregate = AggregateResults(sample_results)
        expected_csv = [
            {"property": k, "value": v} for k, v in aggregate.to_dict().items()
        ]
        assert aggregate.for_csv() == expected_csv


@pytest.fixture
def mock_evaluation_data_file(tmp_path):
    file_path = tmp_path / "evaluation_data.jsonl"
    data = [
        {
            "question": "Question",
            "expected_outcome": True,
            "actual_outcome": True,
        },
        {
            "question": "Question",
            "expected_outcome": True,
            "actual_outcome": False,
        },
    ]

    with open(tmp_path / "evaluation_data.jsonl", "w", encoding="utf8") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

    return file_path


def test_evaluate_and_output_results_writes_results(
    mock_project_root, mock_evaluation_data_file
):
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

    assert "Aggregate Results" in caplog.text
    assert re.search(r"Evaluated\s+\d+", caplog.text)


def test_evaluate_and_output_results_copes_with_empty_data(
    mock_project_root, tmp_path, caplog
):
    caplog.set_level(logging.ERROR)
    file_path = tmp_path / "evaluation_data.jsonl"
    file_path.touch()

    evaluate_and_output_results(mock_project_root, file_path)

    assert "There is no data to evaluate" in caplog.text
