import csv
import json
import re

import numpy as np
import pytest

from govuk_chat_evaluation.output_guardrails.evaluate import (
    AggregateResults,
    EvaluationResult,
    evaluate_and_output_results,
)


class TestEvaluationResult:
    @pytest.mark.parametrize(
        "expected_triggered, actual_triggered, expected_classification_triggered",
        [
            (True, True, "true_positive"),
            (False, False, "true_negative"),
            (False, True, "false_positive"),
            (True, False, "false_negative"),
        ],
    )
    def test_classification_triggered(
        self, expected_triggered, actual_triggered, expected_classification_triggered
    ):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=expected_triggered,
            actual_triggered=actual_triggered,
            expected_exact="",
            actual_exact="",
        )
        assert result.classification_triggered == expected_classification_triggered

    @pytest.mark.parametrize(
        "expected_exact, actual_exact, expected_classification_exact",
        [
            ('True | "1, 2"', 'True | "1, 2"', "true_positive"),
            ("False | None", "False | None", "true_negative"),
            ("False | None", 'True | "1, 2"', "false_positive"),
            ('True | "1, 2"', "False | None", "false_negative"),
        ],
    )
    def test_classification_exact(
        self, expected_exact, actual_exact, expected_classification_exact
    ):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=True,
            actual_triggered=True,
            expected_exact=expected_exact,
            actual_exact=actual_exact,
        )
        assert result.classification_exact == expected_classification_exact

    def test_for_csv(self):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=True,
            actual_triggered=True,
            expected_exact='True | "1, 2"',
            actual_exact='True | "1, 2"',
        )

        assert result.for_csv() == {
            "question": "Test question",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_exact": 'True | "1, 2"',
            "actual_exact": 'True | "1, 2"',
            "classification_triggered": "true_positive",
            "classification_exact": "true_positive",
        }


class TestAggregateResults:
    @pytest.fixture
    def sample_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 2"',
                actual_exact='True | "1, 2"',
            ),  # TP
            EvaluationResult(
                question="Q2",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 2"',
                actual_exact='True | "1, 2"',
            ),  # TP
            EvaluationResult(
                question="Q3",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 2"',
                actual_exact='True | "1, 2"',
            ),  # TP
            EvaluationResult(
                question="Q4",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 2"',
                actual_exact='True | "1, 2"',
            ),  # TP
            EvaluationResult(
                question="Q5",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="False | None",
                actual_exact="False | None",
            ),  # TP
            EvaluationResult(
                question="Q6",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="False | None",
                actual_exact="False | None",
            ),  # TP
            EvaluationResult(
                question="Q7",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="False | None",
                actual_exact="False | None",
            ),  # TP
            EvaluationResult(
                question="Q8",
                expected_triggered=False,
                actual_triggered=True,
                expected_exact="False | None",
                actual_exact='True | "1 ,2 "',
            ),  # TP
            EvaluationResult(
                question="Q9",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact='True | "1, 2"',
                actual_exact="False | None",
            ),  # TP
        ]

    def test_precision_triggered_value(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact="",
                actual_exact="",
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=False,
                actual_triggered=True,
                expected_exact="",
                actual_exact="",
            ),
        ]
        aggregate = AggregateResults(results)
        assert aggregate.precision_triggered() == 0.5

    def test_precision_triggered_nan(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact="",
                actual_exact="",
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="",
                actual_exact="",
            ),
        ]
        aggregate = AggregateResults(results)
        assert np.isnan(aggregate.precision_triggered())

    def test_recall_triggered_value(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact="",
                actual_exact="",
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact="",
                actual_exact="",
            ),
        ]
        aggregate = AggregateResults(results)
        assert aggregate.recall_triggered() == 0.5

    def test_recall_triggered_nan(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="",
                actual_exact="",
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=False,
                actual_triggered=True,
                expected_exact="",
                actual_exact="",
            ),
        ]
        aggregate = AggregateResults(results)
        assert np.isnan(aggregate.recall_triggered())

    def test_to_dict(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.to_dict() == {
            "Evaluated": 9,
            "Precision (triggered)": aggregate.precision_triggered(),
            "Recall (triggered)": aggregate.recall_triggered(),
            "True positives (triggered)": 4,
            "True negatives (triggered)": 3,
            "False positives (triggered)": 1,
            "False negatives (triggered)": 1,
            "Precision (exact)": aggregate.precision_exact(),
            "Recall (exact)": aggregate.recall_exact(),
            "Exact True positives (exact)": 4,
            "True negatives (exact)": 3,
            "Exact False positive (exact)": 1,
            "Exact False negative (exact)": 1,
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
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_exact": 'True | "1, 2"',
            "actual_exact": 'True | "1, 2"',
        },
        {
            "question": "Question",
            "expected_triggered": True,
            "actual_triggered": False,
            "expected_exact": 'True | "1, 2"',
            "actual_exact": "False | None",
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
    mock_project_root, mock_evaluation_data_file, capsys
):
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)

    captured = capsys.readouterr()
    assert "Aggregate Results" in captured.out
    assert re.search(r"Evaluated\s+\d+", captured.out)
