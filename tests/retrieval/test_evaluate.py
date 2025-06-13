import csv
import json
import re
import logging
import numpy as np
import pytest

from govuk_chat_evaluation.retrieval.evaluate import (
    AggregateResults,
    EvaluationResult,
    evaluate_and_output_results,
)


class TestEvaluationResult:
    def test_for_csv(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=["/path1", "/path2", "/path3", "/path4"],
            actual_exact_paths_and_scores=[("/path1", 0.9)],
        )

        assert result.for_csv() == {
            "question": "Test question",
            "expected_exact_paths": ["/path1", "/path2", "/path3", "/path4"],
            "actual_exact_paths_and_scores": [("/path1", 0.9)],
            "precision": 1.0,
            "recall": 0.25,
            "f1_score": 0.4,
            "f2_score": 0.2941,
        }

    def test_recall(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=["/path1", "/path2"],
            actual_exact_paths_and_scores=[("/path1", 0.9)],
        )

        assert result.recall() == 0.5

    def test_recall_zero_division(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=[],
            actual_exact_paths_and_scores=[("/path1", 0.9), ("/path3", 0.8)],
        )

        assert np.isnan(result.recall())

    def test_precision(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=["/path1", "/path2"],
            actual_exact_paths_and_scores=[
                ("/path1", 0.9),
                ("/path2", 0.8),
                ("/path3", 0.7),
                ("/path4", 0.6),
            ],
        )

        assert result.precision() == 0.5

    def test_precision_zero_division(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=["/path1", "/path2"],
            actual_exact_paths_and_scores=[],
        )

        assert np.isnan(result.precision())

    def test_f1_score(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=["/path1", "/path2"],
            actual_exact_paths_and_scores=[
                ("/path1", 0.9),
                ("/path3", 0.8),
                ("/path4", 0.7),
            ],
        )

        assert result.f1_score() == 0.4

    def test_f1_score_zero_division(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=[],
            actual_exact_paths_and_scores=[],
        )

        assert np.isnan(result.f1_score())

    def test_f2_score(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=["/path1", "/path2"],
            actual_exact_paths_and_scores=[("/path1", 0.9), ("/path3", 0.8)],
        )

        assert result.f2_score() == 0.5

    def test_f2_score_zero_division(self):
        result = EvaluationResult(
            question="Test question",
            expected_exact_paths=[],
            actual_exact_paths_and_scores=[],
        )

        assert np.isnan(result.f2_score())


class TestAggregateResults:
    @pytest.fixture
    def sample_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                question="Q1",
                expected_exact_paths=["/path1", "/path2"],
                actual_exact_paths_and_scores=[("/path1", 0.9), ("/path2", 0.8)],
            ),
            EvaluationResult(
                question="Q2",
                expected_exact_paths=["/path1"],
                actual_exact_paths_and_scores=[("/path3", 0.9)],
            ),
            EvaluationResult(
                question="Q3",
                expected_exact_paths=["/path1", "/path2"],
                actual_exact_paths_and_scores=[("/path1", 0.9)],
            ),
            EvaluationResult(
                question="Q4",
                expected_exact_paths=["/path1"],
                actual_exact_paths_and_scores=[
                    ("/path1", 0.9),
                    ("/path2", 0.8),
                    ("/path3", 0.7),
                ],
            ),
            EvaluationResult(
                question="Q5",
                expected_exact_paths=["/path1", "/path2"],
                actual_exact_paths_and_scores=[
                    ("/path1", 0.9),
                    ("/path2", 0.8),
                    ("/path3", 0.7),
                ],
            ),
        ]

    @pytest.fixture
    def aggregate(self, sample_results):
        return AggregateResults(sample_results)

    def test_precision_mean(self, aggregate):
        assert aggregate.precision_mean() == 0.6

    def test_precision_median(self, aggregate):
        assert aggregate.precision_median() == 0.6667

    def test_precision_max(self, aggregate):
        assert aggregate.precision_max() == 1.0

    def test_precision_standard_deviaion(self, aggregate):
        assert aggregate.precision_standard_deviation() == 0.3887

    def test_recall_mean(self, aggregate):
        assert aggregate.recall_mean() == 0.7

    def test_recall_median(self, aggregate):
        assert aggregate.recall_median() == 1.0

    def test_recall_max(self, aggregate):
        assert aggregate.recall_max() == 1.0

    def test_recall_standard_deviaion(self, aggregate):
        assert aggregate.recall_standard_deviation() == 0.40

    def test_f1_mean(self, aggregate):
        assert aggregate.f1_mean() == 0.5933

    def test_f1_median(self, aggregate):
        assert aggregate.f1_median() == 0.6667

    def test_f1_max(self, aggregate):
        assert aggregate.f1_max() == 1.0

    def test_f1_standard_deviaion(self, aggregate):
        assert aggregate.f1_standard_deviation() == 0.3389

    def test_f2_mean(self, aggregate):
        assert aggregate.f2_mean() == 0.6358

    def test_f2_median(self, aggregate):
        assert aggregate.f2_median() == 0.7143

    def test_f2_max(self, aggregate):
        assert aggregate.f2_max() == 1.0

    def test_f2_standard_deviaion(self, aggregate):
        assert aggregate.f2_standard_deviation() == 0.3533

    def test_to_dict(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.to_dict() == {
            "Evaluated": len(sample_results),
            "Precision mean": aggregate.precision_mean(),
            "Precision median": aggregate.precision_median(),
            "Precision max": aggregate.precision_max(),
            "Precision standard deviation": aggregate.precision_standard_deviation(),
            "Recall mean": aggregate.recall_mean(),
            "Recall median": aggregate.recall_median(),
            "Recall max": aggregate.recall_max(),
            "Recall standard deviation": aggregate.recall_standard_deviation(),
            "F1 mean": aggregate.f1_mean(),
            "F1 median": aggregate.f1_median(),
            "F1 max": aggregate.f1_max(),
            "F1 standard deviation": aggregate.f1_standard_deviation(),
            "F2 mean": aggregate.f2_mean(),
            "F2 median": aggregate.f2_median(),
            "F2 max": aggregate.f2_max(),
            "F2 standard deviation": aggregate.f2_standard_deviation(),
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
            "question": "Question 1",
            "expected_exact_paths": ["/path1", "/path2"],
            "actual_exact_paths_and_scores": [("/path1", 0.9), ("/path2", 0.8)],
        },
        {
            "question": "Question 2",
            "expected_exact_paths": ["/path1"],
            "actual_exact_paths_and_scores": [("/path3", 0.9)],
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
