import csv
import json
import re
import logging

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
    def test_classification_triggered(self, expected, actual, expected_classification):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=expected,
            actual_triggered=actual,
            expected_exact="",
            actual_exact="",
        )
        assert result.classification_triggered == expected_classification

    @pytest.mark.parametrize(
        "exact_str, expected_vec",
        [
            ('True | "1, 3"', [1, 0, 1, 0, 0, 0, 0]),
            ('True | "1"', [1, 0, 0, 0, 0, 0, 0]),
            ('True | "5, 6, 7"', [0, 0, 0, 0, 1, 1, 1]),
            ("False | None", None),
        ],
    )
    def test_expected_exact_triggered(self, exact_str, expected_vec):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=True,
            actual_triggered=True,
            expected_exact=exact_str,
            actual_exact="Irrelevant",
        )
        assert result.expected_exact_triggered == expected_vec

    @pytest.mark.parametrize(
        "exact_str, expected_vec",
        [
            ('True | "1, 3"', [1, 0, 1, 0, 0, 0, 0]),
            ('True | "1"', [1, 0, 0, 0, 0, 0, 0]),
            ('True | "5, 6, 7"', [0, 0, 0, 0, 1, 1, 1]),
        ],
    )
    def test_actual_exact_triggered(self, exact_str, expected_vec):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=True,
            actual_triggered=True,
            expected_exact="Irrelevant",
            actual_exact=exact_str,
        )
        assert result.actual_exact_triggered == expected_vec

    def test_for_csv(self):
        result = EvaluationResult(
            question="Test question",
            expected_triggered=True,
            actual_triggered=True,
            expected_exact='True | "1, 3"',
            actual_exact='True | "1, 3"',
        )

        assert result.for_csv() == {
            "question": "Test question",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_exact": 'True | "1, 3"',
            "actual_exact": 'True | "1, 3"',
            "classification": "true_positive",
        }

    @pytest.mark.parametrize(
        "guardrails_str, num_guardrails, vec",
        [
            ("1, 2, 5", 7, [1, 1, 0, 0, 1, 0, 0]),
            ("1, 2, 3, 4, 5, 6, 7", 7, [1, 1, 1, 1, 1, 1, 1]),
            ("2", 7, [0, 1, 0, 0, 0, 0, 0]),
        ],
    )
    def test_guardrail_str_to_vec(self, guardrails_str, num_guardrails, vec):
        assert (
            EvaluationResult.guardrail_str_to_vec(guardrails_str, num_guardrails) == vec
        )

    @pytest.mark.parametrize(
        "input_str, num_guardrails, expected_tuple",
        [
            ('True | "1, 3"', 7, (True, "1, 3")),
            ('True | "1"', 7, (True, "1")),
            ('True | "7"', 7, (True, "7")),
            ('True | "1, 7"', 7, (True, "1, 7")),
            ("False | None", 7, (False, "None")),
        ],
    )
    def test_parse_exact_string(self, input_str, num_guardrails, expected_tuple):
        assert (
            EvaluationResult.parse_exact_string(input_str, num_guardrails)
            == expected_tuple
        )

    @pytest.mark.parametrize(
        "invalid_input_str, num_guardrails",
        [
            ("True | 1, 2, 6", 7),  # Missing double quotes
            ('True | ""', 7),  # Empty string in quotes
            ('True | " "', 7),  # Whitespace string in quotes
            ('False | "None"', 7),  # "None" should not be quoted
            ("True | None", 7),  # True should have quoted digits, not None
            ('False | "1, 2"', 7),  # False should have None, not quoted digits
            ("Gibberish", 7),  # Completely incorrect format
            ("True |", 7),  # Incomplete format
            ("False | ", 7),  # Incomplete format
            ('True | "1, 8"', 7),  # Number out of range (high)
            ('True | "0, 7"', 7),  # Number out of range (low)
            ('True | "1, 1"', 7),  # Duplicate number
        ],
    )
    def test_parse_exact_string_invalid_format(self, invalid_input_str, num_guardrails):
        """Test that parse_exact_string raises ValueError for invalid formats."""

        if invalid_input_str in ('True | ""', 'True | " "'):
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' has 'True' but contains an empty "
                f"quoted string. Expected comma-separated numbers."
            )
        elif invalid_input_str == 'True | "1, 8"':
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' contains numbers outside the "
                f"valid range [1, {num_guardrails}]."
            )
        elif invalid_input_str == 'True | "0, 7"':
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' contains numbers outside the "
                f"valid range [1, {num_guardrails}]."
            )
        elif invalid_input_str == 'True | "1, 1"':
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' contains duplicate guardrail numbers."
            )
        elif invalid_input_str == "True | None":
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' reports being triggered, but is not "
                f"followed by quoted comma-separated numbers (e.g., 'True | \"1,2\"')."
            )
        elif invalid_input_str in ('False | "None"', 'False | "1, 2"'):
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' starts with 'False' but is not "
                f"followed by 'None'. Expected format 'False | None'."
            )
        else:  # General format mismatch
            expected_error_pattern = re.escape(
                f"Guardrail string '{invalid_input_str}' does not match expected format "
                f"'True | \"<comma-separated numbers>\"' or 'False | None'."
            )

        with pytest.raises(ValueError, match=expected_error_pattern):
            EvaluationResult.parse_exact_string(invalid_input_str, num_guardrails)

    def test_exact_triggered_parsing_error(self, caplog):
        """Test that properties return None and log warning on parsing error."""
        invalid_format = "True | Invalid Format"
        result = EvaluationResult(
            question="Test Invalid Q",
            expected_triggered=True,  # Overall trigger is True
            actual_triggered=True,
            expected_exact=invalid_format,
            actual_exact=invalid_format,
        )

        caplog.set_level(logging.WARNING)
        caplog.clear()

        # Check expected_exact_triggered
        assert result.expected_exact_triggered is None
        assert len(caplog.records) == 1
        assert "Failed to parse expected_exact" in caplog.text
        assert invalid_format in caplog.text
        assert result.question in caplog.text
        assert "Treating as None" in caplog.text

        caplog.clear()

        # Check actual_exact_triggered
        assert result.actual_exact_triggered is None
        assert len(caplog.records) == 1
        assert "Failed to parse actual_exact" in caplog.text
        assert invalid_format in caplog.text
        assert result.question in caplog.text
        assert "Treating as None" in caplog.text


class TestAggregateResults:
    @pytest.fixture
    def sample_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
            ),  # TP
            EvaluationResult(
                question="Q2",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
            ),  # TP
            EvaluationResult(
                question="Q3",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
            ),  # TP
            EvaluationResult(
                question="Q4",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
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
                actual_exact='True | "1, 3"',
            ),  # TP
            EvaluationResult(
                question="Q9",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact='True | "1, 3"',
                actual_exact="False | None",
            ),  # TP
        ]

    def test_precision_value(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=False,
                actual_triggered=True,
                expected_exact="False | None",
                actual_exact='True | "1, 3"',
            ),
        ]
        aggregate = AggregateResults(results)
        assert aggregate.precision() == 0.5

    def test_precision_nan(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact='True | "1, 3"',
                actual_exact="False | None",
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="False | None",
                actual_exact="False | None",
            ),
        ]
        aggregate = AggregateResults(results)
        assert np.isnan(aggregate.precision())

    def test_recall_value(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact='True | "1, 3"',
                actual_exact="False | None",
            ),
        ]
        aggregate = AggregateResults(results)
        assert aggregate.recall() == 0.5

    def test_recall_nan(self):
        results = [
            EvaluationResult(
                question="Q1",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="False | None",
                actual_exact="False | None",
            ),
            EvaluationResult(
                question="Q2",
                expected_triggered=False,
                actual_triggered=True,
                expected_exact="False | None",
                actual_exact='True | "1, 3"',
            ),
        ]
        aggregate = AggregateResults(results)
        assert np.isnan(aggregate.recall())

    def test_to_dict(self, sample_results):
        aggregate = AggregateResults(sample_results)
        expected_dict = {
            "Evaluated": 9,
            "Any-triggered Precision": aggregate.precision(),
            "Any-triggered Recall": aggregate.recall(),
            "Any-triggered True positives": 4,
            "Any-triggered True negatives": 3,
            "Any-triggered False positives": 1,
            "Any-triggered False negatives": 1,
            # Expected Per-Guardrail Metrics based on sample_results
            "Precision G1": 0.8,  # TP=4, FP=1 -> 4/5
            "Recall G1": 0.8,  # TP=4, FN=1 -> 4/5
            "F1 G1": 0.8,
            "Precision G2": 0.0,
            "Recall G2": 0.0,
            "F1 G2": 0.0,
            "Precision G3": 0.8,  # TP=4, FP=1 -> 4/5
            "Recall G3": 0.8,  # TP=4, FN=1 -> 4/5
            "F1 G3": 0.8,
            "Precision G4": 0.0,
            "Recall G4": 0.0,
            "F1 G4": 0.0,
            "Precision G5": 0.0,
            "Recall G5": 0.0,
            "F1 G5": 0.0,
            "Precision G6": 0.0,
            "Recall G6": 0.0,
            "F1 G6": 0.0,
            "Precision G7": 0.0,
            "Recall G7": 0.0,
            "F1 G7": 0.0,
        }
        assert aggregate.to_dict() == expected_dict

    def test_for_csv(self, sample_results):
        aggregate = AggregateResults(sample_results)
        expected_csv = [
            {"property": k, "value": v} for k, v in aggregate.to_dict().items()
        ]
        assert aggregate.for_csv() == expected_csv

    @pytest.fixture
    def per_guardrail_results(self) -> list[EvaluationResult]:
        """Fixture providing sample results for per-guardrail testing."""
        return [
            # Case 1: Perfect match
            EvaluationResult(
                question="Q1",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "1, 3"',
                actual_exact='True | "1, 3"',
            ),
            # Case 2: FP (predicted 4)
            EvaluationResult(
                question="Q2",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "2"',
                actual_exact='True | "2, 4"',  # Predicts 4 incorrectly
            ),
            # Case 3: FN (missed 4)
            EvaluationResult(
                question="Q3",
                expected_triggered=True,
                actual_triggered=True,
                expected_exact='True | "4, 5"',
                actual_exact='True | "5"',  # Misses 4
            ),
            # Case 4: Both False
            EvaluationResult(
                question="Q4",
                expected_triggered=False,
                actual_triggered=False,
                expected_exact="False | None",
                actual_exact="False | None",
            ),
            # Case 5: Actual triggered when expected False (FP for guardrail 6)
            EvaluationResult(
                question="Q5",
                expected_triggered=False,
                actual_triggered=True,
                expected_exact="False | None",
                actual_exact='True | "6"',
            ),
            # Case 6: Expected triggered when actual False (FN for guardrail 7)
            EvaluationResult(
                question="Q6",
                expected_triggered=True,
                actual_triggered=False,
                expected_exact='True | "7"',
                actual_exact="False | None",
            ),
        ]

    def test_expected_actual_vectors(self, per_guardrail_results):
        aggregate = AggregateResults(per_guardrail_results)
        expected_vectors, actual_vectors = aggregate._expected_actual_vectors

        # Based on per_guardrail_results fixture
        expected_ground_truth = [
            [1, 0, 1, 0, 0, 0, 0],  # Q1: "1, 3"
            [0, 1, 0, 0, 0, 0, 0],  # Q2: "2"
            [0, 0, 0, 1, 1, 0, 0],  # Q3: "4, 5"
            [0, 0, 0, 0, 0, 0, 0],  # Q4: False
            [0, 0, 0, 0, 0, 0, 0],  # Q5: False
            [0, 0, 0, 0, 0, 0, 1],  # Q6: "7"
        ]
        actual_predictions = [
            [1, 0, 1, 0, 0, 0, 0],  # Q1: "1, 3"
            [0, 1, 0, 1, 0, 0, 0],  # Q2: "2, 4"
            [0, 0, 0, 0, 1, 0, 0],  # Q3: "5"
            [0, 0, 0, 0, 0, 0, 0],  # Q4: False
            [0, 0, 0, 0, 0, 1, 0],  # Q5: "6"
            [0, 0, 0, 0, 0, 0, 0],  # Q6: False
        ]

        assert expected_vectors == expected_ground_truth
        assert actual_vectors == actual_predictions

    def test_expected_actual_vectors_empty(self):
        aggregate = AggregateResults([])
        expected_vectors, actual_vectors = aggregate._expected_actual_vectors
        assert expected_vectors == []
        assert actual_vectors == []

    def test_precision_per_guardrail(self, per_guardrail_results):
        aggregate = AggregateResults(per_guardrail_results)
        precision = aggregate.precision_per_guardrail()
        # Expected precision based on _expected_actual_vectors:
        # G1: TP=1, FP=0 -> P=1/(1+0)=1
        # G2: TP=1, FP=0 -> P=1/(1+0)=1
        # G3: TP=1, FP=0 -> P=1/(1+0)=1
        # G4: TP=0, FP=1 -> P=0/(0+1)=0
        # G5: TP=1, FP=0 -> P=1/(1+0)=1
        # G6: TP=0, FP=1 -> P=0/(0+1)=0
        # G7: TP=0, FP=0 -> P=0/(0+0)=0 (zero_division=0)
        expected_precision = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        assert precision == expected_precision

    def test_precision_per_guardrail_empty(self):
        aggregate = AggregateResults([])
        precision = aggregate.precision_per_guardrail()
        assert all(np.isnan(p) for p in precision)
        assert len(precision) == 7

    def test_recall_per_guardrail(self, per_guardrail_results):
        aggregate = AggregateResults(per_guardrail_results)
        recall = aggregate.recall_per_guardrail()
        # Expected recall based on _expected_actual_vectors:
        # G1: TP=1, FN=0 -> R=1/(1+0)=1
        # G2: TP=1, FN=0 -> R=1/(1+0)=1
        # G3: TP=1, FN=0 -> R=1/(1+0)=1
        # G4: TP=0, FN=1 -> R=0/(0+1)=0
        # G5: TP=1, FN=0 -> R=1/(1+0)=1
        # G6: TP=0, FN=0 -> R=0/(0+0)=0 (zero_division=0)
        # G7: TP=0, FN=1 -> R=0/(0+1)=0
        expected_recall = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        assert recall == expected_recall

    def test_recall_per_guardrail_empty(self):
        aggregate = AggregateResults([])
        recall = aggregate.recall_per_guardrail()
        assert all(np.isnan(r) for r in recall)
        assert len(recall) == 7

    def test_f1_per_guardrail(self, per_guardrail_results):
        aggregate = AggregateResults(per_guardrail_results)
        f1 = aggregate.f1_per_guardrail()
        # Expected F1 based on precision and recall above:
        # G1: P=1, R=1 -> F1=2*(1*1)/(1+1)=1
        # G2: P=1, R=1 -> F1=1
        # G3: P=1, R=1 -> F1=1
        # G4: P=0, R=0 -> F1=0 (zero_division=0)
        # G5: P=1, R=1 -> F1=1
        # G6: P=0, R=0 -> F1=0
        # G7: P=0, R=0 -> F1=0
        expected_f1 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        assert f1 == expected_f1

    def test_f1_per_guardrail_empty(self):
        aggregate = AggregateResults([])
        f1 = aggregate.f1_per_guardrail()
        assert all(np.isnan(f) for f in f1)
        assert len(f1) == 7


@pytest.fixture
def mock_evaluation_data_file(tmp_path):
    file_path = tmp_path / "evaluation_data.jsonl"
    data = [
        {
            "question": "Question 1",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_exact": 'True | "1"',
            "actual_exact": 'True | "1"',
        },
        {
            "question": "Question 2",
            "expected_triggered": True,
            "actual_triggered": False,
            "expected_exact": 'True | "2"',
            "actual_exact": "False | None",
        },
        {
            "question": "Question 3",
            "expected_triggered": False,
            "actual_triggered": False,
            "expected_exact": "False | None",
            "actual_exact": "False | None",
        },
        {
            "question": "Question 4",
            "expected_triggered": False,
            "actual_triggered": True,
            "expected_exact": "False | None",
            "actual_exact": 'True | "3"',
        },
    ]

    with open(file_path, "w", encoding="utf8") as file:
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
