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
    GuardrailParseResult,
    NUM_GUARDRAILS,
)


@pytest.fixture
def result_true_positive() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="TP",
        expected_triggered=True,
        actual_triggered=True,
        expected_exact='True | "1, 3"',
        actual_exact='True | "1, 3"',
    )


@pytest.fixture
def result_false_positive() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="FP",
        expected_triggered=False,
        actual_triggered=True,
        expected_exact="False | None",
        actual_exact='True | "1, 3"',
    )


@pytest.fixture
def result_false_negative() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="FN",
        expected_triggered=True,
        actual_triggered=False,
        expected_exact='True | "1, 3"',
        actual_exact="False | None",
    )


@pytest.fixture
def result_true_negative() -> EvaluationResult:  # type: ignore
    return EvaluationResult(
        question="TN",
        expected_triggered=False,
        actual_triggered=False,
        expected_exact="False | None",
        actual_exact="False | None",
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
            ("False | None", [0, 0, 0, 0, 0, 0, 0]),
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

    def test_exact_triggered_parsing_error(self, caplog):
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
        assert result.expected_exact_triggered == [0, 0, 0, 0, 0, 0, 0]
        assert len(caplog.records) == 0

        # Check actual_exact_triggered
        assert result.actual_exact_triggered == [0, 0, 0, 0, 0, 0, 0]
        assert len(caplog.records) == 0


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

    def test_precision_value(self, result_true_positive, result_false_positive):
        results = [result_true_positive, result_false_positive]
        aggregate = AggregateResults(results)
        assert aggregate.precision() == 0.5

    def test_precision_nan(self, result_false_negative, result_true_negative):
        aggregate = AggregateResults([result_false_negative, result_true_negative])
        assert np.isnan(aggregate.precision())

    def test_recall_value(self, result_true_positive, result_false_negative):
        aggregate = AggregateResults([result_true_positive, result_false_negative])
        assert aggregate.recall() == 0.5

    def test_recall_nan(self, result_true_negative, result_false_positive):
        aggregate = AggregateResults([result_true_negative, result_false_positive])
        assert np.isnan(aggregate.recall())

    def test_to_dict(self, sample_results):
        aggregate = AggregateResults(sample_results)
        result_dict = aggregate.to_dict()

        assert result_dict["Evaluated"] == 9
        assert result_dict["Any-triggered True positives"] == 4
        assert result_dict["Any-triggered True negatives"] == 3
        assert result_dict["Any-triggered False positives"] == 1
        assert result_dict["Any-triggered False negatives"] == 1
        assert result_dict["Precision G1"] == 0.8
        assert result_dict["Recall G1"] == 0.8
        assert result_dict["F1 G1"] == 0.8
        assert result_dict["Precision G3"] == 0.8
        assert result_dict["Recall G3"] == 0.8
        assert result_dict["F1 G3"] == 0.8

        for guardrail in [2, 4, 5, 6, 7]:
            for metric in ["Precision", "Recall", "F1"]:
                key = f"{metric} G{guardrail}"
                assert np.isnan(result_dict[key]) or result_dict[key] == 0.0

    def test_for_csv(self, sample_results):
        aggregate = AggregateResults(sample_results)
        csv_data = aggregate.for_csv()

        # Verify the structure is correct
        assert isinstance(csv_data, list)
        assert all(isinstance(item, dict) for item in csv_data)
        assert all("property" in item and "value" in item for item in csv_data)

        # Check a specific non-nan entry
        precision_g1_entry = next(
            (item for item in csv_data if item["property"] == "Precision G1"), None
        )
        assert precision_g1_entry is not None
        assert precision_g1_entry["value"] == 0.8

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
        # G7: TP=0, FP=0 -> P=nan (zero_division=np.nan)
        expected_precision = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, np.nan]
        # Use np.testing.assert_array_equal to handle NaN values properly
        np.testing.assert_array_equal(precision, expected_precision)

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
        # G6: TP=0, FN=0 -> R=nan (zero_division=np.nan)
        # G7: TP=0, FN=1 -> R=0/(0+1)=0
        expected_recall = [1.0, 1.0, 1.0, 0.0, 1.0, np.nan, 0.0]
        # Use np.testing.assert_array_equal to handle NaN values properly
        np.testing.assert_array_equal(recall, expected_recall)

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
        # G4: P=0, R=0 -> F1=0
        # G5: P=1, R=1 -> F1=1
        # G6: P=0, R=nan -> F1=0
        # G7: P=nan, R=0 -> F1=0
        expected_f1 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        np.testing.assert_array_equal(f1, expected_f1)

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

    captured = caplog.text
    assert "Aggregate Results" in captured
    assert re.search(r"Evaluated\s+\d+", captured)


class TestGuardrailParseResult:
    @pytest.mark.parametrize(
        "exact_str, expected_triggered, expected_numbers, expected_warning_parts",
        [
            (
                'True | "1, 3"',
                True,
                [1, 3],
                [],
            ),
            (
                "False | None",
                False,
                [],
                [],
            ),
            # Duplicate number (1) and out of range numbers (0 and 9)
            (
                'True | "1, 1, 0, 9, 2"',
                True,
                [1, 2],
                [
                    "Removed duplicate numbers: [1]",
                    "Removed numbers outside the valid range",
                ],
            ),
            # Triggered == True but no guardrail numbers supplied -> returns is_triggered False
            (
                "True | None",
                False,
                [],
                ["Triggered == True but no guardrail numbers supplied."],
            ),
            # Triggered == False but guardrail numbers supplied
            (
                'False | "1"',
                False,
                [],
                ["Triggered == False but expected literal 'None'."],
            ),
            # Completely invalid format
            (
                "Unexpected format string",
                False,
                [],
                [
                    "String does not match expected format 'True | \"1,2\"' or 'False | None'."
                ],
            ),
        ],
    )
    def test_parse(
        self, exact_str, expected_triggered, expected_numbers, expected_warning_parts
    ):
        result = GuardrailParseResult.parse(exact_str, NUM_GUARDRAILS)

        assert result.is_triggered is expected_triggered
        assert result.valid_numbers == expected_numbers

        for expected_part in expected_warning_parts:
            assert any(expected_part in warning for warning in result.warnings), (
                f"Expected warning containing '{expected_part}' not found in {result.warnings}"
            )

    def test_to_vector(self):
        exact_str = 'True | "2, 4"'
        parse_result = GuardrailParseResult.parse(exact_str, NUM_GUARDRAILS)

        expected_vector = [0, 1, 0, 1, 0, 0, 0]
        assert parse_result.to_vector(NUM_GUARDRAILS) == expected_vector

        exact_str_false = "False | None"
        parse_result_false = GuardrailParseResult.parse(exact_str_false, NUM_GUARDRAILS)
        assert parse_result_false.to_vector(NUM_GUARDRAILS) == [0] * NUM_GUARDRAILS
