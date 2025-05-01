import pytest
import pandas as pd
import re
import yaml
import logging

from govuk_chat_evaluation.rag_answers.data_models import (
    Config,
    EvaluationResult,
    RunMetricOutput,
)
from govuk_chat_evaluation.rag_answers.evaluate import (
    AggregatedResults,
    evaluate_and_output_results,
)
from tests.conftest import assert_csv_exists_with_headers


@pytest.fixture
def mock_evaluation_config(mock_config_file):
    with open(mock_config_file, "r") as file:
        config_data = yaml.safe_load(file)
        return Config(**config_data)


@pytest.fixture
def mock_run_deepeval_evaluation(mocker, mock_deepeval_results):
    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.run_deepeval_evaluation",
        return_value=mock_deepeval_results,
    )


class TestAggregateResults:
    @pytest.fixture
    def mock_evaluation_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                name="Test1",
                input="Is Vat a tax?",
                actual_output="Yes",
                expected_output="Yes, VAT is a tax.",
                retrieval_context=[],
                run_metric_outputs=[
                    RunMetricOutput(run=0, metric="faithfulness", score=1.0),
                    RunMetricOutput(run=1, metric="faithfulness", score=0.8),
                    RunMetricOutput(run=0, metric="bias", score=0.1),
                    RunMetricOutput(run=0, metric="bias", score=0.0),
                ],
            ),
            EvaluationResult(
                name="Test2",
                input="What is capital of France?",
                actual_output="Paris",
                expected_output="Paris",
                retrieval_context=[],
                run_metric_outputs=[
                    RunMetricOutput(run=0, metric="faithfulness", score=1.0),
                    RunMetricOutput(run=1, metric="faithfulness", score=1.0),
                    RunMetricOutput(run=0, metric="bias", score=0.0),
                    RunMetricOutput(run=0, metric="bias", score=0.0),
                ],
            ),
        ]

    def test_per_input_metric_averages(self, mock_evaluation_results):
        metric_averages = AggregatedResults(
            mock_evaluation_results
        ).per_input_metric_averages
        assert isinstance(metric_averages, pd.DataFrame)
        assert list(metric_averages.columns) == [
            ("name", ""),
            ("input", ""),
            ("mean", "bias"),
            ("mean", "faithfulness"),
            ("std", "bias"),
            ("std", "faithfulness"),
        ]
        assert list(metric_averages[("name", "")]) == ["Test1", "Test2"]
        assert list(metric_averages[("input", "")]) == [
            "Is Vat a tax?",
            "What is capital of France?",
        ]

    def test_summary(self, mock_evaluation_results):
        summary = AggregatedResults(mock_evaluation_results).summary
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == ["median", "mean", "std"]
        assert list(summary.index) == ["bias", "faithfulness"]

    def test_export_to_csvs(self, mock_evaluation_results, tmp_path):
        agg = AggregatedResults(mock_evaluation_results)
        agg.export_to_csvs(tmp_path)

        assert_csv_exists_with_headers(
            tmp_path / "tidy_results.csv", "actual_output", "expected_output"
        )
        assert_csv_exists_with_headers(
            tmp_path / "results_per_input.csv", "mean", "std"
        )
        assert_csv_exists_with_headers(
            tmp_path / "results_summary.csv", "metric", "median"
        )


def test_evaluate_and_output_results_runs_evaluation(
    tmp_path, mock_input_data, mock_evaluation_config, mock_run_deepeval_evaluation
):
    evaluate_and_output_results(tmp_path, mock_input_data, mock_evaluation_config)

    mock_run_deepeval_evaluation.assert_called_once()


@pytest.mark.usefixtures("mock_run_deepeval_evaluation")
def test_evaluate_and_output_results_writes_results_to_disk(
    tmp_path, mock_input_data, mock_evaluation_config
):
    evaluate_and_output_results(tmp_path, mock_input_data, mock_evaluation_config)

    for filename in [
        "tidy_results.csv",
        "results_per_input.csv",
        "results_summary.csv",
    ]:
        assert (tmp_path / filename).exists()


@pytest.mark.usefixtures("mock_run_deepeval_evaluation")
def test_evaluate_and_output_results_prints_summary(
    tmp_path, mock_input_data, mock_evaluation_config, caplog
):
    caplog.set_level(logging.INFO)
    evaluate_and_output_results(tmp_path, mock_input_data, mock_evaluation_config)

    captured = caplog.text
    assert "Evaluation Results:" in captured
    assert re.search(r"median\s+mean\s+std", captured)
