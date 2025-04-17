import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationResult,
    RunMetricOutput,
    EvaluationTestCase,
    StructuredContext,
)
from govuk_chat_evaluation.rag_answers.evaluate import (
    AggregatedResults,
    evaluate_and_output_results,
    DEEPEVAL_EVAL_PARAMETERS,
)


def test_aggregated_results_stats():
    eval_results = [
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

    agg = AggregatedResults(eval_results)
    df_avg = agg.per_input_metric_averages
    df_summary = agg.summary

    assert isinstance(df_avg, pd.DataFrame)
    assert "faithfulness" in df_avg.columns.get_level_values(1)
    assert "bias" in df_avg.columns.get_level_values(1)
    assert df_avg.shape[0] == 2
    assert df_avg.shape[1] == 6  # 4 metrics + input name
    assert set(df_avg.columns.get_level_values(0)) == {
        "input",
        "name",
        "mean",
        "mean",
        "std",
        "std",
    }
    assert set(df_avg.columns.get_level_values(1)) == {
        "",
        "bias",
        "faithfulness",
        "bias",
        "faithfulness",
    }
    assert df_avg["mean"].shape[0] == 2
    assert df_avg["std"].shape[0] == 2
    assert not df_summary.empty
    assert set(df_summary.columns) == {"mean", "median", "std"}
    assert df_summary.shape[0] == 2  # 2 metrics
    assert df_summary.shape[1] == 3  # mean, median, std
    assert df_summary.index.tolist() == ["bias", "faithfulness"]
    assert df_summary["mean"]["faithfulness"] == 0.95
    assert df_summary["mean"]["bias"] == 0.025


def test_export_to_csvs(tmp_path):
    eval_results = [
        EvaluationResult(
            name="Test",
            input="input",
            actual_output="output",
            expected_output="output",
            retrieval_context=[],
            run_metric_outputs=[
                RunMetricOutput(run=0, metric="accuracy", score=1.0),
            ],
        )
    ]

    agg = AggregatedResults(eval_results)
    agg.export_to_csvs(tmp_path)

    assert (tmp_path / "tidy_results.csv").exists()
    assert (tmp_path / "results_per_input.csv").exists()
    assert (tmp_path / "results_summary.csv").exists()


@pytest.fixture
def mock_evaluation_config():
    """Create a mock evaluation configuration."""
    mock_metric = MagicMock()
    mock_metric.__name__ = "MockMetric"
    mock_metric.evaluation_model = "mock-model"
    mock_metric.strict_mode = False
    mock_metric.async_mode = True

    config = MagicMock()
    config.metric_instances.return_value = [mock_metric]
    config.n_runs = 1
    return config


def test_evaluate_and_output_results_function_calls(tmpdir, mock_evaluation_config):
    """Test that evaluate_and_output_results calls all expected functions."""
    # setup
    output_dir = Path(tmpdir)
    eval_data_path = Path(tmpdir) / "test_eval_data.jsonl"

    # mock test cases and results
    mock_test_cases = [MagicMock(), MagicMock()]
    for case in mock_test_cases:
        case.to_llm_test_case.return_value = {"mock": "case"}

    mock_results = [MagicMock(), MagicMock()]
    mock_deepeval_output = [{"mock": "output"}]

    # set up all mocks
    with patch(
        "govuk_chat_evaluation.rag_answers.evaluate.jsonl_to_models",
        return_value=mock_test_cases,
    ) as mock_jsonl_to_models:
        with patch(
            "govuk_chat_evaluation.rag_answers.evaluate.run_deepeval_evaluation",
            return_value=mock_deepeval_output,
        ) as mock_run_eval:
            with patch(
                "govuk_chat_evaluation.rag_answers.evaluate.convert_deepeval_output_to_evaluation_results",
                return_value=mock_results,
            ) as mock_convert:
                with patch(
                    "govuk_chat_evaluation.rag_answers.evaluate.AggregatedResults"
                ) as MockAggregatedResults:
                    mock_aggregation = MagicMock()
                    MockAggregatedResults.return_value = mock_aggregation

                    structured_context = StructuredContext(
                        title="VAT",
                        heading_hierarchy=["Tax", "VAT"],
                        description="VAT overview",
                        html_content="<p>Some HTML about VAT</p>",
                        exact_path="https://gov.uk/vat",
                        base_path="https://gov.uk",
                    )
                    valid_case = EvaluationTestCase(
                        question="What is GOV.UK?",
                        ideal_answer="GOV.UK is the UK government's website for citizens.",
                        llm_answer="It's a site with UK government info.",
                        retrieved_context=[structured_context],
                    )

                    with open(eval_data_path, "w", encoding="utf-8") as f:
                        f.write(valid_case.model_dump_json() + "\n")

                    evaluate_and_output_results(
                        output_dir=output_dir,
                        evaluation_data_path=eval_data_path,
                        evaluation_config=mock_evaluation_config,
                    )

                    assert os.environ["DEEPEVAL_RESULTS_FOLDER"] == str(output_dir)

                    mock_jsonl_to_models.assert_called_once_with(
                        eval_data_path, EvaluationTestCase
                    )

                    mock_run_eval.assert_called_once()
                    call_args = mock_run_eval.call_args[1]
                    assert "cases" in call_args
                    assert "metrics" in call_args
                    assert "n_runs" in call_args
                    assert (
                        call_args["metrics"]
                        == mock_evaluation_config.metric_instances()
                    )
                    assert call_args["n_runs"] == mock_evaluation_config.n_runs
                    for param, value in DEEPEVAL_EVAL_PARAMETERS.items():
                        assert call_args[param] == value

                    # verify convert_deepeval_output_to_evaluation_results was called
                    mock_convert.assert_called_once_with(mock_deepeval_output)

                    # verify AggregatedResults was instantiated and export method was called
                    MockAggregatedResults.assert_called_once_with(mock_results)
                    mock_aggregation.export_to_csvs.assert_called_once_with(output_dir)
