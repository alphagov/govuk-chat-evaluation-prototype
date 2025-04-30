import pytest
from unittest.mock import MagicMock

from deepeval.metrics import BaseMetric
from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    CacheConfig,
    ErrorConfig,
)

from govuk_chat_evaluation.file_system import jsonl_to_models
from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase,
    EvaluationResult,
)
from govuk_chat_evaluation.rag_answers.deepeval_evaluate import (
    run_deepeval_evaluation,
    convert_deepeval_output_to_evaluation_results,
)


@pytest.mark.usefixtures("mock_deepeval_evaluate")
class TestRunDeepEvalEvaluation:
    @pytest.fixture
    def mock_test_cases(self, mock_input_data):
        models = jsonl_to_models(mock_input_data, EvaluationTestCase)
        return [m.to_llm_test_case() for m in models]

    @pytest.fixture
    def mock_metrics(self):
        metric1 = MagicMock(spec=BaseMetric)
        metric1.name = "faithfulness"
        metric1.threshold = 0.5
        metric1.async_mode = False

        metric2 = MagicMock(spec=BaseMetric)
        metric2.name = "bias"
        metric2.threshold = 0.5
        metric2.async_mode = False

        return [metric1, metric2]

    def test_runs_evaluate_defaults_to_1_time(
        self, mock_test_cases, mock_metrics, mock_deepeval_evaluate
    ):
        run_deepeval_evaluation(mock_test_cases, mock_metrics)
        mock_deepeval_evaluate.assert_called_once_with(
            test_cases=mock_test_cases, metrics=mock_metrics
        )

    def test_runs_evaluate_n_times(
        self, mock_test_cases, mock_metrics, mock_deepeval_evaluate
    ):
        run_deepeval_evaluation(mock_test_cases, mock_metrics, n_runs=2)
        mock_deepeval_evaluate.assert_called_with(
            test_cases=mock_test_cases, metrics=mock_metrics
        )

        assert mock_deepeval_evaluate.call_count == 2

    def test_returns_a_list_for_each_n_runs(self, mock_test_cases, mock_metrics):
        results = run_deepeval_evaluation(mock_test_cases, mock_metrics, n_runs=2)
        assert len(results) == 2

    def test_accepts_deepeval_options(
        self, mock_test_cases, mock_metrics, mock_deepeval_evaluate
    ):
        run_deepeval_evaluation(
            mock_test_cases,
            mock_metrics,
            display_config=DisplayConfig(print_results=True),
            async_config=AsyncConfig(max_concurrent=10),
            cache_config=CacheConfig(use_cache=True),
            error_config=ErrorConfig(ignore_errors=False),
        )

        mock_deepeval_evaluate.assert_called_with(
            test_cases=mock_test_cases,
            metrics=mock_metrics,
            display_config=DisplayConfig(print_results=True),
            async_config=AsyncConfig(max_concurrent=10),
            cache_config=CacheConfig(use_cache=True),
            error_config=ErrorConfig(ignore_errors=False),
        )


class TestConvertDeepEvalOutput:
    def test_convert_empty_results(self):
        results = convert_deepeval_output_to_evaluation_results([])
        assert results == []

    def test_converts_to_evaluation_results(self, mock_deepeval_results):
        results = convert_deepeval_output_to_evaluation_results(mock_deepeval_results)

        assert all(isinstance(item, EvaluationResult) for item in results)

        # each item should have the number of metrics multiplied by runs
        metrics_tested = len(mock_deepeval_results[0][0].metrics_data)
        runs = len(mock_deepeval_results)
        expected_metrics = metrics_tested * runs

        assert all(len(item.run_metric_outputs) == expected_metrics for item in results)

    def test_with_none_retrieval_context(self, mock_deepeval_results):
        # modify test data to have None for retrieval_context
        mock_deepeval_results[0][0].retrieval_context = None

        results = convert_deepeval_output_to_evaluation_results(
            [mock_deepeval_results[0]]
        )

        assert results[0].retrieval_context == []
