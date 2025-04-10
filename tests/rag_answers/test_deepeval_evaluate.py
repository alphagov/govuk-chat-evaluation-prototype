import pytest
from unittest.mock import patch, MagicMock
from typing import cast, List

from deepeval.test_case import LLMTestCase
from deepeval.evaluate import TestResult, BaseMetric

from govuk_chat_evaluation.rag_answers.deepeval_evaluate import run_deepeval_evaluation, convert_deepeval_output_to_evaluation_results


class TestRunDeepEvalEvaluation:
    @pytest.fixture
    def mock_test_cases(self):
        # create test cases with required attributes
        test_case1 = MagicMock(spec=LLMTestCase)
        test_case1.input = "test input 1"
        test_case1.actual_output = "test output 1"
        test_case1.expected_output = "expected output 1"
        test_case1.retrieval_context = ["context 1"]
        test_case1.name = "test_case_1"
        test_case1.context = None
        test_case1.tools_called = None
        test_case1.expected_tools = None
        test_case1.token_cost = None
        test_case1.completion_time = None
        test_case1.additional_metadata = {}  
        test_case1.comments = "" 

        test_case2 = MagicMock(spec=LLMTestCase)
        test_case2.input = "test input 2"
        test_case2.actual_output = "test output 2"
        test_case2.expected_output = "expected output 2"
        test_case2.retrieval_context = ["context 2"]
        test_case2.name = "test_case_2"
        test_case2.context = None
        test_case2.tools_called = None
        test_case2.expected_tools = None
        test_case2.token_cost = None
        test_case2.completion_time = None
        test_case2.additional_metadata = {}
        test_case2.comments = ""
        
        return [test_case1, test_case2]

    @pytest.fixture
    def mock_metrics(self):
        metric1 = MagicMock(spec=BaseMetric)
        metric1.name = "faithfulness"
        metric1.threshold = 0.5
        metric1.async_mode=False
        
        metric2 = MagicMock(spec=BaseMetric)
        metric2.name = "bias"
        metric2.threshold = 0.5
        metric2.async_mode=False
        
        return [metric1, metric2]

    @pytest.fixture
    def mock_test_results(self):
        # Create test results with proper metric data
        metric_data1 = MagicMock()
        metric_data1.name = "faithfulness"
        metric_data1.score = 0.9
        metric_data1.reason = "Good faith"
        metric_data1.evaluation_cost = 0.1
        metric_data1.success = True

        metric_data2 = MagicMock()
        metric_data2.name = "bias"
        metric_data2.score = 0.8
        metric_data2.reason = "No Bias"
        metric_data2.evaluation_cost = 0.2
        metric_data2.success = True

        test_result1 = MagicMock(spec=TestResult)
        test_result1.name = "test_case_1"
        test_result1.input = "test input 1"
        test_result1.actual_output = "test output 1"
        test_result1.expected_output = "expected output 1"
        test_result1.retrieval_context = ["context 1"]
        test_result1.metrics_data = [metric_data1, metric_data2]
        
        test_result2 = MagicMock(spec=TestResult)
        test_result2.name = "test_case_2"
        test_result2.input = "test input 2"
        test_result2.actual_output = "test output 2"
        test_result2.expected_output = "expected output 2"
        test_result2.retrieval_context = ["context 2"]
        test_result2.metrics_data = [metric_data1, metric_data2]
        
        return [test_result1, test_result2]

    def test_run_single_evaluation(self, mock_test_cases, mock_metrics, mock_test_results):
        # Use a direct patch to completely bypass the function internals
        with patch('govuk_chat_evaluation.rag_answers.deepeval_evaluate.deepeval_evaluate') as mock_deepeval:
            # setup the mock to return a value with test_results attribute
            mock_evaluation = MagicMock()
            mock_evaluation.test_results = mock_test_results
            mock_deepeval.return_value = mock_evaluation
                
            results = run_deepeval_evaluation(mock_test_cases, mock_metrics, n_runs=1) # type: ignore
                
            # verify the function was called correctly
            mock_deepeval.assert_called_once_with(
                    test_cases=cast(List[LLMTestCase], mock_test_cases),
                    metrics=mock_metrics
                )
                
            # verify the results
            assert results == [mock_test_results]
            assert len(results) == 1

        
    def test_run_multiple_evaluations(self, mock_test_cases, mock_metrics, mock_test_results):
        with patch('govuk_chat_evaluation.rag_answers.deepeval_evaluate.deepeval_evaluate') as mock_deepeval:
            mock_evaluation = MagicMock()
            mock_evaluation.test_results = mock_test_results
            mock_deepeval.return_value = mock_evaluation
            
            results = run_deepeval_evaluation(mock_test_cases, mock_metrics, n_runs=3)
            
            assert mock_deepeval.call_count == 3
            
            assert len(results) == 3
            assert all(r == mock_test_results for r in results)

    def test_with_additional_kwargs(self, mock_test_cases, mock_metrics):
        with patch('govuk_chat_evaluation.rag_answers.deepeval_evaluate.deepeval_evaluate') as mock_deepeval:
            mock_evaluation = MagicMock()
            mock_evaluation.test_results = []
            mock_deepeval.return_value = mock_evaluation
            
            run_deepeval_evaluation(
                mock_test_cases, 
                mock_metrics, 
                n_runs=1, 
                print=False, 
                max_concurrent=10,
            )
            
            mock_deepeval.assert_called_once_with(
                test_cases=cast(List[LLMTestCase], mock_test_cases),
                metrics=mock_metrics,
                print=False, 
                max_concurrent=10,
            )

class TestConvertDeepEvalOutput:
    @pytest.fixture
    def sample_test_results(self):
        # Create a sample TestResult with metrics data
        metric_data1 = MagicMock()
        metric_data1.name = "accuracy"
        metric_data1.score = 0.9
        metric_data1.reason = "Good accuracy"
        metric_data1.evaluation_cost = 0.1
        metric_data1.success = True

        metric_data2 = MagicMock()
        metric_data2.name = "coherence"
        metric_data2.score = 0.8
        metric_data2.reason = "Good coherence"
        metric_data2.evaluation_cost = 0.2
        metric_data2.success = True

        test_result1 = MagicMock(spec=TestResult)
        test_result1.name = "test1"
        test_result1.input = "input1"
        test_result1.actual_output = "actual1"
        test_result1.expected_output = "expected1"
        test_result1.retrieval_context = ["context1"]
        test_result1.metrics_data = [metric_data1]

        test_result2 = MagicMock(spec=TestResult)
        test_result2.name = "test2" 
        test_result2.input = "input2"
        test_result2.actual_output = "actual2"
        test_result2.expected_output = "expected2"
        test_result2.retrieval_context = ["context2"]
        test_result2.metrics_data = [metric_data2]

        return [
            [test_result1, test_result2],  # First run
            [test_result1, test_result2],  # Second run
        ]

    def test_convert_empty_results(self):
        results = convert_deepeval_output_to_evaluation_results([])
        assert results == []

    def test_convert_single_run(self, sample_test_results):
        results = convert_deepeval_output_to_evaluation_results([sample_test_results[0]])
        
        assert isinstance(results, list)
        assert len(results) == 2  # Two test cases
        
        assert results[0].name == "test1"
        assert results[0].input == "input1"
        assert results[0].actual_output == "actual1"
        assert results[0].expected_output == "expected1"
        assert results[0].retrieval_context == ["context1"]
        
        assert len(results[0].run_metric_outputs) == 1
        assert results[0].run_metric_outputs[0].run == 0
        assert results[0].run_metric_outputs[0].metric == "accuracy"
        assert results[0].run_metric_outputs[0].score == 0.9
        assert results[0].run_metric_outputs[0].reason == "Good accuracy"
        assert results[0].run_metric_outputs[0].cost == 0.1
        assert results[0].run_metric_outputs[0].success is True

    def test_convert_multiple_runs(self, sample_test_results):
        results = convert_deepeval_output_to_evaluation_results(sample_test_results)
        
        assert isinstance(results, list)
        assert len(results) == 2  # Two test cases
        
        for result in results:
            assert len(result.run_metric_outputs) == 2
            
            # verify we have run 0 and run 1 metrics
            run_indices = {output.run for output in result.run_metric_outputs}
            assert run_indices == {0, 1}

    def test_with_none_retrieval_context(self, sample_test_results):
        # modify test data to have None for retrieval_context
        sample_test_results[0][0].retrieval_context = None
        
        results = convert_deepeval_output_to_evaluation_results([sample_test_results[0]])
        
        assert results[0].retrieval_context == [] if results is not None else None