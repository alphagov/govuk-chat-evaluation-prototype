import pytest
from pydantic import ValidationError
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase, 
    MetricConfig, 
    EvaluationConfig, 
    StructuredContext, 
    RunMetricOutput, 
    EvaluationResult
)


class TestEvaluationTestCase:
    def test_to_llm_test_case(self):
        structured_context = StructuredContext(
            title="VAT",
            heading_hierarchy=["Tax", "VAT"],
            description="VAT overview",
            html_content="<p>Some HTML about VAT</p>",
            exact_path="https://gov.uk/vat",
            base_path="https://gov.uk"
        )
        
        evaluation_test_case = EvaluationTestCase(
            question="How are you?", 
            ideal_answer="Great", 
            llm_answer="Fine", 
            retrieved_context=[structured_context]
        )

        llm_test_case = evaluation_test_case.to_llm_test_case()

        assert isinstance(llm_test_case, LLMTestCase)
        assert llm_test_case.input == evaluation_test_case.question
        assert llm_test_case.expected_output == evaluation_test_case.ideal_answer
        assert llm_test_case.actual_output == evaluation_test_case.llm_answer
        assert llm_test_case.name is not None
        assert isinstance(llm_test_case.name, str)

        assert isinstance(llm_test_case.retrieval_context, list)
        assert all(isinstance(chunk, str) for chunk in llm_test_case.retrieval_context)
        assert "VAT" in llm_test_case.retrieval_context[0]  
        assert "Some HTML about VAT" in llm_test_case.retrieval_context[0]

class TestStructuredContext:
    def test_to_flattened_string(self):
        structured_context = StructuredContext(
            title="VAT",
            heading_hierarchy=["Tax", "VAT"],
            description="VAT overview",
            html_content="<p>Some HTML about VAT</p>",
            exact_path="https://gov.uk/vat",
            base_path="https://gov.uk"
        )

        flattened_string = structured_context.to_flattened_string()

        assert isinstance(flattened_string, str)
        assert "VAT" in flattened_string
        assert "Tax > VAT" in flattened_string
        assert "VAT overview" in flattened_string
        assert "<p>Some HTML about VAT</p>" in flattened_string

@pytest.mark.parametrize(
    "config_dict, expected_class",
    [
        ({"name": "faithfulness", "threshold": 0.8, "model": "gpt-4o", "temperature": 0.0}, FaithfulnessMetric),
        ({"name": "bias", "threshold": 0.5, "model": "gpt-4o", "temperature": 0.0}, BiasMetric),
    ]
)
def test_get_metric_instance_valid(config_dict, expected_class):
    metric_config = MetricConfig(**config_dict)
    metric = metric_config.to_metric_instance()
    assert isinstance(metric, expected_class)


def test_get_metric_instance_invalid_enum():
    config_dict = {"name": "does_not_exist", "threshold": 0.5, "model": "gpt-4o", "temperature": 0.0}

    with pytest.raises(ValidationError) as exception_info:
        MetricConfig(**config_dict)

    assert "validation error for MetricConfig" in str(exception_info.value)
    assert "does_not_exist" in str(exception_info.value)


class TestEvaluationConfig():
    def test_get_metric_instances(self):
        config_dict = {
            "metrics": [
                {"name": "faithfulness", "threshold": 0.8, "model": "gpt-4o", "temperature": 0.0},
                {"name": "bias", "threshold": 0.5, "model": "gpt-4o", "temperature": 0.5}
            ],
            "n_runs": 3
        }

        evaluation_config = EvaluationConfig(**config_dict)
        metrics = evaluation_config.metric_instances()

        assert len(metrics) == 2
        assert isinstance(metrics[0], FaithfulnessMetric)
        assert isinstance(metrics[1], BiasMetric)
        assert metrics[0].threshold == 0.8
        assert metrics[1].threshold == 0.5
        assert metrics[0].model.get_model_name() == "gpt-4o"
        assert metrics[1].model.get_model_name() == "gpt-4o"
        assert isinstance(evaluation_config.n_runs, int)
        assert evaluation_config.n_runs == 3

def test_run_metric_output_defaults():
    rmo = RunMetricOutput(run=1, metric="faithfulness", score=0.87)
    
    assert rmo.run == 1
    assert rmo.metric == "faithfulness"
    assert rmo.score == 0.87
    assert rmo.cost is None
    assert rmo.reason is None
    assert rmo.success is None

def test_evaluation_result_basic_init():
    rmo = RunMetricOutput(run=1, metric="bias", score=0.9)
    eval_result = EvaluationResult(
        name="test_case_1",
        input="What is the capital of France?",
        actual_output="It is Paris",
        expected_output="Paris",
        retrieval_context=["some relevant text"],
        run_metric_outputs=[rmo],
    )

    assert eval_result.name == "test_case_1"
    assert eval_result.input == "What is the capital of France?"
    assert eval_result.retrieval_context == ["some relevant text"]
    assert isinstance(eval_result.run_metric_outputs[0], RunMetricOutput)


# usage test
def evaluate_successful_runs(evaluation: EvaluationResult) -> list[int]:
    """Return the list of run IDs that were successful."""
    return [
        run_output.run
        for run_output in evaluation.run_metric_outputs
        if run_output.success is True
    ]


def test_evaluate_successful_runs():
    rmo1 = RunMetricOutput(run=1, metric="accuracy", score=0.95, success=True)
    rmo2 = RunMetricOutput(run=2, metric="accuracy", score=0.60, success=False)
    rmo3 = RunMetricOutput(run=3, metric="accuracy", score=0.88, success=True)

    evaluation = EvaluationResult(
        name="MultiRunTest",
        input="Test input",
        actual_output="output",
        expected_output="output",
        retrieval_context=[],
        run_metric_outputs=[rmo1, rmo2, rmo3],
    )

    successful_runs = evaluate_successful_runs(evaluation)
    
    assert successful_runs == [1, 3]