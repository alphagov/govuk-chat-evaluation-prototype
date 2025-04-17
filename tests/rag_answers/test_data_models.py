import pytest
import json
from pydantic import ValidationError
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase,
    MetricConfig,
    Config,
    StructuredContext,
    RunMetricOutput,
    EvaluationResult,
)


@pytest.fixture(autouse=True)
def mock_input_data(tmp_path):
    """Write a valid JSONL input file to use in tests"""
    data = {
        "question": "What is VAT?",
        "llm_answer": "VAT is a tax.",
        "ideal_answer": "VAT is value-added tax.",
        "retrieved_context": [
            {
                "title": "VAT",
                "heading_hierarchy": ["Tax", "VAT"],
                "description": "VAT overview",
                "html_content": "<p>Some HTML about VAT</p>",
                "exact_path": "https://gov.uk/vat",
                "base_path": "https://gov.uk",
            }
        ],
    }

    file_path = tmp_path / "mock_input.jsonl"
    with open(file_path, "w") as f:
        f.write(json.dumps(data) + "\n")

    return file_path


class TestConfig:
    def test_config_requires_provider_for_generate(self, mock_input_data):
        with pytest.raises(ValueError, match="provider is required to generate data"):
            Config(
                what="Test",
                generate=True,
                provider=None,
                input_path=mock_input_data,
                metrics=[],
                n_runs=1,
            )

        # These should not raise
        Config(
            what="Test",
            generate=False,
            provider=None,
            input_path=mock_input_data,
            metrics=[],
            n_runs=1,
        )

        Config(
            what="Test",
            generate=True,
            provider="openai",
            input_path=mock_input_data,
            metrics=[],
            n_runs=1,
        )

    def test_get_metric_instances(self, mock_input_data, mocker):
        mock_faithfulness = mocker.MagicMock(spec=FaithfulnessMetric)
        mock_bias = mocker.MagicMock(spec=BiasMetric)

        # Mock the to_metric_instance method to return the correct mock types
        mock_to_metric_instance = mocker.patch(
            "govuk_chat_evaluation.rag_answers.data_models.MetricConfig.to_metric_instance",
            side_effect=[mock_faithfulness, mock_bias],
        )

        config_dict = {
            "what": "Test",
            "generate": False,
            "provider": None,
            "input_path": mock_input_data,
            "metrics": [
                {
                    "name": "faithfulness",
                    "threshold": 0.8,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                {
                    "name": "bias",
                    "threshold": 0.5,
                    "model": "gpt-4o",
                    "temperature": 0.5,
                },
            ],
            "n_runs": 3,
        }

        evaluation_config = Config(**config_dict)
        metrics = evaluation_config.metric_instances()
        assert len(metrics) == 2
        assert metrics == [mock_faithfulness, mock_bias]
        assert mock_to_metric_instance.call_count == 2


class TestEvaluationTestCase:
    def test_to_llm_test_case(self):
        structured_context = StructuredContext(
            title="VAT",
            heading_hierarchy=["Tax", "VAT"],
            description="VAT overview",
            html_content="<p>Some HTML about VAT</p>",
            exact_path="https://gov.uk/vat",
            base_path="https://gov.uk",
        )

        evaluation_test_case = EvaluationTestCase(
            question="How are you?",
            ideal_answer="Great",
            llm_answer="Fine",
            retrieved_context=[structured_context],
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
            base_path="https://gov.uk",
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
        (
            {
                "name": "faithfulness",
                "threshold": 0.8,
                "model": "gpt-4o",
                "temperature": 0.0,
            },
            FaithfulnessMetric,
        ),
        (
            {"name": "bias", "threshold": 0.5, "model": "gpt-4o", "temperature": 0.0},
            BiasMetric,
        ),
    ],
)
def test_get_metric_instance_valid(config_dict, expected_class, mocker):
    # Mock the llm_judge instantiation
    mock_llm_judge = mocker.MagicMock()
    mocker.patch(
        "govuk_chat_evaluation.rag_answers.data_models.LLMJudgeModelConfig.instantiate_llm_judge",
        return_value=mock_llm_judge,
    )

    # Create a mock metric of the expected class
    mock_metric = mocker.MagicMock(spec=expected_class)
    mocker.patch.object(expected_class, "__new__", return_value=mock_metric)

    metric_config = MetricConfig(**config_dict)
    metric = metric_config.to_metric_instance()

    assert metric is not None


def test_get_metric_instance_invalid_enum():
    config_dict = {
        "name": "does_not_exist",
        "threshold": 0.5,
        "model": "gpt-4o",
        "temperature": 0.0,
    }

    with pytest.raises(ValidationError) as exception_info:
        MetricConfig(**config_dict)

    assert "validation error for MetricConfig" in str(exception_info.value)
    assert "does_not_exist" in str(exception_info.value)


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
