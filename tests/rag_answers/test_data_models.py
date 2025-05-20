import pytest

from pydantic import ValidationError
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.bias.bias import BiasMetric

from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase,
    MetricConfig,
    Config,
    StructuredContext,
)


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

    def test_get_metric_instances(self, mock_input_data):
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
        assert isinstance(metrics[0], FaithfulnessMetric)
        assert isinstance(metrics[1], BiasMetric)


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


class TestMetricConfig:
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
                {
                    "name": "bias",
                    "threshold": 0.5,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                BiasMetric,
            ),
        ],
    )
    def test_to_metric_instance(self, config_dict, expected_class):
        metric_config = MetricConfig(**config_dict)
        assert isinstance(metric_config.to_metric_instance(), expected_class)

    def test_get_metric_instance_invalid_enum(self):
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
