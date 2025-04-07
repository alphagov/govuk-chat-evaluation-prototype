import pytest
from pydantic import ValidationError
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
)
from govuk_chat_evaluation.rag_answers.data_models import EvaluationTestCase, MetricConfig, EvaluationConfig


class TestEvaluationTestCase:
    def test_to_llm_test_case(self):
        evaluation_test_case = EvaluationTestCase(
            question="How are you?", ideal_answer="Great", llm_answer="Fine", retrieved_context=["Context 1", "Context 2"]
        )

        llm_test_case = evaluation_test_case.to_llm_test_case()

        assert isinstance(llm_test_case, LLMTestCase)
        assert llm_test_case.input == evaluation_test_case.question
        assert llm_test_case.expected_output == evaluation_test_case.ideal_answer
        assert llm_test_case.actual_output == evaluation_test_case.llm_answer
        assert llm_test_case.retrieval_context == evaluation_test_case.retrieved_context
        assert llm_test_case.name is not None
        assert isinstance(llm_test_case.name, str)

    def test_flattening_structured_context(self):
        structured = [
            {
                "title": "VAT",
                "heading_hierarchy": ["Tax", "VAT"],
                "description": "VAT overview",
                "html_content": "<p>Some HTML about VAT</p>",
                "exact_path": "https://gov.uk/vat",
                "base_path": "https://gov.uk"
            }
        ]

        case = EvaluationTestCase(
            question="What is VAT?",
            ideal_answer="VAT is a tax.",
            llm_answer="VAT stands for Value Added Tax.",
            retrieved_context=structured # type: ignore
        )

        # Ensure flattening occurred
        assert isinstance(case.retrieved_context, list)
        assert all(isinstance(chunk, str) for chunk in case.retrieved_context)
        assert "VAT" in case.retrieved_context[0]
        assert "Some HTML about VAT" in case.retrieved_context[0]


@pytest.mark.parametrize(
    "config_dict, expected_class",
    [
        ({"name": "faithfulness", "threshold": 0.8,}, FaithfulnessMetric),
        ({"name": "bias", "threshold": 0.5}, BiasMetric),
    ]
)
def test_get_metric_instance_valid(config_dict, expected_class):
    metric_config = MetricConfig(**config_dict)
    metric = metric_config.to_metric_instance()
    assert isinstance(metric, expected_class)


def test_get_metric_instance_invalid_enum():
    config_dict = {"name": "does_not_exist", "threshold": 0.5}

    with pytest.raises(ValidationError) as exception_info:
        MetricConfig(**config_dict)

    assert "validation error for MetricConfig" in str(exception_info.value)
    assert "does_not_exist" in str(exception_info.value)


class TestEvaluationConfig():
    def test_get_metric_instances(self):
        config_dict = {
            "metrics": [
                {"name": "faithfulness", "threshold": 0.8},
                {"name": "bias", "threshold": 0.5}
            ],
            "llm_judge": {
                "model": "gpt-4o",
                "temperature": 0.0
            },
            "n_runs": 3
        }

        evaluation_config = EvaluationConfig(**config_dict)
        metrics = evaluation_config.get_metric_instances()

        assert len(metrics) == 2
        assert isinstance(metrics[0], FaithfulnessMetric)
        assert isinstance(metrics[1], BiasMetric)
        assert metrics[0].threshold == 0.8
        assert metrics[1].threshold == 0.5
        assert evaluation_config.llm_judge_instance is not None
        assert isinstance(evaluation_config.llm_judge_instance, str)
        assert evaluation_config.llm_judge_instance == "gpt-4o"
        assert isinstance(evaluation_config.n_runs, int)
        assert evaluation_config.n_runs == 3
