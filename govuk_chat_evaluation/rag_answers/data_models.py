from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, model_validator
from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Any
import uuid

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    BiasMetric,
)
from deepeval.models.llms.openai_model import GPTModel

from .custom_deepeval.metrics.factual_correctness import (
    FactualCorrectnessMetric,
)
from ..config import BaseConfig


# ----- Input data models -----


class StructuredContext(BaseModel):
    title: str
    heading_hierarchy: list[str]
    description: str
    html_content: str
    exact_path: str
    base_path: str

    def to_flattened_string(self) -> str:
        """Return the flattened string representation of the structure context."""
        return (
            f"{self.title}\n"
            f"{' > '.join(self.heading_hierarchy)}\n"
            f"{self.description}\n\n"
            f"{self.html_content}"
        )


class GenerateInput(BaseModel):
    question: str
    ideal_answer: str
    # TODO: lots more data fields


class EvaluationTestCase(GenerateInput):
    llm_answer: str
    retrieved_context: list[StructuredContext]

    def to_llm_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.question,
            name=str(uuid.uuid4()),
            expected_output=self.ideal_answer,
            actual_output=self.llm_answer,
            retrieval_context=[
                ctx.to_flattened_string() for ctx in self.retrieved_context
            ],
        )


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    BIAS = "bias"
    FACTUAL_CORRECTNESS = "factual_correctness"
    # others to add


class LLMJudgeModel(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    AMAZON_NOVA_MICRO_1 = "eu.amazon.nova-micro-v1:0"
    AMAZON_NOVA_PRO_1 = "eu.amazon.nova-pro-v1:0"
    GEMINI_15_PRO = "gemini-1.5-pro-002"
    GEMINI_15_FLASH = "gemini-1.5-flash-002"


class LLMJudgeModelConfig(BaseModel):
    model: LLMJudgeModel
    temperature: float = 0.0

    def instantiate_llm_judge(self):
        """Return the LLM judge model instance."""
        match self.model:
            case LLMJudgeModel.AMAZON_NOVA_MICRO_1:
                raise NotImplementedError(
                    f"Judge model {self.model} instantiation not implemented."
                )
                # Placeholder for actual class instance - e.g., CustomAmazonNovaJudge(model_name=self.model.value, temperature=self.temperature)
            case LLMJudgeModel.AMAZON_NOVA_PRO_1:
                raise NotImplementedError(
                    f"Judge model {self.model} instantiation not implemented."
                )
            case LLMJudgeModel.GEMINI_15_PRO:
                raise NotImplementedError(
                    f"Judge model {self.model} instantiation not implemented."
                )
            case LLMJudgeModel.GEMINI_15_FLASH:
                raise NotImplementedError(
                    f"Judge model {self.model} instantiation not implemented."
                )
            case LLMJudgeModel.GPT_4O_MINI | LLMJudgeModel.GPT_4O:
                return GPTModel(model=self.model.value, temperature=self.temperature)


class MetricConfig(BaseModel):
    name: MetricName
    threshold: float
    llm_judge: LLMJudgeModelConfig

    @model_validator(mode="before")
    @classmethod
    def inject_llm_judge(cls, values: dict[str, Any]) -> dict[str, Any]:
        # extract model and temperature to build llm_judge
        if "llm_judge" not in values:
            values["llm_judge"] = {
                "model": values.pop("model"),
                "temperature": values.pop("temperature", 0.0),
            }
        return values

    def to_metric_instance(self):
        model = self.llm_judge.instantiate_llm_judge()
        match self.name:
            case MetricName.FAITHFULNESS:
                return FaithfulnessMetric(threshold=self.threshold, model=model)
            case MetricName.RELEVANCE:
                return AnswerRelevancyMetric(threshold=self.threshold, model=model)
            case MetricName.BIAS:
                return BiasMetric(threshold=self.threshold, model=model)
            case MetricName.FACTUAL_CORRECTNESS:
                return FactualCorrectnessMetric(threshold=self.threshold, model=model)


# ----- Configuration models -----


class Config(BaseConfig):
    what: BaseConfig.GenericFields.what
    generate: BaseConfig.GenericFields.generate
    provider: BaseConfig.GenericFields.provider_openai_or_claude
    input_path: BaseConfig.GenericFields.input_path
    metrics: list[MetricConfig]
    n_runs: int

    @model_validator(mode="after")
    def run_validatons(self):
        return self._validate_fields_required_for_generate("provider")

    def metric_instances(self):
        """Return the list of runtime metric objects for evaluation."""
        return [metric.to_metric_instance() for metric in self.metrics]  # type: ignore


# ----- Output data models -----


@dataclass
class RunMetricOutput:
    run: int
    metric: str
    score: float
    cost: float | None = None
    reason: str | None = None
    success: bool | None = None


@dataclass
class EvaluationResult:
    name: str
    input: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str]
    run_metric_outputs: list[RunMetricOutput]
