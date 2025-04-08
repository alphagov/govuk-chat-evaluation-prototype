from deepeval.test_case import LLMTestCase
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Optional
import uuid
from collections import defaultdict
from statistics import mean

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    BiasMetric,
)
from deepeval.models import DeepEvalBaseLLM


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
            name = str(uuid.uuid4()),
            expected_output=self.ideal_answer,
            actual_output=self.llm_answer,
            retrieval_context=[ctx.to_flattened_string() for ctx in self.retrieved_context]
        )


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    BIAS = "bias"
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
                return None  # Placeholder for actual class instance
            case LLMJudgeModel.AMAZON_NOVA_PRO_1:
                return None  # Placeholder
            case LLMJudgeModel.GEMINI_15_PRO:
                return None  # Placeholder
            case LLMJudgeModel.GEMINI_15_FLASH:
                return None  # Placeholder
            case LLMJudgeModel.GPT_4O_MINI | LLMJudgeModel.GPT_4O:
                return self.model.value  # Just returns the string name as they are in-built models for DeepEval llm judge
    

class MetricConfig(BaseModel):
    name: MetricName
    threshold: float
    # model: str | DeepEvalBaseLLM

    def to_metric_instance(self, llm_judge: DeepEvalBaseLLM | None = None):
        match self.name:
            case MetricName.FAITHFULNESS:
                return FaithfulnessMetric(threshold=self.threshold, model=llm_judge)
            case MetricName.RELEVANCE:
                return AnswerRelevancyMetric(threshold=self.threshold, model=llm_judge)
            case MetricName.BIAS:
                return BiasMetric(threshold=self.threshold, model=llm_judge)


class EvaluationConfig(BaseModel):
    metrics: list[MetricConfig]
    llm_judge: LLMJudgeModelConfig
    n_runs: int = 1
    llm_judge_instance: Optional[DeepEvalBaseLLM | str] = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types for llm_judge_instance

    # I am not fully happy with this solution, but it works for now
    # I do not want to instantiate a new llm judge model class for each metric
    def __init__(self, **data):
        super().__init__(**data)
        self.llm_judge_instance = self.llm_judge.instantiate_llm_judge()

    def get_metric_instances(self):
        """Return the list of runtime metric objects for evaluation."""
        return [metric.to_metric_instance(self.llm_judge_instance) for metric in self.metrics]  # type: ignore
    

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
    evaluation_results: list[RunMetricOutput]

    def average_scores_by_metric(self) -> dict[str, float]:
        scores_by_metric = defaultdict(list)
        
        for result in self.evaluation_results:
            scores_by_metric[result.metric].append(result.score)
        
        # Calculate the average for each metric
        return {
            metric: mean(scores)
            for metric, scores in scores_by_metric.items()
            if scores  # guard against empty lists
        }