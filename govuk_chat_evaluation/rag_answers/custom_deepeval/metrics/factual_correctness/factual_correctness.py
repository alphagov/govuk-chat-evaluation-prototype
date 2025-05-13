from typing import Optional, List, Type

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.utils import get_or_create_event_loop

from deepeval.metrics.utils import (
    check_llm_test_case_params,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.telemetry import capture_metric_type

from .template import (
    FactualCorrectnessTemplate,
)
from .schema import ClassifiedFacts, FactClassificationResult
import logging


class FactualCorrectnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        evaluation_template: Type[
            FactualCorrectnessTemplate
        ] = FactualCorrectnessTemplate,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.threshold = 0 if strict_mode else threshold
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.evaluation_template = evaluation_template
        self.evaluation_cost = 0 if self.using_native_model else None
        self.confusion_matrix: Optional[ClassifiedFacts] = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronously evaluate the factual correctness of a test case."""
        check_llm_test_case_params(test_case, self._required_params, self)

        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                return loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                logging.info(
                    "Async mode is disabled. Running the synchronous function."
                )
                self.confusion_matrix = self._classify_statements(
                    test_case.actual_output, test_case.expected_output or ""
                )
                logging.info(f"Confusion matrix: {self.confusion_matrix}")
                return self._finalise_evaluation()

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        """Asynchronously evaluate the factual correctness of a test case."""
        check_llm_test_case_params(test_case, self._required_params, self)
        logging.info(
            "Async mode is enabled. Using event loop to run the async function."
        )

        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            self.confusion_matrix = await self._a_classify_statements(
                test_case.actual_output, test_case.expected_output or ""
            )
            logging.info(f"Confusion matrix: {self.confusion_matrix}")
            return self._finalise_evaluation()

    def _finalise_evaluation(self) -> float:
        """Finalise the evaluation by computing score, reason, and success status."""
        if self.confusion_matrix is not None:
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__, async_mode=self.async_mode)
            return self.score
        else:
            self.error = "Error: confusion_matrix was None"
            return float("nan")

    def _generate_reason(self) -> Optional[str]:
        if not self.include_reason or self.confusion_matrix is None:
            return None
        return f'{{"true_positive_statements": {self.confusion_matrix.TP}, "false_positive_statements": {self.confusion_matrix.FP}}}'

    async def _a_classify_statements(
        self, actual_output: str, expected_output: str
    ) -> ClassifiedFacts:
        prompt = self.evaluation_template.classify_facts(
            answer=actual_output, ground_truth=expected_output
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=FactClassificationResult
            )
            if isinstance(cost, (int, float)):
                self.evaluation_cost = (self.evaluation_cost or 0.0) + cost
            return res.classified_facts  # type: ignore[arg-type]
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=FactClassificationResult
                )
                return res.classified_facts  # type: ignore[arg-type]
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                data_model = FactClassificationResult(**data)
                return data_model.classified_facts
            except Exception as inner_e:
                logging.error("Failed to parse fallback JSON.", exc_info=inner_e)
                return ClassifiedFacts(TP=[], FP=[], FN=[])

    def _classify_statements(
        self, actual_output: str, expected_output: str
    ) -> ClassifiedFacts:
        prompt = self.evaluation_template.classify_facts(
            answer=actual_output, ground_truth=expected_output
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=FactClassificationResult)
            if isinstance(cost, (int, float)):
                self.evaluation_cost = (self.evaluation_cost or 0.0) + cost
            return res.classified_facts  # type: ignore[arg-type]
        else:
            try:
                res = self.model.generate(prompt, schema=FactClassificationResult)
                return res.classified_facts  # type: ignore[arg-type]
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                data_model = FactClassificationResult(**data)
                return data_model.classified_facts
            except Exception as inner_e:
                logging.error("Failed to parse fallback JSON.", exc_info=inner_e)
                return ClassifiedFacts(TP=[], FP=[], FN=[])

    def _calculate_score(self) -> float:
        """
        Calculates the factual-correctness score based on the confusion matrix as a
        float between 0 and 1. The score is calculated as the ratio of the number of
        True Positive statements to the total number of True Positive + False Positive
        statements.

        Returns:
            float: The factual-correctness score.
        """
        if self.confusion_matrix is None:
            return 0.0

        tp = len(self.confusion_matrix.TP)
        fp = len(self.confusion_matrix.FP)
        score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return 1.0 if self.strict_mode and score > self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return bool(self.success)

    @property
    def __name__(self):  # type: ignore[arg-type]
        return "FactualCorrectness"
