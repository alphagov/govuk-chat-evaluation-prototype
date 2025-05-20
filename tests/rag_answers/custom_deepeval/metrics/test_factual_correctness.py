import pytest
import json

from unittest.mock import Mock, AsyncMock, MagicMock, patch
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.errors import MissingTestCaseParamsError

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness import (
    FactualCorrectnessMetric,
)

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.schema import (
    ClassifiedFacts,
    FactClassificationResult,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.template import (
    FactualCorrectnessTemplate,
)


@pytest.fixture
def mock_native_model():
    """Fixture for a mock native model as GPT model"""
    mock = Mock(spec=GPTModel)
    mock.get_model_name.return_value = "gpt-4o"
    return mock


@pytest.fixture
def mock_non_native_model():
    """Fixture for a mock non-native model"""
    mock = Mock(spec=DeepEvalBaseLLM)
    mock.get_model_name.return_value = "a-non-native-model"
    return mock


@pytest.fixture
def test_case():
    """Fixture for a test case"""
    return LLMTestCase(
        input="Input", actual_output="Actual", expected_output="Expected"
    )


@pytest.fixture
def sample_classified_facts():
    """Fixture for sample classified facts"""
    return ClassifiedFacts(TP=["fact1"], FP=["fact2"], FN=["fact3"])


@pytest.fixture
def fact_classification_result():
    """Fixture for a FactClassificationResult"""
    return FactClassificationResult(
        classified_facts=ClassifiedFacts(TP=["fact1"], FP=["fact2"], FN=[])
    )


class TestFactualCorrectness:
    class TestAMeasure:
        @pytest.mark.asyncio
        async def test_invalid_params(self, mock_native_model: Mock):
            metric = FactualCorrectnessMetric(model=mock_native_model)

            invalid_test_case = LLMTestCase(
                input="Input", actual_output="Actual", expected_output=None
            )

            with pytest.raises(MissingTestCaseParamsError, match="cannot be None"):
                await metric.a_measure(invalid_test_case)

        @pytest.mark.asyncio
        @patch(
            "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.factual_correctness.metric_progress_indicator"
        )
        @pytest.mark.parametrize(
            "set_show_progress, expected_show_progress",
            [
                (
                    True,
                    True,
                ),
                (False, False),
            ],
        )
        async def test_show_progress(
            self,
            mock_progress_indicator,
            mock_native_model: Mock,
            test_case: LLMTestCase,
            fact_classification_result: FactClassificationResult,
            set_show_progress: bool,
            expected_show_progress: bool,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(fact_classification_result, 0.5)
            )

            metric = FactualCorrectnessMetric(model=mock_native_model)

            # since we patched metric_progress_indicator, it shuold call the mocked context manager mock_progress_indicator
            await metric.a_measure(test_case, _show_indicator=set_show_progress)

            # test it actually called the mock
            mock_progress_indicator.assert_called_once_with(
                metric,
                async_mode=metric.async_mode,
                _show_indicator=expected_show_progress,
            )

        @pytest.mark.asyncio
        @patch(
            "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.factual_correctness.logging.debug"
        )
        async def test_logs_confusion_matrix(
            self, mock_logging_debug: MagicMock, mock_native_model: Mock, test_case
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(
                    FactClassificationResult(
                        classified_facts=ClassifiedFacts(
                            TP=["fact1"], FP=["fact2"], FN=["fact3"]
                        )
                    ),
                    0.1,
                )
            )

            metric = FactualCorrectnessMetric(model=mock_native_model)

            await metric.a_measure(test_case)

            # verify logging.debug was called once
            mock_logging_debug.assert_called_once()

            # verify ligged message contains the expected objects
            logged_message = mock_logging_debug.call_args[0][0]

            assert "Confusion matrix for test input:" in logged_message
            assert "fact1" in logged_message
            assert "fact2" in logged_message
            assert "fact3" in logged_message

            assert str(test_case.input) in logged_message

        @pytest.mark.asyncio
        @pytest.mark.parametrize(
            "input_TP,input_FP,expected_score",
            [
                (
                    ["fact1", "fact2"],
                    ["fact3"],
                    2 / 3,
                ),
                ([], [], 0.0),
            ],
        )
        async def test_returns_score(
            self,
            mock_native_model: Mock,
            test_case: LLMTestCase,
            input_TP: list,
            input_FP: list,
            expected_score: float,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(
                    FactClassificationResult(
                        classified_facts=ClassifiedFacts(
                            TP=input_TP, FP=input_FP, FN=["fact3"]
                        )
                    ),
                    expected_score,
                )
            )

            metric = FactualCorrectnessMetric(
                model=mock_native_model, threshold=0.7, include_reason=True
            )

            # let _calculate_score run naturally; it uses mocked a_generate
            result = await metric.a_measure(test_case)

            assert round(result, 3) == round(expected_score, 3)

            # assert dependency boundary was called once as expected (with schema)
            mock_native_model.a_generate.assert_awaited_once()
            _, kwargs = mock_native_model.a_generate.call_args
            assert kwargs.get("schema") == FactClassificationResult

        sample_facts = ClassifiedFacts(TP=["fact1", "fact2"], FP=["wrong1"], FN=[])

        @pytest.mark.asyncio
        @pytest.mark.parametrize(
            "threshold, expected_success, facts",
            [
                (0.6, True, sample_facts),
                (0.7, False, sample_facts),
                (0.666, True, sample_facts),
                (0.9, False, sample_facts),
            ],
        )
        async def test_threshold_can_be_configured(
            self,
            threshold: float,
            expected_success: bool,
            facts: ClassifiedFacts,
            mock_native_model: Mock,
            test_case: LLMTestCase,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(FactClassificationResult(classified_facts=facts), 0.1)
            )
            metric = FactualCorrectnessMetric(
                model=mock_native_model, threshold=threshold
            )

            _ = await metric.a_measure(test_case)

            assert metric.success == expected_success

        @pytest.mark.asyncio
        @pytest.mark.parametrize(
            "include_reason, expected_reason",
            [
                (
                    True,
                    {
                        "true_positive_statements": ["fact1"],
                        "false_positive_statements": ["fact2"],
                    },
                ),
                (False, None),
            ],
        )
        async def test_include_reason_can_be_configured(
            self,
            fact_classification_result: FactClassificationResult,
            include_reason: bool,
            expected_reason: dict | None,
            mock_native_model: Mock,
            test_case: LLMTestCase,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(fact_classification_result, 0.1)
            )
            metric = FactualCorrectnessMetric(
                model=mock_native_model, include_reason=include_reason
            )

            _ = await metric.a_measure(test_case)

            if expected_reason is not None:
                assert json.loads(metric.reason.replace("'", '"')) == expected_reason  # type: ignore
            else:
                assert metric.reason == expected_reason  # type: ignore

        @pytest.mark.asyncio
        @pytest.mark.parametrize(
            "strict_mode, threshold, TP, FP, expected_score, expected_success",
            [
                (
                    False,
                    0.7,
                    ["fact1", "fact2"],
                    ["fact3"],
                    0.6667,
                    False,
                ),  # regular mode, return raw score
                (False, 0.7, ["fact1", "fact2"], [], 1.0, True),
                (
                    True,
                    0.7,
                    ["fact1", "fact2", "fact3"],
                    ["fact4"],
                    0.0,
                    False,
                ),  # strict: score < strict-threshold of 1
                (
                    True,
                    0.7,
                    ["fact1", "fact2"],
                    [],
                    1.0,
                    True,
                ),  # strict: score == strict-threshold of 1
                (
                    True,
                    0.7,
                    ["fact1"],
                    ["fact2", "fact3"],
                    0.0,
                    False,
                ),  # strict: score <= threshold so should not trigger strict mode logic
            ],
        )
        async def test_strict_mode_can_be_configured(
            self,
            strict_mode: bool,
            threshold: float,
            TP: list,
            FP: list,
            expected_score: float,
            expected_success: bool,
            mock_native_model: Mock,
            test_case: LLMTestCase,
        ):
            classified_facts = ClassifiedFacts(TP=TP, FP=FP, FN=[])

            mock_native_model.a_generate = AsyncMock(
                return_value=(
                    FactClassificationResult(classified_facts=classified_facts),
                    0.1,
                )
            )

            metric = FactualCorrectnessMetric(
                model=mock_native_model,
                threshold=threshold,
                strict_mode=strict_mode,
            )

            # let _calculate_score and _finalise_evaluation run naturally
            result = await metric.a_measure(test_case)

            assert round(result, 4) == round(expected_score, 4)
            assert metric.success == expected_success

        @pytest.mark.asyncio
        async def test_native_model_sets_evaluation_costs(
            self,
            mock_native_model: Mock,
            test_case: LLMTestCase,
            fact_classification_result: FactClassificationResult,
        ):
            expected_cost = 0.05

            mock_native_model.a_generate = AsyncMock(
                return_value=(fact_classification_result, expected_cost)
            )

            metric = FactualCorrectnessMetric(model=mock_native_model)

            _ = await metric.a_measure(test_case)

            assert metric.evaluation_cost == expected_cost
            mock_native_model.a_generate.assert_awaited_once()

        @pytest.mark.asyncio
        @patch(
            "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.factual_correctness.initialize_model"
        )
        async def test_non_native_model_returns_valid_schema(
            self,
            mock_initialize_model,
            mock_non_native_model: Mock,
            test_case: LLMTestCase,
            fact_classification_result: FactClassificationResult,
        ):
            mock_non_native_model.a_generate = AsyncMock(
                return_value=fact_classification_result
            )

            # force initialize_model to return (mock_model, False)
            mock_initialize_model.return_value = (mock_non_native_model, False)

            metric = FactualCorrectnessMetric(model=mock_non_native_model)  # type: ignore
            _ = await metric.a_measure(test_case)

            assert metric.evaluation_cost is None  # should remain None
            assert isinstance(metric.confusion_matrix, ClassifiedFacts)
            assert (
                metric.confusion_matrix == fact_classification_result.classified_facts
            )

        @pytest.mark.asyncio
        async def test_non_native_model_returns_recoverable_json(
            self, mock_non_native_model: Mock, test_case: LLMTestCase
        ):
            # fallback case: schema call fails, raw JSON string returned
            fixable_json = (
                '{"classified_facts": {"TP": ["fact1"], "FP": ["fact2"], "FN": []}}'
            )

            # first call raises TypeError, second returns fixable string
            mock_non_native_model.a_generate = AsyncMock(
                side_effect=[
                    TypeError("schema parse failed"),
                    fixable_json,
                ]
            )

            metric = FactualCorrectnessMetric(model=mock_non_native_model)  # type: ignore
            _ = await metric.a_measure(test_case)

            # verify a_generate was called twice:
            # 1. with schema
            # 2. without schema (fallback)
            assert mock_non_native_model.a_generate.call_count == 2
            first_call_args = mock_non_native_model.a_generate.call_args_list[
                0
            ]  # should include 'schema'
            second_call_args = mock_non_native_model.a_generate.call_args_list[
                1
            ]  # should not include 'schema'

            assert "schema" in first_call_args.kwargs
            assert "schema" not in second_call_args.kwargs

        @pytest.mark.asyncio
        async def test_non_native_model_returns_unparsable_json(
            self, mock_non_native_model: Mock, test_case: LLMTestCase
        ):
            mock_non_native_model.a_generate = AsyncMock(
                side_effect=Exception("unexpected")
            )
            metric = FactualCorrectnessMetric(model=mock_non_native_model)  # type: ignore
            _ = await metric.a_measure(test_case=test_case)

            assert isinstance(metric.confusion_matrix, ClassifiedFacts)

            assert metric.confusion_matrix.TP == []
            assert metric.confusion_matrix.FP == []

    def test_measure_raises_not_implemented(
        self, mock_native_model: Mock, test_case: LLMTestCase
    ):
        metric = FactualCorrectnessMetric(model=mock_native_model)

        with pytest.raises(
            NotImplementedError, match="Synchronous evaluation is not supported"
        ):
            metric.measure(test_case)

    def test_is_successful(self, mock_native_model: Mock):
        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        metric.success = True
        assert metric.is_successful() is True

        metric.success = False
        assert metric.is_successful() is False

        metric.error = "Some error"
        assert metric.is_successful() is False
