import pytest
import math
import json
import os

from unittest.mock import Mock, AsyncMock, MagicMock, patch
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel, DeepEvalBaseLLM
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness import (
    FactualCorrectnessMetric,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.template import (
    FactualCorrectnessTemplate,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.schema import (
    ClassifiedFacts,
    FactClassificationResult,
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
def mock_template():
    """Fixture for a mock template"""
    mock = MagicMock(spec=FactualCorrectnessTemplate)
    mock.classify_facts.return_value = "mocked_prompt"
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


class TestFactualCorrectnessInitialization:
    """Tests for FactualCorrectnessMetric initialization"""

    def test_initialization_with_native_model(self, mock_native_model):
        metric = FactualCorrectnessMetric(model=mock_native_model)

        assert metric.using_native_model is True
        assert metric.evaluation_model == "gpt-4o"
        assert metric.evaluation_cost == 0

    def test_initialization_with_non_native_model(self, mock_non_native_model):
        metric = FactualCorrectnessMetric(model=mock_non_native_model)

        assert metric.using_native_model is False
        assert metric.evaluation_model == "a-non-native-model"
        assert metric.evaluation_cost != 0
        assert metric.evaluation_cost is None

    @pytest.mark.parametrize(
        "strict_mode, threshold, include_reason, expected_threshold, expected_include_reason",
        [
            (
                True,
                0.7,
                True,
                0,
                True,
            ),  # strict_mode=True should set threshold to 0
            (
                False,
                0.7,
                True,
                0.7,
                True,
            ),  # strict_mode=False should keep threshold as 0.7
            (
                True,
                0.5,
                False,
                0,
                False,
            ),  # strict_mode=True with different params
            (
                False,
                0.9,
                False,
                0.9,
                False,
            ),  # strict_mode=False with different params
        ],
    )
    def test_initialization_all_parameters(
        self,
        mock_native_model,
        strict_mode,
        threshold,
        include_reason,
        expected_threshold,
        expected_include_reason,
    ):
        metric = FactualCorrectnessMetric(
            model=mock_native_model,
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
        )

        assert metric.threshold == expected_threshold
        assert metric.include_reason == expected_include_reason
        assert metric.async_mode  # async_mode should be True by default


class TestFactualCorrectnessClassification:
    """Tests for statement classification functionality"""

    @pytest.mark.asyncio
    async def test_custom_template_is_used(
        self, mock_native_model, mock_template, fact_classification_result
    ):
        mock_native_model.a_generate = AsyncMock(
            return_value=(fact_classification_result, 0.01)
        )

        metric = FactualCorrectnessMetric(
            model=mock_native_model,
            evaluation_template=mock_template,  # type: ignore
        )

        result = await metric._a_classify_statements("actual", "expected")

        mock_template.classify_facts.assert_called_once_with(
            answer="actual", ground_truth="expected"
        )

        assert result == fact_classification_result.classified_facts

    @pytest.mark.asyncio
    async def test_a_classify_statements_using_native_model(
        self, mock_native_model, fact_classification_result, mock_template
    ):
        mock_native_model.a_generate = AsyncMock(
            return_value=(fact_classification_result, 0.5)
        )

        metric = FactualCorrectnessMetric(
            model=mock_native_model, evaluation_template=mock_template
        )  # type: ignore
        result = await metric._a_classify_statements("actual", "expected")

        assert isinstance(result, ClassifiedFacts)
        assert result.TP == ["fact1"]
        assert result.FP == ["fact2"]
        assert metric.evaluation_cost == 0.5
        mock_native_model.a_generate.assert_awaited_once_with(
            "mocked_prompt", schema=FactClassificationResult
        )
        mock_template.classify_facts.assert_called_once_with(
            answer="actual", ground_truth="expected"
        )

    @pytest.mark.asyncio
    async def test_a_classify_statements_non_native_success(
        self, mock_non_native_model, fact_classification_result, mock_template
    ):
        mock_non_native_model.a_generate = AsyncMock(
            return_value=fact_classification_result
        )

        metric = FactualCorrectnessMetric(
            model=mock_non_native_model,
            evaluation_template=mock_template,
        )  # type: ignore
        result = await metric._a_classify_statements("actual", "expected")

        assert result.TP == ["fact1"]
        assert result.FP == ["fact2"]
        assert metric.evaluation_cost is None  # should remain None
        assert isinstance(result, ClassifiedFacts)
        mock_non_native_model.a_generate.assert_awaited_once_with(
            "mocked_prompt", schema=FactClassificationResult
        )

    @pytest.mark.asyncio
    @patch(
        "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness.factual_correctness.trimAndLoadJson"
    )
    async def test_a_classify_statements_non_native_typeerror(
        self, mock_trim, mock_non_native_model, mock_template
    ):
        mock_trim.return_value = {"classified_facts": {"TP": [], "FP": [], "FN": []}}

        # set up the mock to raise TypeError on first call, then return a string on second call
        mock_non_native_model.a_generate = AsyncMock()
        mock_non_native_model.a_generate.side_effect = [
            TypeError("wrong formatted json"),
            '{"classified_facts": {"TP": [], "FP": [], "FN": []}}',
        ]

        metric = FactualCorrectnessMetric(
            model=mock_non_native_model, evaluation_template=mock_template
        )  # type: ignore
        result = await metric._a_classify_statements("actual", "expected")

        assert isinstance(result, ClassifiedFacts)
        assert result.TP == []
        mock_trim.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_classify_statements_non_native_general_exception(
        self, mock_non_native_model, mock_template
    ):
        # set up the mock to raise an exception
        mock_non_native_model.a_generate = AsyncMock(
            side_effect=Exception("unexpected")
        )

        metric = FactualCorrectnessMetric(
            model=mock_non_native_model, evaluation_template=mock_template
        )  # type: ignore
        result = await metric._a_classify_statements("actual", "expected")

        assert isinstance(result, ClassifiedFacts)
        assert result.TP == []
        assert result.FP == []


class TestFactualCorrectnessMeasureMethods:
    """Tests for the measure/a_measure functionality"""

    def test_measure_sync_non_implementation_error(
        self, mock_native_model, test_case, sample_classified_facts
    ):
        metric = FactualCorrectnessMetric(
            model=mock_native_model,
            threshold=0.7,
            include_reason=True,
        )

        with pytest.raises(
            NotImplementedError, match="Synchronous evaluation is not supported"
        ):
            metric.measure(test_case)

    @pytest.mark.asyncio
    async def test_a_measure_async(
        self, mock_native_model, test_case, sample_classified_facts
    ):
        mock_native_model.a_generate = AsyncMock(
            return_value=(Mock(classified_facts="mocked_facts"), 0.01)
        )

        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        metric._a_classify_statements = AsyncMock(return_value=sample_classified_facts)
        metric._calculate_score = Mock(return_value=0.8)

        result = await metric.a_measure(test_case)
        assert result == 0.8
        metric._a_classify_statements.assert_awaited_once_with("Actual", "Expected")


class TestFactualCorrectnessHelperMethods:
    """Tests for helper methods of the FactualCorrectnessMetric class"""

    def test_finalise_evaluation_valid(self, mock_native_model):
        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        # Create a mock confusion matrix
        metric.confusion_matrix = ClassifiedFacts(TP=["fact"], FP=["fact"], FN=[])
        metric._generate_reason = Mock(
            return_value='{"true_positive_statements": 1, "false_positive_statements": 1}'
        )
        metric._calculate_score = Mock(return_value=0.8)

        result = metric._finalise_evaluation()
        assert result == 0.8
        assert (
            metric.reason
            == '{"true_positive_statements": 1, "false_positive_statements": 1}'
        )
        assert metric.success is True

    def test_finalise_evaluation_invalid(self, mock_native_model):
        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        # Set confusion matrix to None
        metric.confusion_matrix = None

        result = metric._finalise_evaluation()
        assert math.isnan(result)
        assert metric.error == "Error: confusion_matrix was None"
        assert metric.reason is None
        assert metric.success is None

    def test_generate_reason(self, mock_native_model):
        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        # Create a mock confusion matrix
        metric.confusion_matrix = ClassifiedFacts(
            TP=["a fact"], FP=["a different fact"], FN=["fact"]
        )

        expected = {
            "true_positive_statements": ["a fact"],
            "false_positive_statements": ["a different fact"],
        }

        result = metric._generate_reason()

        if result is not None:
            result_dict = json.loads(result.replace("'", '"'))
            assert result_dict == expected
        else:
            # handle the case where result is None, if necessary
            assert False, "Result is None"

    def test_calculate_score(self, mock_native_model, sample_classified_facts):
        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        metric.confusion_matrix = sample_classified_facts
        result = metric._calculate_score()
        assert result == 0.5  # TP / (TP + FP) = 1 / (1 + 1) = 0.5

        metric.confusion_matrix = ClassifiedFacts(TP=[], FP=[], FN=[])
        result = metric._calculate_score()
        assert result == 0.0

    def test_is_successful(self, mock_native_model):
        metric = FactualCorrectnessMetric(
            model=mock_native_model, threshold=0.7, include_reason=True
        )

        metric.success = True
        assert metric.is_successful() is True

        metric.success = False
        assert metric.is_successful() is False

        metric.error = "Some error"
        assert metric.is_successful() is False


# -------- DeepEval metrics tests --------


run_openai_tests = os.getenv("RUN_OPENAI_TESTS") == "1"
# to run the tests, set the environment variable RUN_OPENAI_TESTS=1
# to run these tests:
# RUN_OPENAI_TESTS=1 uv run pytest

if run_openai_tests:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUN_OPENAI_TESTS is set, but OPENAI_API_KEY is not defined."
        )


test_1 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Pigs oink and dogs bark.",
        input="What noise do pigs and dogs do?",
    ),
    2 / 2,
)

test_2 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Pigs oink, dogs bark and cats meow.",
        input="What noise do pigs and dogs do?",
    ),
    2 / 3,
)

test_3 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Dogs bark and cats meow.",
        input="What noise do pigs and dogs do?",
    ),
    1 / 2,
)

test_4 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Dogs don't bark.",
        input="What noise do pigs and dogs do?",
    ),
    0 / 3,
)

test_5 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Dogs don't bark and pigs oink.",
        input="What noise do pigs and dogs do?",
    ),
    1 / 2,
)

test_6 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Dogs don't bark and are cute.",
        input="What noise do pigs and dogs do?",
    ),
    0 / 2,
)

test_7 = (
    LLMTestCase(
        expected_output="Pigs oink. Dogs bark.",
        actual_output="Dogs bark and are cute.",
        input="What noise do pigs and dogs do?",
    ),
    1 / 2,
)


@pytest.mark.skipif("not run_openai_tests", reason="openai is expensive")
@pytest.mark.parametrize(
    "llm_test_case, expected_score",
    [test_1, test_2, test_3, test_4, test_5, test_6, test_7],
)
@pytest.mark.asyncio
async def test_factual_correctness_score(llm_test_case, expected_score):
    correctness_metric = FactualCorrectnessMetric(
        model=GPTModel(model="gpt-4o", temperature=0),
        include_reason=False,
    )
    computed_score = await correctness_metric.a_measure(llm_test_case)
    assert computed_score == expected_score


@pytest.mark.skipif("not run_openai_tests", reason="openai is expensive")
def test_factual_correctness_deepeval():
    test_case = LLMTestCase(
        input="What noise do pigs and dogs do?",
        actual_output="Pigs oink and dogs bark.",
        expected_output="Pigs oink. Dogs bark.",
    )
    metric = FactualCorrectnessMetric(
        model=GPTModel(model="gpt-4o", temperature=0),
        include_reason=False,
    )
    assert_test(test_case, [metric])
