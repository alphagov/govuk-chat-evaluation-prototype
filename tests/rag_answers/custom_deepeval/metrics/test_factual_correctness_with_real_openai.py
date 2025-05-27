import pytest
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from deepeval import assert_test
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness import (
    FactualCorrectnessMetric,
)


@pytest.mark.real_openai
class TestFactualCorrectnessRealOpenAI:
    """
    Test the FactualCorrectnessMetric with real OpenAI API calls.
    This test requires the OPENAI_API_KEY environment variable to be set.

    It can be run with the command:
    uv run pytest -m 'real_openai'
    """

    @pytest.mark.parametrize(
        "llm_test_case, expected_score",
        [
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Pigs oink and dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                1.0,
            ),
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Pigs oink, dogs bark and cats meow.",
                    input="What noise do pigs and dogs do?",
                ),
                2 / 3,
            ),
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs bark and cats meow.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs don't bark.",
                    input="What noise do pigs and dogs do?",
                ),
                0.0,
            ),
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs don't bark and pigs oink.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs don't bark and are cute.",
                    input="What noise do pigs and dogs do?",
                ),
                0.0,
            ),
            (
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs bark and are cute.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_factual_correctness_score(
        self, llm_test_case: LLMTestCase, expected_score: float
    ):
        metric = FactualCorrectnessMetric(
            model=GPTModel(model="gpt-4o", temperature=0),
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)  # type: ignore
        assert round(computed_score, 4) == round(expected_score, 4)

    def test_factual_correctness_deepeval(self):
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
