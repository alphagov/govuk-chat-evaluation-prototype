from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class GenerateInput(BaseModel):
    question: str
    ideal_answer: str
    # TODO: lots more data fields


class EvaluationTestCase(GenerateInput):
    llm_answer: str
    # TODO: lots more data fields

    def to_llm_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.question,
            expected_output=self.ideal_answer,
            actual_output=self.llm_answer,
        )
