from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


class EvaluateInput(BaseModel):
    question: str
    ideal_answer: str

    def to_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.question,
            expected_output=self.ideal_answer,
            actual_output="not a real answer",
        )
