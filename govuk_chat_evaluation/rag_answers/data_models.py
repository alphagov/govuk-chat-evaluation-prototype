from deepeval.test_case import LLMTestCase
from pydantic import BaseModel


# I expect this model to be a subclass of the model used for generating answers
class EvaluationTestCase(BaseModel):
    question: str
    ideal_answer: str
    llm_answer: str
    # TODO: lots more data fields

    def to_llm_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.question,
            expected_output=self.ideal_answer,
            actual_output=self.llm_answer,
        )
