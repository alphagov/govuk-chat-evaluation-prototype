from pydantic import BaseModel


class GenerateInput(BaseModel):
    question: str
    ideal_answer: str

    def to_evaluation_test_case(self): ...


class EvaluationTestCase(GenerateInput):
    llm_answer: str
