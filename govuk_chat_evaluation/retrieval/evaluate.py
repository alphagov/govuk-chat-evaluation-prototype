from pydantic import BaseModel


class EvaluationResult(BaseModel):
    question: str
    expected_exact_paths: list[str]
    actual_exact_paths: list[str]
