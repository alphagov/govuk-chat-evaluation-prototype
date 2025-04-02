from pathlib import Path

from pydantic import BaseModel


class EvaluationTestCase(BaseModel):
    question: str
    ideal_answer: str
    ideal_context: list[str]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
    ...
    # models = jsonl_to_models(evaluation_data_path, EvaluationTestCase)
