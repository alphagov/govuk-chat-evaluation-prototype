from pathlib import Path

from ..file_system import jsonl_to_models
from .data_models import EvaluationTestCase


def evaluate_and_output_results(_output_dir: Path, evaluation_data_path: Path):
    jsonl_to_models(evaluation_data_path, EvaluationTestCase)
