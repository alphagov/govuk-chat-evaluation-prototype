from pathlib import Path

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from ..file_system import jsonl_to_models
from .data_models import EvaluationTestCase


# would expect we need to pass config object through if that has metrics configuration
def evaluate_and_output_results(_output_dir: Path, evaluation_data_path: Path):
    models = jsonl_to_models(evaluation_data_path, EvaluationTestCase)
    run_deepeval_evaluation(models)


def run_deepeval_evaluation(models: list[EvaluationTestCase]):
    llm_test_cases = [model.to_llm_test_case() for model in models]

    # contrived example metric
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )

    deepeval_evaluate(llm_test_cases, [correctness_metric], print_results=False)  # type: ignore
