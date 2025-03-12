import questionary
from datetime import datetime
from govuk_chat_evaluation.answer_generation.models import EvaluateInput
from govuk_chat_evaluation.file_system import (
    create_output_directory,
    jsonl_to_models,
)
from deepeval import evaluate as deepeval_evaluate
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import (
    # FaithfulnessMetric,
    # AnswerRelevancyMetric,
    # ToxicityMetric,
    # BiasMetric,
    GEval,
)


def _evaluate(evaluate_path: str):
    models = jsonl_to_models(evaluate_path, EvaluateInput)
    test_cases = [m.to_test_case() for m in models]

    # minimal example of a call to deepeval
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )
    deepeval_evaluate(test_cases, [correctness_metric])  # type: ignore
    # generate test cases
    # run deep eval


def main():
    start_time = datetime.now()

    # TODO: make use of this
    questionary.text("What are you testing?", default="The evaluation software").ask()

    # generate = questionary.confirm(
    #     "Do you want to generate answers?", default=True
    # ).ask()
    #
    # if generate:
    #     provider = questionary.select(
    #         "Which answer strategy do you want to use?", choices=["openai_structured_answer", "claude_structured_answer"]
    #     ).ask()

    evaluate_path = questionary.text(
        "What is the path to your input data",
        default="data/govukchat_groundtruth_feb2025_reduced.jsonl",
    ).ask()

    # output_dir = create_output_directory("answer_generation", start_time)

    _evaluate(evaluate_path)
