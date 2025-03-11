import math
import questionary
import textwrap
from datetime import datetime
from govuk_chat_evaluation.file_system import (
    create_output_directory,
    jsonl_to_models,
    write_generated_to_output,
)
from pathlib import Path
from typing import cast
from .models import GenerateInput, EvaluateInput, Result
from .generate import generate_jailbreak_guardrails_dataset


def _generate_dataset(input_path: str, provider: str, output_dir: Path):
    models = jsonl_to_models(input_path, GenerateInput)
    generated = generate_jailbreak_guardrails_dataset(provider, models)
    return write_generated_to_output(generated, output_dir)


def _evaluate(evaluate_path):
    models = jsonl_to_models(evaluate_path, EvaluateInput)
    return Result(models)


def _result_summary(result):
    precision_str = "N/A" if math.isnan(result.precision) else f"{result.precision:.2%}"
    recall_str = "N/A" if math.isnan(result.recall) else f"{result.recall:.2%}"

    return textwrap.dedent(
        f"""\
            Evaluated: {len(result.evaluations)}
            Precision: {precision_str}
            Recall: {recall_str}
            True positives: {result.true_positives}
            False positives: {result.false_positives}
            True negatives: {result.true_negatives}
            False negatives: {result.false_negatives}"""
    )


def main():
    start_time = datetime.now()
    provider = None

    # TODO: make use of this
    questionary.text("What are you testing?", default="The evaluation software").ask()

    generate = questionary.confirm(
        "Do you want to generate answers?", default=True
    ).ask()

    if generate:
        provider = questionary.select(
            "Which LLM provider do you want to evaluate?", choices=["openai", "claude"]
        ).ask()

    input_path = questionary.text(
        "What is the path to your input data", default="data/jailbreak_guardrails.jsonl"
    ).ask()

    output_dir = create_output_directory("jailbreak_guardrails", start_time)

    if generate:
        evaluate_path = _generate_dataset(input_path, cast(str, provider), output_dir)
    else:
        evaluate_path = input_path

    result = _evaluate(evaluate_path)
    print(_result_summary(result))
