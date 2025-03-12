import math
import textwrap
import yaml
from datetime import datetime
from govuk_chat_evaluation.file_system import create_output_directory, jsonl_to_models
from typing import cast
from .models import Config, EvaluateInput, Result
from .generate import generate_and_write_dataset


# move this to evaluate and intend to mock it in tests
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
    # TODO: make this configurable
    with open("config/defaults/jailbreak_guardrails.yaml", "r") as file:
        config_data = yaml.safe_load(file)

    config = Config(**config_data)

    output_dir = create_output_directory("jailbreak_guardrails", start_time)

    if config.generate:
        evaluate_path = generate_and_write_dataset(
            config.input_path, cast(str, config.provider), output_dir
        )
    else:
        evaluate_path = config.input_path

    result = _evaluate(evaluate_path)
    print(_result_summary(result))
