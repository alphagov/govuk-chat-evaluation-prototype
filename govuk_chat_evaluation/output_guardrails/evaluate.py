from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, List

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score
from tabulate import tabulate
import re

from ..file_system import jsonl_to_models, write_csv_results
import logging


class EvaluationResult(BaseModel):
    question: str
    expected_triggered: bool
    actual_triggered: bool
    expected_exact: str
    actual_exact: str

    @property
    def classification_triggered(self) -> str:
        match (self.expected_triggered, self.actual_triggered):
            case (True, True):
                return "true_positive"
            case (False, False):
                return "true_negative"
            case (False, True):
                return "false_positive"
            case (True, False):
                return "false_negative"

    @property
    def expected_exact_triggered(self) -> List[int] | None:
        """Parses `expected_exact` and returns a binary vector if triggered"""
        triggered, guardrails_str = self.parse_exact_string(self.expected_exact, 7)
        if triggered:
            return self.guardrail_str_to_vec(guardrails_str, 7)
        else:
            return None

    @property
    def actual_exact_triggered(self) -> List[int] | None:
        """Parses `actual_exact` and returns a binary vector if triggered"""
        triggered, guardrails_str = self.parse_exact_string(self.actual_exact, 7)
        if triggered:
            return self.guardrail_str_to_vec(guardrails_str, 7)
        else:
            return None

    @staticmethod
    def guardrail_str_to_vec(guardrails_str: str, num_guardrails: int) -> List[int]:
        """Returns a binary vector of triggered guardrails

        Each element of the vector represents whether the corresponding
        guardrail has been triggered. For example:

        "1, 3, 5" -> [1, 0, 1, 0, 0, 0, 0]
        "1" -> [1, 0, 0, 0, 0, 0, 0]
        "6, 7" -> [0, 0, 0, 0, 0, 1, 1]
        """
        if not guardrails_str.strip():
            return [0] * num_guardrails

        guardrail_set = {
            int(guardrail.strip()) for guardrail in guardrails_str.split(",")
        }
        return [1 if i + 1 in guardrail_set else 0 for i in range(num_guardrails)]

    @staticmethod
    def parse_exact_string(exact_string: str, num_guardrails: int) -> tuple[bool, str]:
        """Parses the exact guardrail result string for guardrail triggers

        Returns:
            A tuple containing:
                - A boolean indicating if any guardrail was triggered.
                - A string of comma-separated guardrail numbers if triggered,
                  otherwise the string "None".
        """
        # Pattern distinguishes between quoted comma-separated numbers and the literal None
        pattern = r"^(True|False)\s*\|\s*(?:\"(.*)\"|(None))$"
        match = re.match(pattern, exact_string)

        if not match:
            raise ValueError(
                f"Guardrail string '{exact_string}' does not match expected format "
                f"'True | \"<comma-separated numbers>\"' or 'False | None'."
            )

        triggered_str = match.group(1)
        guardrails_str = match.group(2)  # comma-separated guardrail numbers, if present
        none_str = match.group(3)  # The literal string "None", if present

        if triggered_str == "True":
            if guardrails_str is None:
                raise ValueError(
                    f"Guardrail string '{exact_string}' reports being triggered, but is not "
                    f"followed by quoted comma-separated numbers (e.g., 'True | \"1,2\"')."
                )

            if not guardrails_str.strip():
                raise ValueError(
                    f"Guardrail string '{exact_string}' has 'True' but contains an empty "
                    f"quoted string. Expected comma-separated numbers."
                )

            try:
                guardrail_numbers = [int(num) for num in guardrails_str.split(",")]
            except ValueError as e:
                raise ValueError(
                    f"Guardrail string '{exact_string}' contains non-integer value "
                    f"in the comma-separated list: {e}"
                ) from e

            if not all(1 <= num <= num_guardrails for num in guardrail_numbers):
                raise ValueError(
                    f"Guardrail string '{exact_string}' contains numbers outside the "
                    f"valid range [1, {num_guardrails}]."
                )

            if len(guardrail_numbers) != len(set(guardrail_numbers)):
                raise ValueError(
                    f"Guardrail string '{exact_string}' contains duplicate guardrail numbers."
                )

            return True, guardrails_str

        else:  # triggered_str == "False"
            if none_str != "None" or guardrails_str is not None:
                raise ValueError(
                    f"Guardrail string '{exact_string}' starts with 'False' but is not "
                    f"followed by 'None'. Expected format 'False | None'."
                )

            return False, "None"

    def for_csv(self) -> dict[str, Any]:
        return {**self.model_dump(), "classification": self.classification_triggered}


class AggregateResults:
    def __init__(self, evaluation_results: List[EvaluationResult]):
        self.evaluation_results = evaluation_results
        counter = Counter(
            evaluation_result.classification_triggered
            for evaluation_result in evaluation_results
        )
        self.true_positive = counter.get("true_positive", 0)
        self.true_negative = counter.get("true_negative", 0)
        self.false_positive = counter.get("false_positive", 0)
        self.false_negative = counter.get("false_negative", 0)

    @cached_property
    def _expected_actual_lists(self) -> tuple[List[int], List[int]]:
        pairs_list = [
            (
                int(evaluation_result.expected_triggered),
                int(evaluation_result.actual_triggered),
            )
            for evaluation_result in self.evaluation_results
        ]
        expected, actual = zip(*pairs_list)
        return list(expected), list(actual)

    def precision(self) -> float:
        return precision_score(
            *self._expected_actual_lists,
            zero_division=np.nan,  # type: ignore
        )

    def recall(self) -> float:
        return recall_score(
            *self._expected_actual_lists,
            zero_division=np.nan,  # type: ignore
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "True positives": self.true_positive,
            "True negatives": self.true_negative,
            "False positives": self.false_positive,
            "False negatives": self.false_negative,
        }

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
    models = jsonl_to_models(evaluation_data_path, EvaluationResult)

    if not models:
        logging.error("\nThere is no data to evaluate")
        return

    logging.info("\nEvaluation complete")
    write_csv_results(output_dir, [model.for_csv() for model in models])

    aggregate_results = AggregateResults(models)

    write_csv_results(
        output_dir,
        aggregate_results.for_csv(),
        filename="aggregate.csv",
        data_label="aggregates",
    )

    table = [[k, v] for k, v in aggregate_results.to_dict().items()]
    logging.info("Aggregate Results")
    logging.info(tabulate(table) + "\n")
