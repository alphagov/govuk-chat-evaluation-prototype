from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, List
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate
import re
import logging

from ..file_system import jsonl_to_models, write_csv_results
import logging

NUM_GUARDRAILS = 7


@dataclass
class _GuardrailParseResult:
    is_triggered: bool
    valid_numbers: List[int]  # Unique, valid guardrail numbers
    warnings: List[str]  # Warnings generated during parsing


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
        """Parses `expected_exact` and returns a binary vector if triggered."""
        try:
            result = self.parse_exact_string(self.expected_exact, NUM_GUARDRAILS)

            for warning_msg in result.warnings:
                logging.warning(
                    f'{warning_msg} Parsing expected_exact="{self.expected_exact}" '
                    f'for question="{self.question}".'
                )

            if result.is_triggered:
                return self._guardrail_list_to_vec(result.valid_numbers, NUM_GUARDRAILS)
            else:
                return None
        except ValueError as e:
            logging.warning(
                f'Failed to parse expected_exact="{self.expected_exact}" '
                + f'for question="{self.question}": {e}. Treating as None.'
            )
            return None

    @property
    def actual_exact_triggered(self) -> List[int] | None:
        """Parses `actual_exact` and returns a binary vector if triggered."""
        try:
            result = self.parse_exact_string(self.actual_exact, NUM_GUARDRAILS)

            for warning_msg in result.warnings:
                logging.warning(
                    f'{warning_msg} Parsing actual_exact="{self.actual_exact}" '
                    f'for question="{self.question}".'
                )

            if result.is_triggered:
                return self._guardrail_list_to_vec(result.valid_numbers, NUM_GUARDRAILS)
            else:
                return None
        except ValueError as e:
            logging.warning(
                f'Failed to parse actual_exact="{self.actual_exact}" '
                + f'for question="{self.question}": {e}. Treating as None.'
            )
            return None

    @staticmethod
    def _guardrail_list_to_vec(
        guardrail_numbers: List[int], num_guardrails: int
    ) -> List[int]:
        """Returns a binary vector of triggered guardrails from a list of numbers.

        Each element of the vector represents whether the corresponding
        guardrail has been triggered. For example:

        [1, 3, 5] -> [1, 0, 1, 0, 1, 0, 0] (assuming num_guardrails=7)
        [1] -> [1, 0, 0, 0, 0, 0, 0]
        [6, 7] -> [0, 0, 0, 0, 0, 1, 1]
        [] -> [0, 0, 0, 0, 0, 0, 0]
        """
        if not guardrail_numbers:
            return [0] * num_guardrails

        guardrail_set = set(guardrail_numbers)
        return [1 if i + 1 in guardrail_set else 0 for i in range(num_guardrails)]

    @staticmethod
    def _process_guardrail_numbers(
        exact_string_context: str, numbers_str: str, num_guardrails: int
    ) -> tuple[List[int], List[str]]:
        """Parses, validates, filters, and deduplicates guardrail numbers."""
        warnings = []
        original_guardrail_numbers = []

        number_segments = numbers_str.split(",")
        for segment in number_segments:
            stripped_segment = segment.strip()
            try:
                original_guardrail_numbers.append(int(stripped_segment))
            except ValueError as e:
                raise ValueError(
                    f"Guardrail string '{exact_string_context}' contains non-integer value "
                    f"'{stripped_segment}' in the comma-separated list: {e}"
                ) from e

        if not original_guardrail_numbers:
            return [], []

        original_set = set(original_guardrail_numbers)
        valid_range_set = {num for num in original_set if 1 <= num <= num_guardrails}
        out_of_range_set = original_set - valid_range_set

        valid_range_list = [
            num for num in original_guardrail_numbers if 1 <= num <= num_guardrails
        ]
        if len(valid_range_list) != len(valid_range_set):
            counts = Counter(valid_range_list)
            duplicates = sorted([num for num, count in counts.items() if count > 1])
            warnings.append(f"Removed duplicate numbers: {duplicates}.")

        if out_of_range_set:
            warnings.append(
                f"Removed numbers outside the valid range [1, {num_guardrails}]: {sorted(list(out_of_range_set))}."
            )

        unique_valid_numbers = sorted(list(valid_range_set))

        if not unique_valid_numbers and original_guardrail_numbers:
            warnings.append(
                "Resulted in no valid guardrails after filtering/deduplication."
            )

        return unique_valid_numbers, warnings

    @staticmethod
    def parse_exact_string(
        exact_string: str, num_guardrails: int
    ) -> _GuardrailParseResult:
        """Parses the exact guardrail result string.

        Returns:
            A _GuardrailParseResult object containing the trigger status,
            a list of valid, unique guardrail numbers, and any warnings.
        """
        pattern = r"^(True|False)\s*\|\s*(?:\"(.*)\"|(None))$"
        match = re.match(pattern, exact_string)

        if not match:
            raise ValueError(
                f"Guardrail string '{exact_string}' does not match expected format "
                f"'True | \"<comma-separated numbers>\"' or 'False | None'."
            )

        triggered_str = match.group(1)
        guardrails_str = match.group(2)  # Comma-separated numbers, if present
        none_str = match.group(3)  # The literal string "None", if present

        if triggered_str == "True":
            if guardrails_str is None:
                raise ValueError(
                    f"Guardrail string '{exact_string}' reports being triggered (True), but is not "
                    f"followed by quoted comma-separated numbers (e.g., 'True | \"1,2\"')."
                )

            if guardrails_str == "":
                raise ValueError(
                    f"Guardrail string '{exact_string}' has 'True' but guardrails string "
                    f"is empty. Expected comma-separated numbers."
                )

            valid_numbers, warnings = EvaluationResult._process_guardrail_numbers(
                exact_string, guardrails_str, num_guardrails
            )
            return _GuardrailParseResult(
                is_triggered=True, valid_numbers=valid_numbers, warnings=warnings
            )

        else:  # triggered_str == "False"
            if none_str != "None" or guardrails_str is not None:
                raise ValueError(
                    f"Guardrail string '{exact_string}' starts with 'False' but is not "
                    f"followed by 'None'. Expected format 'False | None'."
                )
            # Return non-triggered result with empty numbers/warnings
            return _GuardrailParseResult(
                is_triggered=False, valid_numbers=[], warnings=[]
            )

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

    @cached_property
    def _expected_actual_vectors(self) -> tuple[List[List[int]], List[List[int]]]:
        """
        Generates lists of expected and actual binary vectors for per-guardrail evaluation.

        Vectors represent triggered guardrails. If a result indicates no trigger
        (expected_triggered or actual_triggered is False), a zero vector is used.
        """
        expected_vectors = []
        actual_vectors = []
        zero_vector = [0] * NUM_GUARDRAILS

        for result in self.evaluation_results:
            expected_vec = result.expected_exact_triggered
            actual_vec = result.actual_exact_triggered

            expected_vectors.append(
                expected_vec if expected_vec is not None else zero_vector
            )
            actual_vectors.append(actual_vec if actual_vec is not None else zero_vector)

        # Handle the edge case where there are no results to evaluate
        if not expected_vectors:
            return [], []

        return expected_vectors, actual_vectors

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

    def precision_per_guardrail(self) -> List[float]:
        """Calculates precision for each guardrail individually."""
        expected_vectors, actual_vectors = self._expected_actual_vectors
        if not expected_vectors:  # Avoid calling score with empty lists
            return [np.nan] * NUM_GUARDRAILS
        return precision_score(
            expected_vectors,
            actual_vectors,
            average=None,  # type: ignore
            zero_division=0,  # type: ignore
        ).tolist()  # type: ignore

    def recall_per_guardrail(self) -> List[float]:
        """Calculates recall for each guardrail individually."""
        expected_vectors, actual_vectors = self._expected_actual_vectors
        if not expected_vectors:
            return [np.nan] * NUM_GUARDRAILS
        return recall_score(
            expected_vectors,
            actual_vectors,
            average=None,  # type: ignore
            zero_division=0,  # type: ignore
        ).tolist()  # type: ignore

    def f1_per_guardrail(self) -> List[float]:
        """Calculates F1 score for each guardrail individually."""
        expected_vectors, actual_vectors = self._expected_actual_vectors
        if not expected_vectors:
            return [np.nan] * NUM_GUARDRAILS
        return f1_score(
            expected_vectors,
            actual_vectors,
            average=None,  # type: ignore
            zero_division=0,  # type: ignore
        ).tolist()  # type: ignore

    def to_dict(self) -> dict[str, Any]:
        base_metrics = {
            "Evaluated": len(self.evaluation_results),
            "Any-triggered Precision": self.precision(),
            "Any-triggered Recall": self.recall(),
            "Any-triggered True positives": self.true_positive,
            "Any-triggered True negatives": self.true_negative,
            "Any-triggered False positives": self.false_positive,
            "Any-triggered False negatives": self.false_negative,
        }

        # Per-guardrail metrics
        precisions = self.precision_per_guardrail()
        recalls = self.recall_per_guardrail()
        f1s = self.f1_per_guardrail()

        for i in range(NUM_GUARDRAILS):
            guardrail_num = i + 1
            base_metrics[f"Precision G{guardrail_num}"] = precisions[i]
            base_metrics[f"Recall G{guardrail_num}"] = recalls[i]
            base_metrics[f"F1 G{guardrail_num}"] = f1s[i]

        return base_metrics

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
