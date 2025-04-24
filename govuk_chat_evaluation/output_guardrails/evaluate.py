from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, List, cast
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
class GuardrailParseResult:
    is_triggered: bool
    valid_numbers: List[int]
    warnings: List[str]

    @staticmethod
    def _process_guardrail_numbers(
        numbers_str: str, num_guardrails: int
    ) -> tuple[List[int], List[str]]:
        """Parse, validate and deduplicate a comma-separated string of numbers.

        Returns a tuple of (unique_valid_numbers, warnings).
        """

        warnings: list[str] = []

        number_segments = [seg.strip() for seg in numbers_str.split(",") if seg.strip()]
        original_numbers: list[int] = []

        for segment in number_segments:
            try:
                original_numbers.append(int(segment))
            except ValueError:
                warnings.append(f"Non-integer value '{segment}' ignored.")

        if not original_numbers:
            warnings.append("Absence of valid guardrail numbers.")
            return [], warnings

        counts = Counter(original_numbers)
        duplicates = sorted([num for num, c in counts.items() if c > 1])
        if duplicates:
            warnings.append(f"Removed duplicate numbers: {duplicates}.")

        unique_numbers = sorted(set(original_numbers))

        valid_numbers = [n for n in unique_numbers if 1 <= n <= num_guardrails]
        out_of_range = [n for n in unique_numbers if n not in valid_numbers]
        if out_of_range:
            warnings.append(
                f"Removed numbers outside the valid range [1, {num_guardrails}]: {out_of_range}."
            )

        if not valid_numbers:
            warnings.append(
                "Resulted in no valid guardrails after filtering/deduplication."
            )

        return valid_numbers, warnings

    @classmethod
    def parse(cls, exact_string: str, num_guardrails: int) -> "GuardrailParseResult":
        """Parses the exact guardrail result string.

        Returns a GuardrailParseResult object containing the trigger status,
        a list of valid, unique guardrail numbers, and any warnings.
        """
        # Regex pattern requiring either:
        #   True  | "<numbers, spaces, commas>"  OR
        #   False | None
        pattern = r"^\s*(True|False)\s*\|\s*(?:\"([0-9,\s]*)\"|(None))\s*$"
        match = re.match(pattern, exact_string)

        warnings: list[str] = []

        if not match:
            warnings.append(
                "String does not match expected format 'True | \"1,2\"' or 'False | None'."
            )
            return cls(False, [], warnings)

        triggered = match.group(1) == "True"
        guardrails_str = match.group(2)  # Comma-separated numbers, if present
        none_str = match.group(3)  # The literal string "None", if present

        if triggered:
            if guardrails_str is None:
                warnings.append("Triggered == True but no guardrail numbers supplied.")
                return cls(is_triggered=False, valid_numbers=[], warnings=warnings)

            valid_numbers, num_warnings = cls._process_guardrail_numbers(
                guardrails_str, num_guardrails
            )
            warnings.extend(num_warnings)
            return cls(
                is_triggered=True, valid_numbers=valid_numbers, warnings=warnings
            )

        if none_str != "None":
            warnings.append("Triggered == False but expected literal 'None'.")
        return cls(False, [], warnings)

    def to_vector(self, num_guardrails: int) -> List[int]:
        """Return a binary vector indicating which guardrails are triggered.

        Each index in the returned list corresponds to a guardrail (1-indexed),
        holding 1 if that guardrail was triggered for the answer and 0
        otherwise.
        """
        if not self.is_triggered:
            return [0] * num_guardrails

        guardrail_set = set(self.valid_numbers)
        return [1 if i + 1 in guardrail_set else 0 for i in range(num_guardrails)]


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

    @cached_property
    def _parsed_expected(self) -> GuardrailParseResult:
        return GuardrailParseResult.parse(self.expected_exact, NUM_GUARDRAILS)

    @cached_property
    def _parsed_actual(self) -> GuardrailParseResult:
        return GuardrailParseResult.parse(self.actual_exact, NUM_GUARDRAILS)

    @property
    def expected_exact_triggered(self) -> List[int]:
        return self._parsed_expected.to_vector(NUM_GUARDRAILS)

    @property
    def actual_exact_triggered(self) -> List[int]:
        return self._parsed_actual.to_vector(NUM_GUARDRAILS)

    @property
    def warnings(self) -> List[str]:
        return self._parsed_expected.warnings + self._parsed_actual.warnings

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
        expected, actual = zip(
            *(
                (
                    int(evaluation_result.expected_triggered),
                    int(evaluation_result.actual_triggered),
                )
                for evaluation_result in self.evaluation_results
            )
        )
        return list(expected), list(actual)

    @cached_property
    def _expected_actual_vectors(self) -> tuple[List[List[int]], List[List[int]]]:
        """Generates lists of expected and actual binary vectors for per-guardrail evaluation."""
        if not self.evaluation_results:
            return [], []

        expected_vectors = [
            result.expected_exact_triggered for result in self.evaluation_results
        ]
        actual_vectors = [
            result.actual_exact_triggered for result in self.evaluation_results
        ]
        return expected_vectors, actual_vectors

    def _metric_any(self, metric_function):
        return metric_function(*self._expected_actual_lists, zero_division=np.nan)  # type: ignore

    def _metric_per_guardrail(self, metric_function):
        expected_vectors, actual_vectors = self._expected_actual_vectors
        if not expected_vectors:
            return [np.nan] * NUM_GUARDRAILS
        return metric_function(
            expected_vectors, actual_vectors, average=None, zero_division=np.nan
        ).tolist()  # type: ignore

    def precision(self) -> float:
        return cast(float, self._metric_any(precision_score))

    def recall(self) -> float:
        return cast(float, self._metric_any(recall_score))

    def precision_per_guardrail(self) -> List[float]:
        return self._metric_per_guardrail(precision_score)

    def recall_per_guardrail(self) -> List[float]:
        return self._metric_per_guardrail(recall_score)

    def f1_per_guardrail(self) -> List[float]:
        return self._metric_per_guardrail(f1_score)

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
    for model in models:
        for warning in model.warnings:
            logging.warning(
                f'{warning}  Parsing actual_exact="{model.actual_exact}" '
                f'for question="{model.question}".'
            )

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
