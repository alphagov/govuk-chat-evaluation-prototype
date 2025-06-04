from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tabulate import tabulate
import numpy as np

from ..file_system import jsonl_to_models, write_csv_results
import logging


class EvaluationResult(BaseModel):
    question: str
    expected_exact_paths: list[str]
    actual_exact_paths: list[str]

    def for_csv(self) -> dict[str, Any]:
        return {**self.model_dump()}


class AggregateResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results

        self.true_positives = set()
        self.false_positives = set()
        self.false_negatives = set()

        for expected, actual in zip(*self._expected_actual_lists):
            self.true_positives.update(set(actual) & set(expected))
            self.false_positives.update(set(actual) - set(expected))
            self.false_negatives.update(set(expected) - set(actual))

    @cached_property
    def _expected_actual_lists(self) -> tuple[list[list[str]], list[list[str]]]:
        pairs_list = [
            (eval.expected_exact_paths, eval.actual_exact_paths)
            for eval in self.evaluation_results
        ]
        expected, actual = zip(*pairs_list)
        return list(expected), list(actual)

    @cached_property
    def _total_actual_path_count(self) -> int:
        actual_values = self._expected_actual_lists[1]
        return sum(len(sublist) for sublist in actual_values)

    @cached_property
    def _total_expected_path_count(self) -> int:
        expected_values = self._expected_actual_lists[0]
        return sum(len(sublist) for sublist in expected_values)

    def precision(self) -> float:
        if self._total_actual_path_count == 0:
            return np.nan

        return len(self.true_positives) / self._total_actual_path_count

    def recall(self) -> float:
        if self._total_expected_path_count == 0:
            return np.nan

        return len(self.true_positives) / self._total_expected_path_count

    def f1_score(self) -> float:
        try:
            return (
                2
                * len(self.true_positives)
                / (
                    2 * len(self.true_positives)
                    + len(self.false_positives)
                    + len(self.false_negatives)
                )
            )
        except ZeroDivisionError:
            return np.nan

    def f2_score(self) -> float:
        try:
            return (
                (1 + 2**2)
                * len(self.true_positives)
                / (
                    (1 + 2**2) * len(self.true_positives)
                    + ((2**2) * len(self.false_negatives))
                    + (len(self.false_positives))
                )
            )
        except ZeroDivisionError:
            return np.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1 Score": self.f1_score(),
            "F2 Score": self.f2_score(),
        }

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
    """Evaluate the data in the evaluation data file and write result files
    to the output paths, with aggregates written to STDOUT"""

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
    logging.info("\nAggregate Results")
    logging.info(tabulate(table) + "\n")
