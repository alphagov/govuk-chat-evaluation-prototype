from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, List

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score
from tabulate import tabulate

from ..file_system import jsonl_to_models, write_csv_results


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
    def classification_exact(self) -> str:
        is_match = self.expected_exact == self.actual_exact
        expected_is_positive = self.expected_exact.startswith("True")

        if is_match:
            return "true_positive" if expected_is_positive else "true_negative"
        else:
            return "false_positive" if not expected_is_positive else "false_negative"

    def for_csv(self) -> dict[str, Any]:
        return {
            **self.model_dump(),
            "classification_triggered": self.classification_triggered,
            "classification_exact": self.classification_exact,
        }


class AggregateResults:
    def __init__(self, evaluation_results: List[EvaluationResult]):
        self.evaluation_results = evaluation_results
        counter_triggered = Counter(
            evaluation_result.classification_triggered
            for evaluation_result in evaluation_results
        )
        counter_exact = Counter(
            evaluation_result.classification_exact
            for evaluation_result in evaluation_results
        )
        self.true_positive_triggered = counter_triggered.get("true_positive", 0)
        self.true_negative_triggered = counter_triggered.get("true_negative", 0)
        self.false_positive_triggered = counter_triggered.get("false_positive", 0)
        self.false_negative_triggered = counter_triggered.get("false_negative", 0)
        self.true_positive_exact = counter_exact.get("true_positive", 0)
        self.true_negative_exact = counter_exact.get("true_negative", 0)
        self.false_positive_exact = counter_exact.get("false_positive", 0)
        self.false_negative_exact = counter_exact.get("false_negative", 0)

    @cached_property
    def _expected_actual_lists_triggered(self) -> tuple[List[int], List[int]]:
        pairs_list = [
            (
                int(evaluation_result.expected_triggered),
                int(evaluation_result.actual_triggered),
            )
            for evaluation_result in self.evaluation_results
        ]
        expected, actual = zip(*pairs_list)
        return list(expected), list(actual)

    def precision_triggered(self) -> float:
        return precision_score(
            *self._expected_actual_lists_triggered,
            zero_division=np.nan,  # type: ignore
        )

    def recall_triggered(self) -> float:
        return recall_score(
            *self._expected_actual_lists_triggered,
            zero_division=np.nan,  # type: ignore
        )

    def _calculate_metric(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return float("nan")
        return float(numerator) / denominator

    def precision_exact(self) -> float:
        return self._calculate_metric(
            self.true_positive_exact,
            self.true_positive_exact + self.false_positive_exact,
        )

    def recall_exact(self) -> float:
        return self._calculate_metric(
            self.true_positive_exact,
            self.true_positive_exact + self.false_negative_exact,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Precision (triggered)": self.precision_triggered(),
            "Recall (triggered)": self.recall_triggered(),
            "True positives (triggered)": self.true_positive_triggered,
            "True negatives (triggered)": self.true_negative_triggered,
            "False positives (triggered)": self.false_positive_triggered,
            "False negatives (triggered)": self.false_negative_triggered,
            "Precision (exact)": self.precision_exact(),
            "Recall (exact)": self.recall_exact(),
            "Exact True positives (exact)": self.true_positive_exact,
            "True negatives (exact)": self.true_negative_exact,
            "Exact False positive (exact)": self.false_positive_exact,
            "Exact False negative (exact)": self.false_negative_exact,
        }

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
    print("\nEvaluation complete")
    models = jsonl_to_models(evaluation_data_path, EvaluationResult)
    write_csv_results(output_dir, [model.for_csv() for model in models])

    for mode in ("triggered", "exact"):
        aggregate_results = AggregateResults(models)

        write_csv_results(
            output_dir,
            aggregate_results.for_csv(),
            filename="aggregate.csv",
            data_label="aggregates",
        )
        table = [[k, v] for k, v in aggregate_results.to_dict().items()]
        print(f"\nAggregate Results ({mode})")
        print(tabulate(table) + "\n")
