from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate

from ..file_system import jsonl_to_models, write_csv_results
import logging


class EvaluationResult(BaseModel):
    question: str
    expected_triggered: bool
    actual_triggered: bool
    expected_guardrails: dict[str, bool]
    actual_guardrails: dict[str, bool]

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

    def for_csv(self) -> dict[str, Any]:
        return {**self.model_dump(), "classification": self.classification_triggered}


class AggregateResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results
        counter = Counter(
            result.classification_triggered for result in evaluation_results
        )
        self.true_positive = counter.get("true_positive", 0)
        self.true_negative = counter.get("true_negative", 0)
        self.false_positive = counter.get("false_positive", 0)
        self.false_negative = counter.get("false_negative", 0)

        guardrail_set = {
            name
            for result in evaluation_results
            for name in list(result.expected_guardrails.keys())
            + list(result.actual_guardrails.keys())
        }
        self.guardrail_names: list[str] = sorted(guardrail_set)

    @cached_property
    def _expected_actual_triggered_lists(self) -> tuple[list[int], list[int]]:
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
    def _expected_actual_guardrails_vectors(
        self,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Generates lists of expected and actual binary vectors for per-guardrail evaluation."""

        def to_vector(d: dict[str, bool]) -> list[int]:
            return [int(d.get(name, False)) for name in self.guardrail_names]

        expected_vectors = [
            to_vector(result.expected_guardrails) for result in self.evaluation_results
        ]
        actual_vectors = [
            to_vector(result.actual_guardrails) for result in self.evaluation_results
        ]
        return expected_vectors, actual_vectors

    def _triggered_metric(self, metric_function):
        expected, actual = self._expected_actual_triggered_lists
        return metric_function(expected, actual, zero_division=np.nan)  # type: ignore

    def _metric_per_guardrail(self, metric_function):
        expected_vectors, actual_vectors = self._expected_actual_guardrails_vectors

        return metric_function(
            expected_vectors,
            actual_vectors,
            average=None,
            zero_division=np.nan,
        ).tolist()  # type: ignore

    def precision(self) -> float:
        return cast(float, self._triggered_metric(precision_score))

    def recall(self) -> float:
        return cast(float, self._triggered_metric(recall_score))

    def precision_per_guardrail(self) -> list[float]:
        return self._metric_per_guardrail(precision_score)

    def recall_per_guardrail(self) -> list[float]:
        return self._metric_per_guardrail(recall_score)

    def f1_per_guardrail(self) -> list[float]:
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

        precisions = self.precision_per_guardrail()
        recalls = self.recall_per_guardrail()
        f1s = self.f1_per_guardrail()

        for i, name in enumerate(self.guardrail_names):
            base_metrics[f"Precision [{name}]"] = precisions[i]
            base_metrics[f"Recall [{name}]"] = recalls[i]
            base_metrics[f"F1 [{name}]"] = f1s[i]

        return base_metrics

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
    models = jsonl_to_models(evaluation_data_path, EvaluationResult)

    if not models:
        logging.error("\nThere is no data to evaluate")
        return

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
