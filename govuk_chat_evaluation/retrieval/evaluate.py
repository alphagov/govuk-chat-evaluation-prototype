from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from tabulate import tabulate
import numpy as np
from collections.abc import Callable

from ..file_system import jsonl_to_models, write_csv_results
import logging

DECIMAL_PLACES = 4


class EvaluationResult(BaseModel):
    question: str
    expected_exact_paths: list[str]
    actual_exact_paths_and_scores: list[tuple[str, float]]

    @property
    def actual_exact_paths(self) -> list[str]:
        return [path for path, _ in self.actual_exact_paths_and_scores]

    @property
    def all_paths(self) -> list[str]:
        return list(set(self.expected_exact_paths + self.actual_exact_paths))

    @property
    def y_true(self) -> list[int]:
        return [int(path in self.expected_exact_paths) for path in self.all_paths]

    @property
    def y_pred(self) -> list[int]:
        return [int(path in self.actual_exact_paths) for path in self.all_paths]

    def precision(self) -> float:
        return precision_score(
            self.y_true,
            self.y_pred,
            zero_division=np.nan,  # type: ignore
        )

    def recall(self) -> float:
        return recall_score(
            self.y_true,
            self.y_pred,
            zero_division=np.nan,  # type: ignore
        )

    def f1_score(self) -> float:
        return f1_score(
            self.y_true,
            self.y_pred,
            zero_division=np.nan,  # type: ignore
        )

    def f2_score(self) -> float:
        return fbeta_score(
            self.y_true,
            self.y_pred,
            beta=2,
            zero_division=np.nan,  # type: ignore
        )

    def for_csv(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "expected_exact_paths": self.expected_exact_paths,
            "actual_exact_paths_and_scores": self.actual_exact_paths_and_scores,
            "precision": round(self.precision(), DECIMAL_PLACES),
            "recall": round(self.recall(), DECIMAL_PLACES),
            "f1_score": round(self.f1_score(), DECIMAL_PLACES),
            "f2_score": round(self.f2_score(), DECIMAL_PLACES),
        }


class AggregateResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results

    def _aggregate(
        self,
        score_fn: Callable[[Any], float],
        agg_fn: Callable[[list[float]], float],
    ) -> float:
        scores = [score_fn(result) for result in self.evaluation_results]
        result = agg_fn(scores)
        return float(round(result, DECIMAL_PLACES))

    def precision_mean(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.mean)

    def precision_median(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.median)

    def precision_max(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.max)

    def precision_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.std)

    def recall_mean(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.mean)

    def recall_median(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.median)

    def recall_max(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.max)

    def recall_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.std)

    def f1_mean(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.mean)

    def f1_median(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.median)

    def f1_max(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.max)

    def f1_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.std)

    def f2_mean(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.mean)

    def f2_median(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.median)

    def f2_max(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.max)

    def f2_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.std)

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Precision mean": self.precision_mean(),
            "Precision median": self.precision_median(),
            "Precision max": self.precision_max(),
            "Precision standard deviation": self.precision_standard_deviation(),
            "Recall mean": self.recall_mean(),
            "Recall median": self.recall_median(),
            "Recall max": self.recall_max(),
            "Recall standard deviation": self.recall_standard_deviation(),
            "F1 mean": self.f1_mean(),
            "F1 median": self.f1_median(),
            "F1 max": self.f1_max(),
            "F1 standard deviation": self.f1_standard_deviation(),
            "F2 mean": self.f2_mean(),
            "F2 median": self.f2_median(),
            "F2 max": self.f2_max(),
            "F2 standard deviation": self.f2_standard_deviation(),
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
