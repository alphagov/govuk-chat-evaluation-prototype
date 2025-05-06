from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..file_system import jsonl_to_models, write_csv_results
import logging


class EvaluationResult(BaseModel):
    question: str
    expected_outcome: str
    actual_outcome: str
    confidence_score: float

    def for_csv(self) -> dict[str, Any]:
        return {**self.model_dump()}


class AggregateResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results

    @cached_property
    def classification_labels(self) -> list[str]:
        return sorted(
            list(set(item for lst in self._expected_actual_lists for item in lst))
        )

    @cached_property
    def _expected_actual_lists(self) -> tuple[list[str], list[str]]:
        pairs_list = [
            (eval.expected_outcome, eval.actual_outcome)
            for eval in self.evaluation_results
        ]

        expected, actual = zip(*pairs_list)
        return list(expected), list(actual)

    def accuracy(self) -> float:
        return accuracy_score(
            *self._expected_actual_lists,  # type: ignore
        )

    def precision(self) -> float:
        return precision_score(
            *self._expected_actual_lists,
            average="weighted",
            zero_division=np.nan,  # type: ignore
        )

    def recall(self) -> float:
        return recall_score(
            *self._expected_actual_lists,
            average="weighted",
            zero_division=np.nan,  # type: ignore
        )

    def f1_score(self) -> float:
        return f1_score(
            *self._expected_actual_lists,
            average="weighted",
            zero_division=np.nan,  # type: ignore
        )

    def f2_score(self) -> float:
        return fbeta_score(
            *self._expected_actual_lists,
            beta=2,
            average="weighted",
            zero_division=np.nan,  # type: ignore
        )

    def confusion_matrix_data(self) -> list[list[int]]:
        return confusion_matrix(
            *self._expected_actual_lists,
            labels=sorted(list(set(self.classification_labels))),  # type: ignore
        )

    def miscategorised_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "question": result.question,
                "predicted_classification": result.expected_outcome,
                "actual_classification": result.actual_outcome,
                "confidence_score": result.confidence_score,
            }
            for result in self.evaluation_results
            if result.expected_outcome != result.actual_outcome
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Accuracy": self.accuracy(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1 Score": self.f1_score(),
            "F2 Score": self.f2_score(),
            "Miscategorised Cases": len(self.miscategorised_cases()),
        }

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def generate_and_output_confusion_matrix(
    output_dir: Path,
    confusion_matrix_data: list[list[int]],
    confusion_matrix_labels: list[str],
):
    """Takes confusion matrix data (a 2D list) calculated by sklearn
    and a list of labels (strings representing the question routing labels)
    and outputs an confusion matrix PNG image to the output directory"""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        confusion_matrix_data,  # type: ignore
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=confusion_matrix_labels,
        yticklabels=confusion_matrix_labels,
        ax=ax,
        annot_kws={"size": 8},
        cbar=False,
    )
    plt.title("Confusion Matrix", fontsize=8)
    plt.xlabel("Predicted", fontsize=8)
    plt.ylabel("True", fontsize=8)
    ax.xaxis.set_ticklabels(
        ax.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=6
    )
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")


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

    generate_and_output_confusion_matrix(
        output_dir,
        aggregate_results.confusion_matrix_data(),
        aggregate_results.classification_labels,
    )

    write_csv_results(
        output_dir,
        aggregate_results.miscategorised_cases(),
        filename="miscategorised_cases.csv",
        data_label="miscategorised_cases",
    )

    table = [[k, v] for k, v in aggregate_results.to_dict().items()]
    logging.info("\nAggregate Results")
    logging.info(tabulate(table) + "\n")
