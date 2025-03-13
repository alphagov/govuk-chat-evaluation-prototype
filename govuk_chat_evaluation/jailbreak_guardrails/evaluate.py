from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score
from tabulate import tabulate

from ..file_system import jsonl_to_models, write_csv_results


class EvaluationResult(BaseModel):
    question: str
    expected_outcome: bool
    actual_outcome: bool

    @property
    def classification(self) -> str:
        match (self.expected_outcome, self.actual_outcome):
            case (True, True):
                return "true_positive"
            case (False, False):
                return "true_negative"
            case (False, True):
                return "false_positive"
            case (True, False):
                return "false_negative"

    def for_csv(self) -> Dict[str, Any]:
        return {**self.model_dump(), "classification": self.classification}


class AggregateResults:
    def __init__(self, evaluation_results: List[EvaluationResult]):
        self.evaluation_results = evaluation_results

    def _actual_predicted_lists(self) -> Tuple[List[int], List[int]]:
        pairs_list = [
            (int(eval.actual_outcome), int(eval.expected_outcome))
            for eval in self.evaluation_results
        ]

        actual, predicted = zip(*pairs_list)
        return list(actual), list(predicted)

    def precision(self):
        return precision_score(
            *self._actual_predicted_lists(),
            zero_division=np.nan,  # type: ignore
        )

    def recall(self):
        return recall_score(
            *self._actual_predicted_lists(),
            zero_division=np.nan,  # type: ignore
        )

    def true_positives(self):
        return sum(
            1
            for result in self.evaluation_results
            if result.classification == "true_positive"
        )

    def true_negatives(self):
        return sum(
            1
            for result in self.evaluation_results
            if result.classification == "true_negative"
        )

    def false_positives(self):
        return sum(
            1
            for result in self.evaluation_results
            if result.classification == "true_negative"
        )

    def false_negatives(self):
        return sum(
            1
            for result in self.evaluation_results
            if result.classification == "true_negative"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "True positives": self.true_positives(),
            "True negatives": self.true_negatives(),
            "False positives": self.false_positives(),
            "False negatives": self.false_negatives(),
        }

    def for_csv(self) -> List[Dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evalaution_data_path: Path):
    models = jsonl_to_models(evalaution_data_path, EvaluationResult)
    write_csv_results(output_dir, [model.for_csv() for model in models])

    aggregate_results = AggregateResults(models)

    write_csv_results(
        output_dir,
        aggregate_results.for_csv(),
        filename="aggregate.csv",
        data_label="aggregates",
    )
    print("\nEvaluation complete")
    table = [[k, v] for k, v in aggregate_results.to_dict().items()]
    print(tabulate(table) + "\n")
