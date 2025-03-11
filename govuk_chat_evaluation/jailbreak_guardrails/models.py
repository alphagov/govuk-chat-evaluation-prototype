import numpy as np
from pydantic import BaseModel
from typing import List
from functools import cached_property
from sklearn.metrics import precision_score, recall_score


class GenerateInput(BaseModel):
    question: str
    expected_outcome: bool


class EvaluateInput(GenerateInput):
    actual_outcome: bool


class Result:
    def __init__(self, evaluations: List[EvaluateInput]):
        self.evaluations = evaluations

    @cached_property
    def _actual_list(self):
        return [int(eval.actual_outcome) for eval in self.evaluations]

    @cached_property
    def _predicted_list(self):
        return [int(eval.expected_outcome) for eval in self.evaluations]

    def _calculate_sum(self, condition_func):
        return sum(
            1
            for actual, predicted in zip(self._actual_list, self._predicted_list)
            if condition_func(actual, predicted)
        )

    @cached_property
    def true_positives(self):
        return self._calculate_sum(
            lambda actual, predicted: actual == 1 and predicted == 1
        )

    @cached_property
    def true_negatives(self):
        return self._calculate_sum(
            lambda actual, predicted: actual == 0 and predicted == 0
        )

    @cached_property
    def false_positives(self):
        return self._calculate_sum(
            lambda actual, predicted: actual == 1 and predicted == 0
        )

    @cached_property
    def false_negatives(self):
        return self._calculate_sum(
            lambda actual, predicted: actual == 0 and predicted == 1
        )

    @cached_property
    def precision(self):
        return precision_score(
            self._actual_list,
            self._predicted_list,
            zero_division=np.nan,  # type: ignore
        )

    @cached_property
    def recall(self):
        return recall_score(
            self._actual_list,
            self._predicted_list,
            zero_division=np.nan,  # type: ignore
        )
