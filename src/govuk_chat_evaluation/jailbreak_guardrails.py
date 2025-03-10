import asyncio
import json
import math
import numpy as np
import os
import pandas as pd
import textwrap
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from pydantic import BaseModel
from sklearn.metrics import precision_score, recall_score
from typing import List
from tqdm.asyncio import tqdm


class JailbreakGolden(BaseModel):
    question: str
    expected_outcome: bool


@dataclass(frozen=True)
class JailbreakEvaluation:
    question: str
    expected_outcome: bool
    actual_outcome: bool


class Result:
    def __init__(self, evaluations: List[JailbreakEvaluation]):
        self.evaluations = evaluations

    @cached_property
    def _actual_list(self):
        return [1 if eval.actual_outcome else 0 for eval in self.evaluations]

    @cached_property
    def _predicted_list(self):
        return [1 if eval.expected_outcome else 0 for eval in self.evaluations]

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


def _fetch_ground_truth():
    # Need a nicer way to do this
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "jailbreak.csv"
    # could use datasets from deepeval for this but not sure if we gain much using
    # deepeval when not actually doing the eval
    dataframe = pd.read_csv(
        csv_path, converters={"expected_outcome": lambda n: bool(int(n))}
    )

    # would a map be better than this?
    return [JailbreakGolden(**row) for row in dataframe.to_dict(orient="records")]


async def _evaluate_golden(golden, semaphore):
    async with semaphore:
        govuk_chat_dir = Path.home() / "govuk" / "govuk-chat"

        env = os.environ.copy()
        env["INPUT"] = golden.question

        process = await asyncio.create_subprocess_exec(
            "bundle",
            "exec",
            "spring",
            "rake",
            "evaluation:generate_jailbreak_guardrail_response",
            cwd=govuk_chat_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception("Failed to generate a jailbreak response", stderr.decode())

        stdout_json = json.loads(stdout.decode())

        return JailbreakEvaluation(
            question=golden.question,
            expected_outcome=golden.expected_outcome,
            actual_outcome=stdout_json.get("triggered"),
        )


async def _generate_dataset(ground_truth):
    semaphore = asyncio.Semaphore(10)

    tasks = [
        asyncio.create_task(_evaluate_golden(golden, semaphore))
        for golden in ground_truth
    ]
    evaluations = []

    for future in tqdm.as_completed(tasks, total=len(tasks)):
        try:
            evaluation = await future
            evaluations.append(evaluation)
        except Exception as e:
            # Cancel all remaining tasks to ensure clean termination
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            raise e

    return evaluations


def _result_summary(result):
    precision_str = "N/A" if math.isnan(result.precision) else f"{result.precision:.2%}"
    recall_str = "N/A" if math.isnan(result.recall) else f"{result.recall:.2%}"

    return textwrap.dedent(
        f"""\
            Evaluated: {len(result.evaluations)}
            Precision: {precision_str}
            Recall: {recall_str}
            True positives: {result.true_positives}
            False positives: {result.false_positives}
            True negatives: {result.true_negatives}
            False negatives: {result.false_negatives}"""
    )


def main():
    ground_truth = _fetch_ground_truth()
    evaluations = asyncio.run(_generate_dataset(ground_truth))
    result = Result(evaluations)
    print(_result_summary(result))


if __name__ == "__main__":
    main()
