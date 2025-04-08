import os
from pathlib import Path
from typing import cast, Any
from dataclasses import asdict
from functools import cached_property
from pydantic import BaseModel
import pandas as pd

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate import TestResult
from deepeval.test_case import LLMTestCase, MLLMTestCase

from ..timing import print_task_duration
from ..file_system import jsonl_to_models
from .data_models import EvaluationTestCase, EvaluationConfig



# would expect we need to pass config object through if that has metrics configuration
def evaluate_and_output_results(
    _output_dir: Path, 
    evaluation_data_path: Path, 
    evaluation_config: EvaluationConfig, 
    deepeval_evaluate_params: dict
):
    """
    Function to run the evaluation, aggregate the results, and export them to files.

    Args:
        _output_dir: The directory to save the evaluation results.
        evaluation_data_path: Path to the JSONL file containing the evaluation data.
        evaluation_config: Configuration for the evaluation.
        deepeval_evaluate_params: Parameters to be passed to deepeval evaluation.
    """
    # set DeepEval results folder
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = str(_output_dir)

    # convert raw data into model instances for evaluation
    models = jsonl_to_models(evaluation_data_path, EvaluationTestCase)

    # initialise EvaluationResults with the config and the cases
    evaluation_results = EvaluationResults(
        evaluation_config=evaluation_config,
        cases=[model.to_llm_test_case() for model in models]
    )

    # run the DeepEval evaluation
    evaluation_results.run_deepeval_evaluation(**deepeval_evaluate_params)

    # initialise AggregatedResults to aggregate the results
    aggregation = AggregatedResults(evaluation_results.evaluation_results)

    # calculate aggregated results and exports results to CSV files
    aggregation.export_to_csvs(_output_dir, prefix="alessia_test")

    print("Evaluation Results:")
    print(aggregation.summary)


class EvaluationResults:
    def __init__(self, evaluation_config: EvaluationConfig, cases: list[LLMTestCase]):
        self.evaluation_config = evaluation_config
        self.metrics = evaluation_config.get_metric_instances()
        self.llm_judge = evaluation_config.llm_judge_instance
        self.n_runs = evaluation_config.n_runs
        self.cases = cases
        self.evaluation_results : list[TestResult] = []
        
    def run_deepeval_evaluation(self, **kwargs) -> None:
        """"
        Run the Deepval evaluation on the given models and metrics

        Args:
            cases : List of test cases to evaluate
            metrics : List of metrics to use for evaluation
            n_runs : Number of runs to perform for the evaluation
            **kwargs: Additional arguments to pass to the deepeval.evaluation function

        Returns:
            List of evaluation results for each model

        """

        with print_task_duration("Running DeepEval Evaluation"):
            print("Running DeepEval evaluation")

            all_evaluation_runs = []

            for i in range(self.n_runs):
                print(f"Running evaluation iteration {i+1}/{self.n_runs}...")

                evaluation_run = deepeval_evaluate(
                    test_cases=cast(list[LLMTestCase | MLLMTestCase], self.cases), # hint to type checker: this is a List[LLMTestCase | MLLMTestCase], even though it's just List[LLMTestCase]
                    metrics=self.metrics, # type: ignore
                    **kwargs,  # pass additional arguments dynamically 
                )

                all_evaluation_runs.append(evaluation_run.test_results)  # Store results per run

        print("DeepEval evaluation complete")
        
        self.evaluation_results = [results for run in all_evaluation_runs for results in run]

        return None


    def get_results(self):
        """Return the computed evaluation results."""
        return self.evaluation_results
    
    @staticmethod
    def _serialise_eval_result(result: TestResult) -> dict:
        result_dict = asdict(result)
        if "metrics_data" in result_dict:
            result_dict["metrics_data"] = [
                metric.model_dump() if isinstance(metric, BaseModel) else metric
                for metric in result_dict["metrics_data"]
            ]
        return result_dict
    
    def to_serialised_dicts(self) -> list[dict[str, Any]]:
        return [self._serialise_eval_result(result) for result in self.evaluation_results]



class AggregatedResults:
    def __init__(
        self,
        evaluation_results: list
    ):
        """
        Args:
            evaluation_results: List of TestResult objects.
        """
        self.evaluation_results = evaluation_results

    @cached_property
    def flattened_results(self) -> list[dict]:
        """
        Flattens TestResults into a list of dictionaries, one per metric evaluation.

        Example:
            Input:
                [
                    TestResult(name="test_case_1", input="input_1", metrics_data=[...]),
                    TestResult(name="test_case_2", input="input_2", metrics_data=[...]),
                    ...
                ]
            Output:
                [
                    {"name": "test_case_1", "input": "input_1", "metric": "metric_1", "score": 0.9, ...},
                    {"name": "test_case_1", "input": "input_1", "metric": "metric_2", "score": 0.8, ...},
                    ...
                ]
        """
        results = []

        for result in self.evaluation_results:
            input_value = result.input
            for metric in result.metrics_data:
                entry = {
                    "name": result.name,
                    "input": input_value,
                    "metric": metric.name,
                    "score": metric.score,
                    "cost": metric.evaluation_cost,
                    "reason": metric.reason,
                    "rag_answer": result.actual_output,
                    "ideal_answer": result.expected_output,
                }

                results.append(entry)

        return results

    @cached_property
    def per_input_metric_averages(self) -> pd.DataFrame:
        """
        Computes average metric scores per test input.

        Returns:
            DataFrame with rows as test names and columns as metrics.
        """
        df = pd.DataFrame(self.flattened_results)
        return df.groupby(["name", "input", "metric"])["score"].mean().unstack()

    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics across all inputs: median, mean, std per metric.

        Returns:
            DataFrame with metric as index and stats as columns.
        """
        return pd.DataFrame({
            "median": self.per_input_metric_averages.median(),
            "mean": self.per_input_metric_averages.mean(),
            "std": self.per_input_metric_averages.std(),
        })

    def export_to_csvs(self, _output_dir: Path, prefix: str = "metrics") -> None:
        """
        Exports per-input and summary metric statistics to CSV files.

        Args:
            prefix: Prefix for output file names.
        """
        pd.DataFrame(self.flattened_results).to_csv(_output_dir / f"{prefix}_tidy_results.csv")
        self.per_input_metric_averages.to_csv(_output_dir / f"{prefix}_per_input.csv")
        self.summary.to_csv(_output_dir / f"{prefix}_summary.csv")
