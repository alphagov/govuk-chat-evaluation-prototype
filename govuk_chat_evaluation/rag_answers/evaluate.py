import os
from pathlib import Path
from typing import cast
from functools import cached_property
import pandas as pd

from deepeval.metrics import BaseMetric

from .deepeval_evaluate import (
    run_deepeval_evaluation,
    convert_deepeval_output_to_evaluation_results,
)
from ..file_system import jsonl_to_models
from .data_models import EvaluationTestCase, Config, EvaluationResult


DEEPEVAL_EVAL_PARAMETERS = {
    "print_results": False,
    "show_indicator": True,
    "max_concurrent": 40,
    "throttle_value": 5,
    "use_cache": True,
    "ignore_errors": True,
}


# would expect we need to pass config object through if that has metrics configuration
def evaluate_and_output_results(
    output_dir: Path, evaluation_data_path: Path, evaluation_config: Config
):
    """
    Function to run the evaluation, aggregate the results, and export them to files.

    Args:
        output_dir: The directory to save the evaluation results.
        evaluation_data_path: Path to the JSONL file containing the evaluation data.
        evaluation_config: Configuration for the evaluation.
    """
    # set DeepEval results folder
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = str(output_dir)

    models = jsonl_to_models(evaluation_data_path, EvaluationTestCase)

    evaluation_outputs = run_deepeval_evaluation(
        cases=[model.to_llm_test_case() for model in models],
        metrics=cast(list[BaseMetric], evaluation_config.metric_instances()),
        n_runs=evaluation_config.n_runs,
        **DEEPEVAL_EVAL_PARAMETERS,
    )

    evaluation_results = convert_deepeval_output_to_evaluation_results(
        evaluation_outputs
    )

    aggregation = AggregatedResults(evaluation_results)

    # calculate aggregated results and exports results to CSV files
    aggregation.export_to_csvs(output_dir)

    print("Evaluation Results:")
    print(aggregation.summary)


class AggregatedResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results

    @cached_property
    def per_input_metric_averages(self) -> pd.DataFrame:
        """
        Computes average metric scores per test input.

        Returns:
            DataFrame with rows as test names and columns as metrics.
        """
        # flatten the evaluation results
        data = []

        for eval_result in self.evaluation_results or []:
            for evaluation_output in eval_result.run_metric_outputs:
                data.append(
                    {
                        "name": eval_result.name,
                        "input": eval_result.input,
                        "metric": evaluation_output.metric,
                        "score": evaluation_output.score,
                    }
                )

        df = pd.DataFrame(data)

        return (
            df.groupby(["name", "input", "metric"])["score"]
            .agg(["mean", "std"])
            .unstack()
            .reset_index()
        )

    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics across all inputs: median, mean, std per metric.

        Returns:
            DataFrame with metric as index and stats as columns.
        """

        mean_df = self.per_input_metric_averages["mean"]

        return pd.DataFrame(
            {
                "median": mean_df.median(),
                "mean": mean_df.mean(),
                "std": mean_df.std(),
            }
        )

    def export_to_csvs(self, output_dir: Path) -> None:
        """
        Exports per-input and summary metric statistics to CSV files.
        """
        pd.DataFrame(self.evaluation_results).to_csv(output_dir / "tidy_results.csv")
        self.per_input_metric_averages.to_csv(output_dir / "results_per_input.csv")
        self.summary.to_csv(output_dir / "results_summary.csv")
