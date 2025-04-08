import os
from pathlib import Path
from typing import cast, Any
from dataclasses import asdict
from functools import cached_property
from pydantic import BaseModel
import pandas as pd

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import BaseMetric

from .deepeval_evaluate import run_deepeval_evaluation, convert_deepeval_output_to_evaluation_results
from ..file_system import jsonl_to_models
from .data_models import EvaluationTestCase, EvaluationConfig, EvaluationResult



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

    evaluation_outputs = run_deepeval_evaluation(
        cases=[model.to_llm_test_case() for model in models],
        metrics=cast(list[BaseMetric], evaluation_config.metrics),
        n_runs=evaluation_config.n_runs,
        **deepeval_evaluate_params
    )

    # convert the results from DeepEval into a more structured format
    evaluation_results = convert_deepeval_output_to_evaluation_results(evaluation_outputs)
    
    # initialise AggregatedResults to aggregate the results
    aggregation = AggregatedResults(evaluation_results)  # type: ignore

    # calculate aggregated results and exports results to CSV files
    aggregation.export_to_csvs(_output_dir, prefix="alessia_test")

    print("Evaluation Results:")
    print(aggregation.summary)



class AggregatedResults:
    def __init__(
        self,
        evaluation_results: list[EvaluationResult]
    ):
        """
        Args:
            evaluation_results: List of TestResult objects.
        """
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

        for eval_result in self.evaluation_results:
            for evaluation_output in eval_result.evaluation_results:
                data.append({
                    "name": eval_result.name,
                    "input": eval_result.input,
                    "metric": evaluation_output.metric,
                    "score": evaluation_output.score,
                })

        df = pd.DataFrame(data)

        # Aggregate by input and metric, computing both mean and std score
        return df.groupby(["name", "input", "metric"])["score"].agg(['mean', 'std']).unstack().reset_index()

    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics across all inputs: median, mean, std per metric.

        Returns:
            DataFrame with metric as index and stats as columns.
        """

        mean_df = self.per_input_metric_averages["mean"]

        return pd.DataFrame({
            "median": mean_df.median(),
            "mean": mean_df.mean(),
            "std": mean_df.std(),
        })

    def export_to_csvs(self, _output_dir: Path, prefix: str = "metrics") -> None:
        """
        Exports per-input and summary metric statistics to CSV files.

        Args:
            prefix: Prefix for output file names.
        """
        pd.DataFrame(self.evaluation_results).to_csv(_output_dir / f"{prefix}_tidy_results.csv")
        self.per_input_metric_averages.to_csv(_output_dir / f"{prefix}_per_input.csv")
        self.summary.to_csv(_output_dir / f"{prefix}_summary.csv")
