import asyncio
from pathlib import Path

from pydantic import BaseModel

from .evaluate import EvaluationResult
from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output


class GenerateInput(BaseModel):
    question: str
    expected_exact_paths: list[str]


def generate_and_write_dataset(
    input_path: Path, embedding_provider: str, output_dir: Path
):
    models = jsonl_to_models(Path(input_path), GenerateInput)
    generated = generate_inputs_to_evaluation_results(embedding_provider, models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_results(
    embedding_provider: str, generate_inputs: list[GenerateInput]
) -> list[EvaluationResult]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate a result"""

    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question, "EMBEDDING_PROVIDER": embedding_provider}

        result = await run_rake_task(
            "evaluation:search_results_for_question",
            env,
        )
        exact_paths_and_scores = [
            (item["exact_path"], item["weighted_score"]) for item in result
        ]

        return EvaluationResult(
            question=input.question,
            expected_exact_paths=input.expected_exact_paths,
            actual_exact_paths_and_scores=exact_paths_and_scores,
        )

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
