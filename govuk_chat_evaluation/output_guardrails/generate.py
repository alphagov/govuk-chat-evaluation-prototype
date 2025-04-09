import asyncio
from pathlib import Path

from pydantic import BaseModel

from .evaluate import EvaluationResult
from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output


class GenerateInput(BaseModel):
    question: str
    expected_outcome: bool


def generate_and_write_dataset(
    input_path: Path, provider: str, guardrail_type: str, output_dir: Path
):
    models = jsonl_to_models(input_path, GenerateInput)
    generated = generate_inputs_to_evaluation_results(provider, guardrail_type, models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_results(
    provider: str, guardrail_type: str, generate_inputs: list[GenerateInput]
) -> list[EvaluationResult]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate a result"""

    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question}
        result = await run_rake_task(
            f"evaluation:generate_output_guardrail_response[{provider},{guardrail_type}]",
            env,
        )

        return EvaluationResult(
            question=input.question,
            expected_outcome=input.expected_outcome,
            actual_outcome=result["triggered"],
        )

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
