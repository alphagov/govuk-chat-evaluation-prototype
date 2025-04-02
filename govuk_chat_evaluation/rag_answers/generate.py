import asyncio
from pathlib import Path

from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output
from .data_models import GenerateInput, EvaluationTestCase


def generate_and_write_dataset(input_path: Path, output_dir: Path):
    models = jsonl_to_models(Path(input_path), GenerateInput)
    generated = generate_inputs_to_evaluation_test_cases(models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_test_cases(
    generate_inputs: list[GenerateInput],
) -> list[EvaluationTestCase]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate a result"""

    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question}
        result = await run_rake_task(
            f"evaluation:generate_jailbreak_guardrail_response[{provider}]",
            env,
        )

        return input.to_evaluation_test_case()

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
