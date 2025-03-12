import asyncio
from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output
from .models import GenerateInput, EvaluateInput
from pathlib import Path
from typing import List


def generate_and_write_dataset(input_path: str, provider: str, output_dir: Path):
    models = jsonl_to_models(input_path, GenerateInput)
    generated = generate_models_to_evaluate_models(provider, models)
    return write_generated_to_output(generated, output_dir)


def generate_models_to_evaluate_models(
    provider: str, models: List[GenerateInput]
) -> List[EvaluateInput]:
    async def generate_to_evaluate(input: GenerateInput):
        env = {"INPUT": input.question}
        result = await run_rake_task(
            f"evaluation:generate_jailbreak_guardrail_response[{provider}]",
            env,
        )
        return EvaluateInput(
            question=input.question,
            expected_outcome=input.expected_outcome,
            actual_outcome=result["triggered"],
        )

    return asyncio.run(generate_dataset(models, generate_to_evaluate))
