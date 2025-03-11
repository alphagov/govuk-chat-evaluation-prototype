import asyncio
from govuk_chat_evaluation.generate import generate_dataset, run_rake_task
from .models import GenerateInput, EvaluateInput
from typing import List


def generate_jailbreak_guardrails_dataset(
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
