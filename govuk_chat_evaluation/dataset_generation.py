import asyncio
import json
import os
from tqdm.asyncio import tqdm
from pathlib import Path


async def run_rake_task(task_name: str, extra_env: dict[str, str] | None = None):
    govuk_chat_dir = Path.home() / "govuk" / "govuk-chat"

    env = {**os.environ.copy(), **(extra_env or {})}

    process = await asyncio.create_subprocess_exec(
        "bundle",
        "exec",
        "spring",
        "rake",
        task_name,
        cwd=govuk_chat_dir,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise Exception("Failed to generate a jailbreak response", stderr.decode())

    return json.loads(stdout.decode())


async def generate_dataset(ground_truth, generator_func):
    semaphore = asyncio.Semaphore(10)

    async def run_evaluation(item, semaphore):
        async with semaphore:
            return await generator_func(item)

    tasks = [
        asyncio.create_task(run_evaluation(item, semaphore)) for item in ground_truth
    ]
    evaluations = []

    print("Generating dataset")
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
