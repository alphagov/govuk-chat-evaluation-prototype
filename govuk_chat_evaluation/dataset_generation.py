import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

from tqdm.asyncio import tqdm
import logging


async def run_rake_task(task_name: str, env_vars: dict[str, str] | None = None) -> Any:
    """Asynchronously run a rake task on the GOV.UK Chat project expected to be
    running locally. Raises an error if it returns a non 0 return code"""

    govuk_chat_dir = Path.home() / "govuk" / "govuk-chat"

    env = {**os.environ.copy(), **(env_vars or {})}

    process = await asyncio.create_subprocess_exec(
        "bundle",
        "exec",
        "rake",
        task_name,
        cwd=govuk_chat_dir,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError("Failed to successfully run the rake task", stderr.decode())

    return json.loads(stdout.decode())


async def generate_dataset(
    ground_truth: list[Any], generator_func: Callable[[Any], Awaitable[Any]]
) -> list[Any]:
    """Asynchronously generate data for each item in the ground_truth list by
    calling the generator_func with each item. Outputs a progress bar and
    cancels all jobs if one fails."""

    semaphore = asyncio.Semaphore(10)

    async def run_generation_with_limited_async(item, semaphore):
        async with semaphore:
            return await generator_func(item)

    tasks = [
        asyncio.create_task(run_generation_with_limited_async(item, semaphore))
        for item in ground_truth
    ]
    evaluations = []

    logging.info("Generating dataset")
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
