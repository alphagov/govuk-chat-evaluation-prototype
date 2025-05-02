import json
from unittest.mock import AsyncMock, ANY

import pytest

from govuk_chat_evaluation.dataset_generation import run_rake_task, generate_dataset


@pytest.mark.asyncio
async def test_run_rake_task_success(mocker):
    mock_subprocess_exec = mocker.patch("asyncio.create_subprocess_exec")
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        json.dumps({"result": "success"}).encode(),
        b"",
    )
    mock_process.returncode = 0
    mock_subprocess_exec.return_value = mock_process

    result = await run_rake_task("task_name")

    mock_subprocess_exec.assert_called_once_with(
        "bundle",
        "exec",
        "rake",
        "task_name",
        cwd=ANY,
        env=ANY,
        stdout=ANY,
        stderr=ANY,
    )

    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_run_rake_task_accepts_extra_env_vars(mocker):
    mock_subprocess_exec = mocker.patch("asyncio.create_subprocess_exec")
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        json.dumps({"result": "success"}).encode(),
        b"",
    )
    mock_process.returncode = 0
    mock_subprocess_exec.return_value = mock_process

    await run_rake_task("task_name", env_vars={"EXAMPLE": "env var"})

    mock_subprocess_exec.assert_called_once()
    env_args = mock_subprocess_exec.call_args[1]["env"]

    assert env_args["EXAMPLE"] == "env var"


@pytest.mark.asyncio
async def test_run_rake_task_failure(mocker):
    mock_subprocess_exec = mocker.patch("asyncio.create_subprocess_exec")
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Error occurred")
    mock_process.returncode = 1
    mock_subprocess_exec.return_value = mock_process

    with pytest.raises(RuntimeError) as exc_info:
        await run_rake_task("task_name")

    assert "Failed to successfully run the rake task" in str(exc_info.value)
    assert "Error occurred" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_dataset():
    async def mock_generation_func(item):
        if item == "question2":
            return None
        else:
            return {"input": item, "output": f"generated-{item}"}

    ground_truth = ["question1", "question2", "question3"]
    result = await generate_dataset(ground_truth, mock_generation_func)

    expected_result = [
        {"input": "question1", "output": "generated-question1"},
        {"input": "question3", "output": "generated-question3"},
    ]

    # async results don't return in a predictable order
    sorted_result = sorted(result, key=lambda x: x["input"])

    assert sorted_result == expected_result


@pytest.mark.asyncio
async def test_generate_dataset_failure_raises_error():
    async def mock_generation_func(item):
        if item == "fail":
            raise RuntimeError("Contrived failure")
        return {"input": item, "output": f"generated-{item}"}

    ground_truth = ["question1", "fail", "question3"]

    with pytest.raises(RuntimeError, match="Contrived failure"):
        await generate_dataset(ground_truth, mock_generation_func)
