import json
from unittest.mock import AsyncMock

import pytest

from govuk_chat_evaluation.retrieval.generate import (
    generate_inputs_to_evaluation_results,
    generate_and_write_dataset,
    GenerateInput,
    EvaluationResult,
)


@pytest.fixture
def run_rake_task_mock(mocker):
    async def default_side_effect(_, env):
        if env["INPUT"] == "Question 1":
            return [
                {
                    "exact_path": "/foo",
                    "plain_content": "Content for foo",
                    "weighted_score": 1.0,
                },
                {
                    "exact_path": "/bar",
                    "plain_content": "Content for bar",
                    "weighted_score": 0.8,
                },
                {
                    "exact_path": "/baz",
                    "plain_content": "Content for baz",
                    "weighted_score": 0.5,
                },
            ]
        else:
            return [
                {
                    "exact_path": "/path1",
                    "plain_content": "Content for path1",
                    "weighted_score": 1.0,
                },
                {
                    "exact_path": "/path2",
                    "plain_content": "Content for path2",
                    "weighted_score": 0.9,
                },
            ]

    mock = mocker.patch(
        "govuk_chat_evaluation.retrieval.generate.run_rake_task",
        new_callable=AsyncMock,
    )
    mock.side_effect = default_side_effect
    return mock


def test_generate_inputs_to_evaluation_results_returns_evaluation_results(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_exact_paths=["/foo"],
        ),
        GenerateInput(
            question="Question 2",
            expected_exact_paths=["/path1", "/path2"],
        ),
    ]
    expected_results = [
        EvaluationResult(
            question="Question 1",
            expected_exact_paths=["/foo"],
            actual_exact_paths_and_scores=[("/foo", 1.0), ("/bar", 0.8), ("/baz", 0.5)],
        ),
        EvaluationResult(
            question="Question 2",
            expected_exact_paths=["/path1", "/path2"],
            actual_exact_paths_and_scores=[("/path1", 1.0), ("/path2", 0.9)],
        ),
    ]
    actual_results = generate_inputs_to_evaluation_results("titan", generate_inputs)

    assert sorted(expected_results, key=lambda r: r.question) == sorted(
        actual_results, key=lambda r: r.question
    )


def test_generate_inputs_to_evaluation_results_runs_expected_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_exact_paths=["/foo", "/bar"],
        ),
    ]

    generate_inputs_to_evaluation_results("titan", generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:search_results_for_question",
        {"INPUT": "Question 1", "EMBEDDING_PROVIDER": "titan"},
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(mock_input_data, "titan", mock_project_root)
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            assert json.loads(line)
