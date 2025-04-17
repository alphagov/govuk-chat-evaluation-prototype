import json
from unittest.mock import AsyncMock

import pytest

from govuk_chat_evaluation.rag_answers.generate import (
    generate_inputs_to_evaluation_test_cases,
    generate_and_write_dataset,
    GenerateInput,
    EvaluationTestCase,
)


@pytest.fixture
def run_rake_task_mock(mocker):
    mock = mocker.patch(
        "govuk_chat_evaluation.rag_answers.generate.run_rake_task",
        new_callable=AsyncMock,
    )
    mock.side_effect = lambda *_: {"message": "An answer"}
    return mock


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_models_to_evaluation_test_cases_returns_evaluation_test_cases():
    generate_inputs = [
        GenerateInput(question="Question 1", ideal_answer="Answer 1"),
        GenerateInput(question="Question 2", ideal_answer="Answer 2"),
    ]
    expected_results = [
        EvaluationTestCase(
            question="Question 1",
            ideal_answer="Answer 1",
            llm_answer="An answer",
            retrieved_context=[],
        ),
        EvaluationTestCase(
            question="Question 2",
            ideal_answer="Answer 2",
            llm_answer="An answer",
            retrieved_context=[],
        ),
    ]
    actual_results = generate_inputs_to_evaluation_test_cases("openai", generate_inputs)

    assert sorted(expected_results, key=lambda r: r.question) == sorted(
        actual_results, key=lambda r: r.question
    )


def test_generate_models_to_evaluation_test_cases_runs_expected_rake_task(
    run_rake_task_mock,
):
    run_rake_task_mock.side_effect = lambda *_: {"message": "An answer"}
    generate_inputs = [
        GenerateInput(question="Question 1", ideal_answer="Answer"),
    ]
    generate_inputs_to_evaluation_test_cases("openai", generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_rag_structured_answer_response[openai]",
        {"INPUT": "Question 1"},
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(mock_input_data, "openai", mock_project_root)
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            assert json.loads(line)
