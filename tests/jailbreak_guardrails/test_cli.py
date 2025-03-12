import pytest
from govuk_chat_evaluation.jailbreak_guardrails.cli import main
from govuk_chat_evaluation.jailbreak_guardrails.models import EvaluateInput
from govuk_chat_evaluation.file_system import project_root
from pathlib import Path


@pytest.fixture(autouse=True)
def mock_project_root(mocker, tmp_path):
    """use tmp_path fixture for files created in test execution"""
    mocker.patch(
        "govuk_chat_evaluation.file_system.project_root", return_value=tmp_path
    )
    # ensure function is mocked for tests too
    mocker.patch(__name__ + ".project_root", return_value=tmp_path)


@pytest.fixture(autouse=True)
def freeze_time_for_all_tests(freezer):
    """Automatically freeze time for all tests in this file."""
    freezer.move_to("2024-11-11 12:34:56")


@pytest.fixture(autouse=True)
def mock_data_generation(mocker):
    return_value = [
        EvaluateInput(question="Question", expected_outcome=True, actual_outcome=True),
        EvaluateInput(
            question="Question", expected_outcome=False, actual_outcome=False
        ),
    ]

    return mocker.patch(
        "govuk_chat_evaluation.jailbreak_guardrails.generate.generate_models_to_evaluate_models",
        return_value=return_value,
    )


def output_directory() -> Path:
    return project_root() / "results" / "jailbreak_guardrails" / "2024-11-11T12:34:56"


def test_main_creates_output_directory():
    main()

    assert output_directory().exists()
    assert output_directory().is_dir()


# TODO: have different tests to check generate and not
def test_main_generates_results(mock_data_generation):
    main()

    generated_file = output_directory() / "generated.jsonl"

    mock_data_generation.assert_called_once()
    assert generated_file.exists()


# TODO: test we write output files
# def test_main_writes_results(): ...
