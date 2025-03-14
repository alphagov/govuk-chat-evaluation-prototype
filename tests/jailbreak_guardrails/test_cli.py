import sys
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from govuk_chat_evaluation.jailbreak_guardrails.cli import main
from govuk_chat_evaluation.jailbreak_guardrails.evaluate import EvaluationResult
from govuk_chat_evaluation.file_system import project_root


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
def mock_config_file(tmp_path):
    """Write a config file as an input for testing"""
    data = {
        "what": "Testing Jailbreak Guardrail evaluations",
        "generate": False,
        "input_path": str(tmp_path / "input_data.jsonl"),
    }
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as file:
        yaml.dump(data, file)

    yield str(file_path)


@pytest.fixture(autouse=True)
def input_data(tmp_path):
    data = [
        {"question": "Question 1", "expected_outcome": True, "actual_outcome": True},
        {"question": "Question 2", "expected_outcome": False, "actual_outcome": True},
    ]

    with open(tmp_path / "input_data.jsonl", "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")


@pytest.fixture(autouse=True)
def mock_data_generation(mocker):
    return_value = [
        EvaluationResult(
            question="Question", expected_outcome=True, actual_outcome=True
        ),
        EvaluationResult(
            question="Question", expected_outcome=False, actual_outcome=False
        ),
    ]

    return mocker.patch(
        "govuk_chat_evaluation.jailbreak_guardrails.generate.generate_models_to_evaluation_models",
        return_value=return_value,
    )


def output_directory() -> Path:
    return project_root() / "results" / "jailbreak_guardrails" / "2024-11-11T12:34:56"


def test_main_creates_output_files(mock_config_file):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    results_file = output_directory() / "results.csv"
    aggregate_file = output_directory() / "aggregate.csv"
    config_file = output_directory() / "config.yaml"

    assert result.exit_code == 0, result.output
    assert output_directory().exists()
    assert results_file.exists()
    assert aggregate_file.exists()
    assert config_file.exists()


def test_main_generates_results(mock_config_file, mock_data_generation):
    runner = CliRunner()
    result = runner.invoke(
        main, [mock_config_file, "--generate", "--provider", "claude"]
    )

    generated_file = output_directory() / "generated.jsonl"

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_called_once()
    assert generated_file.exists()


def test_main_doesnt_generate_results(mock_config_file, mock_data_generation):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--no-generate"])

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_not_called()
