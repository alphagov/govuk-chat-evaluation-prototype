import json

import pytest
import yaml
from click.testing import CliRunner

from govuk_chat_evaluation.jailbreak_guardrails.cli import main


@pytest.fixture(autouse=True)
def freeze_time_for_all_tests(freezer):
    """Automatically freeze time for all tests in this file."""
    freezer.move_to("2024-11-11 12:34:56")


@pytest.fixture(autouse=True)
def mock_config_file(tmp_path):
    """Write a config file as an input for testing"""
    data = {
        "what": "Testing Jailbreak Guardrail evaluations",
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


@pytest.fixture
def mock_output_directory(mock_project_root):
    return (
        mock_project_root / "results" / "jailbreak_guardrails" / "2024-11-11T12:34:56"
    )


def test_main_creates_output_files(mock_output_directory, mock_config_file):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    config_file = mock_output_directory / "config.yaml"
    results_file = mock_output_directory / "results.csv"
    aggregate_file = mock_output_directory / "aggregate.csv"

    assert result.exit_code == 0, result.output
    assert mock_output_directory.exists()
    assert results_file.exists()
    assert aggregate_file.exists()
    assert config_file.exists()
