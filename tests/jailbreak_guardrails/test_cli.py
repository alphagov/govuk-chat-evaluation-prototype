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
    }
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as file:
        yaml.dump(data, file)

    yield str(file_path)


@pytest.fixture
def mock_output_directory(mock_project_root):
    return (
        mock_project_root / "results" / "jailbreak_guardrails" / "2024-11-11T12:34:56"
    )


def test_main_creates_output_files(mock_output_directory, mock_config_file):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    config_file = mock_output_directory / "config.yaml"

    assert result.exit_code == 0, result.output
    assert mock_output_directory.exists()
    assert config_file.exists()
