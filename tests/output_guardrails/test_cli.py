import pytest
import yaml
from click.testing import CliRunner

from govuk_chat_evaluation.output_guardrails.cli import main, Config
from govuk_chat_evaluation.output_guardrails.evaluate import EvaluationResult


class TestConfig:
    def test_config_requires_provider_for_generate(self, mock_input_data):
        with pytest.raises(ValueError, match="provider is required to generate data"):
            Config(
                what="Test",
                guardrail_type="answer_guardrails",
                generate=True,
                provider=None,
                input_path=mock_input_data,
            )

        Config(
            what="Test",
            guardrail_type="answer_guardrails",
            generate=False,
            provider=None,
            input_path=mock_input_data,
        )

        Config(
            what="Test",
            guardrail_type="answer_guardrails",
            generate=True,
            provider="openai",
            input_path=mock_input_data,
        )


@pytest.fixture(autouse=True)
def mock_config_file(tmp_path, mock_input_data):
    """Write a config file as an input for testing"""
    data = {
        "what": "Testing Output Guardrail evaluations",
        "guardrail_type": "answer_guardrails",
        "generate": True,
        "provider": "claude",
        "input_path": str(mock_input_data),
    }
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as file:
        yaml.dump(data, file)

    yield str(file_path)


@pytest.fixture(autouse=True)
def freeze_time_for_all_tests(freezer):
    """Automatically freeze time for all tests in this file."""
    freezer.move_to("2024-11-11 12:34:56")


@pytest.fixture
def mock_data_generation(mocker):
    return_value = [
        EvaluationResult(
            question="Question",
            expected_triggered=True,
            actual_triggered=True,
            expected_exact="True | None",
            actual_exact="True | None",
        ),
        EvaluationResult(
            question="Question",
            expected_triggered=False,
            actual_triggered=False,
            expected_exact="False | None",
            actual_exact="False | None",
        ),
    ]

    return mocker.patch(
        "govuk_chat_evaluation.output_guardrails.generate.generate_inputs_to_evaluation_results",
        return_value=return_value,
    )


@pytest.fixture
def mock_output_directory(mock_project_root):
    return mock_project_root / "results" / "output_guardrails" / "2024-11-11T12:34:56"


def test_main_creates_output_files(
    mock_output_directory, mock_config_file, mock_data_generation
):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    config_file = mock_output_directory / "config.yaml"
    results_file = mock_output_directory / "results.csv"
    aggregate_file = mock_output_directory / "aggregate.csv"

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_called_once()
    assert mock_output_directory.exists()
    assert results_file.exists()
    assert aggregate_file.exists()
    assert config_file.exists()


def test_main_generates_results(
    mock_output_directory, mock_config_file, mock_data_generation
):
    runner = CliRunner()
    result = runner.invoke(
        main, [mock_config_file, "--generate", "--provider", "claude"]
    )

    generated_file = mock_output_directory / "generated.jsonl"

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_called_once()
    assert generated_file.exists()


@pytest.mark.usefixtures("mock_output_directory")
def test_main_doesnt_generate_results(mock_config_file, mock_data_generation):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--no-generate"])

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_not_called()
