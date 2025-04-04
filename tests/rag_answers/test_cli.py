import os

import pytest
import yaml
from click.testing import CliRunner

from govuk_chat_evaluation.rag_answers.cli import main, Config
from govuk_chat_evaluation.rag_answers.data_models import EvaluationTestCase


class TestConfig:
    def test_config_requires_provider_for_generate(self, mock_input_data):
        with pytest.raises(ValueError, match="provider is required to generate data"):
            Config(
                what="Test",
                generate=True,
                provider=None,
                input_path=mock_input_data,
            )

        Config(
            what="Test",
            generate=False,
            provider=None,
            input_path=mock_input_data,
        )

        Config(
            what="Test",
            generate=True,
            provider="openai",
            input_path=mock_input_data,
        )


@pytest.fixture(autouse=True)
def mock_config_file(tmp_path, mock_input_data):
    """Write a config file as an input for testing"""
    data = {
        "what": "Testing RAG Answer evaluations",
        "generate": True,
        "provider": "openai",
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
        EvaluationTestCase(
            question="Question", ideal_answer="An answer", llm_answer="An answer"
        ),
        EvaluationTestCase(
            question="Question", ideal_answer="An answer", llm_answer="An answer"
        ),
    ]

    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.generate.generate_inputs_to_evaluation_test_cases",
        return_value=return_value,
    )


@pytest.fixture(autouse=True)
def mock_deepeval_evaluate(mocker):
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "mock"})

    mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.deepeval_evaluate",
        return_value=None,
    )


@pytest.fixture
def mock_output_directory(mock_project_root):
    return mock_project_root / "results" / "rag_answers" / "2024-11-11T12:34:56"


@pytest.mark.usefixtures("mock_data_generation")
def test_main_creates_output_files(mock_output_directory, mock_config_file):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    config_file = mock_output_directory / "config.yaml"

    assert result.exit_code == 0, result.output
    assert mock_output_directory.exists()
    assert config_file.exists()


def test_main_generates_results(
    mock_output_directory, mock_config_file, mock_data_generation
):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--generate"])

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
