import pytest
from click.testing import CliRunner

from govuk_chat_evaluation.rag_answers.cli import main
from govuk_chat_evaluation.rag_answers.data_models import EvaluationTestCase

# ─── Fixtures


@pytest.fixture(autouse=True)
def freeze_time_for_all_tests(freezer):
    freezer.move_to("2024-11-11 12:34:56")


@pytest.fixture(autouse=True)
def mock_data_generation(mocker):
    return_value = [
        EvaluationTestCase(
            question="Question",
            ideal_answer="An answer",
            llm_answer="An answer",
            retrieved_context=[],
        )
    ] * 2

    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.generate.generate_inputs_to_evaluation_test_cases",
        return_value=return_value,
    )


@pytest.fixture
def mock_output_directory(mock_project_root):
    return mock_project_root / "results" / "rag_answers" / "2024-11-11T12:34:56"


# ─── Main CLI Tests


@pytest.mark.usefixtures("mock_deepeval_evaluate")
def test_main_creates_output_files(mock_output_directory, mock_config_file):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    assert result.exit_code == 0, result.output

    # just one file from the generation since we're already testing all of them
    # elsewhere
    expected_files = ["config.yaml", "results_summary.csv"]

    for filename in expected_files:
        assert (mock_output_directory / filename).exists()


@pytest.mark.usefixtures("mock_deepeval_evaluate")
def test_main_generates_results(
    mock_output_directory, mock_config_file, mock_data_generation, mocker
):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--generate"])  # type: ignore[arg-type]

    generated_file = mock_output_directory / "generated.jsonl"

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_called_once()
    assert generated_file.exists()


@pytest.mark.usefixtures("mock_deepeval_evaluate", "mock_output_directory")
def test_main_doesnt_generate_results(mock_config_file, mock_data_generation):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--no-generate"])

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_not_called()
