import json
import pytest
from click.testing import CliRunner

from govuk_chat_evaluation.rag_answers.cli import main
from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase,
    StructuredContext,
    EvaluationResult,
    RunMetricOutput,
)


# ─── Fixtures


@pytest.fixture(autouse=True)
def freeze_time_for_all_tests(freezer):
    freezer.move_to("2024-11-11 12:34:56")


@pytest.fixture(autouse=True)
def mock_input_data(tmp_path):
    """Write a valid JSONL input file to use in tests"""
    data = {
        "question": "What is VAT?",
        "llm_answer": "VAT is a tax.",
        "ideal_answer": "VAT is value-added tax.",
        "retrieved_context": [
            {
                "title": "VAT",
                "heading_hierarchy": ["Tax", "VAT"],
                "description": "VAT overview",
                "html_content": "<p>Some HTML about VAT</p>",
                "exact_path": "https://gov.uk/vat",
                "base_path": "https://gov.uk",
            }
        ],
    }

    file_path = tmp_path / "mock_input.jsonl"
    with open(file_path, "w") as f:
        f.write(json.dumps(data) + "\n")

    return file_path


@pytest.fixture(autouse=True)
def mock_data_generation(mocker):
    structured_context = StructuredContext(
        title="VAT",
        heading_hierarchy=["Tax", "VAT"],
        description="VAT overview",
        html_content="<p>Some HTML about VAT</p>",
        exact_path="https://gov.uk/vat",
        base_path="https://gov.uk",
    )

    return_value = [
        EvaluationTestCase(
            question="Question",
            ideal_answer="An answer",
            llm_answer="An answer",
            retrieved_context=[structured_context],
        )
    ] * 2

    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.generate.generate_inputs_to_evaluation_test_cases",
        return_value=return_value,
    )


@pytest.fixture(autouse=True)
def mock_run_rake_task(mocker):
    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.generate.run_rake_task",
        return_value={
            "message": "This is a mocked response",
            "retrieved_context": [
                {
                    "title": "Mock Title",
                    "heading_hierarchy": ["Mock", "Hierarchy"],
                    "description": "Mock description",
                    "html_content": "<p>Mock HTML</p>",
                    "exact_path": "https://gov.uk/mock-path",
                    "base_path": "https://gov.uk",
                }
            ],
        },
    )


@pytest.fixture(autouse=True)
def fake_evaluation_results():
    return [
        EvaluationResult(
            name="test_case_1",
            input="What is the capital of France?",
            actual_output="Paris",
            expected_output="Paris",
            retrieval_context=["France is a country in Europe."],
            run_metric_outputs=[
                RunMetricOutput(
                    run=0,
                    metric="mock_metric",
                    score=1.0,
                    reason="Perfect match",
                    cost=0.0,
                    success=True,
                )
            ],
        )
    ]


@pytest.fixture(autouse=True)
def mock_convert_deepeval_output_to_evaluation_results(mocker):
    """Bypass real conversion logic with a mock EvaluationResult"""
    mock_run_metric_output = RunMetricOutput(
        run=0,
        metric="mock_metric",
        score=1.0,
        reason="Perfect match",
        cost=0.0,
        success=True,
    )

    mock_evaluation_result = EvaluationResult(
        name="input_1",
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris",
        retrieval_context=["France is a country in Europe."],
        run_metric_outputs=[mock_run_metric_output],
    )

    mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.convert_deepeval_output_to_evaluation_results",
        return_value=[mock_evaluation_result],
    )


@pytest.fixture(autouse=True)
def mock_deepeval_evaluate(mocker):
    """Mock the function that runs actual DeepEval evaluation"""
    mock_result = mocker.MagicMock()
    mock_result.test_results = [
        EvaluationResult(
            name="test_case_1",
            input="What is the capital of France?",
            actual_output="Paris",
            expected_output="Paris",
            retrieval_context=["France is a country in Europe."],
            run_metric_outputs=[
                RunMetricOutput(
                    run=0,
                    metric="mock_metric",
                    score=1.0,
                    reason="Perfect match",
                    cost=0.0,
                    success=True,
                )
            ],
        )
    ]
    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.run_deepeval_evaluation",
        return_value=mock_result.test_results,
    )


@pytest.fixture
def mock_output_directory(mock_project_root):
    return mock_project_root / "results" / "rag_answers" / "2024-11-11T12:34:56"


# ─── Main CLI Tests


@pytest.mark.usefixtures("mock_output_directory")
def test_main_creates_output_files(
    mock_output_directory, mock_config_file, mocker, fake_evaluation_results
):
    # Mock the evaluation process to avoid OpenAI API calls
    mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.run_deepeval_evaluation",
        return_value=[],
    )
    mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.convert_deepeval_output_to_evaluation_results",
        return_value=fake_evaluation_results,
    )

    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    assert result.exit_code == 0, result.output
    result_summary = mock_output_directory / "results_summary.csv"
    assert result_summary.exists()


def test_main_generates_results(
    mock_output_directory, mock_config_file, mock_data_generation, mocker
):
    mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.run_deepeval_evaluation",
        return_value=[],
    )

    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--generate"])  # type: ignore[arg-type]

    generated_file = mock_output_directory / "generated.jsonl"

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_called_once()
    assert generated_file.exists()


@pytest.mark.usefixtures("mock_output_directory")
def test_main_doesnt_generate_results(mock_config_file, mock_data_generation, mocker):
    # Mock any OpenAI dependent functions
    mocker.patch(
        "govuk_chat_evaluation.rag_answers.evaluate.run_deepeval_evaluation",
        return_value=[],
    )

    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--no-generate"])

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_not_called()
