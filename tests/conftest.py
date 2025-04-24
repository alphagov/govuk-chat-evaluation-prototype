import csv
from pathlib import Path

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_project_root(mocker, tmp_path):
    """use tmp_path fixture for files created in test execution"""
    mocker.patch(
        "govuk_chat_evaluation.file_system.project_root", return_value=tmp_path
    )
    return tmp_path


@pytest.fixture(autouse=True)
def mock_openai_dependencies(monkeypatch):
    # Mock the OpenAI client class
    mock_openai = MagicMock()
    mock_openai_instance = MagicMock()
    mock_openai.return_value = mock_openai_instance

    # Mock ChatOpenAI
    mock_chat_openai = MagicMock()
    mock_chat_openai_instance = MagicMock()
    mock_chat_openai.return_value = mock_chat_openai_instance

    # Apply mocks
    monkeypatch.setattr("openai.OpenAI", mock_openai)
    monkeypatch.setattr("langchain_openai.chat_models.ChatOpenAI", mock_chat_openai)

    # Mock GPTModel
    mock_gpt_model = MagicMock()
    monkeypatch.setattr("deepeval.models.llms.openai_model.GPTModel", mock_gpt_model)

    # Set a fake API key
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key-for-testing")


def assert_csv_exists_with_headers(file_path: Path, *expected_headers: str):
    assert file_path.exists()
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        assert headers is not None, f"No headers found in {file_path}"

        missing = [h for h in expected_headers if h not in headers]
        assert not missing, f"Missing headers in {file_path}: {missing}"
