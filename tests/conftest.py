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
