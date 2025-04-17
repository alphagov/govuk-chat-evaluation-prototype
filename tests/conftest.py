import pytest


@pytest.fixture
def mock_project_root(mocker, tmp_path):
    """use tmp_path fixture for files created in test execution"""
    mocker.patch(
        "govuk_chat_evaluation.file_system.project_root", return_value=tmp_path
    )
    return tmp_path


@pytest.fixture(autouse=True)
def mock_openai_model(mocker):
    """
    Automatically mocks the OpenAIModel class from deepeval to prevent real API calls
    and environment variable dependencies during tests.

    This fixture patches deepeval.models.OpenAIModel with a MagicMock that provides
    a dummy `get_model_name` method. It ensures that any DeepEval metric instantiation
    (like FaithfulnessMetric or BiasMetric) in tests will not attempt to access actual
    OpenAI resources or require a valid API key.

    Applied automatically to all tests via `autouse=True`.
    """
    mock_model = mocker.MagicMock()
    mock_model.get_model_name.return_value = "mock-model"
    mocker.patch("deepeval.models.OpenAIModel", return_value=mock_model)
