import csv
from pathlib import Path

import pytest


@pytest.fixture
def mock_project_root(mocker, tmp_path):
    """use tmp_path fixture for files created in test execution"""
    mocker.patch(
        "govuk_chat_evaluation.file_system.project_root", return_value=tmp_path
    )
    return tmp_path


@pytest.fixture(autouse=True)
def mock_openai_key(monkeypatch):
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
