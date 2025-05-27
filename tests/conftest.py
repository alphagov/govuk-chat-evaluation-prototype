import csv
import os
from pathlib import Path
from inspect import signature
from typing import Callable
from dotenv import load_dotenv
from unittest.mock import MagicMock

import pytest
from typeguard import check_type, TypeCheckError


load_dotenv()


@pytest.fixture
def mock_project_root(mocker, tmp_path):
    """use tmp_path fixture for files created in test execution"""
    mocker.patch(
        "govuk_chat_evaluation.file_system.project_root", return_value=tmp_path
    )
    return tmp_path


@pytest.fixture(autouse=True)
def mock_or_use_openai_api_key(request, monkeypatch):
    if request.node.get_closest_marker(
        "real_openai"
    ):  # checks if the current test (or its containing class/module) is marked with real_openai
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be defined for real OpenAI tests.")
    else:
        # set a fake API key but only when not running real OpenAI tests
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key-for-testing")


def assert_csv_exists_with_headers(file_path: Path, *expected_headers: str):
    assert file_path.exists()
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        assert headers is not None, f"No headers found in {file_path}"

        missing = [h for h in expected_headers if h not in headers]
        assert not missing, f"Missing headers in {file_path}: {missing}"


def assert_mock_call_matches_signature(
    mock_function: MagicMock, original_function: Callable
) -> None:
    args, kwargs = mock_function.call_args
    sig = signature(original_function)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    for name, value in bound.arguments.items():
        expected_type = sig.parameters[name].annotation
        if expected_type is not sig.empty:
            try:
                check_type(value, expected_type)
            except TypeCheckError as e:
                raise AssertionError(
                    f"Argument '{name}' for {original_function!r} has invalid "
                    f"type: {value!r} (expected {expected_type})"
                ) from e
