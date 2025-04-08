import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {"question": "Question 1", "expected_outcome": True, "actual_outcome": True},
        {"question": "Question 2", "expected_outcome": False, "actual_outcome": True},
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
