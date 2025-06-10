import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "expected_exact_paths": ["/foo", "/bar"],
            "actual_exact_paths": ["/foo", "/bar"],
        },
        {
            "question": "Question 2",
            "expected_exact_paths": ["/foo"],
            "actual_exact_paths": ["/bar"],
        },
        {
            "question": "Question 3",
            "expected_exact_paths": ["/foo"],
            "actual_exact_paths": [],
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
