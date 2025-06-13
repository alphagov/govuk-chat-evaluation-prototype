import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "expected_exact_paths": ["/foo", "/bar"],
            "actual_exact_paths_and_scores": [("/foo", 0.9), ("/bar", 0.8)],
        },
        {
            "question": "Question 2",
            "expected_exact_paths": ["/foo"],
            "actual_exact_paths_and_scores": [("/bar", 0.9)],
        },
        {
            "question": "Question 3",
            "expected_exact_paths": ["/foo"],
            "actual_exact_paths_and_scores": [],
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
