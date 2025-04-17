import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_exact": 'True | "1, 3"',
            "actual_exact": 'True | "1, 3"',
        },
        {
            "question": "Question 2",
            "expected_triggered": False,
            "actual_triggered": True,
            "expected_exact": "False | None",
            "actual_exact": 'True | "1, 3"',
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
