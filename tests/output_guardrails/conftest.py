import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_exact": "1, 5, 7",
            "actual_exact": "1, 5, 7",
        },
        {
            "question": "Question 2",
            "expected_triggered": False,
            "actual_triggered": True,
            "expected_exact": "None",
            "actual_exact": "1",
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return str(path)
