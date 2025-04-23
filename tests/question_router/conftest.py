import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "expected_outcome": "genuine_rag",
            "actual_outcome": "genuine_rag",
            "confidence_score": 0.95,
        },
        {
            "question": "Question 2",
            "expected_outcome": "greetings",
            "actual_outcome": "about_mps",
            "confidence_score": 0.8,
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
