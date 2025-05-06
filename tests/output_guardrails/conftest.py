import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_guardrails": {"appropriate_language": True, "political": True},
            "actual_guardrails": {"appropriate_language": True, "political": True},
        },
        {
            "question": "Question 2",
            "expected_triggered": False,
            "actual_triggered": True,
            "expected_guardrails": {"appropriate_language": False},
            "actual_guardrails": {"appropriate_language": True, "political": True},
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
