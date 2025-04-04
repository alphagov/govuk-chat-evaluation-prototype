import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {"question": "Question 1", "llm_answer": "Hi", "ideal_answer": "Hello"},
        {"question": "Question 2", "llm_answer": "Bye", "ideal_answer": "Bye"},
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return str(path)
