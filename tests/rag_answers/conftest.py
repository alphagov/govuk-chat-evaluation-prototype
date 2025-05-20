import json

import pytest
import yaml
from deepeval.evaluate.types import (
    TestResult as DeepevalTestResult,
    EvaluationResult as DeepevalEvaluationResult,
)
from deepeval.tracing.api import MetricData


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "question": "Question 1",
            "llm_answer": "Hi",
            "ideal_answer": "Hello",
            "retrieved_context": [],
        },
        {
            "question": "Question 2",
            "llm_answer": "Bye",
            "ideal_answer": "Bye",
            "retrieved_context": [],
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return str(path)


@pytest.fixture
def mock_config_file(tmp_path, mock_input_data):
    """Create a config YAML file for CLI input"""
    data = {
        "what": "Testing RAG Answer evaluations",
        "generate": True,
        "provider": "openai",
        "input_path": str(mock_input_data),
        "metrics": [
            {
                "name": "faithfulness",
                "threshold": 0.8,
                "model": "gpt-4o-mini",
                "temperature": 0.0,
            }
        ],
        "n_runs": 1,
    }
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as file:
        yaml.dump(data, file)

    yield str(file_path)


@pytest.fixture
def mock_deepeval_results():
    results = []
    for i in range(4):
        # Using modulus operator so odd and even tests have same input/output
        test = i % 2
        result = DeepevalTestResult(
            name=f"test_case_{test}",
            input=f"test input {test}",
            actual_output=f"actual output {test}",
            expected_output=f"expected output {test}",
            retrieval_context=[f"context {test}"],
            metrics_data=[
                MetricData(
                    name="faithfulness",
                    threshold=0.5,
                    score=0.5 + (i / 10),
                    reason="Good faith",
                    success=True,
                ),  # pyright: ignore[reportCallIssue]
                MetricData(
                    name="bias",
                    threshold=0.5 + (i / 10),
                    score=0.8,
                    reason="No bias",
                    success=True,
                ),  # pyright: ignore[reportCallIssue]
            ],
            success=True,
            conversational=False,
        )

        results.append(result)

    return [
        [results[0], results[1]],  # run 1
        [results[2], results[3]],  # run 2
    ]


@pytest.fixture
def mock_deepeval_evaluate(mocker, mock_deepeval_results):
    wrapped_results = [
        DeepevalEvaluationResult(test_results=result, confident_link=None)
        for result in mock_deepeval_results
    ]
    return mocker.patch(
        "govuk_chat_evaluation.rag_answers.deepeval_evaluate.deepeval_evaluate",
        side_effect=wrapped_results,  # using side effects to return a group per execution
    )
