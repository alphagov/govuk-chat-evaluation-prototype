import json
import pandas as pd
import pytest
from govuk_chat_evaluation.jailbreak_guardrails import main
from unittest.mock import AsyncMock

TRUE_POSITIVE = ("True positive", True, True)
FALSE_POSITIVE = ("False positive", False, True)
TRUE_NEGATIVE = ("True negative", False, False)
FALSE_NEGATIVE = ("False negative", True, False)


def mock_result_generation(responses):
    def mock_async_process(input):
        match = next((r for r in responses if r[0] == input), None)
        if match is None:
            raise Exception(f"Unexpected input {input}")

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            json.dumps({"triggered": match[2]}).encode(),  # stdout
            b"",  # stderr
        )
        mock_process.returncode = 0
        return mock_process

    mock_subprocess_exec = AsyncMock(
        side_effect=lambda *_args, env, **_kwargs: mock_async_process(env["INPUT"])
    )
    return mock_subprocess_exec


@pytest.mark.parametrize(
    "responses, expected_output",
    [
        (
            [
                TRUE_POSITIVE,
                FALSE_POSITIVE,
                TRUE_NEGATIVE,
                FALSE_NEGATIVE,
            ],
            (
                "Evaluated: 4\n"
                "Precision: 50.00%\n"
                "Recall: 50.00%\n"
                "True positives: 1\n"
                "False positives: 1\n"
                "True negatives: 1\n"
                "False negatives: 1\n"
            ),
        ),
        (
            [
                FALSE_NEGATIVE,
                FALSE_NEGATIVE,
                FALSE_NEGATIVE,
            ],
            (
                "Evaluated: 3\n"
                "Precision: 0.00%\n"
                "Recall: N/A\n"
                "True positives: 0\n"
                "False positives: 0\n"
                "True negatives: 0\n"
                "False negatives: 3\n"
            ),
        ),
        (
            [
                FALSE_POSITIVE,
                FALSE_POSITIVE,
            ],
            (
                "Evaluated: 2\n"
                "Precision: N/A\n"
                "Recall: 0.00%\n"
                "True positives: 0\n"
                "False positives: 2\n"
                "True negatives: 0\n"
                "False negatives: 0\n"
            ),
        ),
    ],
)
def test_main_outputs_results(responses, expected_output, mocker, capsys):
    questions = [t[0] for t in responses]
    expected_outcome = [t[1] for t in responses]
    mock_csv_data = pd.DataFrame(
        {"question": questions, "expected_outcome": expected_outcome}
    )

    mocker.patch("pandas.read_csv", return_value=mock_csv_data)
    subprocess_mock = mock_result_generation(responses)
    mocker.patch("asyncio.create_subprocess_exec", new=subprocess_mock)

    main()
    captured = capsys.readouterr()
    assert captured.out == expected_output
