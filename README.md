# GOV.UK Chat Evaluation (Prototype)

A prototype to explore standardising the approach we use to evaluation on [GOV.UK Chat](https://github.com/alphagov/govuk-chat).

## Nomenclature

TBC

## Technical documentation

Install [uv](https://docs.astral.sh/uv/) to get started.
Put a CSV in data (jailbreak.csv) which has columns for question (string) and expected_outcome (int, 0 or 1).

Run `uv sync` to install dependencies.  
Run `uv run pytest` to run tests.  
Run `uv run src/govuk_chat_evaluation/jailbreak_guardrails.py` to run guardrails.

## Licence

[MIT License](LICENCE)
