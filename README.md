# GOV.UK Chat Evaluation (Prototype)

A prototype to explore standardising the approach we use to evaluation on [GOV.UK Chat](https://github.com/alphagov/govuk-chat).

## Nomenclature

TBC

## Technical documentation

Install [uv](https://docs.astral.sh/uv/) to get started.

### Dependencies

Run `uv sync` to install dependencies.  
Run `uv pip install -e .` to install the executable.

### Usage

Run `uv run -m govuk_chat_evaluation` to run an evaluation.

### Development tasks

Run `uv run pytest` to run tests.  
Run `uv run ruff format` to format the code.  
Run `uv run ruff check .` to lint code base.  
Run `uv run pyright` to validate the type hints.

## Licence

[MIT License](LICENCE)
