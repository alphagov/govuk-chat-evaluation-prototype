# GOV.UK Chat Evaluation

A CLI tool to run evaluation tasks on [GOV.UK Chat](https://github.com/alphagov/govuk-chat). Each evaluation task is designed to test the performance of a specific component of the chat system, allowing for targeted diagnostics and improvement. Tasks typically take a dataset of user input and expected output, they then generate actual user output from GOV.UK Chat, finally the actual output is compared with expected output to produce results.

## Nomenclature

- Evaluation - the process of comparing the expected output and actual output to measure the quality of the actual output
- Config - a mechanism to alter parameters and options for an individual evaluation
- Dataset - a collection of data that contains input and expected output, may also contain actual output
- Generate - the process of taking user input and generating the actual output with GOV.UK Chat
- Results - data that is produced from an individual evaluation to provide qualitative insights 

## Technical documentation

Install [uv](https://docs.astral.sh/uv/) to get started.

### Dependencies

Run `uv sync` to install dependencies.  

Evaluation tasks that generate responses from GOV.UK Chat require the application to be available in your `~/govuk` directory.

The means to access input data is documented in [data/README.md](data/README.md).

### Usage

Run `uv run govuk_chat_evaluation` to view available evaluation tasks and options.

### Development tasks

Run `uv run pytest` to run tests.  
Run `uv run ruff format` to format the code.  
Run `uv run ruff check .` to lint code base.  
Run `uv run pyright` to validate the type hints.

## Licence

[MIT License](LICENCE)
