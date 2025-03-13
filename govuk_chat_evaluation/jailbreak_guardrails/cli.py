from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import cast, Optional, Literal, Self

from pydantic import Field, model_validator

from ..config import config_from_cli_args, BaseConfig
from ..file_system import create_output_directory, write_config_file_for_reuse
from .evaluate import evaluate_and_output_results
from .generate import generate_and_write_dataset


class Config(BaseConfig):
    what: str = Field(..., description="what is being evaluated")
    generate: bool = Field(..., description="whether to generate data")
    provider: Optional[Literal["openai", "claude"]] = Field(
        None,
        description="which provider to use for generating the data, openai or claude",
    )
    input_path: str = Field(..., description="path to the data file used to evaluate")

    @model_validator(mode="after")
    def check_provider_required(self) -> Self:
        if self.generate and self.provider is None:
            raise ValueError("Provider is required to generate data")
        return self


def main():
    start_time = datetime.now()
    parser = ArgumentParser(
        prog="uv run -m govuk_chat_evaluation.jailbreak_guardrails",
        description=(
            "This will load a JSONL file of jailbreak guardrails, optionally "
            "allow generating responses from GOV.UK Chat, and write the "
            "results of the evaluation"
        ),
    )
    config: Config = config_from_cli_args(
        parser,
        default_config_path="config/defaults/jailbreak_guardrails.yaml",
        config_cls=Config,
    )

    output_dir = create_output_directory("jailbreak_guardrails", start_time)

    if config.generate:
        evaluate_path = generate_and_write_dataset(
            config.input_path, cast(str, config.provider), output_dir
        )
    else:
        evaluate_path = Path(config.input_path)

    evaluate_and_output_results(output_dir, evaluate_path)
    write_config_file_for_reuse(output_dir, config)
