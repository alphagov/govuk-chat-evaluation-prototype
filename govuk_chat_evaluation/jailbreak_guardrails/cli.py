from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Self, cast

import click
from pydantic import Field, model_validator

from ..config import BaseConfig, config_from_cli_args, apply_click_options_to_command
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


@click.command(name="jailbreak_guardrails")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/defaults/jailbreak_guardrails.yaml",
)
@apply_click_options_to_command(Config)
def main(**cli_args):
    """Run jailbreak guardrails evaluation"""
    start_time = datetime.now()

    config: Config = config_from_cli_args(
        config_path=cli_args["config_path"],
        config_cls=Config,
        cli_args=cli_args,
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
