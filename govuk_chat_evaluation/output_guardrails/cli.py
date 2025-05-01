from datetime import datetime
from pathlib import Path
from typing import Literal, Self, cast

import click
from pydantic import Field, model_validator

from ..config import BaseConfig, config_from_cli_args, apply_click_options_to_command
from ..file_system import write_config_file_for_reuse
from .evaluate import evaluate_and_output_results
from .generate import generate_and_write_dataset
from ..output import initialise_output


class Config(BaseConfig):
    what: BaseConfig.GenericFields.what
    generate: BaseConfig.GenericFields.generate
    provider: BaseConfig.GenericFields.provider_openai_or_claude
    input_path: BaseConfig.GenericFields.input_path
    guardrail_type: Literal["answer_guardrails", "question_routing_guardrails"] = Field(
        ...,
        description="Type of output guardrail to evaluate: 'answer_guardrails' or 'question_router_guardrails'",
    )

    @model_validator(mode="after")
    def run_validatons(self) -> Self:
        return self._validate_fields_required_for_generate("provider")


@click.command(name="output_guardrails")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/defaults/output_guardrails.yaml",
)
@apply_click_options_to_command(Config)
def main(**cli_args):
    """Run output guardrails evaluation"""
    start_time = datetime.now()

    config: Config = config_from_cli_args(
        config_path=cli_args["config_path"],
        config_cls=Config,
        cli_args=cli_args,
    )

    output_dir = initialise_output("output_guardrails", start_time)

    if config.generate:
        evaluate_path = generate_and_write_dataset(
            config.input_path,
            cast(str, config.provider),
            config.guardrail_type,
            output_dir,
        )
    else:
        evaluate_path = config.input_path

    evaluate_and_output_results(output_dir, evaluate_path)

    write_config_file_for_reuse(output_dir, config)
