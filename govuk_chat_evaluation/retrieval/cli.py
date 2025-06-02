from datetime import datetime
from pathlib import Path
from typing import Self, Annotated, Optional, Literal, cast

import click
from pydantic import Field, model_validator

from ..config import BaseConfig, config_from_cli_args, apply_click_options_to_command
from ..file_system import create_output_directory, write_config_file_for_reuse
from .evaluate import evaluate_and_output_results
from .generate import generate_and_write_dataset


class Config(BaseConfig):
    what: BaseConfig.GenericFields.what
    generate: BaseConfig.GenericFields.generate
    embedding_provider: Annotated[
        Optional[Literal["openai", "titan"]],
        Field(
            None,
            description="Which LLM provider to use for generating the embeddings: openai or titan",
        ),
    ]
    input_path: BaseConfig.GenericFields.input_path

    @model_validator(mode="after")
    def run_validatons(self) -> Self:
        return self._validate_fields_required_for_generate("embedding_provider")


@click.command(name="retrieval")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/defaults/retrieval.yaml",
)
@apply_click_options_to_command(Config)
def main(**cli_args):
    """Run retrieval evaluation"""
    start_time = datetime.now()

    config: Config = config_from_cli_args(
        config_path=cli_args["config_path"],
        config_cls=Config,
        cli_args=cli_args,
    )

    output_dir = create_output_directory("retrieval", start_time)

    if config.generate:
        evaluate_path = generate_and_write_dataset(
            config.input_path, cast(str, config.embedding_provider), output_dir
        )
    else:
        evaluate_path = config.input_path

    evaluate_and_output_results(output_dir, evaluate_path)

    write_config_file_for_reuse(output_dir, config)
