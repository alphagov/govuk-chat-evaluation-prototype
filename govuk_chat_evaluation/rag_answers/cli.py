from datetime import datetime
from pathlib import Path
from typing import cast

import click

from ..config import apply_click_options_to_command, config_from_cli_args
from ..file_system import write_config_file_for_reuse
from .evaluate import evaluate_and_output_results
from .generate import generate_and_write_dataset
from .data_models import Config
from ..output import initialise_output


@click.command(name="rag_answers")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config/defaults/rag_answers.yaml",
)
@apply_click_options_to_command(Config)
def main(**cli_args):
    """Run RAG answers evaluation"""
    start_time = datetime.now()

    config: Config = config_from_cli_args(
        config_path=cli_args["config_path"],
        config_cls=Config,
        cli_args=cli_args,
    )

    output_dir = initialise_output("rag_answers", start_time)

    if config.generate:
        evaluate_path = generate_and_write_dataset(
            config.input_path, cast(str, config.provider), output_dir
        )
    else:
        evaluate_path = config.input_path

    evaluate_and_output_results(output_dir, evaluate_path, config)

    write_config_file_for_reuse(output_dir, config)
