from datetime import datetime
from pathlib import Path

from .file_system import create_output_directory
from .logging import setup_logging


def initialise_output(prefix: str, run_time: datetime) -> Path:
    """
    Create the timestamped results directory for this run and
    configure logging so warnings and errors are captured
    in <output_dir>/problems.log.
    """
    output_dir = create_output_directory(prefix, run_time)
    setup_logging(output_dir)
    return output_dir
