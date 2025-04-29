import logging
import sys
from pathlib import Path


def setup_logging(run_output_dir: Path) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_output_dir / "run.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)s  %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)
