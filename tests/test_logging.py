import logging
from pathlib import Path

from govuk_chat_evaluation.logging import setup_logging


def _read_file(path: Path) -> str:
    with open(path, encoding="utf8") as fh:
        return fh.read()


def test_warning_is_written_to_log_file(tmp_path: Path, caplog):
    setup_logging(tmp_path)
    logger = logging.getLogger(__name__)

    with caplog.at_level(logging.WARNING):
        logger.warning("Incorrect LLM response: %s", "bad response")

    log_file = tmp_path / "run.log"
    assert log_file.exists()
    file_contents = _read_file(log_file)
    assert "Incorrect LLM response: bad response" in file_contents
    assert "WARNING" in file_contents


def test_info_does_not_land_in_log_file(tmp_path: Path):
    setup_logging(tmp_path)
    logging.getLogger(__name__).info("stdout info message")

    log_file = tmp_path / "run.log"

    assert log_file.exists()
    assert "stdout info message" not in _read_file(log_file)
