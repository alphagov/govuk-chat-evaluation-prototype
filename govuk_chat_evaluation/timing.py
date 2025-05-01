import time
from contextlib import contextmanager
import logging


@contextmanager
def log_task_duration(label: str = "Execution"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.info(f"[{label}] took {duration:.4f} seconds")
