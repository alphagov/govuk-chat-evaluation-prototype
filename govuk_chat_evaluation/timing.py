import time

class Timer:
    def __init__(self, label: str="Execution") -> None:
        self.label = label
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time
            print(f"[{self.label}] took {self.duration:.4f} seconds")
        else:
            raise ValueError("Start or end time not set")
