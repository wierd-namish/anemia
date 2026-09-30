"""
Execution timing and duration formatting helpers.
"""

import time
from contextlib import contextmanager
from typing import Generator


def format_duration(seconds: float) -> str:
    """Formats seconds into HH:MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@contextmanager
def benchmark_timer() -> Generator[dict, None, None]:
    """Context manager for tracking elapsed time in milliseconds."""
    tracker = {"elapsed_ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield tracker
    finally:
        tracker["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
