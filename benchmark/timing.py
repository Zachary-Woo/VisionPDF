"""
Lightweight timing utilities for the benchmark.

Provides a context manager that records wall-clock time (and optional
CUDA event timing when a GPU is available) and a helper that appends
per-page rows to a CSV log.
"""

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


class TimingResult:
    """Container returned by the ``timed`` context manager."""

    def __init__(self):
        self.wall_seconds: float = 0.0
        self.cuda_seconds: Optional[float] = None


@contextmanager
def timed(use_cuda: bool = False):
    """
    Context manager that measures elapsed time.

    Parameters
    ----------
    use_cuda : bool
        When True and CUDA is available, also record GPU-side time
        via ``torch.cuda.Event``.

    Yields
    ------
    TimingResult
        Populated with ``wall_seconds`` (and optionally ``cuda_seconds``)
        after the block exits.
    """
    result = TimingResult()
    do_cuda = use_cuda and _CUDA_AVAILABLE

    if do_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start = time.perf_counter()
    yield result
    wall_end = time.perf_counter()
    result.wall_seconds = wall_end - wall_start

    if do_cuda:
        end_event.record()
        torch.cuda.synchronize()
        result.cuda_seconds = start_event.elapsed_time(end_event) / 1000.0


def append_timing_row(
    csv_path: Path,
    method: str,
    page_id: str,
    wall_seconds: float,
    cuda_seconds: Optional[float] = None,
):
    """
    Append a single timing row to *csv_path*, creating the file and
    header if it does not yet exist.
    """
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["method", "page_id", "wall_seconds", "cuda_seconds"])
        writer.writerow([method, page_id, f"{wall_seconds:.6f}", cuda_seconds or ""])
