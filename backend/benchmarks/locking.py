"""A per-user lock serializes benchmark CLIs even when databases differ."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def benchmark_lock() -> Iterator[None]:
    path = Path(tempfile.gettempdir()) / f"liol-benchmark-{os.getuid()}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another LIOL benchmark is running for this user") from exc
        yield
    finally:
        os.close(fd)
