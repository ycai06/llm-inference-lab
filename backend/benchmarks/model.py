"""Read-only GGUF metadata inspection and streaming SHA-256 fingerprinting."""

from __future__ import annotations

import hashlib
import struct
import threading
from pathlib import Path
from typing import Any, BinaryIO


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Canceled while fingerprinting model")
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class GGUFReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def read(self, n: int) -> bytes:
        if n < 0 or self.stream.tell() + n > self.size:
            raise ValueError("Truncated GGUF metadata")
        result = self.stream.read(n)
        if len(result) != n:
            raise ValueError("Truncated GGUF metadata")
        return result

    def number(self, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, retain: bool = True) -> str | None:
        length = int(self.number("Q"))
        if length > 64 * 1024 * 1024:
            raise ValueError("GGUF string exceeds inspection limit")
        if retain:
            return self.read(length).decode("utf-8")
        if self.stream.tell() + length > self.size:
            raise ValueError("Truncated GGUF string")
        self.stream.seek(length, 1)
        return None

    def value(self, tag: int, retain: bool = True, depth: int = 0) -> Any:
        if depth > 2:
            raise ValueError("Unsupported nested GGUF array")
        if tag == 8:
            return self.string(retain)
        if tag == 9:
            subtype, count = int(self.number("I")), int(self.number("Q"))
            if count > 10_000_000:
                raise ValueError("GGUF array exceeds inspection limit")
            for _ in range(count):
                self.value(subtype, False, depth + 1)
            return {"array_length": count}
        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }
        if tag not in formats:
            raise ValueError(f"Unknown GGUF metadata type {tag}")
        return self.number(formats[tag])


def inspect_model(path: Path, cancel: threading.Event | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Model does not exist: {path}")
    before = path.stat()
    with path.open("rb") as stream:
        reader = GGUFReader(stream)
        if reader.read(4) != b"GGUF":
            raise ValueError("Model is not a GGUF file")
        version = reader.number("I")
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        tensors, count = reader.number("Q"), reader.number("Q")
        if count > 100_000:
            raise ValueError("GGUF metadata count exceeds inspection limit")
        metadata: dict[str, Any] = {}
        for _ in range(count):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Canceled during model inspection")
            key = reader.string()
            assert key is not None
            retain = key.startswith("general.") or key.endswith((".context_length", ".block_count"))
            value = reader.value(reader.number("I"), retain)
            if retain:
                metadata[key] = value
    checksum = sha256_file(path, cancel)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("Model changed while being fingerprinted")
    return {
        "path": str(path.resolve()),
        "sha256": checksum,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "inode": after.st_ino,
        "gguf_version": version,
        "tensor_count": tensors,
        "metadata": metadata,
        "source_revision": None,
        "license_verified": False,
    }
