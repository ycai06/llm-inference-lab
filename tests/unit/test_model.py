from __future__ import annotations

import hashlib
import struct
import threading
from pathlib import Path

import pytest

from backend.benchmarks.model import GGUFReader, inspect_model, sha256_file


def packed_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def minimal_gguf(entries: list[tuple[str, int, bytes]], version: int = 3) -> bytes:
    return (
        b"GGUF"
        + struct.pack("<IQQ", version, 0, len(entries))
        + b"".join(
            packed_string(key) + struct.pack("<I", tag) + value for key, tag, value in entries
        )
    )


def test_metadata_and_sha256_are_read_from_bytes_without_writes(tmp_path: Path) -> None:
    content = minimal_gguf(
        [
            ("general.architecture", 8, packed_string("unit-test-architecture")),
            ("general.file_type", 4, struct.pack("<I", 12)),
            ("unit.context_length", 4, struct.pack("<I", 2048)),
            (
                "tokenizer.ggml.tokens",
                9,
                struct.pack("<IQ", 8, 2) + packed_string("a") + packed_string("b"),
            ),
        ]
    )
    path = tmp_path / "test-fixture.gguf"
    path.write_bytes(content)
    before = path.stat()
    result = inspect_model(path)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["size_bytes"] == len(content)
    assert result["gguf_version"] == 3
    assert result["metadata"]["general.architecture"] == "unit-test-architecture"
    assert result["metadata"]["general.file_type"] == 12
    assert result["metadata"]["unit.context_length"] == 2048
    assert "tokenizer.ggml.tokens" not in result["metadata"]
    assert result["source_revision"] is None
    assert result["license_verified"] is False
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert path.read_bytes() == content


def test_missing_model_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Model does not exist"):
        inspect_model(tmp_path / "missing.gguf")


@pytest.mark.parametrize(
    "content, message",
    [
        (b"not a GGUF", "not a GGUF"),
        (b"GGUF", "Truncated GGUF"),
        (minimal_gguf([], version=1), "Unsupported GGUF version"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 100_001), "metadata count"),
        (minimal_gguf([("general.name", 8, struct.pack("<Q", 100))]), "Truncated GGUF"),
        (minimal_gguf([("general.name", 100, b"")]), "Unknown GGUF metadata type"),
        (
            minimal_gguf([("general.name", 8, struct.pack("<Q", 64 * 1024 * 1024 + 1))]),
            "string exceeds",
        ),
        (minimal_gguf([("test.array", 9, struct.pack("<IQ", 0, 10_000_001))]), "array exceeds"),
    ],
)
def test_reject_malformed_or_unbounded_metadata(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = tmp_path / "malformed.gguf"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        inspect_model(path)


def test_hash_cancellation(tmp_path: Path) -> None:
    path = tmp_path / "small.gguf"
    path.write_bytes(b"test-only")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError, match="Canceled while fingerprinting"):
        sha256_file(path, cancel)


def test_inspection_cancellation(tmp_path: Path) -> None:
    path = tmp_path / "small.gguf"
    path.write_bytes(minimal_gguf([("general.name", 8, packed_string("unit-test"))]))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError, match="Canceled during model inspection"):
        inspect_model(path, cancel)


def test_negative_read_length_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"test")
    with path.open("rb") as stream:
        with pytest.raises(ValueError, match="Truncated GGUF"):
            GGUFReader(stream).read(-1)
