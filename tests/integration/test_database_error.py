"""A damaged local database should produce a useful CLI error."""

import os
import subprocess
import sys
from pathlib import Path


def test_corrupt_database_does_not_print_traceback(tmp_path: Path) -> None:
    db = tmp_path / "corrupt.sqlite3"
    db.write_text("not a SQLite file")
    result = subprocess.run(
        [sys.executable, "-m", "backend.cli", "--db", str(db), "list"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=os.environ | {"TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 2
    assert "liol:" in result.stderr
    assert "Traceback" not in result.stderr
