"""Recreate reports from a real exported or compact published experiment JSON."""

import argparse
import json
from pathlib import Path

from backend.analysis.report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(json.loads(args.input.read_text()), args.output)


if __name__ == "__main__":
    main()
