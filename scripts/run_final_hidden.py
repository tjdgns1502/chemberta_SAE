#!/usr/bin/env python3
"""Compatibility wrapper for unified CLI: run.py final-hidden."""

from pathlib import Path
import runpy
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "scripts" / "run.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing target script: {target}")

    sys.argv = [str(target), "final-hidden", *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
