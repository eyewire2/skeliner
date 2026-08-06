"""Lint checks matching CI flake8 configuration."""

import subprocess
import sys


def test_flake8_fatal():
    """Catch syntax errors, undefined names, and f-string issues."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flake8",
            "skeliner/",
            "--jobs=1",
            "--select=E9,F63,F7,F82",
            "--count",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"flake8 errors:\n{result.stdout}")
