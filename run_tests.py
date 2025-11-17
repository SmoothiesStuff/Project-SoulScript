########## Test Runner ##########
# Minimal helper to run the project's pytest suite.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Execute pytest across the repository."""

    repo_root = Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "pytest", str(repo_root / "soulscript" / "tests")]
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
