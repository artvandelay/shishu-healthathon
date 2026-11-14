"""Test package path bootstrap for SHISHU suites."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "code" / "shishu-v2"
TESTS_DIR = Path(__file__).resolve().parent

for _path in (str(TESTS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
