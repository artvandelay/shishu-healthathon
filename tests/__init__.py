"""Test package path bootstrap for SHISHU suites."""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "code" / "shishu-v2"
TESTS_DIR = Path(__file__).resolve().parent

for _path in (str(TESTS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TEST_DATA_DIR = tempfile.mkdtemp(prefix="shishu-tests-")
atexit.register(shutil.rmtree, TEST_DATA_DIR, True)

os.environ.setdefault("DATA_DIR", TEST_DATA_DIR)
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")
