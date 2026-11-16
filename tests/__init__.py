"""Isolated temporary storage bootstrap for V3 test discovery."""

import atexit
import os
import shutil
import tempfile

TEST_DATA_DIR = tempfile.mkdtemp(prefix="intake-v3-tests-")
atexit.register(shutil.rmtree, TEST_DATA_DIR, True)

os.environ.setdefault("DATA_DIR", TEST_DATA_DIR)
os.environ.setdefault("APP_PASSCODE", "user-test")
os.environ.setdefault("ADMIN_PASSCODE", "admin-test")
