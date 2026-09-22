"""Importing hubersed.paths must not write to disk."""

import subprocess
import sys

RECORD_WRITES = """
import pathlib

calls = []

def record(path, *args, **kwargs):
    calls.append(str(path))

pathlib.Path.mkdir = record
pathlib.Path.write_text = record
import hubersed.paths

assert not calls, f"importing hubersed.paths wrote to {calls}"
"""


def test_import_writes_nothing():
    """A fresh interpreter importing hubersed.paths calls neither mkdir nor write_text."""
    subprocess.run([sys.executable, "-c", RECORD_WRITES], check=True)
