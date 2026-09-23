"""The Claude Code hooks in .claude/hooks allow and deny the edits they are meant to."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"
HACK = "sys.path." + "insert(0, 'bin')"  # split so the guard does not block this file


def run_hook(name, event):
    """Run one hook script on an event dict and return its parsed JSON output, or None."""
    out = subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out) if out.strip() else None


@pytest.mark.parametrize(
    ("tool", "args", "denied"),
    [
        ("Write", {"file_path": "/r/a.py", "content": "import numpy\n"}, False),
        ("Edit", {"file_path": "/r/a.py", "old_string": "x", "new_string": "y"}, False),
        ("Write", {"file_path": "/r/a.md", "content": HACK}, False),
        ("Write", {"file_path": "/r/a.py", "content": HACK}, True),
        ("Edit", {"file_path": "/r/a.py", "old_string": "x", "new_string": HACK}, True),
        ("Write", {"file_path": "/r/nb/a.ipynb", "content": "{}"}, True),
        ("NotebookEdit", {"notebook_path": "/r/nb/a.ipynb"}, True),
    ],
)
def test_guard_edits(tool, args, denied):
    """guard_edits denies notebooks and new sys.path hacks in .py files, and nothing else."""
    out = run_hook("guard_edits.py", {"tool_name": tool, "tool_input": args})
    assert (out is not None) == denied
    if denied:
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_guard_edits_denies_on_bad_input():
    """A malformed event is denied rather than let through."""
    out = subprocess.run(
        [sys.executable, str(HOOKS / "guard_edits.py")],
        input="not json",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_ruff_edited_reports_problems_only(tmp_path):
    """ruff_edited reports an unused import and stays silent on a clean file."""
    dirty, clean = tmp_path / "dirty.py", tmp_path / "clean.py"
    dirty.write_text("import os\n")
    clean.write_text("x = 1\n")
    event = {"tool_name": "Write", "cwd": str(tmp_path)}
    out = run_hook("ruff_edited.py", {**event, "tool_input": {"file_path": str(dirty)}})
    assert "F401" in out["hookSpecificOutput"]["additionalContext"]
    assert run_hook("ruff_edited.py", {**event, "tool_input": {"file_path": str(clean)}}) is None
