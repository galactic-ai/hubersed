#!/usr/bin/env python3
"""Report ruff problems in a Python file Claude just wrote or edited.

PostToolUse hook for Write and Edit. Runs ``ruff check`` and
``ruff format --check`` on the edited file and passes any findings back to
Claude as additional context. It never changes the file, and it stays silent
when the file is clean or ruff cannot be found.
"""

import json
import os
import shutil
import subprocess
import sys


def find_ruff(cwd):
    """Return the project's ruff, falling back to one on PATH."""
    roots = [cwd, os.environ.get("CLAUDE_PROJECT_DIR", "")]
    for root in roots:
        candidate = os.path.join(root, ".venv", "bin", "ruff")
        if root and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("ruff")


def main():
    """Read the hook event from stdin and print ruff findings as context."""
    event = json.load(sys.stdin)
    path = event.get("tool_input", {}).get("file_path", "")
    cwd = event.get("cwd") or os.getcwd()
    ruff = find_ruff(cwd)
    if not path.endswith(".py") or not os.path.isfile(path) or ruff is None:
        return
    check = subprocess.run(
        [ruff, "check", "--output-format", "concise", path],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    fmt = subprocess.run([ruff, "format", "--check", path], cwd=cwd, capture_output=True, text=True)
    notes = []
    if check.returncode != 0:
        notes.append(check.stdout.strip() or check.stderr.strip())
    if fmt.returncode != 0:
        notes.append(f"Not ruff-formatted. Run: uv run ruff format {path}")
    if notes:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "ruff on the file you just edited:\n"
                        + "\n".join(notes)[:9000],
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
