#!/usr/bin/env python3
"""Deny Claude edits that write notebooks or add sys.path hacks.

PreToolUse hook for Write, Edit and NotebookEdit. Notebooks are not source in
this repo; experiments are ``experiments/*.py`` files with ``# %%`` cells.
Scripts import from the installed ``hubersed`` package, never through a
``sys.path`` insert. Any error inside the hook denies the edit, because a
crashing hook would let it through.
"""

import json
import re
import sys

SYS_PATH = re.compile(r"sys\.path\.(insert|append|extend)")


def reason_to_deny(event):
    """Return a deny reason for this tool call, or None to allow it."""
    tool = event.get("tool_name", "")
    args = event.get("tool_input", {})
    path = args.get("file_path", "")
    if tool == "NotebookEdit" or path.endswith(".ipynb"):
        return (
            "Notebooks are not source here. Write experiments/YYYY-MM-DD_name.py "
            "with # %% cells instead."
        )
    new_text = args.get("content", "") + args.get("new_string", "")
    if path.endswith(".py") and SYS_PATH.search(new_text):
        return "Do not edit sys.path. Import from the hubersed package instead."
    return None


def main():
    """Read the hook event from stdin and print a deny decision if needed."""
    try:
        reason = reason_to_deny(json.load(sys.stdin))
    except Exception as err:
        reason = f"guard_edits hook failed: {err!r}"
    if reason:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"{reason} (.claude/hooks/guard_edits.py)",
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
