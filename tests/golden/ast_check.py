"""Compare a module before and after a src/scripts split.

Usage is uv run python tests/golden/ast_check.py OLD_REV OLD_PATH NEW_SRC_PATH [NEW_SCRIPT_PATH]

Every top-level function and class of the old module other than main must appear AST-identical
in the new src module or, if it moved, in the new script. main is compared as unparsed source and
the diff is printed for review. Exits 1 on any difference outside main.
"""

import ast
import difflib
import subprocess
import sys


def defs(tree):
    """Return top-level function and class nodes by name."""
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def main():
    """Run the comparison."""
    rev, old_path, new_src = sys.argv[1:4]
    new_script = sys.argv[4] if len(sys.argv) > 4 else None
    old = ast.parse(subprocess.check_output(["git", "show", f"{rev}:{old_path}"], text=True))
    new = defs(ast.parse(open(new_src).read()))
    scr = defs(ast.parse(open(new_script).read())) if new_script else {}
    bad = 0
    for name, node in defs(old).items():
        if name == "main":
            continue
        other = new.get(name) or scr.get(name)
        where = "src" if name in new else "script" if name in scr else None
        if other is None:
            print(f"MISSING {name}")
            bad += 1
        elif ast.dump(node) != ast.dump(other):
            print(f"CHANGED {name} ({where})")
            bad += 1
        else:
            print(f"same    {name} ({where})")
    old_main = defs(old).get("main")
    new_main = scr.get("main") or new.get("main")
    if old_main is not None:
        a = ast.unparse(old_main).splitlines()
        b = ast.unparse(new_main).splitlines() if new_main else []
        diff = list(difflib.unified_diff(a, b, "old main", "new main", lineterm="", n=1))
        print("main: identical" if not diff else "main diff:\n" + "\n".join(diff))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
