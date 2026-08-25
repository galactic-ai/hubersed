"""Static check that run_map_fits_outliers has no undefined names in the hot functions.

Three separate runs have now been thrown away by a NameError/KeyError raised AFTER the
expensive fitting was done:

  1. `eline_sigma` -- summary printer assumed the Cue parameter set, KeyError'd after all
     20 per-galaxy pkls were written (2026-08-20b).
  2. `logzsol`     -- same printer, after --fix-from-sample removed logzsol from
     theta_dict; again after all 11 pkls were written (2026-08-20f).
  3. `zcontinuous` -- referenced in fit_one's `rec` dict but threaded only through
     _worker and get_sps, so it raised after all 10 optimizer starts had run.

Every one is a name that does not resolve, and every one is catchable in milliseconds
without FSPS, a fit, or any data. That is what this does.

Run with ``uv run pytest tests/test_fit_one_names_resolve.py`` (never ``uvx``).
"""

import ast
import builtins
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "bin" / "prospector" / "run_map_fits_outliers.py"
# checked for undefined names; these are the ones that do real work before writing output
FUNCS = ["fit_one", "_worker", "map_fit", "main"]


def module_scope(tree):
    """Top-level names: imports, defs, classes, assignments."""
    names = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names |= {t.id for t in ast.walk(node)
                      if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        elif isinstance(node, (ast.For, ast.With, ast.If, ast.Try)):
            names |= {t.id for t in ast.walk(node)
                      if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    return names


def local_scope(fn):
    """Parameters plus anything bound inside, including comprehensions and handlers."""
    a = fn.args
    names = {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs}
    for x in (a.vararg, a.kwarg):
        if x:
            names.add(x.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if node is not fn:
                names |= local_scope(node) if not isinstance(node, ast.ClassDef) else set()
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(x.asname or x.name).split(".")[0] for x in node.names}
    return names


@pytest.mark.parametrize("func", FUNCS)
def test_no_undefined_names(func):
    tree = ast.parse(SRC.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == func), None)
    assert fn is not None, f"{func} not found in {SRC.name}"
    known = module_scope(tree) | local_scope(fn)
    used = {n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(used - known)
    assert not missing, (
        f"{func}() references names that do not resolve: {missing}. "
        "This raises only when that line executes -- which for fit_one is AFTER every "
        "optimizer start has run. Thread the parameter through, do not patch the caller."
    )


def test_fit_one_and_worker_agree():
    """_worker forwards to fit_one; a parameter on one and not the other is the bug."""
    tree = ast.parse(SRC.read_text())
    get = lambda name: next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == name)
    kw = lambda fn: {x.arg for x in fn.args.args + fn.args.kwonlyargs} - {"self"}
    only_in_fit_one = kw(get("fit_one")) - kw(get("_worker")) - {
        "sps", "cue_sps", "lines", "line_waves", "out"}   # supplied from get_sps/paths
    assert not only_in_fit_one, (
        f"fit_one takes {sorted(only_in_fit_one)} that _worker cannot pass, so the "
        "parallel path (-w >1) and the serial path would behave differently.")
