"""Compare two golden-output folders. Exits 1 on any difference."""

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

# wall-clock time, differs every run
VOLATILE = {"seconds"}
SUFFIXES = {".npz", ".h5", ".pkl"}


def load(path):
    """Read an npz, h5 or pickle file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    object
    """
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            return dict(f)
    if path.suffix == ".h5":
        out = {}
        with h5py.File(path, "r") as f:
            out["attrs:/"] = dict(f.attrs)

            def visit(name, obj):
                out[f"attrs:{name}"] = dict(obj.attrs)
                if isinstance(obj, h5py.Dataset):
                    out[name] = obj[()]

            f.visititems(visit)
        return out
    with open(path, "rb") as f:
        return pickle.load(f)


def diff(a, b, where, found):
    """Append each place where ``a`` and ``b`` differ to ``found``. NaN equals NaN.

    Parameters
    ----------
    a, b : object
    where : str
        Location in the file.
    found : list of str
    """
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            found.append(f"{where} keys differ")
            return
        for k in a:
            if k not in VOLATILE:
                diff(a[k], b[k], f"{where}.{k}", found)
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            found.append(f"{where} type or length differs")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diff(x, y, f"{where}[{i}]", found)
    else:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype != b.dtype:
            found.append(f"{where} dtype {a.dtype} became {b.dtype}")
            return
        try:
            same = np.array_equal(a, b, equal_nan=True)
        except TypeError:
            same = np.array_equal(a, b)
        if not same:
            found.append(f"{where} differs")


def compare_dirs(base, new):
    """Compare all npz, h5 and pickle files. Skips folders starting with an underscore.

    Parameters
    ----------
    base, new : Path

    Returns
    -------
    list of str
        One line per difference.
    """

    def files(root):
        return {
            p.relative_to(root)
            for p in root.rglob("*")
            if p.suffix in SUFFIXES
            and not any(s.startswith("_") for s in p.relative_to(root).parts)
        }

    fb, fn = files(base), files(new)
    found = [f"{p} missing from new run" for p in sorted(fb - fn)]
    found += [f"{p} only in new run" for p in sorted(fn - fb)]
    for p in sorted(fb & fn):
        diff(load(base / p), load(new / p), str(p), found)
    return found


if __name__ == "__main__":
    base, new = (Path(a) for a in sys.argv[1:3])
    found = compare_dirs(base, new)
    print("\n".join(found) if found else f"identical: {base} and {new}")
    sys.exit(1 if found else 0)
