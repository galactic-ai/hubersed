"""Project directory paths.

Importing this module does not touch the disk. git keeps an empty results/ in every clone,
so scripts can write there directly. data/ has to be provided.
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def get_paths(base: PathLike | None = None) -> dict[str, Path]:
    """Return the main project directories.

    Parameters
    ----------
    base : str or Path, optional
        Repository root. By default it is two levels above this file, which is the root
        for the src/hubersed layout.

    Returns
    -------
    dict of str to Path
        Paths keyed by ROOT, SRC, DATA, RESULTS, LOGS, CACHE, CONFIG and TMP.
    """
    if base is None:
        base_path = Path(__file__).resolve().parents[2]
    else:
        base_path = Path(base).resolve()

    src = base_path / "src"
    return {
        "ROOT": base_path,
        "SRC": src,
        "DATA": base_path / "data",
        "RESULTS": base_path / "results",
        "LOGS": base_path / "logs",
        "CACHE": base_path / ".cache",
        "CONFIG": base_path / "config",
        "TMP": base_path / "tmp",
    }


def ensure_dirs(paths: Iterable[PathLike], *, create: bool = True) -> dict[str, bool]:
    """Check that each directory exists and can be written to, creating it if asked.

    Parameters
    ----------
    paths : iterable of str or Path
        Directories to check.
    create : bool
        Create missing directories, including their parents.

    Returns
    -------
    dict of str to bool
        True for each directory that exists and passed a test write of a small file.
    """
    status = {}
    for p in paths:
        path = Path(p)
        try:
            if not path.exists():
                if create:
                    path.mkdir(parents=True, exist_ok=True)
            ready = path.is_dir() and os.access(str(path), os.W_OK)
            # also write and delete a small file, a stricter test than os.access
            if ready:
                test_file = path / ".write_test"
                try:
                    test_file.write_text("")  # create/truncate
                    test_file.unlink()
                    status[str(path)] = True
                except Exception:
                    status[str(path)] = False
            else:
                status[str(path)] = False
        except Exception:
            status[str(path)] = False
    return status


PATHS = get_paths()
