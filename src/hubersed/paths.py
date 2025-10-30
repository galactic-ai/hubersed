from pathlib import Path
from typing import Dict, Iterable, Optional, Union
import os

PathLike = Union[str, Path]


def get_paths(base: Optional[PathLike] = None) -> Dict[str, Path]:
    """
    Return a dictionary of useful project paths (as Path objects).
    If base is None the function will assume the repository/project root is
    two parents above this file (typical src/package layout).
    Keys: ROOT, SRC, DATA, RESULTS, LOGS, CACHE, CONFIG, TMP
    """
    if base is None:
        # typically: /.../repo/src/package/paths.py -> repo is parents[2]
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


def ensure_dirs(paths: Iterable[PathLike], *, create: bool = True) -> Dict[str, bool]:
    """
    Ensure each path in `paths` exists and is writable.
    - paths: iterable of Path or string paths (or mapping values).
    - create: if True, missing directories will be created (mkdir(parents=True)).
    Returns a dict mapping string(path) -> bool indicating whether the path is ready (exists and writable).
    """
    status = {}
    for p in paths:
        path = Path(p)
        try:
            if not path.exists():
                if create:
                    path.mkdir(parents=True, exist_ok=True)
            ready = path.is_dir() and os.access(str(path), os.W_OK)
            # final check: try to create and remove a tiny temp file to test writability more reliably
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


# convenience: a ready-to-use PATHS mapping for the current project layout
PATHS = get_paths()

# ensure common directories exist on import (can be turned off by callers using ensure_dirs manually)
_defaults = [PATHS[k] for k in ("DATA", "RESULTS", "LOGS", "CACHE", "TMP", "CONFIG")]
_ensure_status = ensure_dirs(_defaults, create=True)