"""Matplotlib style for figures in ApJ format."""

from importlib.resources import files
from shutil import which

from matplotlib import rcParams
from matplotlib import style as _mpl_style


def use_apj_style():
    """Apply ``plotting/data/apj.mplstyle`` and turn off LaTeX text when latex is not installed.

    This changes matplotlib's global settings for the rest of the process.
    """
    _mpl_style.use(files("hubersed.plotting") / "data" / "apj.mplstyle")

    if which("latex") is None:
        rcParams["text.usetex"] = False
