from importlib.resources import files
from shutil import which

from matplotlib import rcParams
from matplotlib import style as _mpl_style


def use_apj_style():
    _mpl_style.use(files("hubersed.plotting") / "data" / "apj.mplstyle")
    
    if which("latex") is None:
        rcParams["text.usetex"] = False
