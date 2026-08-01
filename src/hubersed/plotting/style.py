from importlib.resources import files
from matplotlib import style as _mpl_style

def use_apj_style():
    _mpl_style.use(files("hubersed.plotting") / "data" / "apj.mplstyle")
