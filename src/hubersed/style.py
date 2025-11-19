from pathlib import Path
import matplotlib as mpl

style_path = Path(__file__).resolve().parents[2] / "styles" / "apj.mplstyle"

mpl.style.use(style_path)