"""Ions of the Cue line list that get free line columns, and their line indices."""

import numpy as np

BALMER_PREFIX = "Ba-"
# Each ion: Cue line names, narrow class, broad width ("balmer", "forb" or None for no broad part).
IONS = {
    "H": dict(lines=None, narrow="tied", broad="balmer"),
    "[OII]": dict(lines=("[O II] 3726", "[O II] 3729"), narrow="strong", broad="forb"),
    "[OIII]": dict(lines=("[O III] 4959", "[O III] 5007"), narrow="strong", broad="forb"),
    "[NII]": dict(lines=("[N II] 6548", "[N II] 6584"), narrow="strong", broad="forb"),
    "[SII]": dict(lines=("[S II] 6716", "[S II] 6731"), narrow="free", broad="forb"),
    "[OI]": dict(lines=("[O I] 6300", "[O I] 6363"), narrow="free", broad="forb"),
    "[ArIII]7135": dict(lines=("[Ar III] 7135",), narrow="free", broad=None),
    "[OIII]4363": dict(lines=("[O III] 4363",), narrow="auroral", broad=None),
    "[OII]7325": dict(lines=("[O II] 7323", "[O II] 7332"), narrow="auroral", broad=None),
    "[NII]5755": dict(lines=("[N II] 5755",), narrow="auroral", broad=None),
    "[NeIII]": dict(lines=("[Ne III] 3869", "[Ne III] 3968"), narrow="auroral", broad=None),
    "[SIII]6312": dict(lines=("[S III] 6312",), narrow="auroral", broad=None),
    "[SIII]9069": dict(lines=("[S III] 9069",), narrow="auroral", broad=None),
}


def ion_lines(names):
    """Return the line indices of each ion of ``IONS``.

    Parameters
    ----------
    names : array_like of str
        Line names of the model, as in the Cue line list.

    Returns
    -------
    dict
        Ion name to an integer array of line indices.

    Raises
    ------
    ValueError
        If a line named in ``IONS`` is not in ``names``.
    """
    names = np.char.strip(np.asarray(names).astype(str))
    out = {}
    for ion, spec in IONS.items():
        if spec["lines"] is None:
            out[ion] = np.where(np.char.startswith(names, BALMER_PREFIX))[0]
            continue
        idx = [np.where(names == n)[0] for n in spec["lines"]]
        missing = [n for n, i in zip(spec["lines"], idx, strict=True) if i.size == 0]
        if missing:
            raise ValueError(f"lines {missing} of {ion} are not in the model's line list")
        out[ion] = np.concatenate(idx)
    return out
