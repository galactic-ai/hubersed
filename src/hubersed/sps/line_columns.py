"""Free line columns for the Cue model: ions, line widths and the prospect columns."""

import numpy as np

BALMER_PREFIX = "Ba-"
SIGMA_NARROW_MIN = 10.0  # km/s
# "split" widths (ladder steps 5-6 of 2026-11-07): narrow below 250 km/s, broad 250-800 km/s
SIGMA_SPLIT = 250.0  # km/s
SIGMA_BROAD_MAX = 800.0  # km/s
# "ratio" widths: broad = r * narrow, bounds from the EmFit widths of the parent sample
# (2026-11-10_line_width_split: narrow Halpha p99 408 km/s; second-component ratio p5 1.1-1.3,
# p99 13.5 (Halpha) to 38 ([OIII]))
SIGMA_NARROW_MAX = 450.0  # km/s
RATIO_MIN, RATIO_MAX = 1.2, 40.0
RATIO_INIT = 2.4  # median Halpha second-component ratio
VBROAD_MAX = 300.0  # km/s, bound on the broad offset
WEAK = ("free", "auroral")  # narrow classes that get a free narrow column
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


def broad_width_h(eline_sigma=None, r_broad=None, **extras):
    """Return the Balmer broad width, ``r_broad`` times the narrow width, in km/s."""
    return eline_sigma * r_broad


def broad_width_forb(eline_sigma=None, r_broad_forb=None, **extras):
    """Return the forbidden broad width, ``r_broad_forb`` times the narrow width, in km/s."""
    return eline_sigma * r_broad_forb


def add_line_params(params, widths="ratio"):
    """Set the narrow line width and add the broad widths and the broad offset.

    With ``widths="ratio"`` the narrow width ``eline_sigma`` gets TopHat(10, 450) km/s and each
    broad width is the narrow width times a free ratio, ``r_broad`` (Balmer) or ``r_broad_forb``
    (forbidden), each TopHat(1.2, 40). With ``widths="split"`` (ladder steps 5-6) the narrow width
    gets TopHat(10, 250) and the broad widths TopHat(250, 800) km/s. The broad offset gets
    TopHat(-300, 300) km/s in both.

    Parameters
    ----------
    params : dict
        Parameter specification with ``eline_sigma``, as from ``build_full_cue_model``.
        Changed in place.
    widths : {"ratio", "split"}
        How the broad widths are set.

    Returns
    -------
    dict
        ``params``.
    """
    from prospect.models import priors

    if widths not in ("ratio", "split"):
        raise ValueError(f"widths must be 'ratio' or 'split', not {widths!r}")
    narrow_max = SIGMA_NARROW_MAX if widths == "ratio" else SIGMA_SPLIT
    params["eline_sigma"] = {
        **params["eline_sigma"],
        "isfree": True,
        "init": 0.5 * (SIGMA_NARROW_MIN + SIGMA_SPLIT),
        "prior": priors.TopHat(mini=SIGMA_NARROW_MIN, maxi=narrow_max),
    }
    broad = (
        ("eline_sigma_broad", "r_broad", broad_width_h),
        ("eline_sigma_broad_forb", "r_broad_forb", broad_width_forb),
    )
    for name, ratio, fn in broad:
        if widths == "split":
            params[name] = dict(
                N=1,
                isfree=True,
                init=0.5 * (SIGMA_SPLIT + SIGMA_BROAD_MAX),
                units="km/s",
                prior=priors.TopHat(mini=SIGMA_SPLIT, maxi=SIGMA_BROAD_MAX),
            )
            continue
        params[ratio] = dict(
            N=1, isfree=True, init=RATIO_INIT, prior=priors.TopHat(mini=RATIO_MIN, maxi=RATIO_MAX)
        )
        params[name] = dict(
            N=1,
            isfree=False,
            init=RATIO_INIT * params["eline_sigma"]["init"],
            units="km/s",
            depends_on=fn,
        )
    params["eline_vbroad"] = dict(
        N=1,
        isfree=True,
        init=0.0,
        units="km/s",
        prior=priors.TopHat(mini=-VBROAD_MAX, maxi=VBROAD_MAX),
    )
    return params


def build_columns(names, limits):
    """Return the free line columns: weak narrow ions and broad components.

    Weak ions (narrow class "free" or "auroral" in ``IONS``) replace their Cue lines and share
    the narrow width ``eline_sigma``. Broad columns add to Cue, with width
    ``eline_sigma_broad`` (H) or ``eline_sigma_broad_forb`` (forbidden) and offset
    ``eline_vbroad``. Lines in a column keep Cue's ratios.

    Parameters
    ----------
    names : array_like of str
        Line names of the model, as in the Cue line list.
    limits : dict
        Upper bound of the uniform equivalent-width prior per column, keyed by ion name for
        weak columns and by "<ion> broad" for broad ones (ion_limits.json of the 11-06
        line_ew_prior experiment).

    Returns
    -------
    list of prospect.models.line_marginalisation.LineColumn
        Weak columns in ``IONS`` order, then broad columns in ``IONS`` order.
    """
    from prospect.models.line_marginalisation import LineColumn

    names = np.char.strip(np.asarray(names).astype(str))
    index = ion_lines(names)
    weak, broad = [], []
    for ion, spec in IONS.items():
        lines = tuple(names[index[ion]])
        if spec["narrow"] in WEAK:
            weak.append(LineColumn(ion, lines, float(limits[ion]), "eline_sigma", None, True))
        if spec["broad"] is not None:
            width = "eline_sigma_broad" if spec["broad"] == "balmer" else "eline_sigma_broad_forb"
            name = f"{ion} broad"
            broad.append(LineColumn(name, lines, float(limits[name]), width, "eline_vbroad", False))
    return weak + broad
