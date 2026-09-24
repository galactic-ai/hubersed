"""Quantities derived from a fitted prospector model."""

import numpy as np


def compute_logssfr(model, theta, to=1e8):
    """Return log10 of the specific star formation rate averaged over the last ``to`` years.

    Parameters
    ----------
    model : prospect.models.sedmodel.SpecModel
        A prospector model with a binned star formation history.
    theta : np.ndarray
        Parameter vector. It is set on ``model``, which changes the model's state.
    to : float
        Averaging time in years.

    Returns
    -------
    float
        log10 sSFR in 1/yr.

    Raises
    ------
    KeyError
        If the model has neither ``mass`` nor ``mass_formed``.

    Notes
    -----
    This pins how the code works today. Three parts are known to be wrong and are
    planned to change.

    A bin counts as recent when its midpoint is at most ``to``, and its whole mass is used.
    The age bins shift with redshift, and the midpoint of the fifth bin crosses 100 Myr
    near z of 0.1 to 0.2. For a constant star formation rate this gives a jump of about
    0.37 dex with redshift. Weighting each bin by its overlap with the window would fix it.

    The current mass uses a fixed return fraction of 0.4. prospect already computes the
    surviving mass fraction ``mfrac`` for the actual model.

    If any bin mass is negative, every bin is treated as log10 mass and exponentiated.
    """
    model.set_parameters(theta)

    # in years
    agebins = 10 ** model.params["agebins"]

    if "mass" in model.params:
        mass_per_bin = np.array(model.params["mass"])
    elif "mass_formed" in model.params:
        mass_per_bin = np.array(model.params["mass_formed"])
    else:
        raise KeyError("No mass parameter found in model.params")

    # in case for some reason it goes to log space
    if np.any(mass_per_bin < 0):
        mass_per_bin = 10**mass_per_bin

    M_formed = np.sum(mass_per_bin)
    R = 0.4  # return fraction for the IMF used
    M_current = M_formed * (1 - R)

    # recent SFR, averaged over the last `to` years
    lookback_mid = np.mean(agebins, axis=1)
    mask_recent = lookback_mid <= to
    M_recent = np.sum(mass_per_bin[mask_recent])
    SFR_recent = M_recent / to  # Msun/yr

    # sSFR and log10
    ssfr_recent = SFR_recent / M_current
    return np.log10(ssfr_recent)
