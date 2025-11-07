import numpy as np

def compute_logssfr(model, theta, to=1e8):
    """
    Compute the log10 specific star formation rate (sSFR) averaged over the last `to` years.

    Parameters
    ----------
    model : prospector.models.sedmodel
        The Prospector model instance.
    theta : _array_like
        The parameter vector for the model.
    to : float, optional
        The time over which to compute the sSFR, by default 1e8 years.

    Returns
    -------
    float
        The log10 specific star formation rate (sSFR) averaged over the last `to` years.

    Raises
    ------
    KeyError
        If no mass parameter is found in model.params.
    """
    model.set_parameters(theta)

    # in years
    agebins = 10 ** model.params['agebins']
    dt = np.diff(agebins, axis=1)[:, 0]

    if "mass" in model.params:
        mass_per_bin = np.array(model.params["mass"])
    elif "mass_formed" in model.params:
        mass_per_bin = np.array(model.params["mass_formed"])
    else:
        raise KeyError("No mass parameter found in model.params")

    # in case for some reason it goes to log space
    if np.any(mass_per_bin < 0):
        mass_per_bin = 10 ** mass_per_bin
    
    M_formed = np.sum(mass_per_bin)
    R = 0.4 # return fraction for the IMF used
    M_current = M_formed * (1 - R)

    # Recent SFR averaged over last 100 Myr
    lookback_mid = np.mean(agebins, axis=1)
    mask_recent = lookback_mid <= to
    M_recent = np.sum(mass_per_bin[mask_recent])
    SFR_recent = M_recent / to  # Msun/yr

    # sSFR and log10
    ssfr_recent = SFR_recent / M_current
    return np.log10(ssfr_recent)