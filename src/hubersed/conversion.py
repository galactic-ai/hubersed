"""Convert spectra between maggies and f_lambda with astropy units.

Every function takes and returns astropy Quantities, so a flux in the wrong unit
raises ``astropy.units.UnitsError`` instead of being scaled silently.
"""

import astropy.units as u

DESI_FLAM = u.def_unit("desi_flam", 1e-17 * u.erg / u.s / u.cm**2 / u.AA)
"""Flux density unit of DESI spectra, 1e-17 erg/s/cm^2/A."""

MAGGIE_ZP = 3631 * u.Jy
"""Flux density of 1 maggie. astropy's ``u.ABflux`` is 3630.78 Jy, so it is not used."""

FLAM = u.erg / u.s / u.cm**2 / u.AA


@u.quantity_input
def to_maggies(wave: u.AA, flux: FLAM) -> u.mgy:
    """Convert f_lambda to flux in maggies.

    Parameters
    ----------
    wave : astropy.units.Quantity
        Wavelength, any length unit.
    flux : astropy.units.Quantity
        f_lambda, for example in ``DESI_FLAM``.

    Returns
    -------
    astropy.units.Quantity
        Flux density in maggies.
    """
    fnu = flux.to(u.Jy, u.spectral_density(wave))
    return fnu.to(u.mgy, u.zero_point_flux(MAGGIE_ZP))


@u.quantity_input
def to_flambda(wave: u.AA, maggies: u.mgy, unit=DESI_FLAM):
    """Convert flux in maggies to f_lambda.

    Parameters
    ----------
    wave : astropy.units.Quantity
        Wavelength, any length unit.
    maggies : astropy.units.Quantity
        Flux density in maggies.
    unit : astropy.units.Unit, optional
        f_lambda unit of the result. The default is ``DESI_FLAM``.

    Returns
    -------
    astropy.units.Quantity
        f_lambda in ``unit``. ``to_maggies`` undoes it.
    """
    fnu = maggies.to(u.Jy, u.zero_point_flux(MAGGIE_ZP))
    return fnu.to(unit, u.spectral_density(wave))


@u.quantity_input
def ivar_to_maggies(wave: u.AA, ivar: FLAM**-2) -> u.mgy**-2:
    """Convert the inverse variance of f_lambda to inverse variance in maggies.

    Parameters
    ----------
    wave : astropy.units.Quantity
        Wavelength, any length unit.
    ivar : astropy.units.Quantity
        Inverse variance of f_lambda, for example in ``DESI_FLAM**-2``.

    Returns
    -------
    astropy.units.Quantity
        Inverse variance in maggies^-2.

    Notes
    -----
    The flux conversion multiplies each pixel by a factor K, so the inverse variance
    is divided by K squared. K is the size in maggies of one unit of ``ivar.unit**-0.5``.
    """
    k = to_maggies(wave, 1.0 * ivar.unit**-0.5)
    return ivar.value / k**2
