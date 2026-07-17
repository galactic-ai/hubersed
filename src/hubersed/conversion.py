import astropy.units as u
from astropy.constants import c

C_AA_PER_S = c.to(u.AA / u.s).value

C_CGS = 2.99792458e10  # cm/s
FNU_PER_MAGGIE = 3631e-23  # erg/s/cm^2/Hz


def maggies_to_flambda(wave, maggies):
    # maggies → f_nu (cgs), then f_lambda
    fnu_cgs = (maggies * 3631.0) * 1e-23
    flam = fnu_cgs * C_AA_PER_S / (wave**2)
    return flam


def flambda_to_maggies(wave_A, flambda):
    # flambda: erg/s/cm^2/Å ; wave_A: Å
    flambda_cgs = (
        flambda * 1e-17
    )  # DESI spectra are in 1e-17 erg/s/cm^2/Å, convert to cgs
    return flambda_cgs * (wave_A**2) * 1e-8 / C_CGS / FNU_PER_MAGGIE


def ivar_flambda_to_ivar_maggies(wave_A, ivar_flambda):
    # maggies = flambda * K  => ivar_maggies = ivar_flambda / K^2
    # because DESI spectra are in 1e-17 erg/s/cm^2/Å,
    # we need to include that factor in the conversion from flambda to maggies
    K = (wave_A**2) * 1e-8 * 1e-17 / C_CGS / FNU_PER_MAGGIE
    return ivar_flambda / (K**2)
