import astropy.units as u
from astropy.constants import c

C_AA_PER_S = c.to(u.AA / u.s).value

def maggies_to_flambda(wave, maggies):
    # maggies → f_nu (cgs), then f_lambda
    fnu_cgs = (maggies * 3631.0) * 1e-23
    flam = fnu_cgs * C_AA_PER_S / (wave**2)
    return flam