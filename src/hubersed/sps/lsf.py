"""DESI line spread function, measured from the per-target resolution matrices."""

import pickle

import astropy.io.fits as fits
import astropy.table as aTable
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from hubersed.paths import PATHS

RESULTS_PATH = PATHS["RESULTS"]

C_KMS = 299792.458
DESI_BASE_URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/"
DESI_WAV = np.linspace(3600.0, 9824.0, 7781, dtype=np.float64)
ZPIX_FILE = DESI_BASE_URL + "zcatalog/v1/zpix-sv3-bright.fits"


def sample_target_ids(pkl_file, n_sample=100, seed=42):
    """Return a random sample of TARGETIDs from one spender chunk file.

    Parameters
    ----------
    pkl_file : str or Path
        A chunk pickle, a list of six tensors with the TARGETIDs fourth.
    n_sample : int
        How many TARGETIDs to return, at most the number in the file.
    seed : int
        Seed for ``np.random.default_rng``.

    Returns
    -------
    np.ndarray
        The sampled TARGETIDs.
    """
    with open(pkl_file, "rb") as f:
        batch = pickle.load(f)
    target_ids = batch[3].numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(target_ids), min(n_sample, len(target_ids)), replace=False)
    return target_ids[idx]


def lookup_healpix(target_ids, zpix_file=ZPIX_FILE):
    """Find the HEALPix pixel of each TARGETID in the DESI zpix catalog.

    Parameters
    ----------
    target_ids : iterable of int
        TARGETIDs to look up.
    zpix_file : str
        Path or URL of the zpix catalog. The default is the DR1 SV3 bright catalog.

    Returns
    -------
    dict
        HEALPix number keyed by TARGETID. Missing TARGETIDs are left out with a warning.
    """
    zpix = aTable.Table.read(zpix_file)
    zpix_tids = zpix["TARGETID"]
    zpix_hpix = zpix["HEALPIX"]

    tid_to_hpix = {}
    for tid in target_ids:
        match = zpix_tids == tid
        if np.any(match):
            tid_to_hpix[int(tid)] = int(zpix_hpix[match][0])
        else:
            print(f"  Warning: TARGETID {tid} not found in zpix")
    return tid_to_hpix


def coadd_url(hpix, survey="sv3", program="bright"):
    """Return the DR1 URL of the coadd FITS file for one HEALPix pixel."""
    filename = f"coadd-{survey}-{program}-{hpix}.fits"
    return f"{DESI_BASE_URL}/healpix/{survey}/{program}/{str(hpix)[:-2]}/{hpix}/{filename}"


def extract_resolution(coadd_file, target_id):
    """Read the resolution matrix of one target in each spectrograph arm.

    Parameters
    ----------
    coadd_file : str or Path
        DESI coadd FITS file.
    target_id : int
        TARGETID to read.

    Returns
    -------
    dict
        ``(wave, res_matrix)`` keyed by the lower-case arm prefix of the HDU name.
        ``res_matrix`` has shape ``(ndiag, nwave_arm)`` in the banded diagonal format.

    Raises
    ------
    ValueError
        If the TARGETID is not in the file.
    """
    hdulist = fits.open(coadd_file, cache=True)
    all_tids = hdulist[1].data["TARGETID"]
    idx = np.where(all_tids == target_id)[0]
    if len(idx) == 0:
        raise ValueError(f"TARGETID {target_id} not found in {coadd_file}")
    idx = idx[0]

    result = {}
    waves = {}
    for h in range(2, len(hdulist)):
        extname = hdulist[h].header["EXTNAME"]
        band = extname.split("_")[0].lower()
        if "WAVELENGTH" in extname:
            waves[band] = hdulist[h].data
        if "RESOLUTION" in extname:
            result[band] = hdulist[h].data[idx]  # (ndiag, nwave_arm)

    hdulist.close()
    return {b: (waves[b], result[b]) for b in result}


def resolution_to_sigma_kms(wave, res_banded):
    """Turn a banded resolution matrix into a Gaussian sigma in km/s.

    Treating each row as a Gaussian, the first off-diagonal divided by the diagonal equals
    ``exp(-0.5 / sigma_pix**2)``, which gives sigma in pixels.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in Angstrom.
    res_banded : np.ndarray
        Resolution matrix in banded format, shape ``(ndiag, nwave)``.

    Returns
    -------
    np.ndarray
        Sigma in km/s at each wavelength. NaN where the ratio is not between 0 and 1.
    """
    ndiag = res_banded.shape[0]
    center = ndiag // 2

    r_center = res_banded[center, :]
    r_off1 = res_banded[center + 1, :]

    # Avoid bad pixels
    valid = (r_center > 1e-6) & (r_off1 > 1e-6)
    ratio = np.full_like(r_center, np.nan)
    ratio[valid] = r_off1[valid] / r_center[valid]

    sigma_pix = np.full_like(ratio, np.nan)
    good = valid & (ratio > 0) & (ratio < 1)
    sigma_pix[good] = np.sqrt(-0.5 / np.log(ratio[good]))

    # Convert pixels to km/s
    dwave = np.gradient(wave)
    sigma_ang = sigma_pix * dwave
    sigma_kms = sigma_ang / wave * C_KMS

    return sigma_kms


def sigma_kms_to_R(wave, sigma_kms):
    """Convert a Gaussian sigma in km/s to resolving power, R = c / (2.355 sigma)."""
    return C_KMS / (2.355 * sigma_kms)


def desi_resolution(wave):
    """Return the median DESI resolving power at each wavelength.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in Angstrom.

    Returns
    -------
    np.ndarray
        Resolving power R, interpolated from ``results/desi_lsf_calibration.npz``.

    Notes
    -----
    The calibration file is written by ``bin/model_seds/build_desi_resolution.py`` from the
    resolution matrices of sampled targets. ``results/`` is not tracked by git, so the file
    must be rebuilt on a new machine. The function prints a line on every call.
    """
    print("Loading calibrated DESI resolution from resolution matrices...")
    cal = np.load(RESULTS_PATH / "desi_lsf_calibration.npz")
    R_median = cal["R_median"]
    wave_cal = cal["wave"]
    return np.interp(wave, wave_cal, R_median)


def build_desi_resolution_matrix(wave=DESI_WAV):
    """Build a sparse matrix that applies the median DESI line spread function.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in Angstrom. The default is the DESI grid.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(nwave, nwave)``. Each row is a Gaussian out to 4 sigma that sums to one.
        Multiply a flux vector by it to blur that flux to DESI resolution.
    """
    c_kms = 299792.458
    R = desi_resolution(wave)

    # resolving power to sigma in km/s, then to sigma in pixels
    sigma_kms = c_kms / (2.355 * R)
    dwave = np.gradient(wave)
    dpix_kms = dwave / wave * c_kms
    sigma_pix = sigma_kms / dpix_kms

    n = len(wave)
    mat = lil_matrix((n, n), dtype=np.float64)

    for i in range(n):
        # Kernel extends ±4 sigma
        hw = int(4 * sigma_pix[i]) + 1
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)

        # Gaussian kernel centered on pixel i
        j = np.arange(lo, hi)
        kernel = np.exp(-0.5 * ((j - i) / sigma_pix[i]) ** 2)
        kernel /= kernel.sum()

        mat[i, lo:hi] = kernel

    return csr_matrix(mat)
