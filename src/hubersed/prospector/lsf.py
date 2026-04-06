import pickle

import numpy as np
import astropy.io.fits as fits
import astropy.table as aTable
from scipy.sparse import lil_matrix, csr_matrix

from hubersed.paths import PATHS

RESULTS_PATH = PATHS["RESULTS"]

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
C_KMS = 299792.458
DESI_BASE_URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/"
DESI_WAV = np.linspace(3600.0, 9824.0, 7781)
ZPIX_FILE = DESI_BASE_URL + 'zcatalog/v1/zpix-sv3-bright.fits'

# ──────────────────────────────────────────────
# Step 1: Load pkl and sample target_ids
# ──────────────────────────────────────────────
def sample_target_ids(pkl_file, n_sample=100, seed=42):
    """Load a batch pkl file and return N random target_ids."""
    with open(pkl_file, 'rb') as f:
        batch = pickle.load(f)
    # batch = [spec, w, z, target_id, norm, zerr]
    target_ids = batch[3].numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(target_ids), min(n_sample, len(target_ids)), replace=False)
    return target_ids[idx]


# ──────────────────────────────────────────────
# Step 2: Look up healpix from zpix catalog
# ──────────────────────────────────────────────
def lookup_healpix(target_ids, zpix_file=ZPIX_FILE):
    """Map target_ids to healpix numbers using zpix catalog.

    Returns dict: {target_id: healpix}
    """
    zpix = aTable.Table.read(zpix_file)
    zpix_tids = zpix['TARGETID']
    zpix_hpix = zpix['HEALPIX']

    tid_to_hpix = {}
    for tid in target_ids:
        match = zpix_tids == tid
        if np.any(match):
            tid_to_hpix[int(tid)] = int(zpix_hpix[match][0])
        else:
            print(f"  Warning: TARGETID {tid} not found in zpix")
    return tid_to_hpix


# ──────────────────────────────────────────────
# Step 3: Build coadd URL
# ──────────────────────────────────────────────
def coadd_url(hpix, survey="sv3", program="bright"):
    """Construct the URL for a DESI coadd FITS file."""
    filename = f"coadd-{survey}-{program}-{hpix}.fits"
    return (f"{DESI_BASE_URL}/healpix/{survey}/{program}/"
            f"{str(hpix)[:-2]}/{hpix}/{filename}")


# ──────────────────────────────────────────────
# Step 4: Extract resolution matrix for one target
# ──────────────────────────────────────────────
def extract_resolution(coadd_file, target_id):
    """Extract per-arm resolution data for a single target.

    Returns dict: {band: (wave, res_matrix)}
        where res_matrix is shape (ndiag, nwave_arm)
    """
    hdulist = fits.open(coadd_file, cache=True)
    all_tids = hdulist[1].data['TARGETID']
    idx = np.where(all_tids == target_id)[0]
    if len(idx) == 0:
        raise ValueError(f"TARGETID {target_id} not found in {coadd_file}")
    idx = idx[0]

    result = {}
    waves = {}
    for h in range(2, len(hdulist)):
        extname = hdulist[h].header['EXTNAME']
        band = extname.split('_')[0].lower()
        if 'WAVELENGTH' in extname:
            waves[band] = hdulist[h].data
        if 'RESOLUTION' in extname:
            result[band] = hdulist[h].data[idx]  # (ndiag, nwave_arm)

    hdulist.close()
    return {b: (waves[b], result[b]) for b in result}


# ──────────────────────────────────────────────
# Step 5: Convert banded resolution to sigma(lambda)
# ──────────────────────────────────────────────
def resolution_to_sigma_kms(wave, res_banded):
    """Convert banded resolution matrix to sigma in km/s.

    The resolution matrix is a banded Gaussian. The ratio of
    the first off-diagonal to the diagonal gives sigma in pixels:
        R[center+1]/R[center] = exp(-0.5 / sigma_pix^2)

    Parameters
    ----------
    wave : 1D array, wavelength in Angstroms
    res_banded : 2D array, shape (ndiag, nwave)

    Returns
    -------
    sigma_kms : 1D array, sigma in km/s at each wavelength
    """
    ndiag = res_banded.shape[0]
    center = ndiag // 2

    r_center = res_banded[center, :]
    r_off1 = res_banded[center + 1, :]

    # Avoid bad pixels
    valid = (r_center > 1e-6) & (r_off1 > 1e-6)
    ratio = np.full_like(r_center, np.nan)
    ratio[valid] = r_off1[valid] / r_center[valid]

    # Gaussian: ratio = exp(-0.5 / sigma_pix^2)
    # => sigma_pix = sqrt(-0.5 / ln(ratio))
    sigma_pix = np.full_like(ratio, np.nan)
    good = valid & (ratio > 0) & (ratio < 1)
    sigma_pix[good] = np.sqrt(-0.5 / np.log(ratio[good]))

    # Convert pixels to km/s
    dwave = np.gradient(wave)
    sigma_ang = sigma_pix * dwave
    sigma_kms = sigma_ang / wave * C_KMS

    return sigma_kms

def sigma_kms_to_R(wave, sigma_kms):
    """Convert sigma in km/s to resolving power R = c / (2.355 * sigma)."""
    return C_KMS / (2.355 * sigma_kms)

def desi_resolution(wave):
    """Calibrated DESI R(lambda) from actual resolution matrices."""
    print("Loading calibrated DESI resolution from resolution matrices...")
    cal = np.load(RESULTS_PATH / "desi_lsf_calibration.npz")
    R_median = cal['R_median']
    wave_cal = cal['wave']
    return np.interp(wave, wave_cal, R_median)

def build_desi_resolution_matrix(wave=DESI_WAV):
    """Build sparse resolution matrix from DESI R(lambda) curve.
    
    Parameters
    ----------
    wave : 1D array
        Wavelength grid in Angstroms.
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Shape (nwave, nwave). Multiply by flux to apply LSF.
    """
    c_kms = 299792.458
    R = desi_resolution(wave)
    
    # R(lambda) -> sigma in km/s -> sigma in pixels
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
        kernel = np.exp(-0.5 * ((j - i) / sigma_pix[i])**2)
        kernel /= kernel.sum()
        
        mat[i, lo:hi] = kernel

    return csr_matrix(mat)