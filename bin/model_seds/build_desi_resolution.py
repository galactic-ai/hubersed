# %%
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from tqdm import tqdm

from hubersed.prospector.lsf import (
    DESI_WAV,
    C_KMS,
    resolution_to_sigma_kms,
    sigma_kms_to_R,
    sample_target_ids,
    lookup_healpix,
    coadd_url,
)
from hubersed.paths import PATHS

DATA_PATH = PATHS["DATA"]
RESULTS_PATH = PATHS["RESULTS"]


def desi_resolution_design(wave):
    """DESI design-spec R(lambda)."""
    R = np.zeros_like(wave, dtype=float)
    b = (wave >= 3600) & (wave < 5930)
    r = (wave >= 5930) & (wave < 7470)
    z = wave >= 7470
    R[b] = 2000 + (3200 - 2000) * (wave[b] - 3600) / (5930 - 3600)
    R[r] = 3200 + (4100 - 3200) * (wave[r] - 5930) / (7720 - 5930)
    R[z] = 4100 + (5100 - 4100) * (wave[z] - 7470) / (9800 - 7470)
    return R


# Main pipeline
def main(n_sample=10):
    files = sorted(glob.glob(str(DATA_PATH / "DESIchunk1024_*.pkl")))
    sample_files = files[::25]  # every 25th file, ~10 files
    print(f"Sampling from {len(sample_files)} files: {sample_files}")
    output_dir = str(RESULTS_PATH)

    # Step 1: Sample target_ids
    # print(f"Step 1: Sampling {n_sample} target_ids from {pkl_file}")
    samples_tids = []

    for pkl_file in sample_files:
        sample_tids = sample_target_ids(pkl_file, n_sample=n_sample)
        samples_tids.extend(sample_tids)
    print(f"  Got {len(samples_tids)} target_ids")

    # Step 2: Look up healpix
    tid_to_hpix = lookup_healpix(samples_tids)
    print(f"  Matched {len(tid_to_hpix)} / {len(samples_tids)} targets")

    # Group by healpix to minimize downloads
    hpix_to_tids = {}
    for tid, hpix in tid_to_hpix.items():
        hpix_to_tids.setdefault(hpix, []).append(tid)
    print(f"  Targets span {len(hpix_to_tids)} unique healpix")

    # Step 3-5: Download and extract resolution matrices
    print("Steps 3-5: Downloading coadds and extracting resolution matrices")
    all_wave = []
    all_sigma = []

    for hpix, tids in tqdm(hpix_to_tids.items(), desc="Processing healpix"):
        try:
            url = coadd_url(hpix)
            hdulist = fits.open(url, cache=True)
        except Exception as e:
            print(f"  Failed to download healpix {hpix}: {e}")
            continue

        # Read all needed data once per healpix
        all_tids = hdulist[1].data["TARGETID"]
        waves = {}
        res_data = {}
        for h in range(2, len(hdulist)):
            extname = hdulist[h].header["EXTNAME"]
            band = extname.split("_")[0].lower()
            if "WAVELENGTH" in extname:
                waves[band] = hdulist[h].data
            if "RESOLUTION" in extname:
                res_data[band] = hdulist[h].data

        for tid in tids:
            idx = np.where(all_tids == tid)[0]
            if len(idx) == 0:
                print(f"  TARGETID {tid} not in healpix {hpix}")
                continue
            idx = idx[0]

            for band in ["b", "r", "z"]:
                if band not in waves or band not in res_data:
                    continue
                wave_arm = waves[band]
                res_arm = res_data[band][idx]  # (ndiag, nwave_arm)
                sigma = resolution_to_sigma_kms(wave_arm, res_arm)
                all_wave.append(wave_arm)
                all_sigma.append(sigma)

        hdulist.close()

    print(f"  Extracted {len(all_sigma)} arm-level resolution curves")

    # Step 6: Compute median R(lambda) on the common grid
    print("Step 6: Computing median sigma(lambda) and R(lambda)")

    # Interpolate all curves onto the common DESI wavelength grid
    sigma_grid = np.full((len(all_sigma), len(DESI_WAV)), np.nan)
    for i, (w, s) in enumerate(zip(all_wave, all_sigma)):
        good = np.isfinite(s)
        if np.sum(good) < 10:
            continue
        sigma_grid[i] = np.interp(DESI_WAV, w[good], s[good], left=np.nan, right=np.nan)

    # Compute statistics
    sigma_median = np.nanmedian(sigma_grid, axis=0)
    sigma_16 = np.nanpercentile(sigma_grid, 16, axis=0)
    sigma_84 = np.nanpercentile(sigma_grid, 84, axis=0)

    R_median = sigma_kms_to_R(DESI_WAV, sigma_median)
    R_design = desi_resolution_design(DESI_WAV)

    # Step 7: Save results
    print(f"Step 7: Saving results to {output_dir}")

    np.savez(
        os.path.join(output_dir, "desi_lsf_calibration.npz"),
        wave=DESI_WAV,
        sigma_median_kms=sigma_median,
        sigma_16_kms=sigma_16,
        sigma_84_kms=sigma_84,
        R_median=R_median,
        R_design=R_design,
    )

    # ── Plot 1: R(lambda) comparison ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(
        DESI_WAV,
        sigma_kms_to_R(DESI_WAV, sigma_84),
        sigma_kms_to_R(DESI_WAV, sigma_16),
        alpha=0.3,
        color="C0",
        label="16-84th percentile",
    )
    ax.plot(DESI_WAV, R_median, "C0-", lw=2, label="Median (actual)")
    ax.plot(DESI_WAV, R_design, "r--", lw=2, label="Design spec")
    ax.set_ylabel("Resolving power R")
    ax.legend()
    ax.set_title("DESI LSF: Actual vs Design Specification")
    ax.axvline(5930, color="gray", ls=":", alpha=0.5, label="B/R boundary")
    ax.axvline(7470, color="gray", ls=":", alpha=0.5, label="R/Z boundary")

    ax = axes[1]
    ax.fill_between(DESI_WAV, sigma_16, sigma_84, alpha=0.3, color="C0")
    ax.plot(DESI_WAV, sigma_median, "C0-", lw=2, label="Median (actual)")
    sigma_design = C_KMS / (2.355 * R_design)
    sigma_design[R_design == 0] = np.nan
    ax.plot(DESI_WAV, sigma_design, "r--", lw=2, label="Design spec")
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("σ_inst [km/s]")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lsf_comparison.png"), dpi=150)
    plt.show()
    print("Done!")

    # ── Print summary ──
    print("\nSummary at key wavelengths:")
    for lam in [3800, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8500]:
        idx = np.argmin(np.abs(DESI_WAV - lam))
        print(
            f"  λ={lam}Å: R_actual={R_median[idx]:.0f}, "
            f"R_design={R_design[idx]:.0f}, "
            f"σ_actual={sigma_median[idx]:.1f} km/s, "
            f"σ_design={sigma_design[idx]:.1f} km/s"
        )


if __name__ == "__main__":
    main()
