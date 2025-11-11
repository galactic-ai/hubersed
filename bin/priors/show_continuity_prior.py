import numpy as np
import matplotlib.pyplot as plt

# Prospector 1.4 priors
from prospect.models import priors

from hubersed.prospector.utils import make_agebins_for_z
from hubersed.paths import PATHS

PLOT_PATH = PATHS["RESULTS"]

def sample_continuity_prior(nbins, dt_yr, nsamp=4000, sigma=0.3, df=2, clip=3.0):
    """
    Draw nsamp samples of log(SFR/Mformed)[yr^-1] under the continuity prior:
    logsfr_ratios ~ StudentT(0, sigma, df), clipped to [-clip, clip].
    """
    # Prospector 1.4 StudentT prior (one scalar; we sample nbins-1 times per realization)
    tprior = priors.StudentT(mean=0.0, scale=sigma, df=df)

    out = np.empty((nsamp, nbins))
    for k in range(nsamp):
        # draw adjacent log(SFR_i / SFR_{i+1}) ratios
        logr = np.array([tprior.sample() for _ in range(nbins-1)], dtype=float)
        # apply your hard bounds [-3, 3]
        logr = np.clip(logr, -clip, clip)

        r = np.exp(logr)  # SFR_i / SFR_{i+1}
        # reconstruct SFR pattern up to a constant
        sfr = np.ones(nbins)
        for i in range(1, nbins):
            sfr[i] = sfr[i-1] / r[i-1]

        # Convert to per-bin formed mass and normalize (total mass cancels later,
        # but normalization avoids numerical drift)
        mass = sfr * dt_yr
        mass /= mass.sum()

        # Compute log10(SFR / Mformed) in yr^-1
        sfr_over_M = (mass / dt_yr) / mass.sum()  # = (mass_i/dt_i) / total_mass
        out[k] = np.log10(np.clip(sfr_over_M, 1e-30, None))
    return out

def plot_continuity_panel(ax, t_mids_gyr, samples, title="continuity prior"):
    # credible bands
    p50 = np.percentile(samples, 50, axis=0)
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    p02 = np.percentile(samples, 2.3, axis=0)
    p97 = np.percentile(samples, 97.7, axis=0)

    # 2σ and 1σ
    ax.fill_between(t_mids_gyr, p02, p97, alpha=0.15)
    ax.fill_between(t_mids_gyr, p16, p84, alpha=0.35)
    # median
    ax.plot(t_mids_gyr, p50, lw=1.6)

    # a few random draws
    rng = np.random.default_rng(123)
    for row in samples[rng.choice(samples.shape[0], size=min(6, samples.shape[0]), replace=False)]:
        ax.plot(t_mids_gyr, row, 'r-', lw=0.8, alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlabel("lookback time [Gyr]")
    ax.set_ylabel(r"log(SFR / M$_{\rm formed}$) [yr$^{-1}$]")
    ax.set_title(title)
    ax.set_ylim(-14.5, -8.0)

# ----------------- run examples -----------------
if __name__ == "__main__":
    for z in (0.01, 0.6):
        edges, mids, dt_yr = make_agebins_for_z(z)
        nbins = len(mids)
        samples = sample_continuity_prior(
            nbins=nbins, dt_yr=dt_yr, nsamp=4000,
            sigma=0.3, df=2, clip=3.0
        )

        plt.figure(figsize=(7.2, 4.2))
        plot_continuity_panel(plt.gca(), mids, samples, title=f"Continuity prior (z={z})")
        # Make the x-limits nice: show your full base range but respect truncation
        xmin = max(0.02, edges[0] + 1e-6)
        xmax = edges[-1]
        plt.xlim(xmin, xmax)
        plt.tight_layout()
        plt.savefig(PLOT_PATH / f"continuity_prior_z{z}.png")
        plt.close()
