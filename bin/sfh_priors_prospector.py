import numpy as np
import matplotlib.pyplot as plt

from prospect.models import priors

from hubersed.paths import PATHS

PLOT_PATH = PATHS['RESULTS']

t_edges = np.array([0, 0.03, 0.10, 0.33, 1.10, 3.60, 11.70, 13.70])  # Gyr
nbins   = len(t_edges) - 1
t_mids  = 0.5 * (t_edges[1:] + t_edges[:-1])
dt_gyr  = t_edges[1:] - t_edges[:-1]
dt_yr   = dt_gyr * 1e9  # convert to years

def log_sfr_over_mformed(mass):
    # SFR_i = mass_i / width_i ; normalize by total formed mass
    sfr = mass / dt_yr
    sfr_mformed = sfr / np.sum(mass)
    return np.log10(np.clip(sfr_mformed, 1e-30, None))

def sample_logM_in_bins(nsamp=4000, logm_min=-3, logm_max=3):
    # independent log-mass per bin
    lu = priors.LogUniform(mini=10**logm_min, maxi=10**logm_max)  # linear mass limits
    out = []
    for _ in range(nsamp):
        m = np.array([lu.sample() for _ in range(nbins)], dtype=float)
        out.append(log_sfr_over_mformed(m))
    return np.vstack(out)


def sample_dirichlet(alpha=1.0, nsamp=4000):
    # fractions add to 1 across bins
    out = []
    for _ in range(nsamp):
        frac = np.random.dirichlet(alpha * np.ones(nbins))
        # convert fractions (of total mass) to per-bin mass (arbitrary total mass)
        mass = frac  # total Mformed = 1
        out.append(log_sfr_over_mformed(mass))
    return np.vstack(out)


def sample_continuity(nsamp=4000, sigma=0.3, df=2):
    # Student-t on log SFR ratios between adjacent bins
    tprior = priors.StudentT(mean=0.0, scale=sigma, df=df)
    out = []
    for _ in range(nsamp):
        # draw log ratios r_j = log(SFR_j/SFR_{j+1})
        log_r = np.array([tprior.sample() for _ in range(nbins-1)], dtype=float)
        r = np.exp(log_r)
        # build SFRs up to normalization
        sfr = np.ones(nbins)
        for i in range(1, nbins):
            sfr[i] = sfr[i-1] / r[i-1]
        # convert to mass by multiplying by bin widths; renormalize to total mass = 1
        mass = sfr * dt_yr
        mass /= mass.sum()
        out.append(log_sfr_over_mformed(mass))
    return np.vstack(out)

def sample_continuity_flex(nsamp=4000, sigma=0.3, df=2):
    # Same log-ratio prior, but *equal mass per bin* (flexible bin widths).
    # For plotting on a fixed x-axis, we approximate by computing the SFR that would
    # yield equal-mass bins given those ratios, then map onto the fixed midpoints.
    tprior = priors.StudentT(mean=0.0, scale=sigma, df=df)
    out = []
    Ttot = t_edges[-1] - t_edges[0]  # total lookback span in Gyr

    for _ in range(nsamp):
        # draw SFR pattern (up to a constant) from continuity ratios
        log_r = np.array([tprior.sample() for _ in range(nbins-1)], float)
        r = np.exp(log_r)
        sfr = np.ones(nbins)
        for i in range(1, nbins):
            sfr[i] = sfr[i-1] / r[i-1]

        # enforce equal mass: Δt_i ∝ 1/sfr_i (in Gyr), normalize to Ttot
        dt_var_gyr = (1.0 / sfr)
        dt_var_gyr *= Ttot / dt_var_gyr.sum()
        dt_var_yr = dt_var_gyr * 1e9

        # make cumulative time edges for this realization
        te = np.concatenate([[0.0], np.cumsum(dt_var_gyr)])
        # step SFR/Mformed in yr^-1 (mass per bin equal -> Mformed_i = 1/nbins)
        # SFR_i/Mformed = ( (1/nbins) / Δt_i(yr) ) / 1  = 1/(nbins*Δt_i)
        step_vals = 1.0 / (nbins * dt_var_yr)  # yr^-1
        log_step  = np.log10(np.clip(step_vals, 1e-30, None))

        # evaluate this step function at the fixed t_mids (rescale to actual span)
        # our flexible timeline spans [0, Ttot]; map t_mids into that span:
        tm = np.clip(t_mids, 0, Ttot)
        # find which flexible bin each midpoint falls in
        idx = np.minimum(np.searchsorted(te[1:], tm), nbins-1)
        out.append(log_step[idx])

    return np.vstack(out)

def panel(ax, samples, title):
    # density via percentiles; overlay a few random draws
    p50 = np.percentile(samples, 50, axis=0)
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    p02 = np.percentile(samples, 2.3, axis=0)
    p97 = np.percentile(samples, 97.7, axis=0)

    # blue bands
    ax.fill_between(t_mids, p02, p97, alpha=0.15)
    ax.fill_between(t_mids, p16, p84, alpha=0.35)
    ax.plot(t_mids, p50, lw=1.5)

    # a few realizations (red)
    for row in samples[np.random.choice(samples.shape[0], 6, replace=False)]:
        ax.plot(t_mids, row, 'r-', lw=0.8, alpha=0.6)

    ax.set_xscale('log')
    ax.set_xlim(0.02, 13.7)
    ax.set_ylim(-14.5, -8.0)
    ax.set_xlabel('lookback time [Gyr]')
    ax.set_title(title)

np.random.seed(7)
fig, axs = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

panel(axs[0], sample_logM_in_bins(logm_min=-15, logm_max=-8), 'log(M) in bins')
panel(axs[1], sample_dirichlet(alpha=1.0),   'Dirichlet α=1.0')
panel(axs[2], sample_dirichlet(alpha=0.2),   'Dirichlet α=0.2')
panel(axs[3], sample_continuity(sigma=0.3),  'continuity')
panel(axs[4], sample_continuity_flex(sigma=0.3),   'flexible time bins')

axs[0].set_ylabel('log(SFR / Mformed)')
plt.tight_layout()
plt.savefig(f'{PLOT_PATH}/sfh_priors_prospector.pdf', dpi=300)


