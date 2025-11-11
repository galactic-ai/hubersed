from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.plotting.utils import sample_prior
from prospect.sources import FastStepBasis
from prospect.models.sedmodel import HyperSpecModel

import numpy as np
import matplotlib.pyplot as plt

from hubersed.prospector.utils import make_stochastic_agebins
from hubersed.prospector.derived_quantities import compute_logssfr
from hubersed.paths import PATHS

PLOT_PATH = PATHS['RESULTS']

base_template = TemplateLibrary['stochastic_sfh']

# age bins N=10, first two bins (1-5) Myr, (5-10)Myr, then remaining bins equally spaced in log time between 10 Myr and 0.95*t_univ

z = 0.7
age_bins_log = make_stochastic_agebins(z)

base_template['zred']['init'] = z
base_template['agebins']['init'] = age_bins_log

# (σreg/dex, τeq/Gyr, τin/Gyr, σdyn/dex, τdyn/Gyr) = (1.0, 1.0, 1.0, 0.5, 0.01)
base_template['sigma_reg']['init'] = 5.0
base_template['tau_eq']['init'] = 1.0
base_template['tau_in']['init'] = 1.0
base_template['sigma_dyn']['init'] = 0.5
base_template['tau_dyn']['init'] = 0.01

# make them fixed
for param in ['sigma_reg', 'tau_eq', 'tau_in', 'sigma_dyn', 'tau_dyn']:
    base_template[param]['isfree'] = False

base_template = adjust_stochastic_params(base_template)

sps = FastStepBasis()

model = HyperSpecModel(base_template)

Ndraws = 10_000
logssfrs = np.zeros(Ndraws)
logpriors = np.zeros(Ndraws)

thetas, labels = sample_prior(model, nsample=Ndraws)
for i in range(Ndraws):
    logssfrs[i] = compute_logssfr(model, thetas[i])
    logpriors[i] = model.prior_product(thetas[i])  # log of total prior probability

plt.figure(figsize=(6, 4))
plt.hist(logssfrs, bins=50, density=True, alpha=0.7, range=(-14.5, -8))
plt.xlabel('log(sSFR / yr$^{-1}$)')
plt.ylabel('Prior probability density')

plt.tight_layout()
plt.savefig(f'{PLOT_PATH}/stochastic_sfh_logssfr_prior.pdf', dpi=300)
