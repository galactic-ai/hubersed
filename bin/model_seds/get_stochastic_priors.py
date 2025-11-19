import sys

import numpy as np

from hubersed.prospector.utils import universe_age_gyr
from hubersed.sampling import *

from hubersed.paths import PATHS

DATA_PATH = PATHS['DATA']

# 500_000 samples
SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000

# from Wan+24 Stochastic prior model

# redshift 0.01 to 0.6 uniform
redshifts = sample_uniform(0.01, 0.6, size=SAMPLE_SIZE)

# stellar mass 9.5 to 12 uniform
stellar_masses = sample_uniform(9.5, 12, size=SAMPLE_SIZE)

# stellar metallicity -1.0 to 0.19 uniform
stellar_metallicities = sample_uniform(-1.0, 0.19, size=SAMPLE_SIZE)

# sigma_reg log uniform 0.1 to 10
sigma_regs = sample_log_uniform(0.1, 10, size=SAMPLE_SIZE)

# tau_eq uniform 0.01 to t_H
t_h = universe_age_gyr(redshifts)
tau_eqs = sample_uniform(0.01, t_h, size=SAMPLE_SIZE)
tau_ins = sample_uniform(0.01, t_h, size=SAMPLE_SIZE)

# sigma_dyn log uniform 0.001 to 0.1
sigma_dyns = sample_log_uniform(0.001, 0.1, size=SAMPLE_SIZE)

# tau_dyn clipped normal min 0.005 max 0.2 mu 0.01 sigma 0.02
tau_dyns = sample_clipped_normal(0.01, 0.02, 0.005, 0.2, size=SAMPLE_SIZE)

# n uniform -1 to 0.4 (dust_index)
ns = sample_uniform(-1.0, 0.4, size=SAMPLE_SIZE)

# tau_dust,2 clipped normal min 0.0 max 4 mu 0.3 sigma 1.0
tau_dust_2s = sample_clipped_normal(0.3, 1.0, 0.0, 4.0, size=SAMPLE_SIZE)

# tau_dust,1 clipped normal min 0.0 max 2 mu 1.0 sigma 0.3 (actually dust_ratio)
tau_dust_1s = sample_clipped_normal(1.0, 0.3, 0.0, 2.0, size=SAMPLE_SIZE)

# U_min clipped normal min 0.1 max 15 mu 2.0 sigma 1.0
u_mins = sample_clipped_normal(2.0, 1.0, 0.1, 15.0, size=SAMPLE_SIZE)

# gamma_e log uniform 1e-4 to 0.1
gamma_es = sample_log_uniform(1e-4, 0.1, size=SAMPLE_SIZE)

# q_pah uniform 0.5 to 7.0
q_pahs = sample_uniform(0.5, 7.0, size=SAMPLE_SIZE)

# sigma_gas uniform 30 to 250
sigma_gass = sample_uniform(30, 250, size=SAMPLE_SIZE)

# gas phase metallicity -2 to 0.5
gas_metallicities = sample_uniform(-2.0, 0.5, size=SAMPLE_SIZE)

# gas ionization parameter -4 to -1
gas_ionization_parameters = sample_uniform(-4.0, -1.0, size=SAMPLE_SIZE)

# save to npz
np.savez(
    f'{DATA_PATH}/stochastic_priors_sample_{SAMPLE_SIZE}.npz',
    redshifts=redshifts,
    stellar_masses=stellar_masses,
    stellar_metallicities=stellar_metallicities,
    sigma_regs=sigma_regs,
    tau_eqs=tau_eqs,
    tau_ins=tau_ins,
    sigma_dyns=sigma_dyns,
    tau_dyns=tau_dyns,
    ns=ns,
    tau_dust_2s=tau_dust_2s,
    tau_dust_1s=tau_dust_1s,
    u_mins=u_mins,
    gamma_es=gamma_es,
    q_pahs=q_pahs,
    sigma_gass=sigma_gass,
    gas_metallicities=gas_metallicities,
    gas_ionization_parameters=gas_ionization_parameters,
)