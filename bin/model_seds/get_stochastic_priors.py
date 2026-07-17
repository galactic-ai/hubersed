import argparse
import sys
from pathlib import Path

import numpy as np

from hubersed.distributions import (
    sample_clipped_normal,
    sample_log_uniform,
    sample_uniform,
)
from hubersed.paths import PATHS
from hubersed.prospector.utils import universe_age_gyr

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Draw the stochastic-SFH prior sample used for generating mock SEDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--sample-size", type=int, default=500_000, help="Number of samples to draw")
    p.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--cue", help="Use Cue (Li+24) nebular model with free N/O, C/O, nH", default=True)
    p.add_argument("-o", "--out", type=Path, default=None, help="Output file path (default: DATA_PATH/prospector_model/stochastic_priors_sample[_cue]_{n}.npz)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing output file if it exists")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    n = args.n
    rng = np.random.default_rng(args.seed)

    data_path = PATHS["DATA"] / "prospector_model"
    data_path.mkdir(parents=True, exist_ok=True)

    out = args.out or data_path / (
        f"stochastic_priors_sample_cue_{n}.npz" if args.cue 
        else f"stochastic_priors_sample_{n}.npz"
    )
    if out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing file {out}. Use --force to overwrite."
        )

    # from Wan+24 Stochastic prior model

    # redshift 0.01 to 0.6 uniform
    redshifts = sample_uniform(0.01, 0.6, size=n, rng=rng)

    # stellar mass 7 to 12 uniform
    stellar_masses = sample_uniform(7, 12, size=n, rng=rng)

    # stellar metallicity -1.5 to 0.4 uniform
    stellar_metallicities = sample_uniform(-1.5, 0.4, size=n, rng=rng)

    # sigma_reg log uniform 0.1 to 5
    sigma_regs = sample_log_uniform(0.1, 5, size=n, rng=rng)

    # tau_eq uniform 0.01 to t_H
    t_h = universe_age_gyr(redshifts)
    tau_eqs = sample_uniform(0.01, t_h, size=n, rng=rng)
    tau_ins = sample_uniform(0.01, t_h, size=n, rng=rng)

    # sigma_dyn log uniform 0.001 to 0.5
    sigma_dyns = sample_log_uniform(0.001, 0.5, size=n, rng=rng)

    # tau_dyn clipped normal min 0.005 max 0.2 mu 0.01 sigma 0.02
    tau_dyns = sample_clipped_normal(0.01, 0.02, 0.005, 0.2, size=n, rng=rng)

    # n uniform -1 to 0.4 (dust_index)
    ns = sample_uniform(-1.0, 0.4, size=n, rng=rng)

    # tau_dust,2 clipped normal min 0.0 max 4 mu 0.3 sigma 1.0
    tau_dust_2s = sample_clipped_normal(0.3, 1.0, 0.0, 4.0, size=n, rng=rng)

    # tau_dust,1 clipped normal min 0.0 max 2 mu 1.0 sigma 0.3 (actually dust_ratio)
    tau_dust_1s = sample_clipped_normal(1.0, 0.3, 0.0, 2.0, size=n, rng=rng)

    # U_min clipped normal min 0.1 max 15 mu 2.0 sigma 1.0
    u_mins = sample_clipped_normal(2.0, 1.0, 0.1, 15.0, size=n, rng=rng)

    # gamma_e log uniform 1e-4 to 0.1
    gamma_es = sample_log_uniform(1e-4, 0.1, size=n, rng=rng)

    # q_pah uniform 0.5 to 7.0
    q_pahs = sample_uniform(0.5, 7.0, size=n, rng=rng)

    # sigma_gas uniform 20 to 250
    sigma_gass = sample_uniform(20, 250, size=n, rng=rng)

    # gas phase metallicity (O/H); Cue grid allows -2.2 (Byler/FSPS used -2.0)
    gas_metallicities = sample_uniform(-2.2 if args.cue else -2.0, 0.5, size=n, rng=rng)

    # gas ionization parameter -4 to -1
    gas_ionization_parameters = sample_uniform(-4.0, -1.0, size=n, rng=rng)

    # top hat min 10 max 400 (not used in Wan+24 but included for completeness)
    sigma_smooths = sample_uniform(10, 400, size=n, rng=rng)

    # save to npz (separate filename for the Cue sample)
    arrays = dict(
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
        sigma_smooths=sigma_smooths,
    )

    # --- Cue (Li+24) free nebular params (arXiv:2405.04598 Table 1) ---
    # gas_logno/gas_logco are LOG10 of (N/O)/(N-O)_sun ; grid is linear [0.1, 5.4] -> log [-1, log10(5.4)]
    if args.cue:
        arrays["gas_lognHs"] = sample_uniform(1.0, 4.0, size=n, rng=rng)  # log nH [cm^-3]
        arrays["gas_lognos"] = sample_uniform(-1.0, np.log10(5.4), size=n, rng=rng)  # log [N/O]
        arrays["gas_logcos"] = sample_uniform(-1.0, np.log10(5.4), size=n, rng=rng)  # log [C/O]

    arrays["_seed"] = args.seed
    arrays["_sample_size"] = n
    arrays["_cue"] = args.cue

    np.savez(out, **arrays)
    print(f"saved {out}  (n={n}, cue={args.cue}, seed={args.seed}, keys={len(arrays)})")
    return 0

    # hffs.put(
    #     f'{DATA_PATH}/stochastic_priors_sample_{n}.npz',
    #     f"buckets/nikhil0504/hubersed-data/prospector_model/stochastic_priors_sample_{n}.npz",
    # )

if __name__ == "__main__":
    sys.exit(main())