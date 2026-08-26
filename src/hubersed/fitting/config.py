import copy

import numpy as np

from prospect.models.priors import TopHat, ClippedNormal, LogUniform, Uniform
from prospect.models.transforms import dustratio_to_dust1
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.models.sedmodel import HyperSpecModel

from hubersed.prospector.utils import make_stochastic_agebins, universe_age_gyr

# ── Default initial values ───────────────────────────────────────────────────
DEFAULT_SET_VALS = {
    "logmass": 8.7,
    "logzsol": -0.1,
    "sigma_reg": 0.17,
    "tau_eq": 2.5,
    "sigma_dyn": 0.005,
    "tau_dyn": 0.025,
    "dust_index": 0.0,
    "dust2": 0.1,
    "dust_ratio": 1.0,
    "tau_in": 0.1,
}


def get_priors(redshift):
    tau_max = universe_age_gyr(redshift)
    return {
        "logmass": Uniform(mini=7.0, maxi=12.0),
        "logzsol": Uniform(mini=-2.5, maxi=0.5),
        "sigma_reg": LogUniform(mini=0.1, maxi=5.0),
        "tau_eq": Uniform(mini=0.01, maxi=tau_max),
        "tau_in": Uniform(mini=0.01, maxi=tau_max),
        "sigma_dyn": LogUniform(mini=0.001, maxi=0.5),
        "tau_dyn": ClippedNormal(mean=0.01, sigma=0.02, mini=0.005, maxi=0.2),
        "dust_index": TopHat(mini=-2.5, maxi=0.4),
        "dust2": ClippedNormal(mean=0.3, sigma=1.0, mini=0.0, maxi=4.0),
        "dust_ratio": ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0),
    }


def build_continuum_model(redshift, logmass_init=None):
    """Build the continuum-only HyperSpecModel."""
    tau_in = universe_age_gyr(redshift)
    set_vals = DEFAULT_SET_VALS.copy()
    set_vals["tau_in"] = tau_in
    if logmass_init is not None:
        set_vals["logmass"] = logmass_init
    set_priors = get_priors(redshift)

    template = copy.deepcopy(TemplateLibrary["stochastic_sfh"])
    dust_template = copy.deepcopy(TemplateLibrary["dust_emission"])
    template.update(dust_template)

    template["zred"]["init"] = redshift
    template["zred"]["isfree"] = False

    template["dust_ratio"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "units": "ratio of birth-cloud to diffuse dust",
        "prior": set_priors["dust_ratio"],
    }
    template["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": np.float64(0.0),
        "units": "power-law multiplication of Calzetti",
        "prior": set_priors["dust_index"],
    }
    template["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": dustratio_to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
    }

    for key in set_vals:
        if key in template:
            template[key]["init"] = set_vals[key]
            template[key]["prior"] = set_priors[key]
            template[key]["isfree"] = key not in [
                "tau_eq",
                "tau_in",
                "sigma_dyn",
                "tau_dyn",
                "sigma_reg",
            ]
            # The continuum is dominated by old stars that accumulated over billions of years
            # — it cares about total mass at broad age ranges, not fine-grained SFH variations.
            # The hyperparameters control bin-to-bin correlations and burstiness, which barely
            # affect the cumulative light. Emission lines are the opposite — they trace stars
            # younger than ~10 Myr, where the exact recent SFH shape
            # (controlled by the hyperparameters) matters enormously.

    template["agebins"] = {
        "N": 10,
        "isfree": False,
        "init": make_stochastic_agebins(redshift),
        "units": "log10(yr)",
    }
    template["dust_type"]["init"] = 4
    template["sigma_smooth"] = {
        "N": 1,
        "isfree": True,
        "init": 200.0,
        "units": "km/s",
        "prior": TopHat(mini=10.0, maxi=400.0),
    }
    template["smoothtype"] = {"N": 1, "isfree": False, "init": "vel"}
    template["fftsmooth"] = {"N": 1, "isfree": False, "init": True}

    template = adjust_stochastic_params(template)
    return HyperSpecModel(template), template


def build_full_model(continuum_template, theta_best_cont, cont_model, redshift):
    """Build the full nebular HyperSpecModel seeded from continuum MAP."""
    nebular_template = copy.deepcopy(TemplateLibrary["nebular"])
    full_template = copy.deepcopy(continuum_template)
    full_template.update(nebular_template)

    # Prospector handles emission lines separately from FSPS
    # so we can apply independent gas velocity dispersion
    full_template["nebemlineinspec"] = {
        "N": 1,
        "isfree": False,
        "init": False,
    }

    vary_params = [
        "logsfr_ratios",
        "gas_logz",
        "gas_logu",
        "dust2",
        "dust_ratio",
        "dust_index",
        "sigma_reg",
        "tau_eq",
        "sigma_dyn",
        "tau_dyn",
        "tau_in",
    ]
    for key in full_template:
        full_template[key]["isfree"] = key in vary_params

    # Seed from continuum best-fit
    for key in ["logmass", "logzsol", "sigma_smooth"]:
        full_template[key]["isfree"] = True
        full_template[key]["init"] = float(
            theta_best_cont[cont_model.theta_index[key]][0]
        )

    full_template["gas_logz"] = {
        "N": 1,
        "isfree": True,
        "init": 0.0,
        "prior": TopHat(mini=-2.0, maxi=0.5),
    }
    full_template["gas_logu"] = {
        "N": 1,
        "isfree": True,
        "init": -2.5,
        "prior": TopHat(mini=-4.0, maxi=-1.0),
    }

    # Gas velocity dispersion — independent from stellar sigma_smooth
    full_template["eline_sigma"] = {
        "N": 1,
        "isfree": True,
        "init": 200.0,
        "units": "km/s",
        "prior": TopHat(mini=10.0, maxi=250.0),
    }


    full_template = adjust_stochastic_params(full_template)
    return HyperSpecModel(full_template), full_template


# use Cue
def build_full_cue_model(continuum_template, theta_best_cont, cont_model, redshift):
    full_template = copy.deepcopy(continuum_template)
    nebular = copy.deepcopy(TemplateLibrary["cue_stellar_nebular"])
    full_template.update(nebular)

    # independent handle of emission lines in spectrum
    full_template["nebemlineinspec"] = {
            "N": 1,
            "isfree": False,
            "init": False,
    }

    # vary same set as before plus her gas params
    vary_params = [
        "logsfr_ratios",
        "gas_logu",
        "gas_logz",
        "gas_lognH",
        "gas_logno",
        "gas_logco",
        "gas_logqion",
        "dust2",
        "dust_ratio",
        "dust_index",
        "sigma_reg",
        "tau_eq",
        "sigma_dyn",
        "tau_dyn",
        "tau_in",
    ]
    for key in full_template:
        full_template[key]["isfree"] = key in vary_params

    for key in ["logmass", "logzsol", "sigma_smooth"]:
        full_template[key]["isfree"] = True
        full_template[key]["init"] = float(
            theta_best_cont[cont_model.theta_index[key]][0]
        )

    full_template["eline_sigma"] = {
        "N": 1,
        "isfree": True,
        "init": 100.0,
        "units": "km/s",
        "prior": TopHat(mini=10.0, maxi=250.0),
    }

    # full_template["gas_logqion"] = {
    #     "N": 1,
    #     "isfree": True,
    #     "init": 49.5,
    #     "prior": TopHat(mini=46.0, maxi=52.0),
    #     "units": "log10(ionizing photons / s)",
    # }

    full_template = adjust_stochastic_params(full_template)
    return HyperSpecModel(full_template), full_template
