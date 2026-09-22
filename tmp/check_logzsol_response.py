"""Does FSPS respond to logzsol above the old +0.19 ceiling -- AT THE OPERATING POINT THAT MATTERS?

v1 of this script was WRONG: it fixed logsfr_ratios=0 (constant SFH -> Dn4000~1.22, a YOUNG
population) and concluded "SATURATED / NO-OP". Dn4000's metallicity leverage comes from metal-line
blanketing in COOL stellar atmospheres, so it is weak for young populations and strong for old
ones. Measuring the derivative at Dn4000~1.22 says nothing about galaxies at Dn4000~1.9-2.1.

results/mualpha_step1_zmet.pkl already proves the point (medians, no new compute):
    d(Dn4000) for logzsol 0.00 -> +0.30:  +0.047 at mu_a=-0.5  vs  +0.139 at mu_a=+1.0
    full logz range (-1.0 -> +0.3)     :  +0.141 at mu_a=-0.5  vs  +0.459 at mu_a=+1.0
    Dn4000(mu_a=+1.0, logz=+0.30) = 2.043   vs   Dn4000(mu_a=+1.0, logz=0.00) = 1.905
So the widening past +0.19 IS real. What that grid does NOT cover: it stops at +0.30. This script
tests +0.30 -> +0.40, at alpha where the real massive galaxies live.

Run from hubersed root (venv):  python tmp/check_logzsol_response.py
"""

import sys, copy, warnings
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models.priors import TopHat
from hubersed.prospector.utils import make_stochastic_agebins

Z = 0.075
ALPHA_GRID = [0.0, 1.0, 2.0]  # 0 = control; +1/+2 = real galaxies
ZGRID = [-1.0, -0.5, 0.0, 0.19, 0.25, 0.30, 0.35, 0.40]
OLD_CEILING, NEW_CEILING = 0.19, 0.40

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True
_c, CTEMP = FC.build_continuum_model(Z)

ft = copy.deepcopy(CTEMP)
ft.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
ft["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
ft["use_stellar_ionizing"]["init"] = True
nrat = len(ft["agebins"]["init"]) - 1
ft["logsfr_ratios"]["isfree"] = True
ft["logsfr_ratios"]["init"] = np.zeros(nrat)
ft["logsfr_ratios"]["prior"] = TopHat(mini=np.full(nrat, -5.0), maxi=np.full(nrat, 5.0))
fmodel = SpecModel(ft)
ti = fmodel.theta_index
obs0 = P.build_obs(
    spec=np.ones(len(P.WAVE_OBS)),
    unc=np.ones(len(P.WAVE_OBS)),
    mask=np.ones(len(P.WAVE_OBS), bool),
    resolution=res_lsf,
)
W = P.WAVE_OBS
R = W / (1 + Z)
bmask = (R >= 3850) & (R <= 3950)
rmask = (R >= 4000) & (R <= 4100)

# same alpha parameterization as mualpha_base_set.py / make_cue_model_sed.py
AB = 10 ** make_stochastic_agebins(Z)
MID = AB.mean(1)
DLOG = np.log10(MID[:-1] / MID[1:])


def dn_at(alpha, lz):
    th = np.array(fmodel.theta, dtype=float)
    th[ti["logsfr_ratios"]] = np.clip(
        alpha * DLOG, -10, 10
    )  # mean SFH tilt, no ACF scatter
    th[ti["logmass"]] = 10.0
    th[ti["logzsol"]] = lz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr, _ = fmodel.predict(th, observations=obs0, sps=sps)
    s = np.asarray(pr[0], float).reshape(-1)
    s[s <= 0] = np.nan  # maggies == f_nu -> Balogh OK
    return float(np.nanmean(s[rmask]) / np.nanmean(s[bmask]))


T = {a: {lz: dn_at(a, lz) for lz in ZGRID} for a in ALPHA_GRID}
print("Dn4000, all else FIXED:\n")
print("  logzsol " + "".join(f"  alpha={a:+.1f}" for a in ALPHA_GRID))
for lz in ZGRID:
    mark = "  <- old ceiling" if lz == OLD_CEILING else ""
    print(f"  {lz:+7.2f} " + "".join(f"{T[a][lz]:10.4f}" for a in ALPHA_GRID) + mark)

print(
    f"\n{'alpha':>7}{'Dn(+0.19)':>11}{'Dn(+0.40)':>11}{'delta ABOVE ceiling':>21}{'delta 0.00->0.19':>18}"
)
verdict_alpha = max(ALPHA_GRID)
for a in ALPHA_GRID:
    hi = T[a][NEW_CEILING] - T[a][OLD_CEILING]
    lo = T[a][OLD_CEILING] - T[a][0.0]
    print(
        f"{a:+7.1f}{T[a][OLD_CEILING]:11.4f}{T[a][NEW_CEILING]:11.4f}{hi:+21.4f}{lo:+18.4f}"
    )

hi_old = T[verdict_alpha][NEW_CEILING] - T[verdict_alpha][OLD_CEILING]
print(
    f"\nVERDICT is read at alpha={verdict_alpha:+.1f} (old population; where massive DESI galaxies live),"
)
print("NOT at alpha=0 -- that was v1's mistake.")
if abs(hi_old) < 0.01:
    print(f"*** SATURATED above +0.19 even for OLD populations (delta={hi_old:+.4f}).")
    print(
        "    The +0.19 -> +0.40 widening is a NO-OP; revert logzsol in get_stochastic_priors.py"
    )
    print(
        "    and rely on alpha alone. (The zmet grid's 0.00->+0.30 gain would then need explaining.)"
    )
elif abs(hi_old) < 0.03:
    print(
        f"*** MARGINAL: delta={hi_old:+.4f} above the old ceiling. Widening buys little; alpha does the work."
    )
else:
    print(
        f"OK: delta={hi_old:+.4f} above the old ceiling for old populations. Widening is real -> proceed."
    )
