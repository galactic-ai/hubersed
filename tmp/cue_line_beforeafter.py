import warnings; warnings.filterwarnings("ignore")
import importlib.util as U, numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prospect.sources import NebStepBasis

IDX=278060
H5="data/prospector_model/prospector_stochastic_model_seds_cue_500000.h5"
NEW="/Users/ng27753/Documents/UberSED/patches/cue_emlines_info.dat"

s=U.spec_from_file_location("mms","bin/model_seds/make_model_seds.py")
m=U.module_from_spec(s); s.loader.exec_module(m)
m._init_worker("cue",500000,42)

with h5py.File(H5,"r") as f:
    z=float(f["priors/redshifts"][IDX]); stored=f["fluxes"][IDX][:]; old_wave=f["line_wave"][:]
DW=np.asarray(m.DESI_WAV,float); R=m.build_desi_resolution_matrix(DW); obs=m.make_obs(DW.size)
new_wave=np.genfromtxt(NEW,delimiter=",",usecols=0)

def run(patch):
    sps=NebStepBasis()
    if patch: sps.emline_wavelengths=new_wave.copy()
    parset,_=m.build_parset_for_index(IDX)
    mod=m.HyperSpecModel(configuration=parset)
    preds,_=mod.predict(mod.theta,[obs],sps=sps)
    return R.dot(preds[0])

f_old=run(False); f_new=run(True)
print("old vs stored max|Δ|:",np.max(np.abs(f_old-stored)))

from hubersed.plotting.style import use_apj_style
from hubersed.plotting.spectra import plot_spectrum, REST_WAVE_LABEL
use_apj_style()
def near(arr,w): return float(arr[np.argmin(np.abs(arr-w))])
models=[{"flux":f_new,"label":"after (corrected)","color":"C3","lw":1.0},
        {"flux":f_old,"label":"before (buggy)","color":"0.35","ls":"--","lw":1.0}]
zooms=[("[OII] 3726,3729",[3726,3729]),("Hb + [OIII]",[4862.68,4960.29,5008.24]),
("Ha + [NII]",[6549.86,6564.61,6585.27]),("[SII] 6716,6731",[6718.29,6732.67])]

fig=plt.figure(figsize=(9,5.2)); gs=fig.add_gridspec(2,4,height_ratios=[2,1.3],hspace=0.32,wspace=0.32)
ax=fig.add_subplot(gs[0,:])
plot_spectrum(ax,DW,z=z,models=models)
ax.set_xlim(3850/(1+z),7150/(1+z)); ax.set_xlabel(REST_WAVE_LABEL)
for j,(t,ws) in enumerate(zooms):
    a=fig.add_subplot(gs[1,j])
    plot_spectrum(a,DW,z=z,models=models); a.get_legend().remove()
    for w in ws:
        a.axvline(near(old_wave,w),ls="--",lw=0.8,c="0.4",alpha=0.8)
        a.axvline(near(new_wave,w),ls="--",lw=0.8,c="C3",alpha=0.8)
    a.set_xlim(min(ws)-6,max(ws)+6); a.set_title(t,fontsize=8); a.set_xlabel(REST_WAVE_LABEL,fontsize=8)
    if j>0: a.set_ylabel("")
fig.savefig("tmp/cue_line_beforeafter.png",dpi=150,bbox_inches="tight")
print("saved tmp/cue_line_beforeafter.png")
