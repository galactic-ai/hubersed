#!/bin/bash
# Build python-fsps with C3K_ER, MIST and AFE_FLAG for the ER runs, without touching the project
# venv. The source is galactic-ai/python-fsps exp/c3k-er-init at d2e87e2, which adds C3K_ER and
# allocates speclib at runtime. The other static arrays still pass 2 GB with ER and AFE_FLAG
# (spec_ssp_zz alone is 2.07 GB), so the build also needs -mcmodel=medium, as the 2026-10-02 HR
# alpha build did. The wheel goes into a site dir that run.sh puts first on PYTHONPATH, with
# SPS_HOME set to sps_home_er. Run on a compute node.
# The check at the end prints the library, its resolution, and an SSP at afe 0 and 0.4.
set -euo pipefail
B=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
SHA=d2e87e280ea8b4cfe5740d3159d23ab461976b75
mkdir -p $B
cd $B
[ -d src ] || { curl -sL https://github.com/galactic-ai/python-fsps/archive/$SHA.tar.gz | tar xz \
    && mv python-fsps-$SHA src; }
if ! ls wheel/fsps-*.whl >/dev/null 2>&1; then
    FFLAGS="-DC3K_LR=0 -DC3K_HR=0 -DC3K_ER=1 -DMILES=0 -DMIST=1 -DAFE_FLAG=1 -mcmodel=medium" \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0 \
        uv build --wheel -o $B/wheel $B/src > build.log 2>&1 || { tail -30 build.log; exit 1; }
fi
[ -d fsps_site/fsps ] || uv pip install --python $REPO/.venv/bin/python --no-deps \
    --target $B/fsps_site wheel/fsps-*.whl
export OMP_NUM_THREADS=1
PYTHONPATH=$B/fsps_site SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er \
    /usr/bin/time -v $REPO/.venv/bin/python -c '
import numpy as np
import fsps
sp = fsps.StellarPopulation(zcontinuous=1)
print(fsps.__file__, sp.libraries, sp.wavelengths.size)
w, r = sp.wavelengths, sp.resolutions
for lam in (3200, 3300, 3301, 5000, 9999, 10001):
    i = np.argmin(abs(w - lam))
    print(f"{w[i]:8.1f} A  sigma {r[i]:8.2f} km/s")
_, s0 = sp.get_spectrum(tage=10.0, peraa=True)
sp.params["afe"] = 0.4
_, s4 = sp.get_spectrum(tage=10.0, peraa=True)
k = (w > 3300) & (w < 9000)
print("finite", np.isfinite(s0[k]).all(), np.isfinite(s4[k]).all(),
      "afe 0.4 / 0 median ratio", np.median(s4[k] / s0[k]))
' 2>&1 | grep -E "fsps|sigma|finite|Maximum resident|Elapsed"
