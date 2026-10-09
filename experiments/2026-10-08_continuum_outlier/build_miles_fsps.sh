#!/bin/bash
# Build python-fsps with MILES + MIST for the MILES runs, without touching the project venv.
# The source is the unmodified c3k-nzinit 8b14205 copy used for the alpha build (see
# knowledge/_evidence/2026-10-01_afe_build/README.md). The wheel goes into a site dir that run.sh
# puts first on PYTHONPATH. Run inside a compute-node step on ls6. Built 2026-10-08.
# The check at the end prints the library and its resolution. FSPS miles.res gives FWHM 2.54 A
# over 3530-7490 A and a negative value outside.
set -euo pipefail
M=/scratch/11006/nikhilgaruda/hubersed_tmp/miles_build
SRC=/scratch/11006/nikhilgaruda/hubersed_tmp/afe_build/pyfsps_orig
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
mkdir -p $M
[ -d $M/src ] || cp -a $SRC $M/src
cd $M
if ! ls wheel/fsps-*.whl >/dev/null 2>&1; then
    FFLAGS="-DC3K_HR=0 -DC3K_LR=0 -DMILES=1 -DMIST=1" SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0 \
        uv build --wheel -o $M/wheel $M/src > build.log 2>&1 || { tail -30 build.log; exit 1; }
fi
[ -d fsps_site/fsps ] || uv pip install --python $REPO/.venv/bin/python --no-deps \
    --target $M/fsps_site wheel/fsps-*.whl
export OMP_NUM_THREADS=1
PYTHONPATH=$M/fsps_site SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5 $REPO/.venv/bin/python -c '
import numpy as np
import fsps
sp = fsps.StellarPopulation(zcontinuous=1)
print(fsps.__file__, sp.libraries)
w, r = sp.wavelengths, sp.resolutions
for lam in (3500, 3550, 3601.8, 4500, 5500, 7400.8, 7490, 7600):
    i = np.argmin(abs(w - lam))
    print(f"{w[i]:8.1f} A  sigma {r[i]:8.2f} km/s  FWHM {r[i] * w[i] / 299792.458 * 2.355:6.2f} A")
'
