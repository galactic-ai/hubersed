#!/bin/bash
# Build python-fsps with C3K_HR, MIST and AFE_FLAG for the ladder steps c3k_hr and c3k_hr_afe, as
# 2026-10-09_c3k_er/build_er_fsps.sh with the library flags changed. The source is a copy of
# fsps_builds/c3k_er_afe_v3/src (C3K_ER, allocatable speclib, build_ssps), so the code is that of
# the ER fits. SPS_HOME is sps_home_hr_afe: a symlink copy of fsps_5 with the 52 non-solar afe
# files of c3khr_allfal_v2.3_final added to SPECTRA/C3K/c3k_hr. The check at the end prints the
# library, nafe, its resolution and an SSP at afe 0 and 0.4. Submit from the repository root:
#   sbatch experiments/2026-11-01_poco_repeats/build_hr_afe_fsps.slurm
#SBATCH -J hrafebuild
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/hrafebuild_%j.log
set -euo pipefail
R=/work/11006/nikhilgaruda/ls6/research
B=$R/fsps_builds/c3k_hr_afe
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
export SPS_HOME=$R/sps_home_hr_afe
echo "job ${SLURM_JOB_ID:-none} on $(hostname), start $(date)"
echo "c3k_hr spec files: $(ls $SPS_HOME/SPECTRA/C3K/c3k_hr/*.spec.bin | wc -l)"
mkdir -p $B
cd $B
[ -d src ] || cp -r $R/fsps_builds/c3k_er_afe_v3/src src
if ! ls wheel/fsps-*cp313*.whl >/dev/null 2>&1; then
    FFLAGS="-DC3K_LR=0 -DC3K_HR=1 -DC3K_ER=0 -DMILES=0 -DMIST=1 -DAFE_FLAG=1 -mcmodel=medium" \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0 \
        uv build --python $PY --wheel -o $B/wheel $B/src > build.log 2>&1 || { tail -30 build.log; exit 1; }
fi
[ -d fsps_site/fsps ] || uv pip install --python $PY --no-deps --target $B/fsps_site wheel/fsps-*cp313*.whl
export OMP_NUM_THREADS=1
PYTHONPATH=$B/fsps_site /usr/bin/time -v $PY -c '
import numpy as np
import fsps
sp = fsps.StellarPopulation(zcontinuous=1)
print(fsps.__file__, sp.libraries, sp.wavelengths.size, "nafe", fsps.fsps.driver.get_nafe())
w, r = sp.wavelengths, sp.resolutions
for lam in (3000, 3001, 5000, 9999, 10001):
    i = np.argmin(abs(w - lam))
    print(f"{w[i]:8.1f} A  sigma {r[i]:8.2f} km/s")
_, s0 = sp.get_spectrum(tage=10.0, peraa=True)
sp.params["afe"] = 0.4
_, s4 = sp.get_spectrum(tage=10.0, peraa=True)
k = (w > 3300) & (w < 9000)
print("finite", np.isfinite(s0[k]).all(), np.isfinite(s4[k]).all(),
      "afe 0.4 / 0 median ratio", np.median(s4[k] / s0[k]))
' 2>&1 | grep -E "fsps|sigma|finite|Maximum resident|Elapsed"
echo "end $(date)"
