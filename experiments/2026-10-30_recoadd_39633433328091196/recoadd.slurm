#!/bin/bash
# Recoadd the continuum outlier 39633433328091196 (SV3 bright, 3 exposures) on stampede3, as
# experiments/2026-10-29_continuum_recoadd/recoadd.slurm did for 39627757533007793, and save its
# single-exposure spectra (exposures.py). No reference rerun: the 10-29 job checked the stampede3
# setup against ls6. Inputs come from $SCR (stage.sh on the login node first). The DR1
# resolution_matrix.npz of 2026-10-29_continuum_39633433328091196_map goes into the output folder,
# for coadd_rows.py to check its column-weighted rebuild against.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-30_recoadd_39633433328091196/recoadd.slurm
#SBATCH -J recoadd1030
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/recoadd1030_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
E=experiments/2026-10-09_nautilus_c3k_er_broadline
HERE=experiments/2026-10-30_recoadd_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
DESI=/work/11006/nikhilgaruda/ls6/research/desi_env_0562
DESIPY=$DESI/venv_shared/bin/python
SCR=/scratch/11006/nikhilgaruda/hubersed_data
export IRON_EXPOSURES=$SCR/desi_iron_exposures
SPECTRA=$SCR/desi_iron_spectra/spectra-sv3-bright-10231.fits.gz
OUT=results/2026-10-30_recoadd_39633433328091196/sv3
TID=39633433328091196
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RECOADD_SURVEY=sv3

echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $HERE/recoadd.slurm $HERE/exposures.py $E/coadd_rows.py $E/fluxcal_rerun.py \
    $E/ivar_check.py $E/ivar_recompute.py $E/targets.py
mkdir -p $OUT
cp results/2026-10-29_continuum_39633433328091196_map/resolution_matrix.npz $OUT/
$PY $HERE/exposures.py "$SPECTRA"
$PY $E/coadd_rows.py "$SPECTRA" --tid $TID
DESIMODEL=$DESI/desimodel $DESIPY $E/fluxcal_rerun.py --tid $TID
$PY $E/ivar_check.py --tid $TID
$PY $E/ivar_recompute.py --tid $TID
echo "DONE $(date)"
