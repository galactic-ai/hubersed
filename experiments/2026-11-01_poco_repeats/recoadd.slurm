#!/bin/bash
# SV3 recoadd of one poco_repeats target, as 2026-10-30_recoadd_39633433328091196/recoadd.slurm
# without exposures.py. The four scripts of 2026-10-09_nautilus_c3k_er_broadline run from SRC, a
# copy of the local repository holding that folder with the six targets in targets.py, and whose
# results/ links to the shared clone's results/, since the shared clone holds older versions.
# Inputs come from /scratch (stage_recoadd.sh first). The target's DR1 resolution_matrix.npz goes
# into the output folder, for coadd_rows.py to check its column-weighted rebuild against.
# Submit from the repository root on stampede3:
#   TID=<TARGETID> SRC=<copy> sbatch --export=ALL -J rec_<TID> experiments/2026-11-01_poco_repeats/recoadd.slurm
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/recoadd_%x_%j.log

set -euo pipefail
[ -n "${TID:-}" ] && [ -n "${SRC:-}" ] || { echo "set TID and SRC"; exit 1; }
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
E=$SRC/experiments/2026-10-09_nautilus_c3k_er_broadline
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
DESI=/work/11006/nikhilgaruda/ls6/research/desi_env_0562
DESIPY=$DESI/venv_shared/bin/python
SCR=/scratch/11006/nikhilgaruda/hubersed_data
export IRON_EXPOSURES=$SCR/desi_iron_exposures RECOADD_SURVEY=sv3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HP=$(awk -v t=$TID '$1 == t {print $2; exit}' $REPO/experiments/2026-11-01_poco_repeats/recoadd_exposures.txt)
SPECTRA=$SCR/desi_iron_spectra/spectra-sv3-bright-$HP.fits.gz
OUT=$REPO/results/2026-11-01_poco_repeats/$TID/recoadd
cd $SRC
echo "job $SLURM_JOB_ID TID $TID healpix $HP on $(hostname), start $(date)"
sha256sum $REPO/experiments/2026-11-01_poco_repeats/recoadd.slurm $E/coadd_rows.py \
    $E/fluxcal_rerun.py $E/ivar_check.py $E/ivar_recompute.py $E/targets.py
mkdir -p $OUT
cp $REPO/results/2026-11-01_poco_repeats/$TID/resolution_matrix.npz $OUT/
$PY $E/coadd_rows.py "$SPECTRA" --tid $TID
DESIMODEL=$DESI/desimodel $DESIPY $E/fluxcal_rerun.py --tid $TID
$PY $E/ivar_check.py --tid $TID
$PY $E/ivar_recompute.py --tid $TID
ls -la $OUT
echo "DONE $(date)"
