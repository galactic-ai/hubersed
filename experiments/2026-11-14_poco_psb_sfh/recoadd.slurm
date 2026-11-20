#!/bin/bash
# SV3 recoadd of one emission-line outlier, as 2026-11-01_poco_repeats/recoadd.slurm. The four scripts
# of 2026-10-09_nautilus_c3k_er_broadline run from the $RUN copy, whose targets.py holds the five
# targets; RECOADD_OUT points them at results/2026-11-14_poco_psb_sfh/<TID>/recoadd. Inputs come from
# /scratch (stage_recoadd.sh first). The DR1 resolution_matrix.npz of the target goes into the output
# folder, for coadd_rows.py to check its column-weighted rebuild against.
# Submit from $RUN:
#   TID=<TARGETID> sbatch --export=ALL -J rec_<TID> experiments/2026-11-14_poco_psb_sfh/recoadd.slurm
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/recoadd_%x_%j.log

set -euo pipefail
[ -n "${TID:-}" ] || { echo "set TID"; exit 1; }
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
RUN=/scratch/11006/nikhilgaruda/claude_runs/2026-11-14_poco_psb_sfh
H=$RUN/experiments/2026-11-14_poco_psb_sfh
E=$RUN/experiments/2026-10-09_nautilus_c3k_er_broadline
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
DESI=/work/11006/nikhilgaruda/ls6/research/desi_env_0562
DESIPY=$DESI/venv_shared/bin/python
SCR=/scratch/11006/nikhilgaruda/hubersed_data
HP=$(awk -v t=$TID '$1 == t {print $2; exit}' $H/recoadd_exposures.txt)
SPECTRA=$SCR/desi_iron_spectra/spectra-sv3-bright-$HP.fits.gz
OUT=$REPO/results/2026-11-14_poco_psb_sfh/$TID/recoadd
export IRON_EXPOSURES=$SCR/desi_iron_exposures RECOADD_SURVEY=sv3 RECOADD_OUT=$OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RUN
echo "job $SLURM_JOB_ID TID $TID healpix $HP on $(hostname), start $(date)"
sha256sum $H/recoadd.slurm $E/coadd_rows.py $E/fluxcal_rerun.py $E/ivar_check.py \
    $E/ivar_recompute.py $E/targets.py
mkdir -p $OUT
cp $REPO/results/2026-11-14_poco_psb_sfh/$TID/resolution_matrix.npz $OUT/
$PY $E/coadd_rows.py "$SPECTRA" --tid $TID
DESIMODEL=$DESI/desimodel $DESIPY $E/fluxcal_rerun.py --tid $TID
$PY $E/ivar_check.py --tid $TID
$PY $E/ivar_recompute.py --tid $TID
ls -la $OUT
echo "DONE $(date)"
