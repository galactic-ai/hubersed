#!/bin/bash
# Recoadd the continuum outlier 39627757533007793 (SV3 and main) on stampede3, as
# experiments/2026-10-18_candidate_recoadd/recoadd.slurm did for the line outliers on ls6.
#
# 1. Rerun the four 2026-10-09 scripts for the #14 reference 39627770174637084 into its own folder
#    (RECOADD_OUT) and compare with the ls6 outputs (compare_reference.py). The comparison is
#    reported and does not stop the job; the first line of coadd_rows.txt names the input path,
#    which differs here.
# 2. Run them for 39627757533007793, SV3 (RECOADD_SURVEY=sv3) and main (RECOADD_SURVEY=main).
#
# Compute nodes do not mount Corral, so the iron spectra and exposure files are read from the
# copies in $SCR (staged on the login node first). Each target folder gets the DR1
# resolution_matrix.npz that coadd_rows.py checks its column-weighted rebuild against.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-29_continuum_recoadd/recoadd.slurm
#SBATCH -J recoadd1029
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/recoadd1029_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
E=experiments/2026-10-09_nautilus_c3k_er_broadline
HERE=experiments/2026-10-29_continuum_recoadd
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
DESI=/work/11006/nikhilgaruda/ls6/research/desi_env_0562
DESIPY=$DESI/venv_shared/bin/python
SCR=/scratch/11006/nikhilgaruda/hubersed_data
export IRON_EXPOSURES=$SCR/desi_iron_exposures
SPEC=$SCR/desi_iron_spectra
OUT=results/2026-10-29_continuum_recoadd
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $HERE/recoadd.slurm $E/coadd_rows.py $E/fluxcal_rerun.py $E/ivar_check.py \
    $E/ivar_recompute.py $E/targets.py experiments/2026-10-18_candidate_recoadd/compare_reference.py
$DESIPY -c "import desispec, numpy, sys; print('desispec', desispec.__version__, 'numpy', numpy.__version__, sys.version.split()[0])"

run_target() {
    local tid=$1 spectra=$2
    echo "=== $tid survey ${RECOADD_SURVEY:-sv3} out ${RECOADD_OUT:-default} $(date)"
    $PY $E/coadd_rows.py "$spectra" --tid "$tid"
    DESIMODEL=$DESI/desimodel $DESIPY $E/fluxcal_rerun.py --tid "$tid"
    $PY $E/ivar_check.py --tid "$tid"
    $PY $E/ivar_recompute.py --tid "$tid"
}

# 1. reference check
REF=results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084
mkdir -p $OUT/reference_stampede3
cp $REF/resolution_matrix.npz $OUT/reference_stampede3/
export RECOADD_OUT=$REPO/$OUT/reference_stampede3
run_target 39627770174637084 "$SPEC/spectra-sv3-bright-26004.fits.gz"
unset RECOADD_OUT
$PY experiments/2026-10-18_candidate_recoadd/compare_reference.py $REF $OUT/reference_stampede3 || \
    echo "compare_reference reported differences (see above)"

# 2. the continuum outlier
mkdir -p $OUT/sv3 $OUT/main
cp results/2026-10-28_continuum_c3k_er_rmatrix_map/resolution_matrix.npz $OUT/sv3/
cp results/2026-10-29_continuum_main_survey_map/resolution_matrix.npz $OUT/main/
export RECOADD_SURVEY=sv3
run_target 39627757533007793 "$SPEC/spectra-sv3-bright-25598.fits.gz"
export RECOADD_SURVEY=main
run_target 39627757533007793 "$SPEC/spectra-main-bright-25598.fits.gz"
echo "DONE $(date)"
