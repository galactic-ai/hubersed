#!/bin/bash
# Recoadd four line outliers with the patched flux-calibration ivar, after checking the reference.
#
# 1. Back up the reference outputs of the 2026-10-09 scripts (only if no backup exists yet).
# 2. Rerun coadd_rows, fluxcal_rerun, ivar_check and ivar_recompute for 39627770174637084.
# 3. compare_reference.py must find every output unchanged, or the job stops.
# 4. Run the four scripts for each candidate of targets.py.
#
# Submit from the folder that holds this script and compare_reference.py:
#   cd experiments/2026-10-18_candidate_recoadd && sbatch recoadd.slurm
#SBATCH -J recoadd
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A AST25022
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/recoadd_%j.log

set -euo pipefail
HERE=$SLURM_SUBMIT_DIR
cd /work/11006/nikhilgaruda/ls6/research/hubersed
E=experiments/2026-10-09_nautilus_c3k_er_broadline
PY=.venv/bin/python
DESI=/work/11006/nikhilgaruda/ls6/research/desi_env_0562
SPEC=/corral/utexas/AST25023/nikhilgaruda/hubersed/data/desi_iron_spectra
REF=results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084
BACKUP=$REF/backup_before_59b386c

echo "host $(hostname), repo $(cat .git/refs/heads/exp/14-17-c3k-er), start $(date)"
free -g
sha256sum "$HERE/recoadd.slurm" "$HERE/compare_reference.py" $E/coadd_rows.py $E/fluxcal_rerun.py \
    $E/ivar_check.py $E/ivar_recompute.py $E/targets.py

run_target() {
    local tid=$1 hp=$2
    echo "=== $tid (healpix $hp) $(date)"
    $PY $E/coadd_rows.py "$SPEC/spectra-sv3-bright-$hp.fits.gz" --tid "$tid"
    DESIMODEL=$DESI/desimodel $DESI/venv/bin/python $E/fluxcal_rerun.py --tid "$tid"
    $PY $E/ivar_check.py --tid "$tid"
    $PY $E/ivar_recompute.py --tid "$tid"
}

if [ ! -d "$BACKUP" ]; then
    mkdir "$BACKUP"
    cp -a $REF/coadd_rows.txt $REF/resolution_matrix_rows.npz $REF/fluxcal_rerun.txt \
        $REF/fluxcalib_rerun $REF/ivar_check.txt $REF/ivar_recompute.txt \
        $REF/resolution_matrix_rows_ivar.npz "$BACKUP"/
fi

run_target 39627770174637084 26004
$PY "$HERE/compare_reference.py" "$BACKUP" "$REF"

run_target 39628357133926597 28115
run_target 39633278973513279 11404
run_target 39632961393395307 10512
run_target 39633001071509599 10518
echo "DONE $(date)"
