#!/bin/bash
# Exact corrected variance vs floor models on the VALIDATE units, development queue.
# Runs fluxcal_sample.py on validate_units.csv without --discard, so the iron, unpatched and
# patched fluxcalib stay in $SCRATCH/fluxcal_sample/validate/, then validate_floor.py.
# Needs results/2026-10-13_fluxcal_sample/all/ with fibers_part0.fits, fibers_part1.fits and
# f_table.fits on ls6. A resubmission skips finished units.
# Submit from this folder: sbatch validate.slurm
#SBATCH -J fluxcal_validate
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A AST25022
#SBATCH -o fluxcal_validate_%j.out

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
D=experiments/2026-10-13_fluxcal_sample
ALL=results/2026-10-13_fluxcal_sample/all
for f in $ALL/fibers_part0.fits $ALL/fibers_part1.fits $ALL/f_table.fits $D/validate_units.csv; do
    [ -f $f ] || { echo "missing $f"; exit 1; }
done

export DESIMODEL=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/desimodel
PY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv/bin/python
$PY $D/fluxcal_sample.py $D/validate_units.csv validate --s3-streams 50 --units-parallel 16 || exit 1

# the venv's python directly, not uv run, as in the other slurm scripts. Its python finds no CA
# certificates on ls6 (https fails with CERTIFICATE_VERIFY_FAILED), so point it at the system bundle.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
$REPO/.venv/bin/python $D/validate_floor.py --unit-root $SCRATCH/fluxcal_sample/validate \
    --streams 32 --workers 32
