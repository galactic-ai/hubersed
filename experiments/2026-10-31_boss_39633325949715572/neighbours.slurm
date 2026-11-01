#!/bin/bash
# Unzip the targets.txt spectra files and run neighbours.py, after stage.sh, as
# 2026-10-31_screen_all/sky_extract.slurm. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_boss_39633325949715572/neighbours.slurm
#SBATCH -J bossnb1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/bossnb1031_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
HERE=experiments/2026-10-31_boss_39633325949715572
DESIPY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv_shared/bin/python
SPEC=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
UNZ=$SPEC/unzipped
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $HERE/neighbours.py $HERE/neighbours.slurm $HERE/targets.txt
mkdir -p $UNZ
while read -r tid hp; do
    f=spectra-sv3-bright-$hp.fits
    [ -s $UNZ/$f ] || gzip -dc $SPEC/$f.gz > $UNZ/$f
done < $HERE/targets.txt
echo "unzipped $(date)"
$DESIPY $HERE/neighbours.py $UNZ results/$(basename $HERE)
echo "end $(date)"
