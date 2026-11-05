#!/bin/bash
# Unzip the spectra files of urls.txt and extract 39627770174637084 (extract.py), after stage.sh.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-11-05_exposures_39627770174637084/extract.slurm
#SBATCH -J exp084
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/exp084_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
HERE=experiments/2026-11-05_exposures_39627770174637084
DESIPY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv_shared/bin/python
SPEC=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
UNZ=$SPEC/unzipped
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $HERE/extract.py $HERE/extract.slurm $HERE/urls.txt
mkdir -p $UNZ
for f in $(sed 's#.*/##' $HERE/urls.txt); do
    case $f in
        *.gz) [ -s $UNZ/${f%.gz} ] || gzip -dc $SPEC/$f > $UNZ/${f%.gz} ;;
        *) ln -sf $SPEC/$f $UNZ/$f ;;
    esac
done
echo "unzipped $(date)"
$DESIPY $HERE/extract.py $UNZ results/2026-11-05_exposures_39627770174637084/extract.npz
echo "end $(date)"
