#!/bin/bash
# Unzip the urls.txt files in parallel and extract the targets.txt galaxies (extract.py),
# after stage.sh. Submit from the repository root on stampede3:
#   sbatch experiments/2026-11-02_exposure_checks/extract.slurm
#SBATCH -J expchk
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/expchk_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
HERE=experiments/2026-11-02_exposure_checks
DESIPY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv_shared/bin/python
SPEC=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
UNZ=$SPEC/unzipped
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $HERE/extract.py $HERE/extract.slurm $HERE/targets.txt
mkdir -p $UNZ
sed 's#.*/##; s#\.gz$##' $HERE/urls.txt \
    | xargs -P 20 -I{} sh -c "[ -s $UNZ/{} ] || gzip -dc $SPEC/{}.gz > $UNZ/{}"
echo "unzipped $(date)"
$DESIPY $HERE/extract.py $UNZ results/2026-11-02_exposure_checks/exposures
echo "end $(date)"
