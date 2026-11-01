#!/bin/bash
# Unzip the sky_urls.txt files in parallel and extract the sky_targets.txt galaxies (sky_extract.py),
# after sky_stage.sh. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_screen_all/sky_extract.slurm
#SBATCH -J skyext1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/skyext1031_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
HERE=experiments/2026-10-31_screen_all
DESIPY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv_shared/bin/python
SPEC=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
UNZ=$SPEC/unzipped
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $HERE/sky_extract.py $HERE/sky_extract.slurm $HERE/sky_targets.txt
mkdir -p $UNZ
sed 's#.*/##; s#\.gz$##' $HERE/sky_urls.txt \
    | xargs -P 20 -I{} sh -c "[ -s $UNZ/{} ] || gzip -dc $SPEC/{}.gz > $UNZ/{}"
echo "unzipped $(date)"
$DESIPY $HERE/sky_extract.py $UNZ results/2026-10-31_screen_all/exposures
echo "end $(date)"
