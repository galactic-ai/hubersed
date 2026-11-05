#!/bin/bash
# Extract the single-exposure spectra of the exp_targets.txt galaxies (exp_extract.py), after
# exp_stage.sh. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_logp_vs_colour/exp_extract.slurm
#SBATCH -J expext1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/expext1031_%j.log

set -euo pipefail
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
cd $REPO
HERE=experiments/2026-10-31_logp_vs_colour
DESIPY=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv_shared/bin/python
SPEC=/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra
UNZ=$SPEC/unzipped
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $HERE/exp_extract.py $HERE/exp_extract.slurm $HERE/exp_targets.txt
# reading a .fits.gz decompresses it serially, so unzip all files at once first
mkdir -p $UNZ
sed 's#.*/##; s#\.gz$##' $HERE/exp_urls.txt \
    | xargs -P 16 -I{} sh -c "[ -s $UNZ/{} ] || gzip -dc $SPEC/{}.gz > $UNZ/{}"
echo "unzipped $(date)"
$DESIPY $HERE/exp_extract.py $UNZ results/2026-10-31_logp_vs_colour/exposures
echo "end $(date)"
