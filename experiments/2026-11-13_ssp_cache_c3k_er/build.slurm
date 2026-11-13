#!/bin/bash
# Build and check the C3K ER SSP cache (build.py) with the environment of the mock and ladder jobs.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-11-13_ssp_cache_c3k_er/build.slurm
# The cache is written to /scratch (Corral is not mounted on compute nodes); copy it to
# data/ssp_cache/ on the login node afterwards.
#SBATCH -J sspcache
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/sspcache_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
HERE=experiments/2026-11-13_ssp_cache_c3k_er
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
BUILD=c3k_er_afe_v3
CACHE=/scratch/11006/nikhilgaruda/hubersed_data/ssp_cache/$BUILD.npz
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export PYTHONPATH=$FSPS/$BUILD/fsps_site
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $HERE/build.py $HERE/build.slurm src/hubersed/sps/ssp_cache.py
$PY $HERE/build.py build $CACHE || exit 1
$PY $HERE/build.py check $CACHE || exit 1
ls -la $CACHE
echo "end $(date)"
