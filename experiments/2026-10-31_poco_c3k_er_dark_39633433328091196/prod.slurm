#!/bin/bash
# Production pocoMC run of fit.py runfork (option 2) on spr: one MPI rank per node builds every SSP
# slot, frees speclib and forks a pool sized from the node's commit room; rank 0 also runs pocoMC.
# Output poco_neff1024. It resumes from the newest state there. If the folder has no state and
# SEED_STATE names one (for example the last state of the skx-dev test poco_neff1024_forktest), that
# file is copied in first and its name and sha256 are logged. A job that ends before convergence
# is continued by submitting this script again. Submit from the repository root on stampede3:
#   SEED_STATE=<state file> sbatch --export=ALL experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/prod.slurm
#SBATCH -J pocoprod1031
#SBATCH -p spr
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/pocoprod1031_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
OUTDIR=$REPO/results/2026-10-31_poco_c3k_er_dark_39633433328091196/poco_neff1024
TLIM=42600  # s, leaves 10 min of the 12 h
cd $REPO
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_PIN=0 PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/fsps_mem.py $EXP/prod.slurm \
    experiments/2026-10-30_continuum_39633433328091196_dark_map/fit.py \
    experiments/2026-10-29_continuum_39633433328091196_map/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
mkdir -p $OUTDIR
if ! ls $OUTDIR/pmc_*.state >/dev/null 2>&1 && [ -n "$SEED_STATE" ]; then
    cp -p "$SEED_STATE" $OUTDIR/ || exit 1
    echo "seeded from $SEED_STATE"
    sha256sum "$SEED_STATE" $OUTDIR/$(basename "$SEED_STATE")
fi
(while sleep 600; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
timeout --kill-after=60 $TLIM ibrun $PY $EXP/fit.py runfork --n-effective 1024 --n-active 512 --save-every 1
echo "python exit $?, end $(date)"
ls -la $OUTDIR | tail -3
kill $MON
