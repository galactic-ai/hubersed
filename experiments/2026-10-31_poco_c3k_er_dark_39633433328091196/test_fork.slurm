#!/bin/bash
# skx-dev test of fit.py runfork (option 2): one MPI rank per node builds every SSP slot, frees
# speclib (fsps_mem.py) and forks a pool sized from the node's commit room; rank 0 also runs pocoMC.
# Two nodes, so blocks go over MPI. Run 1 for RUN1 s, killed by timeout, then a resume run for RUN2 s;
# each run builds the slots again (~15 min). Production sampler settings, written to
# poco_neff1024_forktest. I_MPI_PIN=0 leaves the affinity to fit.py, which pins each worker to a core.
# Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/test_fork.slurm
#SBATCH -J pocofork1031
#SBATCH -p skx-dev
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/pocofork1031_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
RUN1=2700  # s of the first run
RUN2=1800  # s of the resume run
cd $REPO
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_PIN=0 PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/fsps_mem.py $EXP/test_fork.slurm \
    experiments/2026-10-30_continuum_39633433328091196_dark_map/fit.py \
    experiments/2026-10-29_continuum_39633433328091196_map/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
(while sleep 300; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!

OPTS="--n-effective 1024 --n-active 512 --save-every 1 --tag _forktest"
echo "run 1, $RUN1 s, start $(date)"
timeout --kill-after=60 $RUN1 ibrun $PY $EXP/fit.py runfork $OPTS
echo "run 1 exit $?, end $(date)"
echo "run 2 (resume), $RUN2 s, start $(date)"
timeout --kill-after=60 $RUN2 ibrun $PY $EXP/fit.py runfork $OPTS
echo "run 2 exit $?, end $(date)"
ls -la $REPO/results/2026-10-31_poco_c3k_er_dark_39633433328091196/poco_neff1024_forktest | tail -5
kill $MON
echo "end $(date)"
