#!/bin/bash
# skx-dev test of fit.py run (one sampler rank and NWORK worker ranks, no fork) before the
# production job. Steps: check in one process; an MPI hello through ibrun; run for RUN1 s, killed
# by timeout; then a resume run for RUN2 s. Sampler settings are the production ones, written to
# poco_neff1024_devtest so production never resumes from them. Node Committed_AS is logged every
# 5 min. Env as 2026-10-30_nodust_miles_39633433328091196_dark/map.slurm (C3K ER). Submit from the
# repository root on stampede3:
#   sbatch experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/test.slurm
#SBATCH -J pocodev1031
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 9
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/pocodev1031_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
RUN1=3600  # s of the first run
RUN2=900   # s of the resume run
cd $REPO
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/test.slurm experiments/2026-10-30_continuum_39633433328091196_dark_map/fit.py \
    experiments/2026-10-29_continuum_39633433328091196_map/fit.py \
    experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
(while sleep 300; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!

taskset -c 0 $PY $EXP/fit.py check --n-calls 3 || { kill $MON; exit 1; }
ibrun -n 2 $PY -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size(), MPI.Get_library_version().splitlines()[0])" || { kill $MON; exit 1; }

OPTS="--n-effective 1024 --n-active 512 --save-every 1 --init-width 2 --tag _devtest"
echo "run 1, $RUN1 s, start $(date)"
timeout --kill-after=60 $RUN1 ibrun $PY $EXP/fit.py run $OPTS
echo "run 1 exit $?, end $(date)"
grep -E "Committed_AS" /proc/meminfo
echo "run 2 (resume), $RUN2 s, start $(date)"
timeout --kill-after=60 $RUN2 ibrun $PY $EXP/fit.py run $OPTS
echo "run 2 exit $?, end $(date)"
ls -la $REPO/results/2026-10-31_poco_c3k_er_dark_39633433328091196/poco_neff1024_devtest | tail -5
kill $MON
echo "end $(date)"
