#!/bin/bash
# Why did forked pool workers die after fsps_mem.free_speclib on 2026-10-21? Runs the 10-21
# test_free_speclib.py (on /work only, uncommitted) with POOL workers (default 4), PYTHONFAULTHANDLER=1 so a worker
# that gets a fatal signal prints its Python stack, output not piped, and a process and
# Committed_AS log every 30 s. FIT_TID is the 10-21 default target, whose spender files are staged.
# A hung Pool.map is ended by timeout after TLIM s. Submit from the repository root on stampede3:
#   sbatch experiments/2026-10-31_free_speclib_debug/debug.slurm
#SBATCH -J speclibdbg
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/speclibdbg_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
D=experiments/2026-10-21_continuum_poco
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
TLIM=3000
cd $REPO
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site:$D
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export FIT_TID=39633315497510808
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $D/test_free_speclib.py $D/fsps_mem.py $D/model_setup.py experiments/2026-10-31_free_speclib_debug/debug.slurm
(while sleep 30; do
    echo "mon $(date +%T) $(grep Committed_AS /proc/meminfo | tr -s ' ')"
    ps -u $USER -o pid,ppid,stat,etime,time,rss,args --sort=pid | grep -E "test_free_speclib" | grep -v grep
done) &
MON=$!
timeout --kill-after=30 $TLIM $PY $D/test_free_speclib.py --build 10 --pool ${POOL:-4} --ncall 5
echo "python exit $?, end $(date)"
kill $MON
