#!/bin/bash
# Run fit.py collect on its own skx-dev node, for example while map.slurm still runs some starts.
# collect skips starts with no output yet and lists them as missing. A later collect overwrites
# the files.
# Submit from the repository root on stampede3:
#   sbatch --export=ALL,NEB=on experiments/2026-10-28_continuum_c3k_er_rmatrix_map/collect.slurm
#SBATCH -J er1028col
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/er1028col_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-28_continuum_c3k_er_rmatrix_map
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
NEB=${NEB:-off}
NSTART=${NSTART:-7}
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
echo "job $SLURM_JOB_ID on $(hostname), nebular $NEB, start $(date)"
sha256sum $EXP/fit.py $EXP/collect.slurm
taskset -c 0 $PY $EXP/fit.py collect --n-starts $NSTART --nebular $NEB
echo "end $(date)"
