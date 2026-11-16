#!/bin/bash
# predict.py for every target and model, in one process on one skx-dev node, after prod.slurm.
# Environment as check.slurm. TIDS and MODELS (space-separated) restrict it to those targets and models.
# Submit from $RUN:
#   [TIDS="<TARGETID> ..." MODELS="<model> ..."] sbatch --export=ALL experiments/2026-11-14_poco_psb_sfh/predict.slurm
#SBATCH -J psbpred
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/psbpred_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
RUN=/scratch/11006/nikhilgaruda/claude_runs/2026-11-14_poco_psb_sfh
EXP=$RUN/experiments/2026-11-14_poco_psb_sfh
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"  # Cue (JAX), as the emline ladder
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $EXP/fit.py $EXP/predict.py $EXP/predict.slurm
timeout 2500 taskset -c 0 $PY $EXP/predict.py ${TIDS:-} ${MODELS:+--model $MODELS}
echo "python exit $?, end $(date)"
