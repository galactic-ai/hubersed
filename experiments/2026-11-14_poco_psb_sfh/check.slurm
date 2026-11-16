#!/bin/bash
# Setup check on one skx-dev node, after stage.sh: the DR1 resolution matrix of every target, then
# fit.py repro (model A reproduces the poco_repeats max-L lnlike of 39633332836764152), then fit.py
# check for every target and model (SSP cache loaded, key held, lnL parent == pool, one pocoMC batch).
# The scripts run from the scratch copy $RUN; results go to the repository's results folder.
# TARGETS names the target list (default targets.txt; targets_99.txt for 99 alone), REPRO=0 skips repro,
# MODELS the models (default "A B"), GIB the commit per worker (10 for the Cue models), RMAIN=1 also
# saves the main-coadd matrix.
# Submit from $RUN:
#   [TARGETS=<file> REPRO=0] sbatch --export=ALL experiments/2026-11-14_poco_psb_sfh/check.slurm
#SBATCH -J psbcheck
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/psbcheck_%j.log

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
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
LIST=$EXP/${TARGETS:-targets.txt}
sha256sum $EXP/*.py $EXP/*.slurm $EXP/targets*.txt
while read -r tid hp chunk; do
    echo "== R $tid $(date +%T)"
    $PY $EXP/resolution_matrix.py --tid $tid || exit 1
    if [ "${RMAIN:-0}" = 1 ]; then
        $PY $EXP/resolution_matrix.py --tid $tid --coadd main || exit 1
    fi
done < $LIST
if [ "${REPRO:-1}" = 1 ]; then
    echo "== repro $(date +%T)"
    timeout 1800 $PY $EXP/fit.py repro || exit 1
fi
while read -r tid hp chunk; do
    for model in ${MODELS:-A B}; do
        echo "== check $tid $model $(date +%T)"
        timeout 1200 $PY $EXP/fit.py check --tid $tid --model $model --gib-worker ${GIB:-6.5} || exit 1
    done
done < $LIST
echo "end $(date)"
