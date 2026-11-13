#!/bin/bash
# Mock PSB coverage (sfh.py, rebuild.py, indices.py) from a scratch run directory; mocks.py and its
# environment are those of the 2026-11-09 mock run (mocks.slurm). Copy this folder to $RUN/experiments
# and the PSB spectra to $RUN/results, stage the mock file with stage.sh, then from $RUN:
#   MODE=test sbatch -p skx-dev -t 00:45:00 experiments/2026-11-13_mock_psb_coverage/run.slurm
#   MODE=full sbatch experiments/2026-11-13_mock_psb_coverage/run.slurm
# test: rebuild 8 mocks and run the reproduction check; full: sfh.py on all mocks, rebuild 20,000.
#SBATCH -J mockpsb
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/mockpsb_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
RUN=/scratch/11006/nikhilgaruda/claude_runs/2026-11-13_mock_psb_coverage
HERE=$RUN/experiments/2026-11-13_mock_psb_coverage
OUT=$RUN/results/2026-11-13_mock_psb_coverage
MOCKS_H5=/scratch/11006/nikhilgaruda/hubersed_data/mocks_v2_500000.h5
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
BUILD=c3k_er_afe_v3
MODE=${MODE:-test}
GIB=${GIB:-8}
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export PYTHONPATH=$FSPS/$BUILD/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONFAULTHANDLER=1
mkdir -p $OUT
echo "job $SLURM_JOB_ID mode $MODE on $SLURM_JOB_NODELIST, start $(date)"
sha256sum $HERE/*.py $HERE/run.slurm $REPO/experiments/2026-11-09_mocks_v2_outliers/mocks.py
if [ "$MODE" = test ]; then
    $PY $HERE/rebuild.py $REPO/experiments/2026-11-09_mocks_v2_outliers/mocks.py $OUT/rebuilt_test.h5 --n 8 --gib-worker $GIB || exit 1
    $PY $HERE/indices.py $OUT/rebuilt_test.h5 $MOCKS_H5 $RUN/results/2026-11-13_mock_psb_coverage/psb_spectra.npz $OUT/test || exit 1
else
    $PY $HERE/sfh.py $MOCKS_H5 $OUT || exit 1
    $PY $HERE/rebuild.py $REPO/experiments/2026-11-09_mocks_v2_outliers/mocks.py $OUT/rebuilt.h5 --n 20000 --gib-worker $GIB || exit 1
    $PY $HERE/indices.py $OUT/rebuilt.h5 $MOCKS_H5 $RUN/results/2026-11-13_mock_psb_coverage/psb_spectra.npz $OUT || exit 1
fi
echo "end $(date)"
