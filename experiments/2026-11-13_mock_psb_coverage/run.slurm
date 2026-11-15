#!/bin/bash
# Mock PSB coverage (sfh.py, rebuild.py, indices.py) from a scratch run directory; mocks.py and its
# environment are those of the 2026-11-09 mock run (mocks.slurm). Copy this folder to $RUN/experiments
# and the PSB spectra to $RUN/results, stage the mock file with stage.sh, then from $RUN:
#   MODE=test sbatch -p skx-dev -t 00:45:00 experiments/2026-11-13_mock_psb_coverage/run.slurm
#   MODE=full sbatch -t 06:00:00 experiments/2026-11-13_mock_psb_coverage/run.slurm
#   MODE=part PART=k NPARTS=K sbatch experiments/2026-11-13_mock_psb_coverage/run.slurm  (k = 0..K-1)
#   MODE=finish sbatch -t 00:30:00 --dependency=afterok:<part job ids> .../run.slurm
# test: rebuild 8 mocks and the reproduction check; full: sfh.py, rebuild 20,000, indices.py on one
# node; part: rebuild slice k of K; finish: sfh.py and indices.py over every part. The SSP cache
# (2026-11-13_ssp_cache_c3k_er) is loaded instead of building the SSPs when CACHE exists.
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
CACHE=/scratch/11006/nikhilgaruda/hubersed_data/ssp_cache/c3k_er_afe_v3.npz
PSB=$OUT/psb_spectra.npz
MOCKS_PY=$REPO/experiments/2026-11-09_mocks_v2_outliers/mocks.py
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
BUILD=c3k_er_afe_v3
MODE=${MODE:-test}
GIB=${GIB:-6.5}
PART=${PART:-0}
NPARTS=${NPARTS:-1}
CACHE_ARG=""
[ -s "$CACHE" ] && CACHE_ARG="--cache $CACHE"
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export PYTHONPATH=$FSPS/$BUILD/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONFAULTHANDLER=1
mkdir -p $OUT
echo "job $SLURM_JOB_ID mode $MODE part $PART/$NPARTS gib $GIB cache '$CACHE_ARG' on $SLURM_JOB_NODELIST, start $(date)"
sha256sum $HERE/*.py $HERE/run.slurm $MOCKS_PY
case $MODE in
    test)
        $PY $HERE/rebuild.py $MOCKS_PY $OUT/rebuilt_test.h5 --n 8 --gib-worker $GIB $CACHE_ARG || exit 1
        $PY $HERE/indices.py $OUT/rebuilt_test.h5 $MOCKS_H5 $PSB $OUT/test || exit 1 ;;
    full)
        $PY $HERE/sfh.py $MOCKS_H5 $OUT || exit 1
        $PY $HERE/rebuild.py $MOCKS_PY $OUT/rebuilt.h5 --n 20000 --gib-worker $GIB $CACHE_ARG || exit 1
        $PY $HERE/indices.py $OUT/rebuilt.h5 $MOCKS_H5 $PSB $OUT || exit 1 ;;
    part)
        $PY $HERE/rebuild.py $MOCKS_PY $OUT/rebuilt_part$PART.h5 --n 20000 --part $PART \
            --nparts $NPARTS --gib-worker $GIB $CACHE_ARG || exit 1 ;;
    finish)
        $PY $HERE/sfh.py $MOCKS_H5 $OUT || exit 1
        $PY $HERE/indices.py "$OUT/rebuilt_part*.h5" $MOCKS_H5 $PSB $OUT || exit 1 ;;
    *) echo "unknown MODE $MODE"; exit 1 ;;
esac
echo "end $(date)"
