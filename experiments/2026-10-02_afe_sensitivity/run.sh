#!/bin/bash
# Run check.py on ls6 with an alpha-enabled FSPS, without touching the project venv or SPS_HOME.
#
# FSPS: python-fsps c3k-nzinit built with -DAFE_FLAG=1 -mcmodel=medium (wheel in $AFE/wheel_medium),
# installed into $AFE/fsps_site and put first on PYTHONPATH so it shadows the venv's fsps.
# SPS_HOME: $AFE/sps_home, symlinks into fsps_5 plus the alpha C3K_HR spectra.
# Run from the repository root inside a compute-node step, for example
#   srun --overlap --jobid=<job> -A AST26017 -p development -N1 -n1 -c1 -t 01:00:00 \
#       bash experiments/2026-10-02_afe_sensitivity/run.sh
set -euo pipefail
AFE=/scratch/11006/nikhilgaruda/hubersed_tmp/afe_build
if [ ! -d "$AFE/fsps_site/fsps" ]; then
    uv pip install --python .venv/bin/python --no-deps --target "$AFE/fsps_site" \
        "$AFE"/wheel_medium/fsps-0.5.0-cp313-cp313-linux_x86_64.whl
fi
export PYTHONPATH="$AFE/fsps_site" SPS_HOME="$AFE/sps_home"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
.venv/bin/python experiments/2026-10-02_afe_sensitivity/check.py
