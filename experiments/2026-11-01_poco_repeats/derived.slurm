#!/bin/bash
# derived.py fsps for one target, run by run in turn on one node, each run with the FSPS build and
# SPS_HOME of its fit (ladder.slurm, prod.slurm). RUNS lists the runs, N the draws per kind (all if
# unset), BUILD the SSP build helpers. Submit from the repository root on stampede3, after the
# draws_<RUN>.npz files are in results/2026-11-01_poco_repeats/<TID>/derived/:
#   TID=<TARGETID> RUNS="miles c3k_hr_afe dr1" N=50 BUILD=8 sbatch --export=ALL \
#       experiments/2026-11-01_poco_repeats/derived.slurm
#SBATCH -J derived
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/derived_%j.log

[ -n "${TID:-}" ] && [ -n "${RUNS:-}" ] || { echo "set TID and RUNS"; exit 1; }
REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
R=/work/11006/nikhilgaruda/ls6/research
EXP=experiments/2026-11-01_poco_repeats
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID TID $TID RUNS $RUNS N ${N:-all} BUILD ${BUILD:-0} on $(hostname), start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/derived.py $EXP/derived.slurm $EXP/fit_ladder.py $EXP/fit.py
for run in $RUNS; do
    case $run in
        miles) BUILDDIR=miles_ls6; SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5 ;;
        c3k_hr | c3k_hr_afe) BUILDDIR=c3k_hr_afe; SPS_HOME=$R/sps_home_hr_afe ;;
        dr1 | recoadd) BUILDDIR=c3k_er_afe_v3; SPS_HOME=$R/sps_home_er ;;
        *) echo "unknown run $run"; exit 1 ;;
    esac
    export SPS_HOME PYTHONPATH=$R/fsps_builds/$BUILDDIR/fsps_site
    echo "$(date +%T) start $run, build $BUILDDIR"
    $PY $EXP/derived.py fsps --tid $TID --run $run ${N:+--n $N} --build ${BUILD:-0}
    echo "$(date +%T) $run python exit $?"
done
echo "end $(date)"
