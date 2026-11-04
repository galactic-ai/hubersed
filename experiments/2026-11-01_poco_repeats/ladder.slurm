#!/bin/bash
# One fit_ladder.py action for one target and step, with that step's FSPS build. runfork is the
# production pocoMC run on 8 spr nodes as prod.slurm; check and predict run one process and are
# submitted to one skx-dev node. Submit from the repository root on stampede3:
#   TID=<TARGETID> STEP=<miles|c3k_hr|c3k_hr_afe|recoadd> ACTION=runfork sbatch --export=ALL \
#       -J lad_<STEP>_<TID> experiments/2026-11-01_poco_repeats/ladder.slurm
#   POOL=<n> sets the workers per node, TLIM=<s> the fit time (resubmit to resume), CORE=<n> the core of check or predict. TID=... STEP=... ACTION=check sbatch --export=ALL -p skx-dev -N 1 -t 01:00:00 ... ladder.slurm
#SBATCH -p spr
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/ladder_%x_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
R=/work/11006/nikhilgaruda/ls6/research
EXP=experiments/2026-11-01_poco_repeats
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
TLIM=${TLIM:-42600}  # s of fit, 42600 leaves 10 min of 12 h; set it for shorter jobs
[ -n "$TID" ] && [ -n "$STEP" ] && [ -n "$ACTION" ] || { echo "set TID, STEP and ACTION"; exit 1; }
case $STEP in
    miles) BUILD=miles_ls6; SPS_HOME=/home1/11006/nikhilgaruda/research/fsps_5; EXTRA=--no-free ;;
    c3k_hr | c3k_hr_afe) BUILD=c3k_hr_afe; SPS_HOME=$R/sps_home_hr_afe; EXTRA= ;;
    recoadd) BUILD=c3k_er_afe_v3; SPS_HOME=$R/sps_home_er; EXTRA= ;;
    *) echo "unknown STEP $STEP"; exit 1 ;;
esac
[ -n "${POOL:-}" ] && EXTRA="$EXTRA --pool $POOL"  # workers per node, else set from commit room
cd $REPO
export SPS_HOME PYTHONPATH=$R/fsps_builds/$BUILD/fsps_site
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_PIN=0 PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID TID $TID STEP $STEP ACTION $ACTION build $BUILD on $SLURM_JOB_NODELIST, start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit_ladder.py $EXP/fit.py $EXP/ladder.slurm experiments/2026-10-08_continuum_outlier/fit.py \
    experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/fit.py
OUTDIR=results/$(basename $EXP)/$TID/$STEP
if [ "$ACTION" = runfork ] && [ -s $OUTDIR/poco_neff1024/pmc_final.state ]; then
    echo "pmc_final.state exists, nothing to resume"
elif [ "$ACTION" = runfork ]; then
    (while sleep 600; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
    MON=$!
    timeout --kill-after=60 $TLIM ibrun $PY $EXP/fit_ladder.py runfork --tid $TID --step $STEP $EXTRA
    echo "python exit $?, end $(date)"
    ls -la $OUTDIR/poco_neff1024 | tail -3
    kill $MON
else
    taskset -c ${CORE:-0} $PY $EXP/fit_ladder.py $ACTION --tid $TID --step $STEP
    echo "python exit $?, end $(date)"
fi
