#!/bin/bash
# Production fit of one target and model: fit.py run --mpi on spr, one MPI rank per node, each with
# a pool forked after the SSP cache load and the speclib free; rank 0 also runs pocoMC. Resumes from
# the newest state in results/2026-11-14_poco_psb_sfh/<TID>/<MODEL>/. Python stops TLIM s into the
# job; an unfinished run then requeues itself, at most 5 times (REQUEUE=0 turns that off). As the
# emline ladder prod.slurm. After check.slurm. Use GIB=10 for the Cue models. A finished fit is then
# predicted in the same job (predict.py --model). Submit from $RUN:
#   TID=<TARGETID> MODEL=<model> sbatch --export=ALL -J psb_<TID>_<MODEL> experiments/2026-11-14_poco_psb_sfh/prod.slurm
#SBATCH -p spr
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -A TG-PHY240056
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /scratch/11006/nikhilgaruda/logs/psbprod_%x_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
RUN=/scratch/11006/nikhilgaruda/claude_runs/2026-11-14_poco_psb_sfh
EXP=$RUN/experiments/2026-11-14_poco_psb_sfh
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
GIB=${GIB:-6.5}
TLIM=${TLIM:-13800}  # s, leaves 10 min of the 4 h
RUNNO=${SLURM_RESTART_COUNT:-0}
[ -n "$TID" ] && [ -n "$MODEL" ] || { echo "TID and MODEL must be set"; exit 1; }
cd $REPO
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site:$REPO/experiments/2026-10-15_broadline_free_amplitude:$REPO/experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_PIN=0 PYTHONFAULTHANDLER=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"  # Cue (JAX), as the emline ladder
echo "job $SLURM_JOB_ID TID $TID model $MODEL run $RUNNO on $SLURM_JOB_NODELIST, repo $(git rev-parse --short HEAD), GIB $GIB, start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/prod.slurm
(while sleep 600; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
timeout --kill-after=60 $TLIM ibrun $PY $EXP/fit.py run --tid $TID --model $MODEL --mpi --gib-worker $GIB
status=$?
kill $MON
if [ $status -eq 124 ] || [ $status -eq 137 ]; then
    echo "stopped at the time limit, end $(date)"
    if [ "${REQUEUE:-1}" = 1 ] && [ "$RUNNO" -lt 5 ]; then
        echo "requeueing job $SLURM_JOB_ID (run $RUNNO of at most 5 requeues)"
        scontrol requeue $SLURM_JOB_ID || echo "requeue failed"
    fi
else
    echo "python exited with status $status, end $(date)"
    if [ $status -eq 0 ]; then
        # predict.py for this fit in one process (added 2026-11-15), as predict.slurm
        sha256sum $EXP/predict.py
        timeout 2500 taskset -c 0 $PY $EXP/predict.py $TID --model $MODEL
        echo "predict exited with status $?, end $(date)"
    fi
fi
exit $status
