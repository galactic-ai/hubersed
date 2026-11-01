#!/bin/bash
# Production pocoMC run of fit.py runfork for one target on 8 spr nodes, as the 10-31 prod.slurm of
# 39633433328091196. Output results/2026-10-31_poco_quiescent/<TID>/poco_neff1024; a resubmission
# resumes from the newest state there. After prep.slurm. Submit from the repository root on stampede3:
#   TID=<TARGETID> sbatch --export=ALL -J poco_<TID> experiments/2026-10-31_poco_quiescent/prod.slurm
#SBATCH -p spr
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/quies_%x_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-10-31_poco_quiescent
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
FSPS=/work/11006/nikhilgaruda/ls6/research/fsps_builds
TLIM=42600  # s, leaves 10 min of the 12 h
[ -n "$TID" ] || { echo "TID not set"; exit 1; }
cd $REPO
export PYTHONPATH=$FSPS/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export I_MPI_PIN=0 PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID TID $TID on $SLURM_JOB_NODELIST, start $(date)"
grep -E "CommitLimit|Committed_AS|MemTotal" /proc/meminfo
sha256sum $EXP/fit.py $EXP/prod.slurm experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/fit.py \
    experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/fsps_mem.py \
    results/$(basename $EXP)/$TID/resolution_matrix.npz
(while sleep 600; do echo "mem $(date +%T) $(grep -E 'Committed_AS|MemAvailable' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; done) &
MON=$!
timeout --kill-after=60 $TLIM ibrun $PY $EXP/fit.py runfork --tid $TID --n-effective 1024 \
    --n-active 512 --save-every 1
echo "python exit $?, end $(date)"
ls -la results/$(basename $EXP)/$TID/poco_neff1024 | tail -3
kill $MON
