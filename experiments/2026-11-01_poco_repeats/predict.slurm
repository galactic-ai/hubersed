#!/bin/bash
# Max-likelihood spectra (predict.py) of the targets given as arguments, one process per skx-dev node.
# With more targets than nodes, a node runs its targets in turn. Submit from the repository root on
# stampede3, after their prod jobs, with one node per target:
#   sbatch -N <n> -n <n> experiments/2026-11-01_poco_repeats/predict.slurm <TARGETID> [<TARGETID> ...]
#SBATCH -J predr1101
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/predr1101_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=experiments/2026-11-01_poco_repeats
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
sha256sum $EXP/predict.py $EXP/fit.py $EXP/predict.slurm
mapfile -t NODES < <(scontrol show hostnames $SLURM_JOB_NODELIST)
i=0
for tid in "$@"; do
    node=${NODES[$((i % ${#NODES[@]}))]}
    i=$((i + 1))
    (srun -N 1 -n 1 -w $node $PY $EXP/predict.py --tid $tid || echo "predict failed for $tid") &
done
wait
echo "end $(date)"
