#!/bin/bash
# fit.py collect for the SV3 and main afe = 0 MAPs on one stampede3 node, over whichever of the 7
# starts have finished. Run it again once every start is done; each run overwrites the last.
#   sbatch -p spr --export=ALL,SRC=<copy> collect.slurm
#SBATCH -J afe0col
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A TG-PHY240056
#SBATCH -o /work/11006/nikhilgaruda/ls6/research/hubersed/logs/afe0col_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
EXP=${SRC:?set SRC to the experiments copy}/2026-11-01_recoadd_afe0
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
cd $REPO
export PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er
export SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname), repo $(git rev-parse --short HEAD), start $(date)"
sha256sum $EXP/fit.py $SRC/2026-10-29_continuum_recoadd/fit.py \
    $SRC/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py
for v in sv3 main; do
    echo "--- collect $v $(date +%T)"
    RECOADD_SURVEY=$v taskset -c $([ $v = sv3 ] && echo 0 || echo 16) $PY $EXP/fit.py collect --n-starts 7 --nebular off &
    sleep 20
done
wait
echo "end $(date)"
