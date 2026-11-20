#!/bin/bash
# Build the spender Parquet datasets on /scratch. ACTION picks one step:
#   env      install the locked packages (spender v0.4, pyarrow, datasets, desitarget) into the venv
#   convert  scripts/chunks_to_parquet.py for the DESI chunks and the v2 and baseline noisy mocks
# The mock pickles must be staged first (stage.sh). Outputs that already exist stop the job.
# Pick -p at submit (test-only first):
#   ACTION=env sbatch --export=ALL -p skx-dev -t 00:30:00 experiments/2026-11-20_corral_parquet/convert.slurm
#SBATCH -J parquet
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A TG-PHY240056
#SBATCH -o /scratch/11006/nikhilgaruda/logs/parquet_%j.log

REPO=/work/11006/nikhilgaruda/ls6/research/hubersed
PY=/work2/11006/nikhilgaruda/stampede3/venvs/hubersed/bin/python
UV=$HOME/.local/bin/uv
DESI=/work/11006/nikhilgaruda/ls6/research/spender/DATA
P=/scratch/11006/nikhilgaruda/hubersed_data/pickles
D=/scratch/11006/nikhilgaruda/hubersed_data/datasets
ACTION=${ACTION:-convert}
cd $REPO
export PYTHONFAULTHANDLER=1
echo "job $SLURM_JOB_ID $ACTION on $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION), repo $(git rev-parse --short HEAD), start $(date)"
set -o pipefail

convert() {  # source dir, glob, dataset name
    if [ -e $D/$3 ]; then echo "$D/$3 exists, not overwriting"; return 1; fi
    $PY scripts/chunks_to_parquet.py $1 $D/$3 --pattern "$2" | tail -1
}

case $ACTION in
    env)
        $UV export --frozen --no-hashes --no-emit-project -o /scratch/11006/nikhilgaruda/hubersed_data/requirements.txt &&
        $UV pip install --python $PY -r /scratch/11006/nikhilgaruda/hubersed_data/requirements.txt &&
        $PY -c "import importlib.metadata as m; print({p: m.version(p) for p in ('spender', 'pyarrow', 'datasets', 'desitarget')})" ;;
    convert)
        mkdir -p $D
        convert $DESI 'DESIchunk*.pkl' desi_sv3_bright_v1 &&
        convert $P/noised_cuev2 'DESIcuev2_*.pkl' mocks_cuev2_noised &&
        convert $P/noised_cue_meanzero_wide 'DESIcueprospector1024_*.pkl' mocks_cue_noised &&
        du -sh $D/* ;;
    *) echo "unknown ACTION $ACTION"; exit 1 ;;
esac
status=$?
echo "exit $status, end $(date)"
exit $status
