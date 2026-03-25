#!/bin/bash
#SBATCH --job-name=prospector_fits
#SBATCH --array=0-999%50        # max 1000, max 50 concurrent
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=garuda@utexas.edu
#SBATCH --partition=normal

module load impi
unset PYTHONPATH

cd /home1/11006/nikhilgaruda/research/hubersed/
source .venv/bin/activate
cd bin/prospector

mkdir -p logs


export SPS_HOME=/work/11006/nikhilgaruda/ls6/research/fsps/

# OFFSET lets you reuse the same script for both batches
OFFSET=${OFFSET:-0}
TASK_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

python run_fits.py \
    --task_id      $TASK_ID \
    --output_dir   ./results/ \
    --run_full \
    --cont_nseeds  5 \
    --full_nseeds  5 \
    --cont_nprod   1000 \
    --full_nprod   3000

echo "Finished task $TASK_ID at $(date)"
