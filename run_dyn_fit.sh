#!/bin/bash
#SBATCH -J run-full-dynesty-fit           # Job name
#SBATCH -o /home1/11006/nikhilgaruda/research/hubersed/%j.o       # Name of stdout output file
#SBATCH -e /home1/11006/nikhilgaruda/research/hubersed/%j.e       # Name of stderr error file
#SBATCH -p normal         # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 72              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 15:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=garuda@utexas.edu

# Any other commands must follow all #SBATCH directives...


module load impi
unset PYTHONPATH

cd /home1/11006/nikhilgaruda/research/hubersed/
source .venv/bin/activate
cd bin/prospector
mpirun -np 72 python run_dynesty_fit.py

