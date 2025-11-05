#!/bin/bash
#SBATCH --job-name=issf7_sampling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#module load python
source .venv/bin/activate

python run_monte_carlo.py --exec_type sampling --id a --restart --n_cores 4