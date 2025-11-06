#!/bin/bash
#SBATCH --job-name=issf7_sampling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00

#module load python
source .venv/bin/activate

python run_monte_carlo.py --exec_type sampling --id a --restart --batch_size 4 --n_cores 8 --exec_path /lustre/home/jhorne/projects/moose/test/moose_test-opt