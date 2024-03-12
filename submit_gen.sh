#!/bin/bash
#
#SBATCH --job-name=Gen_numpy
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
source env/bin/activate
python3 gen_npy_files.py
