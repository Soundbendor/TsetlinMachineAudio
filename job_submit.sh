#!/bin/bash
#
#SBATCH --job-name=TMAudio
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
module load gcc/11.4
module load python/3.10
source env/bin/activate
python3 main.py 25

# ARGS: clauses, s, T, weights, epochs, config_file
# Vowel Set up: 2500, 5, 200, False, 100, config_main.json