#!/bin/bash
#
#SBATCH --job-name=TMAudio
#SBATCH --partition=dgxs,dgxh,dgx2
#SBATCH --gres=gpu:2
#SBATCH --constraint=el9
#SBATCH --mem=10G
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err



source env/bin/activate
module load cuda/12.2
python tmu_test.py

# ARGS: clauses, s, T, weights, epochs, config_file
# Vowel Set up: 2500, 5, 200, False, 100, config_main.json






#module load slurm
#module load cuda/12.2
#module load gcc/11.4
#module load python/3.10
#source env/bin/activate
#python3 main.py 2500 5 200 False 100 config_main.json

# ARGS: clauses, s, T, weights, epochs, config_file
# Vowel Set up: 2500, 5, 200, False, 100, config_main.json


#SBATCH --job-name=TMU-TEST

#SBATCH -o sbatch_logs/main.out 
#SBATCH -e sbatch_logs/main.err


#source env/bin/activate
#module load cuda/12.2
#python tmu_test.py