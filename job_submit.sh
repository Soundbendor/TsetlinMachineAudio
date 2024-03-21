#!/bin/bash
#
#SBATCH --job-name=TMAudio
#SBATCH --partition=dgxs,dgxh,dgx2,soundbendor
#SBATCH --soundbendor
#SBATCH --mem=10G
#SBATCH -o main.out
#SBATCH -e main.err

source env/bin/activate
python3 tune.py




#module load slurm
#module load cuda/12.2
#module load gcc/11.4
#module load python/3.10
#source env/bin/activate
#python3 main.py 2500 5 200 False 100 config_main.json

# ARGS: clauses, s, T, weights, epochs, config_file
# Vowel Set up: 2500, 5, 200, False, 100, config_main.json





#SBATCH --job-name=TMAudio
#SBATCH --partition=dgxs,dgxh,dgx2
#SBATCH --gres=gpu:2
#SBATCH --constraint=el9
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err



#source activate test_env/bin/activate
#module load cuda/12.2
#python tmu_test.py