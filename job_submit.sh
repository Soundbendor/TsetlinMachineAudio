#!/bin/bash
#
#SBATCH --job-name=TMU-TEST
#SBATCH --partition=dgxs,dgxh,dgx2
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err


source activate env/bin/activate
module load cuda/12.2
python tmu_test.py


# ARGS: clauses, s, T, weights, epochs, config_file
# Vowel Set up: 2500, 5, 200, False, 100, config_main.json



#SBATCH --job-name=TMAudio
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err


#module load slurm
#module load cuda/12.2
#module load gcc/11.4
#module load python/3.10
#source env/bin/activate
#python3 main.py 2500 5 200 False 100 config_main.json

# ARGS: clauses, s, T, weights, epochs, config_file
# Vowel Set up: 2500, 5, 200, False, 100, config_main.json