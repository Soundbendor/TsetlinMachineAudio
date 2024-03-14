#!/bin/bash
#
#SBATCH --job-name=TMAudio
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
module load gcc/12.2
module load python/3.10
module load openssl/1.1.1w
source env/bin/activate
python3 main.py


### OLD

#SBATCH --job-name=TMAudio_test
#SBATCH --partition=dgxs,dgx2,dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=el9
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err