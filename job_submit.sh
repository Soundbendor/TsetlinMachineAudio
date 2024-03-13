#!/bin/bash
#
#SBATCH --job-name=TMAudio_test
#SBATCH --partition=dgxs,dgx2
#SBATCH --gres=gpu:1
#SBATCH --constraint=el9
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
module load gcc/13.2
module load python/3.10
module load openssl/1.1.1w
source env/bin/activate
python3 tmu_test.py
