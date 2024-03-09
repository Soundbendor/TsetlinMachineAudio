#!/bin/bash
#
#SBATCH --job-name=TMAudio
#SBATCH --partition="dgx2,dgxh,dgxs"
#SBATCH --gres=gpu:1
#SBATCH --constraint=el9
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
module load gcc/11.4
source env/bin/activate
python3 main.py
