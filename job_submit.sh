#!/bin/bash
#
#SBATCH --job-name=TMAudio
<<<<<<< HEAD
#SBATCH --partition="dgx2,dgxh,dgxs"
#SBATCH --gres=gpu:1
=====


#SBATCH -A soundbendor
>>>>>>> ca5c2677b8210c3eeea800e409db1c836f8e4b8c
#SBATCH --constraint=el9
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
module load gcc/11.4
source env/bin/activate
python3 main.py
