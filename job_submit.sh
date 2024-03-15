#!/bin/bash
#
#SBATCH --job-name=TM_test
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
module load gcc/12.2
module load python/3.10
module load openssl/1.1.1w
source env/bin/activate
module list
which gcc
gcc --version
which nvcc
nvcc --version
which python3
python3 --version
python3 tmu_test.py


### CURRENT
#SBATCH --job-name=TMAudio
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

#module load slurm
#module load cuda/12.2
#module load gcc/12.2
#module load python/3.10
#module load openssl/1.1.1w
#source env/bin/activate
#ython3 main.py



### OLDER

#SBATCH --job-name=TMAudio_test
#SBATCH --partition=dgxs,dgx2,dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=el9
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err