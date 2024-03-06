#!/bin/bash
#
#SBATCH --job-name=TMAudio
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_output_%j.err

srun hostname
srun lscpu
srun sleep 60