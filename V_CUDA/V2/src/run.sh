#!/bin/bash
#SBATCH --job-name=v1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=30:00

module load gcc/10.2.0
module load cuda/11.1.0

nvcc main.cu -o main 
./main