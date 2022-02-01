#!/bin/bash
#SBATCH --job-name=b_test
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

module load gcc/10.2.0
module load cuda/11.1.0

nvcc main.cu -o main 
./main