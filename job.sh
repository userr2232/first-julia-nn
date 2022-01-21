#!/bin/bash
#SBATCH --job-name HPO_CUDA
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --mem-per-cpu=1gb
#SBATCH --gres=gpu:1

module load cuda
python main.py root=$(pwd) action=run_study
