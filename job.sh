#!/bin/bash
#SBATCH --job-name HPO_CUDA
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --mem-per-cpu=2gb
#SBATCH --gres=gpu:1

module load cuda
for i in {1..100}
do
    srun python main.py root=$(pwd) action=run_study &
done
wait
