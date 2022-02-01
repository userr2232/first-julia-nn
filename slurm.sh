#!/bin/bash
#SBATCH --job-name HPO_CUDA
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module load cuda
if [ "$1" = -drop ]; then 
    echo Dropping database
    mysql < db/drop.sql
fi
mysql < db/create.sql
for i in {1..64}
do
    srun python main.py root=$(pwd) action=run_study &
done
wait
