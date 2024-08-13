#!/bin/bash
#SBATCH --job-name=PMMoTo
#SBATCH -N 10
#SBATCH -p 528_queue
#SBATCH -n 216
#SBATCH --ntasks-per-node=22
#SBATCH --time=00:15:00

module load openmpi_4.0.1/gcc_11.2.0
module load python/3.10.14

date

mpirun -np 216 python3 testOpen.py

date
