#!/bin/bash
#SBATCH --job-name=PMMoTo
#SBATCH -N 1
#SBATCH -p debug_queue
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

module load openmpi_4.0.1/gcc_11.2.0
module load python/3.10.14

date

mpirun -np 8 python3 testOpen_pack1.py

date
