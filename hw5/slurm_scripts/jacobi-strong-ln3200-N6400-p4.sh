#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=jacobi-strong-ln3200-N6400-p4
#SBATCH --mail-type=END
#SBATCH --mail-user=cy1505@nyu.edu
#SBATCH --output=jacobi-strong-ln3200-N6400-p4.out

module purge
module load openmpi/gnu/4.0.2
mpiexec ../jacobi-mpi 3200 20000