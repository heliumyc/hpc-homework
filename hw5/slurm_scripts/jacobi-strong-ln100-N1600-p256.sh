#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=jacobi-strong-ln100-N1600-p256
#SBATCH --mail-type=END
#SBATCH --mail-user=cy1505@nyu.edu
#SBATCH --output=jacobi-strong-ln100-N1600-p256.out

module purge
module load openmpi/gnu/4.0.2
mpiexec ../jacobi-mpi 100 20000